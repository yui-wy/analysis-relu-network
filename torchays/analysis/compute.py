import time
import multiprocessing as mp
from collections import deque
from typing import Callable, Deque, Iterable, List, Tuple, TypeAlias

import numpy as np
import torch

from ..nn import Module
from ..nn.modules import BIAS_GRAPH, WEIGHT_GRAPH
from ..utils import get_logger
from .handler import BaseHandler, DefaultHandler
from .model import Model
from .optimization import (
    constraint,
    fun_bound,
    jac_bound,
    jac_linear,
    jac_radius,
    jac_radius_constraint,
    jac_square,
    linear,
    linear_error,
    minimize,
    radius,
    radius_constraint,
    square,
)
from .util import get_regions, find_projection, vertify


class RegionSet:
    def __init__(self) -> None:
        self._regions: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = deque()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._regions.popleft()
        except:
            raise StopIteration

    def __len__(self) -> int:
        return len(self._regions)

    def register(
        self,
        functions: torch.Tensor,
        region: torch.Tensor,
        inner_point: torch.Tensor,
        depth: int,
    ):
        self._regions.append((functions, region, inner_point, depth))

    def registers(self, regions: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
        self._regions.extend(regions)

    def __str__(self):
        return self._regions.__str__()


class WapperRegion:
    """
    Get the region(sign) of the function list.
    *  1 : f(x) > 0
    * -1 : f(x) <= 0
    """

    def __init__(self, regions: List[torch.Tensor] = None):
        self.filter = torch.Tensor().type(torch.int8)
        self.regions: Deque[torch.Tensor] = deque()
        if regions is not None:
            self.registers(regions)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        try:
            region = self.regions.popleft()
            while self._check(region):
                region = self.regions.popleft()
            return region
        except:
            raise StopIteration

    def _update_list(self):
        self.output: torch.Tensor = self.regions.popleft()

    def _check(self, region: torch.Tensor):
        try:
            a = ((self.filter.abs() * region) - self.filter).abs().sum(dim=1)
            return 0 in a
        except:
            return False

    def update_filter(self, region: torch.Tensor):
        if self._check(region):
            return
        self.filter = torch.cat([self.filter, region.unsqueeze(0)], dim=0)

    def register(self, region: torch.Tensor):
        if not self._check(region):
            self.regions.append(region)

    def registers(self, regions: List[torch.Tensor] | torch.Tensor):
        for region in regions:
            self.register(region)


def _log_time(fun_name: str, indent: int = 0, is_log: bool = True):
    def wapper(fun: Callable):
        def new_func(self, *args, **kwargs):
            if self.is_log_time:
                return result
            start = time.time()
            result = fun(self, *args, **kwargs)
            t = time.time() - start
            self.logger.info(f"{' '*indent}[{fun_name}] took time: {t}s.")
            return result

        if is_log:
            return new_func
        return fun

    return wapper


class ReLUNets:
    """
    ReLUNets needs to ensure that the net has the function:
        >>> def forward_layer(*args, depth=depth):
        >>>     ''' layer is a "int" before every ReLU module. "Layer" can get the layer weight and bias graph.'''
        >>>     if depth == 1:
        >>>         return output

    args:
        device: torch.device
            GPU or CPU to get the graph from the network;
        logger: def info(...)
            print the information (Default: print in console)(logger.info(...)).
    """

    def __init__(
        self,
        workers: int = 1,
        device=torch.device("cpu"),
        logger=None,
        is_log_time: bool = False,
    ):
        self.workers = workers if workers > 0 else 1
        self.device = device
        self.one = torch.ones(1).double()
        self.logger = logger or get_logger("AnalysisReLUNetUtils-Console")
        self.is_log_time = is_log_time

    def _find_functions(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Get the list of the linear function before ReLU layer.
        """
        x = x.float().to(self.device)
        x = x.reshape(*self.input_size).unsqueeze(dim=0)
        with torch.no_grad():
            _, graph = self.net.forward_layer(x, depth=depth)
            # (1, *output.size(), *input.size())
            weight_graph, bias_graph = graph[WEIGHT_GRAPH], graph[BIAS_GRAPH]
            # (output.num, input.num)
            weight_graph = weight_graph.reshape(-1, x.size()[1:].numel())
            # (output.num, 1)
            bias_graph = bias_graph.reshape(-1, 1)
            # (output.num, input.num + 1)
        return torch.cat([weight_graph, bias_graph], dim=1).cpu()

    @_log_time("find child region inner point", 2, True)
    def _find_region_inner_point(self, functions: torch.Tensor, init_point: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Calculate the max radius of the insphere in the region to make the radius less than the distance of the insphere center to the all functions
        which can express a liner region.

        maximizing the following function.
        * max_{x,r} r
        * s.t.  (AX+B-r||A|| >= 0)

        To minimize:
        * min_{x,r} (-r)
        * s.t.  (Ax+B-r||A|| >= 0)
        *       r > 0
        """
        functions, init_point = functions.numpy(), init_point.numpy()
        r = np.random.uniform(0, 1.0)
        x = np.append(init_point, r)
        norm = np.linalg.norm(functions[:, :-1], ord=2, axis=1)
        constraints = [
            constraint(
                radius_constraint(functions[i], norm[i]),
                jac_radius_constraint(functions[i], norm[i]),
            )
            for i in range(functions.shape[0])
        ]
        constraints.extend(self.constraints)
        _, result = minimize(radius(), x, constraints, jac_radius(functions[0]), tol=1e-10)
        inner_point, r = result[:-1], result[-1]
        result = np.matmul(functions[:, :-1], inner_point.T) + functions[:, -1]
        result = np.where(result >= -1e-16, 1, 0)
        if (0 in result) or r < 1e-16:
            return None
        return torch.from_numpy(inner_point).float()

    @_log_time("optimize child region", 2, True)
    def _optimize_region(
        self,
        funcs: torch.Tensor,
        region: torch.Tensor,
        constraint_funcs: torch.Tensor,
        c_region: torch.Tensor,
        c_inner_point: torch.Tensor,
    ):
        # Get the bound hyperplanes which can filter the same regions, and find the neighbor regions.
        c_bound_funcs, c_bound_region, neighbor_regions = [], [], []
        # 优化方法前置
        constraint_funcs, optim_x = constraint_funcs.numpy(), c_inner_point.double().numpy()
        filter_region = torch.zeros_like(c_region).type(torch.int8)
        constraints = [
            constraint(
                linear_error(constraint_funcs[i]),
                jac_linear(constraint_funcs[i]),
            )
            for i in range(constraint_funcs.shape[0])
        ]
        constraints.extend(self.constraints)
        # 投影方法前置
        p_points = find_projection(c_inner_point, funcs)
        # print("-----------=-=-=-----------------")
        for i in range(constraint_funcs.shape[0]):
            if not vertify(p_points[i], funcs, region):
                # 没有验证成功，使用优化方法
                constraints[i]["fun"] = linear(constraint_funcs[i])
                err, _ = minimize(square(constraint_funcs[i]), optim_x, constraints, jac_square(constraint_funcs[i]))
                if err > 1e-15:
                    continue
            c_bound_funcs.append(funcs[i])
            c_bound_region.append(region[i])
            # Find the neighbor regions.
            if i < c_region.shape[0]:
                neighbor_region = c_region.clone()
                neighbor_region[i] = -region[i]
                neighbor_regions.append(neighbor_region)
                filter_region[i] = region[i]
        c_bound_funcs = torch.stack(c_bound_funcs)
        c_bound_region = torch.tensor(c_bound_region, dtype=torch.int8)
        # print(c_bound_funcs.shape[0])
        # print("-----------=-=-=-----------------")
        return c_bound_funcs, c_bound_region, filter_region, neighbor_regions

    def _optimize_child_region(
        self,
        c_funcs: torch.Tensor,
        c_region: torch.Tensor,
        p_funcs: torch.Tensor,
        p_region: torch.Tensor,
        inner_point: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        1. 获取区域是否存在; 2. 区域存在, 获取其中一个点; 3. 获取区域的相邻区域;
        """
        funcs, region = torch.cat([c_funcs, p_funcs], dim=0), torch.cat([c_region, p_region], dim=0)
        constraint_funcs = region.view(-1, 1) * funcs
        # 1. Check whether the region exists, the inner point will be obtained if existed.
        c_inner_point: torch.Tensor | None = self._find_region_inner_point(constraint_funcs, inner_point)
        if c_inner_point is None:
            return None, [], [], None, []
        # 2. find the least edges functions to express this region and obtain neighbor regions.
        optimize_child_region_result = self._optimize_region(funcs, region, constraint_funcs, c_region, c_inner_point)
        return c_inner_point, *optimize_child_region_result

    @_log_time("find intersect", 2)
    def _find_intersect(
        self,
        p_funcs: torch.Tensor,
        p_regions: torch.Tensor,
        x: torch.Tensor,
        funcs: torch.Tensor,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        存在一个f(x)与f(x0)符号相反或者f(x)=0

        计算神经元方程构成的区域边界是否穿越父区域（目标函数有至少一点在区域内）:

        *    min(func(x)^2);
        *    s.t. pFunc(x) >= 0
        """
        intersect_funs, inner_points = [], []
        # 优化方法前置
        optim_funcs, optim_x = funcs.numpy(), x.double().numpy()
        pn_funcs = (p_regions.view(-1, 1) * p_funcs).numpy()
        constraints = [
            constraint(
                linear(pn_funcs[i]),
                jac_linear(pn_funcs[i]),
            )
            for i in range(pn_funcs.shape[0])
        ]
        constraints.extend(self.constraints)
        # Is the linear function though the region.
        # 投影点前置
        p_points = find_projection(x, funcs)
        for i in range(funcs.size(0)):
            if torch.isnan(p_points[i]).all():
                continue
            if vertify(p_points[i], p_funcs, p_regions):
                # 判断投影点
                inner_points.append(p_points[i])
            else:
                # 优化方法
                err, result = minimize(square(optim_funcs[i]), optim_x, constraints, jac_square(optim_funcs[i]))
                if err > 1e-16:
                    continue
                inner_points.append(torch.from_numpy(result).float())
            intersect_funs.append(funcs[i])
        if len(intersect_funs) == 0:
            return None, None
        intersect_funs = torch.stack(intersect_funs, dim=0)
        inner_points = torch.stack(inner_points, dim=0)
        return intersect_funs, inner_points

    @_log_time("handler regions", 2)
    def _handler_regions(
        self,
        p_funcs: torch.Tensor,
        p_region: torch.Tensor,
        p_inner_point: torch.Tensor,
        c_funcs: torch.Tensor,
        depth: int,
    ):
        """
        验证hyperplane arrangement的存在性(子平面, 是否在父区域上具有交集)
        """
        next_regions = RegionSet()
        counts, n_regions = 0, 0
        intersect_funcs, intersect_funcs_points = self._find_intersect(p_funcs, p_region, p_inner_point, c_funcs)
        if intersect_funcs is None:
            n_regions = 1
            counts += self._nn_region_counts(p_funcs, p_region, p_inner_point, depth, next_regions.register)
        else:
            c_regions = get_regions(intersect_funcs_points, intersect_funcs)
            # Register some regions in WapperRegion for iterate.
            layer_regions = WapperRegion(c_regions)
            for c_region in layer_regions:
                # Check and get the child region. Then, the neighbor regions will be found.
                c_inner_point, c_bound_funcs, c_bound_region, filter_region, neighbor_regions = self._optimize_child_region(intersect_funcs, c_region, p_funcs, p_region, p_inner_point)
                if c_inner_point is None:
                    continue
                # Add the region to prevent counting again.
                layer_regions.update_filter(filter_region)
                # Register new regions for iterate.
                layer_regions.registers(neighbor_regions)
                # Count the number of the regions in the current parent region.
                n_regions += 1
                # Handle the child region.
                counts += self._nn_region_counts(c_bound_funcs, c_bound_region, c_inner_point, depth, next_regions.register)
        # Collect the information of the current parent region including region functions, child functions, intersect functions and number of the child regions.
        self.handler.inner_hyperplanes_handler(p_funcs, p_region, c_funcs, intersect_funcs, n_regions, depth)
        return counts, next_regions

    def _nn_region_counts(
        self,
        funcs: torch.Tensor,
        region: torch.Tensor,
        inner_point: torch.Tensor,
        depth: int,
        register: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], None],
    ) -> int:
        if depth == self.last_depth:
            # If current layer is the last layer, the region is the in count.
            # Collect the information of the final region.
            self.handler.region_handler(funcs, region, inner_point)
            return 1
        # If not the last layer, the region will be parent region in the next layer.
        register(funcs, region, inner_point, depth + 1)
        return 0

    def _scale_functions(self, functions: torch.Tensor) -> torch.Tensor:
        def _scale_function(function: torch.Tensor):
            v, _ = function.abs().max(dim=0)
            while v < 1:
                function *= 10
                v *= 10
            return function

        for i in range(functions.shape[0]):
            functions[i] = _scale_function(functions[i])
        return functions

    def _functions(self, x: torch.Tensor, depth: int):
        # Get the list of the linear functions from DNN.
        functions = self._find_functions(x, depth)
        # Scale the functions.
        return self._scale_functions(functions)

    def _single_get_counts(self, region_set: RegionSet) -> int:
        counts: int = 0
        current_depth: int = -1
        for p_funcs, p_region, p_inner_point, depth in region_set:
            if depth != current_depth:
                current_depth = depth
                self.logger.info(f"Depth: {current_depth+1}")
            c_funcs = self._functions(p_inner_point, depth)
            self.logger.info(f"Start to get regions. Depth: {depth+1}, ")
            # Get the number of the parent region.
            count, next_regions = self._handler_regions(p_funcs, p_region, p_inner_point, c_funcs, depth)
            region_set.registers(next_regions)
            counts += count
        return counts

    def _work(
        self,
        q: mp.SimpleQueue,
        p_funcs: torch.Tensor,
        p_region: torch.Tensor,
        p_inner_point: torch.Tensor,
        c_funcs: torch.Tensor,
        depth: int,
    ):
        self.logger.info(f"Start to get regions. Depth: {depth+1}. ")
        # Get the number of the parent region.
        print(f"Start to get regions. Depth: {depth+1}. ")
        count, next_regions = self._handler_regions(p_funcs, p_region, p_inner_point, c_funcs, depth)
        q.put((count, next_regions))
        print(f"End processing")

    def _multiprocess_get_counts(self, region_set: RegionSet) -> int:
        counts: int = 0
        current_depth: int = -1
        q = mp.Manager().Queue(self.workers)
        jobs: List[mp.Process] = list()
        for p_funcs, p_region, p_inner_point, depth in region_set:
            if depth != current_depth:
                current_depth = depth
                self.logger.info(f"Depth: {current_depth+1}")
            c_funcs = self._functions(p_inner_point, depth)
            p = mp.Process(target=self._work, args=(q, p_funcs, p_region, p_inner_point, c_funcs, depth))
            jobs.append(p)
            p.start()
            print(f"-----process start! {p.pid}------")
            if len(jobs) >= self.workers or len(region_set) == 0:
                # 当jobs满的时候或者没有后续的时候, 需要等待jobs结束
                print("wait process end.")
                while True:
                    idx: List[int] = list()
                    for i in range(len(jobs)):
                        job = jobs[i]
                        if job.is_alive():
                            continue
                        # 若job结束
                        job.close()
                        idx.append(i)
                        count, next_regions = q.get()
                        counts += count
                        region_set.registers(next_regions)
                    if len(idx) == 0:
                        continue
                    for i in idx:
                        jobs.pop(i)
                    if len(region_set) > 0:
                        # 若jobs空出来,并且继续循环, 则break
                        break
                    if len(jobs) == 0:
                        # 无循环, 等待全部结束break
                        break
                print(f"process num {len(jobs)}")
        return counts

    @_log_time("Region counts")
    def _get_counts(self, region_set: RegionSet) -> int:
        # if self.workers > 1:
        #     return self._multiprocess_get_counts(region_set)
        return self._single_get_counts(region_set)

    def get_region_counts(
        self,
        net: Model,
        bounds: float | int | Tuple[float, float] | Tuple[Tuple[float, float]] = 1.0,
        depth: int = -1,
        input_size: tuple = (2,),
        handler: BaseHandler = DefaultHandler(),
    ):
        """
        目前只支持方形的输入空间画图，需要修改。
        """
        assert isinstance(net, Module), "the type of net must be \"BaseModule\"."
        assert depth >= 0, "countLayers must >= 0."
        # Initialize the settings
        self.net = net.to(self.device).graph()
        self.last_depth = depth
        self.input_size = input_size
        self.handler = handler
        self._init_constraints()
        # Initialize the parameters
        dim = torch.Size(input_size).numel()
        p_funcs, p_region, p_inner_point = _generate_bound_regions(bounds, dim)
        # Initialize the region set.
        region_set = RegionSet()
        region_set.register(p_funcs, p_region, p_inner_point, 0)
        # Start to get the NN regions.
        self.logger.info("Start Get region number.")
        counts = self._get_counts(region_set)
        self.logger.info(f"Region counts: {counts}.")
        return counts

    def _init_constraints(self):
        self.constraints = [constraint(fun_bound, jac_bound)]


BoundTypes: TypeAlias = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _generate_bound_regions(
    bounds: float | int | Tuple[float, float] | List[Tuple[float, float]],
    dim: int = 2,
) -> BoundTypes:
    assert dim > 0, f"dim must be more than 0. [dim = {dim}]"
    handler = _number_bound
    if isinstance(bounds, tuple):
        if isinstance(bounds[0], tuple):
            handler = _tuple_bounds
        else:
            handler = _tuple_bound
    return handler(bounds, dim)


def _tuple_bound(bounds: Tuple[float, float], dim: int = 2) -> BoundTypes:
    low, upper = _bound(bounds)
    inner_point = torch.ones(dim) * (low + upper) / 2
    lows, uppers = torch.zeros(dim) + low, torch.zeros(dim) + upper
    return *_bound_regions(lows, uppers, dim), inner_point


def _tuple_bounds(bounds: Tuple[Tuple[float, float]], dim: int = 2) -> BoundTypes:
    assert len(bounds) == dim, f"length of the bounds must match the dim [dim = {dim}]."
    inner_point = torch.zeros(dim)
    lows, uppers = torch.zeros(dim), torch.zeros(dim)
    for i in range(dim):
        low, upper = _bound(bounds[i])
        lows[i], uppers[i] = low, upper
        inner_point[i] = (low + upper) / 2
    return *_bound_regions(lows, uppers, dim), inner_point


def _number_bound(bound: float | int, dim: int = 2) -> BoundTypes:
    assert len(bound) == 2, f"length of the bounds must be 2. len(bounds) = {len(bound)}."
    bounds = (-bound, bound)
    return _tuple_bound(bounds, dim)


def _bound(bound: Tuple[float, float]) -> Tuple[float, float]:
    low, upper = bound
    if upper < low:
        low, upper = upper, low
    return low, upper


def _bound_regions(lows: torch.Tensor, uppers: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    low_bound_functions = torch.cat([torch.eye(dim), -lows.unsqueeze(1)], dim=1)
    upper_bound_functions = torch.cat([-torch.eye(dim), uppers.unsqueeze(1)], dim=1)
    funcs = torch.cat([low_bound_functions, upper_bound_functions], dim=0)
    region = torch.ones(dim * 2, dtype=torch.int8)
    return funcs, region
