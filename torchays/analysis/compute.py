import time
from collections import deque
from typing import Callable, Deque, List, Tuple, TypeAlias

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
from .util import get_regions


class RegionSet:
    def __init__(
        self,
        inner_point: torch.Tensor,
        functions: torch.Tensor,
        region: torch.Tensor,
        depth: int = 0,
    ) -> None:
        self._regions: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = deque()
        self.register(inner_point, functions, region, depth)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._regions.popleft()
        except:
            raise StopIteration

    def register(
        self,
        inner_point: torch.Tensor,
        functions: torch.Tensor,
        region: torch.Tensor,
        depth: int,
    ):
        self._regions.append((inner_point, functions, region, depth))


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
            self.regist_regions(regions)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        try:
            region = self.regions.popleft()
            while self._check_region(region):
                region = self.regions.popleft()
            return region
        except:
            raise StopIteration

    def _update_list(self):
        self.output: torch.Tensor = self.regions.popleft()

    def _check_region(self, region: torch.Tensor):
        try:
            a = ((self.filter.abs() * region) - self.filter).abs().sum(dim=1)
            return 0 in a
        except:
            return False

    def update_filter(self, region: torch.Tensor):
        if self._check_region(region):
            return
        self.filter = torch.cat([self.filter, region.unsqueeze(0)], dim=0)

    def regist_region(self, region: torch.Tensor):
        if not self._check_region(region):
            self.regions.append(region)

    def regist_regions(self, regions: List[torch.Tensor] | torch.Tensor):
        for region in regions:
            self.regist_region(region)


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
        device=torch.device("cpu"),
        logger=None,
        is_log_time: bool = False,
    ):
        self.device = device
        self.one = torch.ones(1).double()
        self.logger = logger or get_logger("AnalysisReLUNetUtils-Console")
        self.is_log_time = is_log_time

    def _find_functions(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        """
        Get the list of the linear function before ReLU layer.
        """
        x = x.to(self.device)
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
        * s.t.  (AX+B-r||A|| >= 0)
        """
        # ||A||
        functions, init_point = functions.numpy(), init_point.numpy()
        r = np.random.uniform(0, 1.0)
        x = np.append(init_point, r)
        norm_A = np.linalg.norm(functions[:, :-1], ord=2, axis=1)
        constraints = [
            constraint(
                radius_constraint(functions[i], norm_A[i]),
                jac_radius_constraint(functions[i], norm_A[i]),
            )
            for i in range(functions.shape[0])
        ]
        constraints.extend(self.constraints)
        _, result = minimize(radius(), x, constraints, jac_radius(functions[0]), tol=1e-10)
        inner_point, r = result[:-1], result[-1]
        result = np.matmul(functions[:, :-1], inner_point.T) + functions[:, -1]
        result = np.where(result >= -1e-16, 1, 0)
        if (0 in result) or r < 1e-10:
            return None
        return torch.from_numpy(inner_point)

    @_log_time("optimize child region", 2, True)
    def _optimize_region(
        self,
        functions: torch.Tensor,
        region: torch.Tensor,
        constraint_functions: torch.Tensor,
        c_region: torch.Tensor,
        c_inner_point: torch.Tensor,
    ):
        # 计算当前区域的区域边界函数，并且获取“邻居区域”
        # 边界区域可以用来过滤重复区域。
        constraint_functions, c_inner_point = constraint_functions.numpy(), c_inner_point.numpy()
        o_functions, o_region, neighbor_regions = [], [], []
        filter_region = torch.zeros_like(c_region).type(torch.int8)
        constraints = [
            constraint(
                linear_error(constraint_functions[i]),
                jac_linear(constraint_functions[i]),
            )
            for i in range(constraint_functions.shape[0])
        ]
        constraints.extend(self.constraints)
        for i in range(constraint_functions.shape[0]):
            function = square(constraint_functions[i])
            constraints[i]["fun"] = linear(constraint_functions[i])
            # TODO: 优化寻找区域边界的速率
            err = minimize(function, c_inner_point, constraints, jac_square(constraint_functions[i]))
            if err > 1e-15:
                continue
            o_functions.append(functions[i])
            o_region.append(region[i])
            # Find the neighbor rigon.
            if i < c_region.shape[0]:
                neighbor_region = c_region.clone()
                neighbor_region[i] = -region[i]
                neighbor_regions.append(neighbor_region)
                filter_region[i] = region[i]
        o_functions = torch.stack(o_functions)
        o_region = torch.tensor(o_region, dtype=torch.int8)
        return o_functions, o_region, filter_region, neighbor_regions

    def _optimize_child_region(
        self,
        c_functions: torch.Tensor,
        c_region: torch.Tensor,
        p_functions: torch.Tensor,
        p_region: torch.Tensor,
        inner_point: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        1. 获取区域是否存在; 2. 区域存在, 获取其中一个点; 3. 获取区域的相邻区域;
        """
        functions, region = torch.cat([c_functions, p_functions], dim=0), torch.cat([c_region, p_region], dim=0)
        constraint_functions = region.view(-1, 1) * functions
        # 1. checking whether the region exists, the inner point will be obtained if existed.
        c_inner_point: torch.Tensor | None = self._find_region_inner_point(constraint_functions, inner_point)
        if c_inner_point is None:
            return None, [], [], None, []
        # 2. find the least edges functions to express this region and obtain neighbor regions.
        optimize_child_region_result = self._optimize_region(functions, region, constraint_functions, c_region, c_inner_point)
        return c_inner_point, *optimize_child_region_result

    @_log_time("compute intersect", 2)
    def _compute_intersect(
        self,
        functions: torch.Tensor,
        p_functions: torch.Tensor,
        p_regions: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        存在一个f(x)与f(x0)符号相反或者f(x)=0

        计算神经元方程构成的区域边界是否穿越父区域（目标函数有至少一点在区域内）:

        *    min(func(x)^2);
        *    s.t. pFunc(x) >= 0
        """
        pn_functions = p_regions.view(-1, 1) * p_functions
        pn_functions, functions, x = pn_functions.numpy(), functions.numpy(), x.numpy()
        new_functions, inner_points = [], []
        constraints = [
            constraint(
                linear(pn_functions[i]),
                jac_linear(pn_functions[i]),
            )
            for i in range(pn_functions.shape[0])
        ]
        constraints.extend(self.constraints)
        # Is the linear function though the region.
        # TODO: 需要优化:太耗时
        for i in range(functions.shape[0]):
            err, result = minimize(square(functions[i]), x, constraints, jac_square(functions[i]))
            if err > 1e-16:
                continue
            new_functions.append(torch.from_numpy(functions[i]))
            inner_points.append(torch.from_numpy(result))
        if len(new_functions) == 0:
            return None, None
        else:
            new_functions = torch.stack(new_functions, dim=0)
            inner_points = torch.stack(inner_points, dim=0)
        return new_functions, inner_points

    @_log_time("get layer regions", 2)
    def _get_layer_regions(
        self,
        c_functions: torch.Tensor,
        p_functions: torch.Tensor,
        p_region: torch.Tensor,
        p_inner_point: torch.Tensor,
        depth: int,
        set_register: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], None],
    ):
        """
        验证hyperplane arrangement的存在性(子平面, 是否在父区域上具有交集)
        """
        intersect_funs, intersect_funs_points = self._compute_intersect(c_functions, p_functions, p_region, p_inner_point)
        counts, n_regions = 0, 0
        if intersect_funs is None:
            n_regions = 1
            counts += self._layer_region_counts(p_functions, p_region, p_inner_point, depth, set_register)
        else:
            c_regions = get_regions(intersect_funs_points, intersect_funs)
            # Regist some areas in WapperRegion for iterate.
            layer_regions = WapperRegion(c_regions)
            # TODO: Muti-Processes
            for c_region in layer_regions:
                c_inner_point, c_m_functions, c_m_region, filter_region, neighbor_regions = self._optimize_child_region(intersect_funs, c_region, p_functions, p_region, p_inner_point)
                if len(c_m_functions) == 0:
                    continue
                # Add the region to prevent counting again.
                layer_regions.update_filter(filter_region)
                # Regist new regions for iterate.
                layer_regions.regist_regions(neighbor_regions)
                n_regions += 1
                counts += self._layer_region_counts(c_m_functions, c_m_region, c_inner_point, depth, set_register)
        self.handler.inner_hyperplanes_handler(p_functions, p_region, c_functions, intersect_funs, n_regions, depth)
        return counts

    def _layer_region_counts(
        self,
        functions: torch.Tensor,
        region: torch.Tensor,
        inner_point: torch.Tensor,
        depth: int,
        set_register: Callable[[np.ndarray, torch.Tensor, torch.Tensor, int], None],
    ) -> int:
        if depth == self.last_depth:
            # last layer
            self.handler.region_handler(functions, region, inner_point)
            return 1
        set_register(inner_point, functions, region, depth + 1)
        return 0

    @_log_time("Region counts")
    def _get_counts(self, region_set: RegionSet) -> int:
        counts: int = 0
        current_depth: int = -1
        for p_inner_point, p_functions, p_region, depth in region_set:
            if depth != current_depth:
                current_depth = depth
                self.logger.info(f"Depth: {current_depth}")
            child_functions = self._functions(p_inner_point, depth)
            # get the region number of one layer.
            self.logger.info(f"Start get layers. Depth: {depth}, ")
            counts += self._get_layer_regions(
                child_functions,
                p_functions,
                p_region,
                p_inner_point,
                depth,
                region_set.register,
            )
        return counts

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
        # get the list of the linear functions from DNN.
        functions = self._find_functions(x, depth)
        # scale the functions
        return self._scale_functions(functions)

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
        # initialize the settings
        self.logger.info("Start Get region number.")
        self.last_depth = depth
        self.net = net.to(self.device).graph()
        self.input_size = input_size
        self.handler = handler
        self._init_constraints()
        # initialize the parameters
        dim = torch.Size(input_size).numel()
        p_functions, p_region, p_inner_point = _generate_bound_regions(bounds, dim)
        # start
        region_set = RegionSet(p_inner_point, p_functions, p_region)
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
    functions = torch.cat([low_bound_functions, upper_bound_functions], dim=0)
    region = torch.ones(dim * 2, dtype=torch.int8)
    return functions, region
