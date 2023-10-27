from collections import deque
import math
import time
from typing import Callable, Deque, List, Tuple

import numpy as np
import torch
from scipy.optimize import minimize

from torchays.analysis.model import Model
from torchays.analysis.optimization import (
    constraint,
    fun_bound,
    jac_bound,
    jac_linear,
    jac_radius,
    jac_radius_constraint,
    jac_square,
    linear,
    linear_error,
    radius,
    radius_constraint,
    square,
)
from torchays.nn import Module
from torchays.utils import get_logger


class RegionSet:
    def __init__(
        self,
        inner_point: np.ndarray,
        functions: torch.Tensor,
        region: torch.Tensor,
        depth: int = 0,
    ) -> None:
        self._regions: Deque[Tuple[np.ndarray, torch.Tensor, torch.Tensor, int]] = deque()
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
        inner_point: np.ndarray,
        functions: torch.Tensor,
        region: torch.Tensor,
        depth: int,
    ):
        self._regions.append((inner_point, functions, region, depth))


class WapperRegion:
    """
    Get the region(sign) of the funtion list.
    *  1 : f(x) > 0
    * -1 : f(x) <= 0
    """

    def __init__(self, regions: List[Tuple[torch.Tensor, np.ndarray]] = None):
        self.region_sign = torch.Tensor().type(torch.int8)
        self.regions: Deque[Tuple[torch.Tensor, np.ndarray]] = deque()
        if regions is not None:
            self.regist_regions(regions)

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, np.ndarray]:
        try:
            region, inner_point = self.regions.popleft()
            while self._check_region(region):
                region, inner_point = self.regions.popleft()
            return region, inner_point
        except:
            raise StopIteration

    def _update_list(self):
        self.output: torch.Tensor = self.regions.popleft()

    def _check_region(self, region: torch.Tensor):
        try:
            a = ((self.region_sign.abs() * region) - self.region_sign).abs().sum(dim=1)
            return 0 in a
        except:
            return False

    def update_index(self, region_sign: torch.Tensor):
        if self._check_region(region_sign):
            return
        self.region_sign = torch.cat([self.region_sign, region_sign.view(1, -1)], dim=0)

    def regist_region(self, region_sign: torch.Tensor, inner_point: np.ndarray):
        if not self._check_region(region_sign):
            self.regions.append((region_sign, inner_point))

    def regist_regions(self, regions: List[Tuple[torch.Tensor, np.ndarray]]):
        for region, inner_point in regions:
            self.regist_region(region, inner_point)


class ReLUNets:
    """
    ReLUNets needs to ensure that the net has the function:
        >>> def forward_graph_Layer(*args, depth=depth):
        >>>     ''' layer is a "int" before every ReLU module. "Layer" can get the layer weight and bias graph.'''
        >>>     if depth == 1:
        >>>         return output

    args:
        device: torch.device
            GPU or CPU to get the graph from the network;
        logger: def info(...)
            print the information (Default: print in console)(logger.info(...)).
    """

    def __init__(self, device=torch.device("cpu"), logger=None):
        self.device = device
        self.one = torch.ones(1).double()
        self.logger = logger or get_logger("AnalysisReLUNetUtils-Console")

    def _compute_functions(self, x: np.ndarray, depth: int) -> torch.Tensor:
        """
        Get the list of the linear function before ReLU layer.
        """
        x = torch.from_numpy(x).float().to(self.device)
        x = x.reshape(*self.input_size).unsqueeze(dim=0)
        with torch.no_grad():
            _, graph = self.net.forward_graph_Layer(x, depth=depth)
            # (1, *output.size(), *input.size())
            weight_graph, bias_graph = graph["weight_graph"], graph["bias_graph"]
            # (output.num, input.num)
            weight_graph = weight_graph.reshape(-1, x.size()[1:].numel())
            # (output.num, 1)
            bias_graph = bias_graph.reshape(-1, 1)
            # (output.num, input.num + 1)
        return torch.cat([weight_graph, bias_graph], dim=1).cpu()

    def _minimize(self, function, x0, constraints, jac, method="SLSQP", tol=1e-20, options={"maxiter": 100}):
        return minimize(function, x0, method=method, constraints=constraints, jac=jac, tol=tol, options=options)

    def _compute_radius(self, functions: np.ndarray, x: np.ndarray):
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
        norm_A = np.linalg.norm(functions[:, :-1], ord=2, axis=1)
        constraints = [
            constraint(
                radius_constraint(functions[i], norm_A[i]),
                jac_radius_constraint(functions[i], norm_A[i]),
            )
            for i in range(functions.shape[0])
        ]
        constraints.extend(self.constraints)
        result = self._minimize(radius(), x, constraints, jac_radius(functions[0]), tol=1e-10)
        x, r = result.x[:-1], result.x[-1]
        return x, r

    def _compute_region_inner_point(
        self,
        child_functions: torch.Tensor,
        child_region: torch.Tensor,
        child_inner_point: np.ndarray,
        parent_functions: torch.Tensor,
        parent_region: torch.Tensor,
    ):
        """
        若区域存在, 则计算构成区域的最小数量的边界, 并且获得相邻区域中的点;
        *   min_{x} (aX + b);
        *   s.t. AX + B >= 0;
        """
        functions, region = torch.cat([child_functions, parent_functions], dim=0), torch.cat([child_region, parent_region], dim=0)
        constraint_functions = region.view(-1, 1) * functions
        functions, constraint_functions, region = functions.numpy(), constraint_functions.numpy(), region.numpy()
        constraint_area = np.ones_like(region)
        # initialize output
        next_functions, next_region, neighbor_region_inner_points = [], [], []

        # 1. checking whether the region exists, the inner point will be obtained if existed.
        r = np.random.uniform(0, self.bound)
        next_inner_point, r = self._compute_radius(constraint_functions, np.append(child_inner_point, r))
        result = np.matmul(constraint_functions[:, :-1], next_inner_point.T) + constraint_functions[:, -1]
        result = np.where(result >= -1e-16, 1, 0)
        if not np.array_equal(result, constraint_area) or r < 10e-10 or r > self.bound:
            # no region and return
            return None, next_functions, next_region, None, neighbor_region_inner_points

        # 2. find the least edges functions to express this region and obtain the inner points of the neighbor regions.
        # TODO: 需要优化:太耗时
        constraints = [
            constraint(
                linear_error(constraint_functions[i]),
                jac_linear(constraint_functions[i]),
            )
            for i in range(constraint_functions.shape[0])
        ]
        constraints.extend(self.constraints)
        new_child_region = torch.zeros_like(child_region).type(torch.int8)
        for i in range(constraint_functions.shape[0]):
            function = square(constraint_functions[i])
            constraints[i]["fun"] = linear(constraint_functions[i])
            result = self._minimize(function, next_inner_point, constraints, jac_square(constraint_functions[i]))
            if result.fun > 1e-15:
                continue
            next_functions.append(torch.from_numpy(functions[i]))
            next_region.append(region[i])
            # Find the points of neighbor rigon.
            if i < child_region.shape[0]:
                new_child_region[i] = region[i]
                x, a, b = result.x, constraint_functions[i, :-1], constraint_functions[i, -1]
                if result.fun > 0:
                    k = -2 * (np.matmul(a, x) + b) / (np.matmul(a, a))
                    k2 = -2 * (np.matmul(a, x) + b + 1e-10) / (np.matmul(a, a))
                else:
                    k = -(np.matmul(a, x) + b + 1e-10) / (np.matmul(a, a))
                    k2 = -2 * (np.matmul(a, x) + b + 1e-15) / (np.matmul(a, a))
                x_p = x + k * a
                x_p2 = x + k2 * a
                neighbor_region_inner_points.append(torch.from_numpy(x_p))
                neighbor_region_inner_points.append(torch.from_numpy(x_p2))
        next_functions = torch.stack(next_functions)
        neighbor_region_inner_points = torch.stack(neighbor_region_inner_points)
        next_region = torch.tensor(next_region, dtype=torch.int8)
        # self.logger.info(f"Smallest function size: {next_functions.size()};")
        return next_inner_point, next_functions, next_region, new_child_region, neighbor_region_inner_points

    def _compute_intersect(
        self,
        functions: torch.Tensor,
        parent_functions: torch.Tensor,
        parent_regions: torch.Tensor,
        x: np.ndarray,
    ):
        """
        TODO: 需要优化:太耗时
        存在一个f(x)与f(x0)符号相反或者f(x)=0

        计算神经元方程构成的区域边界是否穿越父区域（目标函数有至少一点在区域内）:

        *    min(func(x)^2);
        *    s.t. pFunc(x) >= 0
        """
        parenet_functions = parent_regions.view(-1, 1) * parent_functions
        parenet_functions, functions = parenet_functions.numpy(), functions.numpy()
        new_functions, inner_points = [], []
        constraints = [
            constraint(
                linear(parenet_functions[i]),
                jac_linear(parenet_functions[i]),
            )
            for i in range(parenet_functions.shape[0])
        ]
        constraints.extend(self.constraints)
        # Is the linear function though the region.
        for i in range(functions.shape[0]):
            result = self._minimize(square(functions[i]), x, constraints, jac_square(functions[i]))
            if result.fun > 1e-16:
                continue
            new_functions.append(torch.from_numpy(functions[i]))
            inner_points.append(torch.from_numpy(result.x))
        if len(new_functions) == 0:
            return None, None
        else:
            new_functions = torch.stack(new_functions, dim=0)
            inner_points = torch.stack(inner_points, dim=0)
        return new_functions, inner_points

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

    def _get_layer_regions(
        self,
        child_functions: torch.Tensor,
        parent_functions: torch.Tensor,
        parent_region: torch.Tensor,
        inner_point: np.ndarray,
        depth: int,
        set_register: Callable[[np.ndarray, torch.Tensor, torch.Tensor, int], None],
    ):
        """
        验证切割的超平面在区域中的存在性:
        """
        counts = 0
        t = time.time()
        child_functions, child_inner_points = self._compute_intersect(child_functions, parent_functions, parent_region, inner_point)
        t = time.time() - t
        self.logger.info(f"     _compute_intersect: {t}s")
        if child_functions is None:
            counts += self._layer_region_counts(inner_point, parent_functions, parent_region, depth, set_register)
        else:
            # Regist some areas in WapperRegion for iterate.
            child_regions = self._get_regions(child_inner_points, child_functions)
            layer_regions = WapperRegion(child_regions)
            for child_region, child_inner_point in layer_regions:
                next_inner_point, next_functions, next_region, new_child_region, neighbor_region_inner_points = self._compute_region_inner_point(
                    child_functions, child_region, child_inner_point, parent_functions, parent_region
                )
                if len(next_functions) == 0:
                    continue
                # Add the region to prevent counting again.
                layer_regions.update_index(new_child_region)
                # Regist new regions for iterate.
                neighbor_regions = self._get_regions(neighbor_region_inner_points, child_functions)
                layer_regions.regist_regions(neighbor_regions)
                counts += self._layer_region_counts(next_inner_point, next_functions, next_region, depth, set_register)
        return counts

    def _get_regions(self, x: torch.Tensor, functions: torch.Tensor) -> torch.Tensor:
        A, B = functions[:, :-1].double(), functions[:, -1].double()
        regions = torch.sign(x @ A.T + B)
        regions = torch.where(regions == 0, -self.one, regions).type(torch.int8)
        return list(zip(regions, x.numpy()))

    def _layer_region_counts(
        self,
        inner_point: np.ndarray,
        functions: torch.Tensor,
        region: torch.Tensor,
        depth: int,
        set_register: Callable[[np.ndarray, torch.Tensor, torch.Tensor, int], None],
    ) -> int:
        if depth == self.last_depth:
            # last layer
            self.region_handler(inner_point, functions, region)
            return 1
        set_register(inner_point, functions, region, depth + 1)
        return 0

    def _get_counts(self, region_set: RegionSet) -> int:
        counts: int = 0
        current_depth: int = -1
        for inner_point, parent_functions, parent_region, depth in region_set:
            if depth != current_depth:
                current_depth = depth
                self.logger.info(f"Depth: {current_depth}. ={'='*current_depth}>")
            t = time.time()
            child_functions = self._functions(inner_point, depth)
            # get the region number of one layer.
            counts += self._get_layer_regions(
                child_functions,
                parent_functions,
                parent_region,
                inner_point,
                depth,
                region_set.register,
            )
            t = time.time() - t
            self.logger.info(f"get child regions: {t}s")
        return counts

    def _functions(self, x: np.ndarray, depth: int):
        # get the list of the linear functions from DNN.
        functions = self._compute_functions(x, depth)
        # scale the functions
        return self._scale_functions(functions)

    def get_region_counts(
        self,
        net: Model,
        bound: float = 1.0,
        depth: int = -1,
        input_size: tuple = (2,),
        region_handler: Callable[[np.ndarray, torch.Tensor, torch.Tensor], None] = None,
    ):
        """
        目前只支持方形的输入空间画图，需要修改。
        """
        assert isinstance(net, Module), "the type of net must be \"BaseModule\"."
        assert depth >= 0, "countLayers must >= 0."
        assert bound > 0, "Please set the bound > 0."
        # initialize the settings
        self.logger.info("Start Get region number.")
        self.bound = bound
        self.last_depth = depth
        self.net = net.to(self.device).graph()
        self.input_size = input_size
        self.region_handler = region_handler or self._default_handler()
        self._init_constraints()

        # initialize the parameters
        size_prod = torch.Size(input_size).numel()
        low_bound_parent_functions = -torch.cat([torch.eye(size_prod), torch.zeros(size_prod, 1) - self.bound], dim=1)
        upper_bound_parent_functions = torch.cat([torch.eye(size_prod), torch.zeros(size_prod, 1) + self.bound], dim=1)
        parent_functions = torch.cat([low_bound_parent_functions, upper_bound_parent_functions], dim=0)
        parent_region = torch.ones(size_prod * 2, dtype=torch.int8)
        inner_point = np.random.uniform(-self.bound, self.bound, size=(size_prod,))
        # start
        s_time = time.time()
        region_set = RegionSet(inner_point, parent_functions, parent_region, 0)
        counts = self._get_counts(region_set)
        e_time = time.time()

        self.logger.info(f"Region counts: {counts}. Time: {e_time-s_time}s")
        return counts

    def _init_constraints(self):
        self.constraints = [
            constraint(fun_bound, jac_bound),
        ]

    def _default_handler(self, point: np.ndarray, functions: torch.Tensor, region: torch.Tensor) -> None:
        return
