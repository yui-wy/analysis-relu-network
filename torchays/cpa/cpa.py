import multiprocessing as mp
import os
from collections import deque
from logging import Logger
from multiprocessing.pool import AsyncResult
from multiprocessing.reduction import ForkingPickler
from typing import Callable, Deque, Tuple

import numpy as np
import torch

from ..nn import Module
from ..nn.modules import BIAS_GRAPH, WEIGHT_GRAPH
from ..utils import get_logger
from .handler import BaseHandler
from .model import Model
from .optimization import cheby_ball, lineprog_intersect
from .regions import CPACache, RegionSet, WapperRegion
from .util import find_projection, generate_bound_regions, get_regions, log_time, vertify


class CPA:
    """
    CPA needs to ensure that the net has the function:
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
        device: torch.device = torch.device("cpu"),
        logger: Logger = None,
        logging: bool = True,
    ):
        self.workers = workers if workers > 1 else 1
        self.device = device
        self.one = torch.ones(1).double()
        self.logger = logger or get_logger("CPAExploration-Console", multi=(workers > 1))
        self.logging = logging

    def start(
        self,
        net: Model,
        bounds: float | int | Tuple[float, float] | Tuple[Tuple[float, float]] = 1.0,
        depth: int = -1,
        input_size: tuple = (2,),
        handler: BaseHandler = None,
        logger: Logger = None,
    ):
        assert isinstance(net, Module), "the type of net must be \"BaseModule\"."
        # Initialize the settings
        self.last_depth = depth if depth >= 0 else net.n_relu
        self.input_size = input_size
        self.handler = handler
        if logger is not None:
            self.logger = logger
        # Initialize the parameters
        dim = torch.Size(input_size).numel()
        p_funcs, p_region, p_inner_point, self.o_bounds = generate_bound_regions(bounds, dim)
        # Initialize the region set.
        region_set = RegionSet()
        region_set.register(p_funcs, p_region, p_inner_point, 0)
        # Start to get the NN regions.
        self.logger.info("Start Get region number.")
        net = net.graph().to(self.device)
        counts = self._get_counts(net, region_set)
        self.logger.info(f"Region counts: {counts}.")
        return counts

    @log_time("Region counts")
    def _get_counts(self, net: Model, region_set: RegionSet) -> int:
        if self.workers == 1:
            return self._single_get_counts(net, region_set)
        # Multi-process
        # Change the ForkingPickler, and stop share memory of torch.Tensor when using multiprocessiong.
        # If share_memory is used, the large number of the fd will be created and lead to OOM.
        _save_reducers = ForkingPickler._extra_reducers
        ForkingPickler._extra_reducers = {}
        counts = self._multiprocess_get_counts(net, region_set)
        ForkingPickler._extra_reducers = _save_reducers
        return counts

    def _single_get_counts(self, net: Model, region_set: RegionSet) -> int:
        counts: int = 0
        cpa_caches = CPACache(self.handler)
        for p_funcs, p_region, p_inner_point, depth in region_set:
            c_funcs = self._functions(net, p_inner_point, depth)
            self.logger.info(f"Start to get regions. Depth: {depth+1}, ")
            # Find the child regions or get the region counts.
            count, child_regions, cpa_cache = self._handler_region(p_funcs, p_region, p_inner_point, c_funcs, depth)
            counts += count
            cpa_caches.extend(cpa_cache)
            region_set.extend(child_regions)
        cpa_caches()
        return counts

    def _work(self, p_funcs: torch.Tensor, p_region: torch.Tensor, p_inner_point: torch.Tensor, c_funcs: torch.Tensor, depth: int):
        self.logger.info(f"Start to get regions. Depth: {depth+1}, Process-PID: {os.getpid()}. ")
        return self._handler_region(p_funcs, p_region, p_inner_point, c_funcs, depth)

    def _multiprocess_get_counts(self, net: Model, region_set: RegionSet) -> int:
        """This method of multi-process implementation will result in the inability to use multi-processing when searching the first layer."""
        counts: int = 0
        cpa_caches = CPACache(self.handler)

        def callback(args) -> None:
            nonlocal counts
            count, child_regions, cpa_cache = args
            counts += count
            cpa_caches.extend(cpa_cache)
            region_set.extend(child_regions)

        def err_callback(msg):
            print(msg)

        pool = mp.Pool(processes=self.workers)
        results: Deque[AsyncResult] = deque()
        for p_funcs, p_region, p_inner_point, depth in region_set:
            # We do not calculate the weigh and bias in the sub-processes.
            # It will use the GPUs and CUDA, and there are many diffcult problems (memory copying, "spawn" model...) that need to be solved.
            c_funcs = self._functions(net, p_inner_point, depth)
            # Multi-processing to search the CPAs.
            res = pool.apply_async(
                func=self._work,
                args=(p_funcs, p_region, p_inner_point, c_funcs, depth),
                callback=callback,
                error_callback=err_callback,
            )
            # Clean finished processes.
            results = [r for r in results if not r.ready()]
            results.append(res)
            if len(region_set) != 0:
                continue
            for res in results:
                res.wait()
                if len(region_set) > 0:
                    break
        results.clear()
        pool.close()
        pool.join()
        cpa_caches()
        return counts

    def _functions(self, net: Model, x: torch.Tensor, depth: int):
        # Get the list of the linear functions from DNN.
        functions = self._net_2_cpa(net, x, depth)
        # Scale the functions.
        return self._scale_functions(functions)

    def _net_2_cpa(self, net: Model, x: torch.Tensor, depth: int) -> torch.Tensor:
        x = x.float().to(self.device)
        x = x.reshape(*self.input_size).unsqueeze(dim=0)
        with torch.no_grad():
            _, graph = net.forward_layer(x, depth=depth)
            # (1, *output.size(), *input.size())
            weight_graph, bias_graph = graph[WEIGHT_GRAPH], graph[BIAS_GRAPH]
            # (output.num, input.num)
            weight_graph = weight_graph.reshape(-1, x.size()[1:].numel())
            # (output.num, 1)
            bias_graph = bias_graph.reshape(-1, 1)
            # (output.num, input.num + 1)
        return torch.cat([weight_graph, bias_graph], dim=1).cpu()

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

    @log_time("handler region", 2, logging=True)
    def _handler_region(
        self,
        p_funcs: torch.Tensor,
        p_region: torch.Tensor,
        p_inner_point: torch.Tensor,
        c_funcs: torch.Tensor,
        depth: int,
    ):
        """
        Search the child regions.
        """
        child_regions, cpa_cache = RegionSet(), CPACache(self.handler)
        counts, n_regions = 0, 0
        intersect_funcs = self._find_intersect(p_funcs, p_region, p_inner_point, c_funcs)
        if intersect_funcs is None:
            n_regions = 1
            counts += self._nn_region_counts(p_funcs, p_region, p_inner_point, depth, child_regions.register, cpa_cache.cpa)
        else:
            c_regions = get_regions(p_inner_point.reshape(1, -1), intersect_funcs)
            # Register some regions in WapperRegion for iterate.
            layer_regions = WapperRegion(c_regions[0])
            for c_region in layer_regions:
                # Check and get the child region. Then, the neighbor regions will be found.
                c_inner_point, c_edge_funcs, c_edge_region, filter_region, neighbor_regions = self._optimize_child_region(intersect_funcs, c_region, p_funcs, p_region)
                if c_inner_point is None:
                    continue
                # Add the region to prevent counting again.
                layer_regions.update_filter(filter_region)
                # Register new regions for iterate.
                layer_regions.extend(neighbor_regions)
                # Count the number of the regions in the current parent region.
                n_regions += 1
                # Handle the child region.
                counts += self._nn_region_counts(c_edge_funcs, c_edge_region, c_inner_point, depth, child_regions.register, cpa_cache.cpa)
        # Collect the information of the current parent region including region functions, child functions, intersect functions and number of the child regions.
        cpa_cache.hyperplane(p_funcs, p_region, c_funcs, intersect_funcs, n_regions, depth)
        return counts, child_regions, cpa_cache

    @log_time("find intersect", 2, logging=False)
    def _find_intersect(
        self,
        p_funcs: torch.Tensor,
        p_regions: torch.Tensor,
        x: torch.Tensor,
        funcs: torch.Tensor,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """Find the hyperplanes intersecting the region."""
        intersect_funs = []
        optim_funcs, optim_x = funcs.numpy(), x.double().numpy()
        pn_funcs = (p_regions.view(-1, 1) * p_funcs).numpy()
        p_points = find_projection(x, funcs)
        for i in range(funcs.size(0)):
            if torch.isnan(p_points[i]).all() or torch.isinf(p_points[i]).all():
                continue
            if vertify(p_points[i], p_funcs, p_regions) or lineprog_intersect(optim_funcs[i], pn_funcs, optim_x, self.o_bounds):
                intersect_funs.append(funcs[i])
        if len(intersect_funs) == 0:
            return None
        intersect_funs = torch.stack(intersect_funs, dim=0)
        return intersect_funs

    def _optimize_child_region(
        self,
        c_funcs: torch.Tensor,
        c_region: torch.Tensor,
        p_funcs: torch.Tensor,
        p_region: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        1. Check if the region is existed.
        2. Get the neighbor regions and the functions of the region edges.;
        """
        funcs, region = torch.cat([c_funcs, p_funcs], dim=0), torch.cat([c_region, p_region], dim=0)
        # ax+b >= 0
        constraint_funcs = region.view(-1, 1) * funcs
        # 1. Check whether the region exists, the inner point will be obtained if existed.
        c_inner_point: torch.Tensor | None = self._find_region_inner_point(constraint_funcs)
        if c_inner_point is None:
            return None, [], [], None, []
        # 2. find the least edges functions to express this region and obtain neighbor regions.
        optimize_child_region_result = self._optimize_region(funcs, region, constraint_funcs, c_region, c_inner_point)
        return c_inner_point, *optimize_child_region_result

    @log_time("find child region inner point", 2, False)
    def _find_region_inner_point(self, functions: torch.Tensor) -> Tuple[np.ndarray, float]:
        """
        Calculate the max radius of the insphere in the region to make the radius less than the distance of the insphere center to the all functions
        which can express a liner region.

        minimizing the following function.

        To minimize:
        * min_{x,r} (-r)
        * s.t.  (-Ax+r||A|| <= B)
        *       r > 0
        """
        funcs = functions.numpy()
        x, _, success = cheby_ball(funcs)
        if not success or x is None:
            return None
        return torch.from_numpy(x).float()

    @log_time("optimize child region", 2, False)
    def _optimize_region(
        self,
        funcs: torch.Tensor,
        region: torch.Tensor,
        constraint_funcs: torch.Tensor,
        c_region: torch.Tensor,
        c_inner_point: torch.Tensor,
    ):
        """Get the bound hyperplanes which can filter the same regions, and find the neighbor regions."""
        c_edge_funcs, c_edge_region, neighbor_regions = [], [], []
        filter_region = torch.zeros_like(c_region).type(torch.int8)

        optim_funcs, optim_x = constraint_funcs.numpy(), c_inner_point.double().numpy()
        p_points = find_projection(c_inner_point, funcs)
        for i in range(optim_funcs.shape[0]):
            if not vertify(p_points[i], funcs, region):
                pn_funcs = np.delete(optim_funcs, i, axis=0)
                success = lineprog_intersect(optim_funcs[i], pn_funcs, optim_x, self.o_bounds)
                if not success:
                    continue
            c_edge_funcs.append(funcs[i])
            c_edge_region.append(region[i])
            # Find the neighbor regions.
            if i < c_region.shape[0]:
                neighbor_region = c_region.clone()
                neighbor_region[i] = -region[i]
                neighbor_regions.append(neighbor_region)
                filter_region[i] = region[i]
        c_edge_funcs = torch.stack(c_edge_funcs)
        c_edge_region = torch.tensor(c_edge_region, dtype=torch.int8)
        return c_edge_funcs, c_edge_region, filter_region, neighbor_regions

    def _nn_region_counts(
        self,
        funcs: torch.Tensor,
        region: torch.Tensor,
        inner_point: torch.Tensor,
        depth: int,
        register: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], None],
        cpa: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None],
    ) -> int:
        if depth == self.last_depth:
            # If current layer is the last layer, the region is the in count.
            # Collect the information of the final region.
            cpa(funcs, region, inner_point)
            return 1
        # If not the last layer, the region will be parent region in the next layer.
        register(funcs, region, inner_point, depth + 1)
        return 0
