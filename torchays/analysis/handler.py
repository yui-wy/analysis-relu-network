import numpy as np
import torch


class Handler:
    def __init__(self) -> None:
        self._init_region_handler()._init_inner_hypeplanes_handler()

    def _init_region_handler(self):
        self.funs = list()
        self.regions = list()
        self.points = list()
        return self

    def region_handler(
        self,
        point: np.ndarray,
        fun: torch.Tensor,
        region: torch.Tensor,
    ) -> None:
        self.funs.append(fun.numpy())
        self.regions.append(region.numpy())
        self.points.append(point)

    def _init_inner_hypeplanes_handler(self):
        return self

    def inner_hypeplanes_handler(
        self,
        c_funs: torch.Tensor,
        p_funs: torch.Tensor,
        p_regions: torch.Tensor,
        intersect_funs: torch.Tensor,
        depth: int,
    ) -> None:
        # 获取与p_regions有交集的超平面的数据
        # 穿越数量
        n_intersect_funs = intersect_funs.size(0)
        # 与神经元数量比
        rate_intersect = n_intersect_funs / c_funs.size(0)
        return
