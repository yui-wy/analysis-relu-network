import numpy as np
import torch


class BaseHandler:
    def region_handler(
        self,
        fun: torch.Tensor,
        region: torch.Tensor,
        point: np.ndarray,
    ):
        raise NotImplementedError

    def inner_hyperplanes_handler(
        self,
        p_funs: torch.Tensor,
        p_regions: torch.Tensor,
        c_funs: torch.Tensor,
        intersect_funs: torch.Tensor | None,
        n_regions: int,
        depth: int,
    ) -> None:
        raise NotImplementedError


class DefaultHandler(BaseHandler):
    def region_handler(
        self,
        fun: torch.Tensor,
        region: torch.Tensor,
        point: np.ndarray,
    ):
        return None

    def inner_hyperplanes_handler(
        self,
        p_funs: torch.Tensor,
        p_regions: torch.Tensor,
        c_funs: torch.Tensor,
        intersect_funs: torch.Tensor | None,
        n_regions: int,
        depth: int,
    ) -> None:
        return None
