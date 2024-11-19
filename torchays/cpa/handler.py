import torch


class BaseHandler:
    def region(
        self,
        fun: torch.Tensor,
        region: torch.Tensor,
        point: torch.Tensor,
    ):
        raise NotImplementedError()

    def inner_hyperplanes(
        self,
        p_funs: torch.Tensor,
        p_regions: torch.Tensor,
        c_funs: torch.Tensor,
        intersect_funs: torch.Tensor | None,
        n_regions: int,
        depth: int,
    ) -> None:
        raise NotImplementedError()
