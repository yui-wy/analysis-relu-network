import torch
import torch.nn as nn

from .base import Module, get_size_to_one


class _BatchNorm(Module, nn.modules.batchnorm._BatchNorm):
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device=None, dtype=None) -> None:
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        assert self.track_running_stats, "Please set track_running_stats = True"

    def forward_graph(self, x, weight_graph=None, bias_graph=None):
        """
        Analyzing BatchNorm2d \n
        track_running_stats = True->Using saving var and mean.
        graph_size: ((*x.shape)), (*origin_size))
        """
        bias_graph = torch.zeros_like(x, device=x.device) if bias_graph is None else bias_graph
        origin_size = self._origin_size(x, weight_graph)
        bias_graph = nn.modules.batchnorm._BatchNorm.forward(self, bias_graph)

        size = list(get_size_to_one(x.size()[1:]))
        size[0] = -1

        weight = self.weight if self.affine else torch.ones(self.num_features, device=x.device)
        real_weight = (weight / torch.sqrt(self.running_var + self.eps)).view(*size, *get_size_to_one(origin_size))
        weight_graph *= real_weight

        return weight_graph, bias_graph


class BatchNormNone(_BatchNorm):
    def forward(self, input):
        return input

    def forward_graph(self, x, weight_graph=None, bias_graph=None):
        return weight_graph, bias_graph


class BatchNorm1d(_BatchNorm):
    __doc__ = nn.BatchNorm1d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class BatchNorm2d(_BatchNorm):
    __doc__ = nn.BatchNorm2d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class BatchNorm3d(_BatchNorm):
    __doc__ = nn.BatchNorm3d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
