import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

from ..functional import conv2d
from .base import Module


class Conv2d(Module, nn.Conv2d):
    __doc__ = nn.Conv2d.__doc__

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        assert self.padding_mode == 'zeros', "Dont support other mode"
        assert self.dilation == _pair(1), "Dont support other dilation."
        # TODO: 支持groups
        assert self.groups == 1, "Dont support other mode"

    def forward_graph(self, input: torch.Tensor, weight_graph=None, bias_graph=None):
        # bias_graph
        bias_graph = torch.zeros_like(input, device=input.device) if bias_graph is None else bias_graph
        bias_graph = nn.Conv2d.forward(self, bias_graph)
        # weight_graph
        origin_size = self._origin_size(input, weight_graph)
        weight_graph = conv2d(
            weight_graph,
            self.weight,
            self.kernel_size,
            origin_size,
            input.size(),
            bias_graph.size(),
            self.out_channels,
            self.in_channels,
            self.padding,
            self.stride,
            self.dilation,
            input.device,
        )
        return weight_graph, bias_graph
