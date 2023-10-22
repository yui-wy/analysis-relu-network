from typing import Callable, Optional
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.common_types import _size_2_t, _size_any_t

from torchays.nn.modules import base
from torch.nn.modules.utils import _pair

from torchays.nn.functional import avg_pool_2d, max_pool_2d


class AvgPool2d(nn.AvgPool2d, base.Module):
    __doc__ = nn.AvgPool2d.__doc__

    def __init__(
        self, kernel_size: _size_2_t, stride: _size_2_t | None = None, padding: _size_2_t = 0, ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: int | None = None
    ) -> None:
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        assert not self.ceil_mode, "'ceil_mode' must be False."
        assert self.count_include_pad, "'count_include_pad' must be True."
        self._padding = _pair(padding)
        self._stride = _pair(stride)
        self._kernel_size = _pair(self.kernel_size)

    def forward(self, input):
        return self._forward(super().forward, input)

    def forward_graph(self, input: Tensor, weight_graph=None, bias_graph=None):
        # bias_graph
        bias_graph = torch.zeros_like(input, device=input.device) if bias_graph is None else bias_graph
        bias_graph = super().forward(bias_graph)
        # weight_graph
        origin_size = self._get_origin_size(input, weight_graph)
        divisor = self.divisor_override or self._kernel_size[0] * self._kernel_size[1]
        output_size = bias_graph.size()
        channels = output_size[1]
        kernel_weight = torch.eye(channels, device=input.device).view(channels, channels, 1, 1) / divisor
        weight_graph = avg_pool_2d(
            weight_graph,
            kernel_weight,
            self._kernel_size,
            origin_size,
            input.size(),
            output_size,
            channels,
            self._padding,
            self._stride,
            input.device,
        )
        return weight_graph, bias_graph


class MaxPool2d(nn.MaxPool2d, base.Module):
    __doc__ = nn.AvgPool2d.__doc__

    def __init__(self, kernel_size: _size_any_t, stride: _size_any_t | None = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        assert not self.ceil_mode, "'ceil_mode' must be False."
        assert dilation == 1, "Dont support other dilation."
        self._padding = _pair(padding)
        self._stride = _pair(stride)
        self._kernel_size = _pair(self.kernel_size)
        self._dilation = _pair(dilation)

    def forward(self, input: Tensor):
        return self._forward(super().forward, input)

    def forward_graph(self, input: Tensor, weight_graph: Tensor = None, bias_graph: Tensor = None):
        # weight_graph
        origin_size = self._get_origin_size(input, weight_graph)
        output_size = bias_graph.size()
        channels = output_size[2]
        weight_graph = max_pool_2d(
            weight_graph,
            self._kernel_weight(input),
            self._kernel_size,
            origin_size,
            input.size(),
            output_size,
            channels,
            self._padding,
            self._stride,
            self._dilation,
            input.device,
        )
        return weight_graph, bias_graph

    def _kernel_weight(input: Tensor) -> Callable[[int], torch.Tensor]:
        def kw(w: int):
            pass

        return kw
