from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_2_t, _size_any_t
from torch.nn.modules.utils import _pair

from ..functional import avg_pool_2d, max_pool_2d
from .base import Module


class AvgPool2d(Module, nn.AvgPool2d):
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

    def forward_graph(self, input: Tensor, weight_graph=None, bias_graph=None):
        # bias_graph
        bias_graph = torch.zeros_like(input, device=input.device) if bias_graph is None else bias_graph
        bias_graph = nn.AvgPool2d.forward(self, bias_graph)
        # weight_graph
        origin_size = self._origin_size(input, weight_graph)
        divisor = self.divisor_override or self._kernel_size[0] * self._kernel_size[1]
        output_size = bias_graph.size()
        kernel_weight = torch.ones(self._kernel_size, device=input.device) / divisor
        weight_graph = avg_pool_2d(
            weight_graph,
            kernel_weight,
            self._kernel_size,
            origin_size,
            input.size(),
            output_size,
            self._padding,
            self._stride,
            input.device,
        )
        return weight_graph, bias_graph


class MaxPool2d(Module, nn.MaxPool2d):
    __doc__ = nn.MaxPool2d.__doc__

    def __init__(self, kernel_size: _size_any_t, stride: _size_any_t | None = None, padding: _size_any_t = 0, dilation: _size_any_t = 1, return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        assert not self.ceil_mode, "'ceil_mode' must be False."
        assert dilation == 1, "Dont support other dilation."
        self._padding = _pair(padding)
        self._stride = _pair(stride)
        self._kernel_size = _pair(self.kernel_size)
        self._dilation = _pair(dilation)

    def forward_graph(self, input: Tensor, weight_graph: Tensor = None, bias_graph: Tensor = None):
        save = self.return_indices
        if not self.return_indices:
            self.return_indices = True
        output, indices = nn.MaxPool2d.forward(self, input)
        origin_size = self._origin_size(input, weight_graph)
        weight_graph, bias_graph = max_pool_2d(
            indices,
            weight_graph,
            bias_graph,
            self._kernel_size,
            origin_size,
            output.size(),
            self._padding,
            self._stride,
            self._dilation,
            input.device,
        )
        self.return_indices = save
        return weight_graph, bias_graph
