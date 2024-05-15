from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.nn import init
from torch import nn
from torch.nn.parameter import Parameter

from .base import Module, get_size_to_one


class _norm(nn.Module):
    def __init__(
        self,
        num_features: int,
        freeze: bool = False,
        set_parameters: Callable[[Parameter, Parameter], None] = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_features = num_features
        self.set_parameters = self._set_parameters if set_parameters is None else set_parameters
        self.freeze = freeze
        self.weight = Parameter(torch.empty(num_features, **factory_kwargs))
        self.bias = Parameter(torch.empty(num_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.set_parameters(self.weight, self.bias)
        if self.freeze:
            self.weight.requires_grad = False
            self.bias.requires_grad = False

    def _set_parameters(self, weight: Parameter, bias: Parameter) -> None:
        init.ones_(weight)
        init.zeros_(bias)

    def _check_input_dim(self, input: torch.Tensor):
        raise NotImplementedError

    def forward(self, input: torch.Tensor):
        self._check_input_dim(input)
        # input: (n, num_features, ...)
        w_size = [1] * len(input.size())
        w_size[1] = -1
        # weight: (num_features)
        # bias,weight: (num_features)
        weight = self.weight.view(*w_size)
        bias = self.bias.view(*w_size)
        return input * weight + bias


class _Norm(Module, _norm):
    def __init__(self, num_features: int, freeze: bool = False, set_parameters: Callable[[Parameter, Parameter], None] = None, device=None, dtype=None) -> None:
        super().__init__(num_features, freeze, set_parameters, device, dtype)

    def forward_graph(self, input: Tensor, weight_graph: Tensor = None, bias_graph: Tensor = None) -> Tuple[Tensor]:
        bias_graph = torch.zeros_like(input, device=input.device, dtype=input.dtype) if bias_graph is None else bias_graph
        origin_size = self._origin_size(input, weight_graph)
        bias_graph = _norm.forward(self, bias_graph)
        size = list(get_size_to_one(input.size()[1:]))
        size[0] = -1

        weight = self.weight
        weight_graph *= weight.view(*size, *get_size_to_one(origin_size))
        return weight_graph, bias_graph


class NormNone(_Norm):
    def forward(self, input):
        return input

    def forward_graph(self, _: Tensor, weight_graph: Tensor = None, bias_graph: Tensor = None) -> Tuple[Tensor, Tensor]:
        return weight_graph, bias_graph


class Norm1d(_Norm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class Norm2d(_Norm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class Norm3d(_Norm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
