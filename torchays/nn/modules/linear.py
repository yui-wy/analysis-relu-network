from typing import Tuple
import torch
from torch import Tensor
import torch.nn as nn

from .base import Module


class Linear(Module, nn.Linear):
    __doc__ = nn.Linear.__doc__

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward_graph(self, input: Tensor, weight_graph: Tensor = None, bias_graph: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        Analyzing linear \n
        graph_size: (n, out_feature, (*origin_size)))
        """
        # bias_graph
        bias_graph = torch.zeros_like(input, device=input.device, dtype=input.dtype) if bias_graph is None else bias_graph
        bias_graph = nn.Linear.forward(self, bias_graph)
        # weight_graph
        origin_size = self._origin_size(input, weight_graph)
        # create hook of x
        # (n, out_features, in_features)
        hook_x = torch.zeros(input.size(0), self.out_features, self.in_features, device=input.device, dtype=input.dtype) + self.weight
        if weight_graph is None:
            weight_graph = torch.zeros(((*bias_graph.size(), *origin_size)), device=input.device, dtype=input.dtype)
            weight_graph += hook_x
        else:
            weight_graph_s = weight_graph.reshape(weight_graph.size(0), self.in_features, -1)
            weight_graph = torch.matmul(hook_x, weight_graph_s).reshape(-1, self.out_features, *origin_size)

        return weight_graph, bias_graph
