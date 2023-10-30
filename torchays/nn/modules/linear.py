import torch
import torch.nn as nn

from .base import Module


class Linear(Module, nn.Linear):
    __doc__ = nn.Linear.__doc__

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward_graph(self, x, weight_graph=None, bias_graph=None):
        """
        Analyzing linear \n
        graph_size: (n, out_feature, (*origin_size)))
        """
        # bias_graph
        bias_graph = torch.zeros_like(x, device=x.device) if bias_graph is None else bias_graph
        bias_graph = nn.Linear.forward(self, bias_graph)
        # weight_graph
        origin_size = self._origin_size(x, weight_graph)
        graph_size = (*bias_graph.size(), *origin_size)
        # create weight_graph
        # (n, out_features, (*origin_size))
        wg = torch.zeros(graph_size, device=x.device)
        # create hook of x
        # (n, out_features, in_features)
        hook_x = torch.zeros(x.size(0), self.out_features, self.in_features, device=x.device) + self.weight

        if weight_graph is None:
            wg += hook_x
        else:
            weight_graph_s = weight_graph.view(weight_graph.size(0), self.in_features, -1)
            wg = torch.matmul(hook_x, weight_graph_s).view(-1, self.out_features, *origin_size)

        return wg, bias_graph
