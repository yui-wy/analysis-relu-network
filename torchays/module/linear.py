import torch
import torch.nn as nn
from torchays.module import base


class AysLinear(nn.Linear, base.AysBaseModule):
    __doc__ = nn.Linear.__doc__

    def forward(self, input: torch.Tensor, weight_graph=None, bias_graph=None):
        output = super().forward(input)
        if self._is_graph:
            weight_graph, bias_graph = self.forward_graph(input, weight_graph, bias_graph)
            return output, weight_graph, bias_graph
        return output

    def forward_graph(self, x, weight_graph=None,  bias_graph=None):
        """
        Analyzing linear \n
        graph_size: (n, out_feature, (*input_size)))
        """
        # bias_graph
        bias_graph = torch.zeros_like(x, device=x.device) if bias_graph is None else bias_graph
        bias_graph = super().forward(bias_graph)
        # weight_graph
        input_size = x.size() if weight_graph is None else weight_graph.size()[len(x.size()):]
        graph_size = (*bias_graph.size(), *input_size)
        # create weight_graph
        # (n, out_features, (*input_size))
        wg = torch.zeros(graph_size, device=self._device)
        # create hook of x
        # (n, out_features, in_features)
        hook_x = torch.zeros(x.size(0), self.out_features, self.in_features, device=x.device) + self.weight

        if weight_graph is None:
            wg += hook_x
        else:
            weight_graph_s = weight_graph.view(weight_graph.size(0), self.in_features, -1)
            wg = torch.matmul(hook_x, weight_graph_s).view(-1, self.out_feature, *input_size)

        return wg, bias_graph
