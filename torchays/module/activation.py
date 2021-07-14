import torch
import torch.nn as nn
from torchays.module import base


class AysReLU(nn.ReLU, base.AysBaseModule):
    __doc__ = nn.ReLU.__doc__

    def forward(self, input: torch.Tensor, weight_graph=None, bias_graph=None):
        output = super().forward(input)
        if self._is_graph:
            weight_graph, bias_graph = self.forward_graph(input, weight_graph, bias_graph)
            return output, weight_graph, bias_graph
        return output

    def forward_graph(self, x, weight_graph=None,  bias_graph=None):
        """
        Analyzing ReLU \n
        Using inplace = False \n
        graph_size: ((*x.shape)), (*input_size))
        """
        if self.inplace:
            print('WARNING: \'inplace\' of ReLU is True!!!')
        input_size = x.size() if weight_graph is None else weight_graph.size()[len(x.size()):]
        graph_size = weight_graph.size()
        # ((*x.shape)), (*input_size)), ((*x.shape))
        wg, bg = torch.zeros(graph_size, device=x.device), torch.zeros(*x.size(), device=x.device)
        one_tensor, zero_tensor = torch.ones((1), device=x.device), torch.zeros((1), device=x.device)
        # ((*x.shape))
        x_relu_hot = torch.where(x > 0, one_tensor, zero_tensor)
        wg += x_relu_hot.view(*x_relu_hot.size(), self._get_size_to_one(input_size))
        bg += x_relu_hot
        if weight_graph is None:
            weight_graph, bias_graph = 1, 0
        wg *= weight_graph
        bg *= bias_graph

        return wg, bg
