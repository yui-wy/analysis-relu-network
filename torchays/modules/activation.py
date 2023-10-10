import torch
import torch.nn as nn
from torchays.modules import base


class ParamReLU(base.BaseModule):
    def __init__(self, active_slope: float = 1.0, negative_slope: float = 0.0) -> None:
        super().__init__()
        self._active_slope = active_slope
        self._negative_slope = negative_slope

    def forward(self, _):
        return NotImplementedError

    def forward_graph(self, x, weight_graph=None, bias_graph=None):
        input_size = self._get_input_size(x, weight_graph)
        graph_size = weight_graph.size()
        # ((*x.shape)), (*input_size)), ((*x.shape))
        wg, bg = torch.zeros(graph_size, device=x.device), torch.zeros(
            *x.size(), device=x.device
        )
        active_slope = torch.ones((1), device=x.device) * self._active_slope
        negative_slope = torch.ones((1), device=x.device) * self._negative_slope
        x_relu_hot = torch.where(x > 0, active_slope, negative_slope)
        wg += x_relu_hot.view(*x_relu_hot.size(), *self._get_size_to_one(input_size))
        bg += x_relu_hot
        if weight_graph is None:
            weight_graph, bias_graph = 1, 0
        wg *= weight_graph
        bg *= bias_graph

        return wg, bg

    def train(self, mode: bool = True):
        return base.BaseModule.train(self, mode=mode)

    def eval(self):
        return base.BaseModule.eval(self)


class ReLU(nn.ReLU, ParamReLU):
    __doc__ = nn.ReLU.__doc__

    def __init__(self, inplace: bool = False) -> None:
        nn.ReLU.__init__(self, inplace)
        ParamReLU.__init__(self, 1, 0)

    def forward(self, input):
        return self._forward(super().forward, input)


class LeakyRule(nn.LeakyReLU, ParamReLU):
    __doc__ = nn.LeakyReLU.__doc__

    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        nn.LeakyReLU.__init__(self, negative_slope, inplace)
        ParamReLU.__init__(self, 1, negative_slope)

    def forward(self, input):
        return self._forward(super().forward, input)
