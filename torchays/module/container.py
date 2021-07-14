import torch
import torch.nn as nn
from torchays.module import base


class AysSequential(nn.Sequential, base.AysBaseModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input: torch.Tensor, weight_graph=None, bias_graph=None):
        if self._is_graph:
            return self.forward_graph(input, weight_graph, bias_graph)
        else:
            return super().forward(input)

    def forward_graph(self, x, weight_graph=None,  bias_graph=None):
        for child_modules in self._modules.values():
            assert isinstance(child_modules, base.AysBaseModule), "child modules must be aysBaseModule."
            output, weight_graph, bias_graph = child_modules(output, weight_graph, bias_graph)
        return output, weight_graph, bias_graph
