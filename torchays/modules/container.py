import torch
import torch.nn as nn
from torchays.modules import base


class AysSequential(nn.Sequential, base.AysBaseModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input):
        if self.graphing:
            for child_modules in self._modules.values():
                assert isinstance(child_modules, base.AysBaseModule), "child modules must be AysBaseModule."
                input = child_modules(input)
            return input
        else:
            return super().forward(input)

    # def forward(self, input):
    #     if self.graphing:
    #         args, kwargs = self.get_input(input)
    #         output = (super().forward(*args), self.get_graph(*args, **kwargs))
    #     else:
    #         output = super().forward(*input)
    #     return output

    def train(self, mode: bool = True):
        return base.AysBaseModule.train(self, mode=mode)

    def eval(self):
        return base.AysBaseModule.eval(self)
