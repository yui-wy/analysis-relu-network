import torch.nn as nn

from torchays.modules import base


class Sequential(nn.Sequential, base.BaseModule):
    """This module does not test. It may have some bug."""

    __doc__ = nn.Sequential.__doc__

    def forward(self, input):
        if self.graphing:
            for child_modules in self._modules.values():
                assert isinstance(
                    child_modules, base.BaseModule
                ), "child modules must be BaseModule."
                input = child_modules(input)
            return input
        else:
            return super().forward(input)

    def train(self, mode: bool = True):
        return base.BaseModule.train(self, mode=mode)
