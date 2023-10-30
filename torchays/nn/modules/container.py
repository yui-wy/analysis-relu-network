import torch.nn as nn

from .base import Module


class Sequential(Module, nn.Sequential):
    __doc__ = nn.Sequential.__doc__

    def __init__(self, *arg) -> None:
        super(Sequential, self).__init__(*arg)
        self._check_modules()

    def forward(self, input):
        return nn.Sequential.forward(self, input)

    def _check_modules(self):
        for module in self:
            assert isinstance(module, Module), "child modules must be BaseModule."
