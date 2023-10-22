import torch.nn as nn

from torchays.nn.modules import base


class Sequential(nn.Sequential, base.Module):
    __doc__ = nn.Sequential.__doc__

    def __init__(self, *arg) -> None:
        super(Sequential, self).__init__(*arg)
        self._check_modules()

    def _check_modules(self):
        for module in self:
            assert isinstance(module, base.BaseModule), "child modules must be BaseModule."
