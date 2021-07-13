import torch
import torch.nn as nn
from torchays.module import base


class Conv2d(nn.Conv2d, base.BaseModule):
    __doc__ = nn.Conv2d.__doc__

    def __init__(self):
        
        pass