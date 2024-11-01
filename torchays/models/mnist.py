import torch

from .. import nn
from ..nn.modules import BIAS_GRAPH, WEIGHT_GRAPH, get_input
from ..analysis.model import Model

class MNISTNet(Model):
    def __init__(self, norm_layer=nn.BatchNorm1d):
        super(MNISTNet, self).__init__()


    def forward(self, x):
        pass



    def forward_layer(self, x, depth):
        return x