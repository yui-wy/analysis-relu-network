from torch import Tensor
import torch
import torchays.modules as ann
from typing import List


class TestResNet(ann.AysBaseModule):
    def __init__(self, in_features: int, layers: List[int], first_features: int = 32, n_classes: int = 2):
        super(TestResNet, self).__init__(self)
        self.num_layers = len(layers)
        self.relu = ann.AysReLU()
        self._norm_layer = ann.AysBatchNorm1d
        self.in_features = in_features
        self.linear1 = ann.AysLinear(in_features, first_features)
        self.bn1 = self._norm_layer(first_features)
        self.last_linear = ann.AysLinear(layers[-1], n_classes)
        self.add_module(f"{self.numLayers}", ann.AysLinear(layers[-1], n_classes))
        self._make_layers(first_features, layers)

    def _make_layers(self, first_features: int, layers: List[int]):
        self.linear_res = ann.AysLinear(first_features, layers[0])
        for i in range(len(layers)-1):
            self.add_module(f"linear_{i}_1", ann.AysLinear(layers[i], layers[i], bias=False))
            self.add_module(f"bn_{i}_1", self._norm_layer(layers[i]))
            self.add_module(f"linear_{i}_2", ann.AysLinear(layers[i], layers[i+1], bias=False))
            self.add_module(f"bn_{i}_2", self._norm_layer(layers[i+1]))
            # downsample
            self.add_module(f"linear_{i}_d", ann.AysLinear(layers[i], layers[i+1], bias=False))
            self.add_module(f"bn_{i}_d", self._norm_layer(layers[i+1]))

    def forward(self, x: Tensor):
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear_res(out)
        # ================================
        # res
        for i in range(self.num_layers):
            out1 = self._modules[f"linear_{i}_1"](out)
            out1 = self._modules[f"bn_{i}_1"](out1)
            out1 = self.relu(out1)
            out1 = self._modules[f"linear_{i}_2"](out1)
            out1 = self._modules[f"bn_{i}_2"](out1)
            # downsample
            out1 = self._modules[f"linear_{i}_d"](out1)
            out1 = self._modules[f"bn_{i}_d"](out1)

            out += out1
            out = self.relu(out)
        # ================================
        out = self.last_linear(out)

        return out

    def forward_graph_Layer(self, x: Tensor, layer=-1):
        assert layer >= 0, "'layer' must be greater than 0."
        out = self.linear1(x)
        out = self.bn1(out)
        if layer == 0:
            return out
        out = self.relu(out)
        out = self.linear_res(out)
        # ================================
        # res
        for i in range(self.num_layers):
            out1 = self._modules[f"linear_{i}_1"](out)
            out1 = self._modules[f"bn_{i}_1"](out1)
            # relu
            if layer == (i*2+1):
                return out1
            out1 = self.relu(out1)
            out1 = self._modules[f"linear_{i}_2"](out1)
            out1 = self._modules[f"bn_{i}_2"](out1)
            # downsample
            out1 = self._modules[f"linear_{i}_d"](out1)
            out1 = self._modules[f"bn_{i}_d"](out1)

            out = self._forward_plus(out1, out)
            # relu
            if layer == (i*2+2):
                return out
            out = self.relu(out)
        # ================================
        out = self.last_linear(out)

        return out

    def _forward_plus(self, input, identity):
        if self.graphing:
            x, graph = self.get_input(input)
            id_x, id_graph = self.get_input(identity)
            o1 = x + id_x
            o2 = graph["weight_graph"] + id_graph["weight_graph"]
            o3 = graph["bias_graph"] + id_graph["bias_graph"]
            output = (o1, {
                "weight_graph": o2,
                "bias_graph": o3,
            })
        else:
            if not isinstance(input, tuple):
                output = input + identity
        return output
