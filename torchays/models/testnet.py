from typing import List

import torch

from .. import nn
from ..nn.modules import BIAS_GRAPH, WEIGHT_GRAPH, get_input


class TestTNetLinear(nn.Module):
    def __init__(
        self,
        in_features=2,
        layers: tuple = [32, 32, 32],
        n_classes=2,
        norm_layer=nn.BatchNorm1d,
    ):
        super(TestTNetLinear, self).__init__()
        self.name = f"Linear-{layers}".replace(' ', '')
        self.n_layers = len(layers)
        self.n_relu = self.n_layers - 1
        self.relu = nn.ReLU()
        self._norm_layer = norm_layer
        self.add_module("0", nn.Linear(in_features, layers[0], bias=True))
        self.add_module(f"{0}_norm", self._norm_layer(layers[0]))
        for i in range(self.n_layers - 1):
            self.add_module(f"{i+1}", nn.Linear(layers[i], layers[i + 1], bias=True))
            self.add_module(f"{i+1}_norm", self._norm_layer(layers[i + 1]))
        self.add_module(f"{self.n_layers}", nn.Linear(layers[-1], n_classes, bias=True))

    def forward(self, x):
        x = self._modules['0'](x)
        x = self._modules["0_norm"](x)
        x = self.relu(x)
        for i in range(1, self.n_layers):
            x = self._modules[f'{i}'](x)
            x = self._modules[f"{i}_norm"](x)
            x = self.relu(x)
        x = self._modules[f"{self.n_layers}"](x)
        return x

    def forward_graph_Layer(self, x, depth=0):
        assert depth >= 0, "'layer' must be greater than 0."
        x = self._modules['0'](x)
        x = self._modules["0_norm"](x)
        if depth == 0:
            return x
        x = self.relu(x)
        for i in range(1, self.n_layers):
            x = self._modules[f'{i}'](x)
            x = self._modules[f"{i}_norm"](x)
            if depth == i:
                return x
            x = self.relu(x)
        x = self._modules[f"{self.n_layers}"](x)
        return x


class TestResNet(nn.Module):
    """Not cnn, use linear."""

    def __init__(
        self,
        in_features: int,
        layers: List[int],
        first_features: int = 32,
        n_classes: int = 2,
        norm_layer=nn.BatchNorm1d,
        is_no_res: bool = False,
    ):
        super(TestResNet, self).__init__()
        self.name = f"Resnet-Full-{layers}".replace(' ', '')
        self._is_no_res = is_no_res
        self.n_layers = len(layers)
        self.n_relu = (self.n_layers - 1) * 2 + 1
        self.relu = nn.ReLU()
        self._norm_layer = norm_layer
        self.in_features = in_features
        self.linear1 = nn.Linear(in_features, first_features)
        self.norm1 = self._norm_layer(first_features)
        self.last_linear = nn.Linear(layers[-1], n_classes)
        self._make_layers(first_features, layers)

    def _make_layers(self, first_features: int, layers: List[int]):
        self.linear_res = nn.Linear(first_features, layers[0])
        for i in range(self.n_layers - 1):
            self.add_module(f"linear_{i}_1", nn.Linear(layers[i], layers[i], bias=True))
            self.add_module(f"norm_{i}_1", self._norm_layer(layers[i]))
            self.add_module(f"linear_{i}_2", nn.Linear(layers[i], layers[i + 1], bias=True))
            self.add_module(f"norm_{i}_2", self._norm_layer(layers[i + 1]))
            # downsample
            if layers[i] != layers[i + 1]:
                self.add_module(f"linear_{i}_d", nn.Linear(layers[i], layers[i + 1], bias=True))
                self.add_module(f"norm_{i}_d", self._norm_layer(layers[i + 1]))

    def forward(self, x: torch.Tensor):
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.linear_res(out)
        # ================================
        # res
        for i in range(self.n_layers - 1):
            out1 = self._modules[f"linear_{i}_1"](out)
            out1 = self._modules[f"norm_{i}_1"](out1)
            out1 = self.relu(out1)
            out1 = self._modules[f"linear_{i}_2"](out1)
            out1 = self._modules[f"norm_{i}_2"](out1)
            if self._is_no_res:
                out = out1
            else:
                # downsample
                if self._modules.get(f"linear_{i}_d") != None and self._modules.get(f"norm_{i}_d") != None:
                    out = self._modules[f"linear_{i}_d"](out)
                    out = self._modules[f"norm_{i}_d"](out)

                out = self._forward_plus(out1, out)
            out = self.relu(out)
        # ================================
        out = self.last_linear(out)

        return out

    def forward_graph_Layer(self, x: torch.Tensor, depth=-1):
        assert depth >= 0, "'layer' must be greater than 0."
        out = self.linear1(x)
        out = self.norm1(out)
        if depth == 0:
            return out
        out = self.relu(out)
        out = self.linear_res(out)
        # ================================
        # res
        for i in range(self.n_layers - 1):
            out1 = self._modules[f"linear_{i}_1"](out)
            out1 = self._modules[f"norm_{i}_1"](out1)
            # relu
            if depth == (i * 2 + 1):
                return out1
            out1 = self.relu(out1)
            out1 = self._modules[f"linear_{i}_2"](out1)
            out1 = self._modules[f"norm_{i}_2"](out1)
            if self._is_no_res:
                out = out1
            else:
                # downsample
                if self._modules.get(f"linear_{i}_d") != None and self._modules.get(f"norm_{i}_d") != None:
                    out = self._modules[f"linear_{i}_d"](out)
                    out = self._modules[f"norm_{i}_d"](out)

                out = self._forward_plus(out1, out)
            # relu
            if depth == (i * 2 + 2):
                return out
            out = self.relu(out)
        # ================================
        out = self.last_linear(out)

        return out

    def _forward_plus(self, input, identity):
        if self.graphing:
            x, graph = get_input(input)
            id_x, id_graph = get_input(identity)
            o1 = x[0] + id_x[0]
            o2 = graph[WEIGHT_GRAPH] + id_graph[WEIGHT_GRAPH]
            o3 = graph[BIAS_GRAPH] + id_graph[BIAS_GRAPH]
            return o1, {
                WEIGHT_GRAPH: o2,
                BIAS_GRAPH: o3,
            }
        else:
            if not isinstance(input, tuple):
                return input + identity
