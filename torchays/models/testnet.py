import torchays.modules as ays


class TestTNetLinear(ays.BaseModule):
    def __init__(
        self,
        in_features=2,
        layers: tuple = [32, 32, 32],
        n_classes=2,
        norm_layer=ays.BatchNorm1d,
    ):
        super(TestTNetLinear, self).__init__()
        self.n_layers = len(layers)
        self.n_relu = self.n_layers - 1
        self.relu = ays.ReLU()
        self._norm_layer = norm_layer
        self.add_module("0", ays.Linear(in_features, layers[0], bias=True))
        self.add_module(f"{0}_norm", self._norm_layer(layers[0]))
        for i in range(self.n_layers - 1):
            self.add_module(f"{i+1}", ays.Linear(layers[i], layers[i + 1], bias=True))
            self.add_module(f"{i+1}_norm", self._norm_layer(layers[i + 1]))
        self.add_module(
            f"{self.n_layers}", ays.Linear(layers[-1], n_classes, bias=True)
        )

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

    def forward_graph_Layer(self, x, layer=0):
        assert layer >= 0, "'layer' must be greater than 0."
        x = self._modules['0'](x)
        if layer == 0:
            return x
        x = self.relu(x)
        for i in range(1, self.n_layers):
            x = self._modules[f'{i}'](x)
            x = self._modules[f"{i}_norm"](x)
            if layer == i:
                return x
            x = self.relu(x)
        x = self._modules[f"{self.n_layers}"](x)
        return x
