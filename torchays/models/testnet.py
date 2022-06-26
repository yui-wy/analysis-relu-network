import torchays.modules as ann


class TestTNetLinear(ann.AysBaseModule):
    def __init__(
            self,
            in_features=2,
            nNum: tuple = [32, 32, 32],
            n_classes=2,
            norm_layer=ann.AysBatchNorm1d):
        super(TestTNetLinear, self).__init__()
        self.numLayers = len(nNum)
        self.reLUNum = self.numLayers-1
        self.relu = ann.AysReLU()
        self._norm_layer = norm_layer
        self.add_module("0", ann.AysLinear(in_features, nNum[0], bias=True))
        self.add_module(f"{0}_norm", self._norm_layer(nNum[0]))
        for i in range(self.numLayers-1):
            self.add_module(f"{i+1}", ann.AysLinear(nNum[i], nNum[i+1], bias=False))
            self.add_module(f"{i+1}_norm", self._norm_layer(nNum[i+1]))
        self.add_module(f"{self.numLayers}", ann.AysLinear(nNum[-1], n_classes, bias=True))

    def forward(self, x):
        x = self._modules['0'](x)
        x = self.relu(x)
        for i in range(1, self.numLayers):
            x = self._modules[f'{i}'](x)
            x = self._modules[f"{i}_norm"](x)
            x = self.relu(x)
        x = self._modules[f"{self.numLayers}"](x)
        return x

    def forward_graph_Layer(self, x, layer=0):
        assert layer >= 0, "'layer' must be greater than 0."
        x = self._modules['0'](x)
        if layer == 0:
            return x
        x = self.relu(x)
        for i in range(1, self.numLayers):
            x = self._modules[f'{i}'](x)
            x = self._modules[f"{i}_norm"](x)
            if layer == i:
                return x
            x = self.relu(x)
        x = self._modules[f"{self.numLayers}"](x)
        return x
