from analysis_lib import analysisNet
import torch.nn as nn


class TestTNetLinear(analysisNet.AnalysisNet):
    def __init__(self, input_size=(2,), nNum: tuple = [32, 32, 32], n_classes=2):
        super(TestTNetLinear, self).__init__(input_size)
        self.numLayers = len(nNum)
        self.reLUNum = self.numLayers-1
        self.add_module("0", nn.Linear(input_size[0], nNum[0], bias=True))
        self.relu = nn.ReLU()
        for i in range(self.numLayers-1):
            fc = nn.Linear(nNum[i], nNum[i+1], bias=True)
            self.add_module(f"{i+1}", fc)
        self.add_module(f"{self.numLayers}", nn.Linear(nNum[-1], n_classes, bias=True))

    def forward(self, x):
        for i in range(self.numLayers):
            x = self._modules[f'{i}'](x)
            x = self.relu(x)
        x = self._modules[f"{self.numLayers}"](x)
        return x

    def forward_graph_Layer(self, x, layer=0, pre_weight_graph=None, pre_bias_graph=None):
        assert layer >= 0, "'layer' must be greater than 0."
        out, weight_graph, bias_graph = x, pre_weight_graph, pre_bias_graph
        for i in range(self.numLayers):
            out, weight_graph, bias_graph = self.analysis_module(out, self._modules[f'{i}'], weight_graph, bias_graph)
            if layer == i:
                return out, weight_graph, bias_graph
            out, weight_graph, bias_graph = self.analysis_module(out, self.relu, weight_graph, bias_graph)
        out, weight_graph, bias_graph = self.analysis_module(out, self._modules[f"{self.numLayers}"], weight_graph, bias_graph)
        return out, weight_graph, bias_graph
