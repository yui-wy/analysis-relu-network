import torch
from analysis_lib import analysisNet
import torch.nn as nn


class TestTNetLinear(analysisNet.AnalysisNet):
    def __init__(self, input_size=(2,)):
        super(TestTNetLinear, self).__init__(input_size)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size[0], 16, bias=True)
        self.fc2 = nn.Linear(16, 16, bias=True)
        self.fc3 = nn.Linear(16, 16, bias=True)
        self.fc4 = nn.Linear(16, 2, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        return x

    def forward_graph_Layer(self, x, Layer=-1, pre_weight_graph=None, pre_bias_graph=None):
        out, weight_graph, bias_graph = self.analysis_module(x, self.fc1, pre_weight_graph, pre_bias_graph)
        if Layer == 0:
            return out, weight_graph, bias_graph
        out, weight_graph, bias_graph = self.analysis_module(out, self.relu, weight_graph, bias_graph)
        out, weight_graph, bias_graph = self.analysis_module(out, self.fc2, weight_graph, bias_graph)
        if Layer == 1:
            return out, weight_graph, bias_graph
        out, weight_graph, bias_graph = self.analysis_module(out, self.relu, weight_graph, bias_graph)
        out, weight_graph, bias_graph = self.analysis_module(out, self.fc3, weight_graph, bias_graph)
        if Layer == 2:
            return out, weight_graph, bias_graph
        out, weight_graph, bias_graph = self.analysis_module(out, self.relu, weight_graph, bias_graph)
        out, weight_graph, bias_graph = self.analysis_module(out, self.fc4, weight_graph, bias_graph)

        return out, weight_graph, bias_graph


class TestTNetCNN(analysisNet.AnalysisNet):
    def __init__(self, input_size=(2,)):
        super(TestTNetCNN, self).__init__(input_size)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size[0], 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, 32, bias=True)
        self.fc4 = nn.Linear(32, 2, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)

        return x

    def forward_graph_Layer(self, x, Layer=-1, pre_weight_graph=None, pre_bias_graph=None):
        out, weight_graph, bias_graph = self.analysis_module(x, self.fc1, pre_weight_graph, pre_bias_graph)
        if Layer == 0:
            return out, weight_graph, bias_graph
        out, weight_graph, bias_graph = self.analysis_module(out, self.relu, weight_graph, bias_graph)
        out, weight_graph, bias_graph = self.analysis_module(out, self.fc2, weight_graph, bias_graph)
        if Layer == 1:
            return out, weight_graph, bias_graph
        out, weight_graph, bias_graph = self.analysis_module(out, self.relu, weight_graph, bias_graph)
        out, weight_graph, bias_graph = self.analysis_module(out, self.fc3, weight_graph, bias_graph)
        if Layer == 2:
            return out, weight_graph, bias_graph
        out, weight_graph, bias_graph = self.analysis_module(out, self.relu, weight_graph, bias_graph)
        out, weight_graph, bias_graph = self.analysis_module(out, self.fc4, weight_graph, bias_graph)

        return out, weight_graph, bias_graph
