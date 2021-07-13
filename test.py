import numpy as np
import torch
import torch.nn as nn
from analysis_lib import analysisNet
import torchvision.models.resnet

GPU_ID = 0
device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(5)
torch.cuda.manual_seed_all(5)
np.random.seed(5)


class TestNet(analysisNet.AnalysisNet):
    def __init__(self, input_size=(2,)):
        super(TestNet, self).__init__(input_size)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.avg = nn.AvgPool2d(2, 1)
        self.linear = nn.Linear(16, 2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.avg(x)

        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x

    def forward_graph(self, x, pre_weight_graph=None, pre_bias_graph=None):
        out, weight_graph, bias_graph = self.analysis_module(x, self.conv1, pre_weight_graph, pre_bias_graph)
        out, weight_graph, bias_graph = self.analysis_module(out, self.bn1, weight_graph, bias_graph)
        out, weight_graph, bias_graph = self.analysis_module(out, self.relu, weight_graph, bias_graph)
        out, weight_graph, bias_graph = self.analysis_module(out, self.conv2, weight_graph, bias_graph)
        out, weight_graph, bias_graph = self.analysis_module(out, self.avg, weight_graph, bias_graph)
        out = out.reshape(bias_graph.size(0), -1)
        bias_graph = bias_graph.reshape(bias_graph.size(0), -1)
        weight_graph = weight_graph.reshape(weight_graph.size(0), -1, *self._input_size)
        out, weight_graph, bias_graph = self.analysis_module(out, self.linear, weight_graph, bias_graph)
        return out, weight_graph, bias_graph


net = TestNet((3, 8, 8)).to(device)
net.eval()

with torch.no_grad():
    data = torch.randn(2, 3, 8, 8)
    output, weight_graph, bias_praph = net.forward_graph(data)
    print(output)
    # print(weight_graph)
    # print(bias_praph)
    for i in range(output.size(0)):
        output = (weight_graph[i] * data[i]).sum(dim=(1, 2, 3)) + bias_praph[i]
        print(output)
