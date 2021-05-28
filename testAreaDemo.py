import time
import numpy as np
import polytope as pc
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from analysis_lib import analysisNet
from analysis_lib.utils import areaUtils

GPU_ID = 0
device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(5)
torch.cuda.manual_seed_all(5)
np.random.seed(5)


class TestNet(analysisNet.AnalysisNet):
    def __init__(self, input_size=(2,)):
        super(TestNet, self).__init__(input_size)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2, 32, bias=True)
        self.fc2 = nn.Linear(32, 32, bias=True)
        self.fc3 = nn.Linear(32, 32, bias=True)
        self.fc4 = nn.Linear(32, 3, bias=True)

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


net = TestNet((2,)).to(device)

au = areaUtils.AnalysisReLUNetUtils(device=device)
num = au.getAreaNum(net, 1, countLayers=3, saveArea=True)
print(num)
funcs, areas, points = au.getAreaData()

ax = plt.subplot()
for i in range(num):
    #  to <= 0
    func, area, point = funcs[i], areas[i], points[i]
    # print(f"Func: {func}, area: {area}, point: {point}")
    func = (1 - area * 2).view(-1, 1) * func
    func = func.numpy()
    A, B = func[:, :-1], -func[:, -1]
    p = pc.Polytope(A, B)
    p.plot(ax, color=np.random.uniform(0.0, 1., 3), alpha=1, linestyle='-', linewidth=0.2, edgecolor='w')

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()

# [60,60,60,60] 21min
