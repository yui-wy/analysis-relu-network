import matplotlib.pyplot as plt
import numpy as np
import polytope as pc
import torch

import torchays.modules as ays
from torchays.utils import areaUtils

GPU_ID = 0
device = (
    torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
)
torch.manual_seed(5)
torch.cuda.manual_seed_all(5)
np.random.seed(5)


class TestNet(ays.BaseModule):
    def __init__(self, input_size=(2,)):
        super(TestNet, self).__init__()
        self.relu = ays.ReLU()
        self.fc1 = ays.Linear(input_size[0], 16, bias=True)
        self.fc2 = ays.Linear(16, 16, bias=True)
        self.fc4 = ays.Linear(16, 3, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc4(x)

        return x

    def forward_graph_Layer(self, x, layer=-1):
        x = self.fc1(x)
        if layer == 0:
            return x
        x = self.relu(x)
        x = self.fc2(x)
        if layer == 1:
            return x
        x = self.relu(x)
        x = self.fc4(x)
        return x


net = TestNet((2,)).to(device)

au = areaUtils.AnalysisReLUNetUtils(device=device)
num = au.getAreaNum(net, 1, countLayers=1, isSaveArea=True)
funcs, areas, points = au.getAreaData()

ax = plt.subplot()
for i in range(num):
    #  to <= 0
    func, area, point = funcs[i], areas[i], points[i]
    # print(f"Func: {func}, area: {area}, point: {point}")
    func = -area.view(-1, 1) * func
    func = func.numpy()
    A, B = func[:, :-1], -func[:, -1]
    p = pc.Polytope(A, B)
    p.plot(
        ax,
        color=np.random.uniform(0.0, 0.95, 3),
        alpha=1.0,
        linestyle='-',
        linewidth=0.2,
        edgecolor='w',
    )

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.axis('off')
plt.show()
