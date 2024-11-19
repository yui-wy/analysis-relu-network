import matplotlib.pyplot as plt
import numpy as np
import polytope as pc
import torch

from torchays import nn
from torchays.cpa import CPA
from torchays.cpa.handler import BaseHandler

GPU_ID = 0
device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(5)
torch.cuda.manual_seed_all(5)
np.random.seed(5)


# 通过本代码实现的网络结构模块，来构成一个需要分析的网络。
# 这个网络可以加载同结构的网络的参数。
class TestNet(nn.Module):
    def __init__(self, input_size=(2,)):
        super(TestNet, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size[0], 16, bias=True)
        self.fc2 = nn.Linear(16, 16, bias=True)
        self.fc4 = nn.Linear(16, 3, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc4(x)

        return x

    # 实现CPA分析模块必须的函数，用来标识分段激活函数前的网络层的输出。
    def forward_layer(self, x, depth=-1):
        x = self.fc1(x)
        if depth == 0:
            return x
        x = self.relu(x)
        x = self.fc2(x)
        if depth == 1:
            return x
        x = self.relu(x)
        x = self.fc4(x)
        return x


net = TestNet((2,)).to(device)


# 给出一个"handler", 用来处理生成的CPA的数据。
class Handler(BaseHandler):
    def __init__(self) -> None:
        self._init_region()

    def _init_region(self):
        self.funs = list()
        self.regions = list()
        self.points = list()
        return self

    def region(self, fun: torch.Tensor, region: torch.Tensor, point: torch.Tensor) -> None:
        self.funs.append(fun.cpu().numpy())
        self.regions.append(region.cpu().numpy())
        self.points.append(point.cpu().numpy())

    def inner_hyperplanes(self, p_funs: torch.Tensor, p_regions: torch.Tensor, c_funs: torch.Tensor, intersect_funs: torch.Tensor | None, n_regions: int, depth: int) -> None:
        return


# 初始化CPA模块分析的类
cpa = CPA(device=device)

handler = Handler()
# 对网络进行cpa的分析。
num = cpa.start(net, 1, handler=handler)

ax = plt.subplot()
for i in range(num):
    func, region, point = handler.funs[i], handler.regions[i], handler.points[i]
    func = -region.view(-1, 1) * func
    func = func.numpy()
    A, B = func[:, :-1], -func[:, -1]

    # 在输入为2D的情况下，绘制区域图片
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
