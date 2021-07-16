import numpy as np
import torch
from torchays import modules


GPU_ID = 0
device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(5)
torch.cuda.manual_seed_all(5)
np.random.seed(5)


class TestNet(modules.AysBaseModule):
    def __init__(self):
        super(TestNet, self).__init__()
        self.relu = modules.AysReLU()
        self.conv1 = modules.AysConv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.bn1 = modules.AysBatchNorm2d(num_features=8)
        self.conv2 = modules.AysConv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.avg = modules.AysAvgPool2d(2, 1)
        self.linear = modules.AysLinear(16, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.avg(x)

        x = self.easy_forward(lambda x: torch.flatten(x, 1), x)

        x = self.linear(x)
        return x

    def forward_graph(self, x, weight_graph=None, bias_graph=None):
        input_size = self._get_input_size(x, weight_graph)
        bias_graph = bias_graph.reshape(bias_graph.size(0), -1)
        weight_graph = weight_graph.reshape(weight_graph.size(0), -1, *input_size)
        return weight_graph, bias_graph


net = TestNet().to(device)
data = torch.randn(2, 3, 8, 8)

print(net(data))

net.eval()
with torch.no_grad():
    output, graph = net(data)
    weight_graph, bias_graph = graph['weight_graph'], graph['bias_graph']
    print(output)
    # print(weight_graph.size())
    # print(bias_praph.size())
    for i in range(output.size(0)):
        output = (weight_graph[i] * data[i]).sum(dim=(1, 2, 3)) + bias_graph[i]
        print(output)
