import torch

from .. import nn
from ..cpa.model import Model


class LeNet(Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.n_relu = 4
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.avg1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avg2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)

        x = self.relu(x)
        x = self.avg1(x)
        x = self.conv2(x)

        x = self.relu(x)
        x = self.avg2(x)
        x = self.flatten(x)
        x = self.linear1(x)

        x = self.relu(x)
        x = self.linear2(x)

        x = self.relu(x)
        x = self.linear3(x)
        return x

    def forward_layer(self, x, depth):
        if depth < 0:
            depth = 4
        x = self.conv1(x)
        if depth == 0:
            return x
        x = self.relu(x)
        x = self.avg1(x)
        x = self.conv2(x)
        if depth == 1:
            return x
        x = self.relu(x)
        x = self.avg2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        if depth == 2:
            return x
        x = self.relu(x)
        x = self.linear2(x)
        if depth == 3:
            return x
        x = self.relu(x)
        x = self.linear3(x)
        return x
