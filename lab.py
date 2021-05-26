import math
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import dataloader

from analysis_lib.utils import areaUtils
from dataset import cifar, mnist
from nets.TestNet import TestMNISTNet

TAG = "Linear-16x4"
GPU_ID = 0
MAX_EPOCH = 100
BATCH_SIZE = 128
LR = 1e-3
ROOT_DIR = osp.abspath("./")
DATA_ROOT_DIR = osp.join(ROOT_DIR, 'data', 'mnist')
SAVE_DIR = osp.join(ROOT_DIR, 'cache', 'mnist', TAG)
MODEL_DIR = osp.join(SAVE_DIR, 'model')
if not osp.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
if not osp.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')

torch.manual_seed(5)
torch.cuda.manual_seed_all(5)
np.random.seed(5)


def transform_test(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.1307, ], [0.3081, ])
        # torchvision.transforms.Normalize([0, 0, 0], [1, 1, 1])
    ])
    img = transform(img)
    return img


def accuracy(x, classes):
    arg_max = torch.argmax(x, dim=1).long()
    eq = torch.eq(classes, arg_max)
    return torch.sum(eq).float()


def val_net(net, val_dataloader):
    net.eval()
    with torch.no_grad():
        val_accuracy_sum = 0
        for _, (x, y) in enumerate(val_dataloader, 1):
            x, y = x.to(device), y.long().to(device)
            x = net(x.view(-1, 784))
            val_acc = accuracy(x, y)
            val_accuracy_sum += val_acc
        val_accuracy_sum /= len(val_dataloader.dataset)
    return val_accuracy_sum


def getRegion(net, name, logPath, au, countLayers):
    num = au.getAreaNum(net, 1, countLayers=countLayers)
    logPath.write(f"Key: {name}; RegionNum: {num}\n")
    logPath.flush()
    return num


if __name__ == "__main__":
    regionLog = open(os.path.join(SAVE_DIR, "region.log"), 'w')

    net = TestMNISTNet((784,)).to(device)
    au = areaUtils.AnalysisReLUNetUtils(device=device)
    # num = au.getAreaNum(net, 1, countLayers=4)

    train_mnist_set = mnist.MNIST(DATA_ROOT_DIR, transform=transform_test)
    val_mnist_set = mnist.MNIST(DATA_ROOT_DIR, transform=transform_test, train=False)

    trainLoader = dataloader.DataLoader(train_mnist_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    valLoader = dataloader.DataLoader(val_mnist_set, batch_size=BATCH_SIZE, pin_memory=True, num_workers=2)

    optim = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    regionNum = {}
    steps = [10, 30, 70, 100, 150]
    epochs = [1, 2, 4, 6, 8, 10, 15, 20]

    # regionNum[f"epoch-0"] = getRegion(net, f"epoch-0", regionLog, au, 3)
    for i in range(MAX_EPOCH):
        net.train()
        totalStep = math.ceil(len(train_mnist_set) / BATCH_SIZE)
        for j, (x, y) in enumerate(trainLoader, 1):
            x, y = x.to(device), y.long().to(device)
            x = net(x.view(-1, 784))
            loss = ce(x, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            acc = accuracy(x, y) / x.size(0)

            if i == 1 and (j in steps):
                net.eval()
                # regionNum[f"epoch-{i}-{j}"] = getRegion(net, f"epoch-{i}-{j}", regionLog, au, 3)
            print(f"Epoch: {i+1} / {MAX_EPOCH}, Loss: {loss:.4f}, Acc: {acc:.4f}")

        if (i in epochs):
            net.eval()
            # regionNum[f"epoch-{i}"] = getRegion(net, f"epoch-{i}", regionLog, au, 3)

        val_acc = val_net(net, valLoader)
        print(f"Epoch: {i+1} / {MAX_EPOCH}, Val_Acc: {val_acc:.4f}")
        if (i + 1) % 5 == 0:
            print("Save net....")
            torch.save(net.state_dict(), osp.join(MODEL_DIR, f'net_{i+1}.pth'))

    torch.save(regionNum, osp.join(SAVE_DIR, "regionNum.pkl"))