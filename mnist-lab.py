import logging
import math
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import dataloader

from analysis_lib.utils import areaUtils
from dataset import mnist
from analysis_lib.models.testnet import TestTNetLinear

TAG = "Linear-32x3"
N_NUM = [32, 32, 32]
GPU_ID = 0
MAX_EPOCH = 100
BATCH_SIZE = 256
LR = 1e-3
ROOT_DIR = osp.abspath("./")
DATA_ROOT_DIR = osp.join(ROOT_DIR, 'data', 'mnist')
SAVE_DIR = osp.join(ROOT_DIR, 'cache', 'mnist', TAG)
MODEL_DIR = osp.join(SAVE_DIR, 'model')
LAB_DIR = os.path.join(SAVE_DIR, 'lab')
if not osp.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
if not osp.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LAB_DIR):
    os.makedirs(LAB_DIR)

device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')

torch.manual_seed(5)
torch.cuda.manual_seed_all(5)
np.random.seed(5)


def transform_test(img):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.1307, ], [0.3081, ])
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


def train():

    net = TestTNetLinear((784,), N_NUM, 10).to(device)

    train_mnist_set = mnist.MNIST(DATA_ROOT_DIR, transform=transform_test)
    totalStep = math.ceil(len(train_mnist_set) / BATCH_SIZE)
    val_mnist_set = mnist.MNIST(DATA_ROOT_DIR, transform=transform_test, train=False)

    trainLoader = dataloader.DataLoader(train_mnist_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    valLoader = dataloader.DataLoader(val_mnist_set, batch_size=BATCH_SIZE, pin_memory=True)

    optim = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    stepEpoch = [0.1, 0.2, 0.3, 0.5, 0.8]
    steps = [math.floor(v * totalStep) for v in stepEpoch]
    torch.save(net.state_dict(), osp.join(MODEL_DIR, f'net_0.pth'))
    for i in range(MAX_EPOCH):
        net.train()
        for j, (x, y) in enumerate(trainLoader, 1):
            x, y = x.to(device), y.long().to(device)
            x = net(x.view(-1, 784))
            loss = ce(x, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            acc = accuracy(x, y) / x.size(0)

            if i == 0 and (j in steps):
                net.eval()
                idx = steps.index(j)
                val_acc = val_net(net, valLoader)
                torch.save(net.state_dict(), osp.join(MODEL_DIR, f'net_{stepEpoch[idx]}.pth'))
            print(f"Epoch: {i+1} / {MAX_EPOCH}, Loss: {loss:.4f}, Acc: {acc:.4f}")

        val_acc = val_net(net, valLoader)
        print(f"Epoch: {i+1} / {MAX_EPOCH}, Val_Acc: {val_acc:.4f}")
        if (i + 1) % 1 == 0:
            print("Save net....")
            torch.save(net.state_dict(), osp.join(MODEL_DIR, f'net_{i+1}.pth'))


def lab():
    val_mnist_set = mnist.MNIST(DATA_ROOT_DIR, transform=transform_test, train=False)
    valLoader = dataloader.DataLoader(val_mnist_set, batch_size=BATCH_SIZE, pin_memory=True)
    net = TestTNetLinear((784,), N_NUM, 10).to(device)
    net.eval()
    au = areaUtils.AnalysisReLUNetUtils(device=device)
    epoch = [0, 0.2, 0.5, 0.8, 1, 5, 8, 10, 20, 30, 50, 80, 100]
    modelList = os.listdir(MODEL_DIR)
    with torch.no_grad():
        for modelName in modelList:
            sign = float(modelName[4:-4])
            if sign not in epoch:
                continue
            print(f"Solve fileName: {modelName} ....")
            saveDir = os.path.join(LAB_DIR, os.path.splitext(modelName)[0])
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            au.logger = getLogger(saveDir, f"region-{os.path.splitext(modelName)[0]}")
            modelPath = os.path.join(MODEL_DIR, modelName)
            net.load_state_dict(torch.load(modelPath,  map_location='cpu'))
            net = net.to(device)
            acc = val_net(net, valLoader).cpu().numpy()
            print(f'Accuracy: {acc:.4f}')
            regionNum = au.getAreaNum(net, 1, countLayers=2, saveArea=True)
            funcs, areas, points = au.getAreaData()
            # 保存数据
            dataSaveDict = {
                "funcs": funcs,
                "areas": areas,
                "points": points,
                "regionNum": regionNum,
                "accuracy": acc,
            }
            torch.save(dataSaveDict, os.path.join(saveDir, "dataSave.pkl"))


def getLogger(saveDir, loggerName):
    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] - %(name)s : %(message)s')
    logName = "region.log"
    fh = logging.FileHandler(os.path.join(saveDir, logName), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


if __name__ == "__main__":
    # train()
    lab()
