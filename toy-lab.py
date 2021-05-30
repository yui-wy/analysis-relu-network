import logging
import math
import os
import time
import polytope as pc
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import *
from torch.utils.data import dataloader

from analysis_lib.utils import areaUtils
from nets.TestNet import TestTNetLinear

TAG = "Linear-16x3"
GPU_ID = 0
MAX_EPOCH = 100
BATCH_SIZE = 32
LR = 1e-3
ROOT_DIR = os.path.abspath("./")
SAVE_DIR = os.path.join(ROOT_DIR, 'cache', 'toy', TAG)
MODEL_DIR = os.path.join(SAVE_DIR, 'model')
LAB_DIR = os.path.join(SAVE_DIR, 'lab')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LAB_DIR):
    os.makedirs(LAB_DIR)

device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')

torch.manual_seed(5)
torch.cuda.manual_seed_all(5)
np.random.seed(5)


class ToyDateBase(dataloader.Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.x = (x - np.min(x)) / (np.max(x) - np.min(x))
        self.y = y

    def __getitem__(self, index):
        x, target = torch.from_numpy(self.x[index]), self.y[index]
        return x, target

    def __len__(self):
        return self.x.shape[0]


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


def getDataSet(n_sample, noise, random_state):
    x, y = make_moons(n_sample, noise=noise, random_state=random_state)
    dataset = ToyDateBase(x, y)
    return dataset


def train():
    dataset = getDataSet(2000, 0.2, 5)
    totalStep = math.ceil(len(dataset) / BATCH_SIZE)
    trainLoader = dataloader.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    net = TestTNetLinear((2,)).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4, betas=[0.9, 0.999])
    ce = torch.nn.CrossEntropyLoss()
    stepEpoch = [0.1, 0.2, 0.3, 0.5, 0.8]
    steps = [math.floor(v * totalStep) for v in stepEpoch]
    torch.save(net.state_dict(), os.path.join(MODEL_DIR, f'net_0.pth'))
    for i in range(MAX_EPOCH):
        net.train()
        for j, (x, y) in enumerate(trainLoader, 1):
            x, y = x.float().to(device), y.long().to(device)
            x = net(x)
            loss = ce(x, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            acc = accuracy(x, y) / x.size(0)

            if i == 1 and (j in steps):
                net.eval()
                idx = steps.index(j)
                torch.save(net.state_dict(), os.path.join(MODEL_DIR, f'net_{stepEpoch[idx]}.pth'))
            print(f"Epoch: {i+1} / {MAX_EPOCH}, Step: {j} / {totalStep}, Loss: {loss:.4f}, Acc: {acc:.4f}")

        print(f"Epoch: {i+1} / {MAX_EPOCH}")
        if (i + 1) % 1 == 0:
            print("Save net....")
            torch.save(net.state_dict(), os.path.join(MODEL_DIR, f'net_{i+1}.pth'))


def lab():
    dataset = getDataSet(2000, 0.2, 5)
    net = TestTNetLinear((2,))
    net.eval()
    au = areaUtils.AnalysisReLUNetUtils(device=device)
    epoch = [0, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 100]
    # epoch = [6, 8, 30, 80]
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
            regionNum = au.getAreaNum(net, 1, countLayers=2, saveArea=True)
            funcs, areas, points = au.getAreaData()
            # 绘图, 保存
            drawRegionImg(regionNum, funcs, areas, points, saveDir)
            # 保存数据
            dataSaveDict = {
                "funcs": funcs,
                "areas": areas,
                "points": points,
                "regionNum": regionNum,
            }
            torch.save(dataSaveDict, os.path.join(saveDir, "dataSave.pkl"))


def drawRegionImg(regionNum, funcs, areas, points, saveDir):
    fig = plt.figure(0, figsize=(8, 7), dpi=600)
    ax = fig.subplots()
    ax.cla()
    ax.tick_params(labelsize=15)
    for i in range(regionNum):
        func, area, point = funcs[i], areas[i], points[i]
        func = -area.view(-1, 1) * func
        func = func.numpy()
        A, B = func[:, :-1], -func[:, -1]
        p = pc.Polytope(A, B)
        p.plot(ax, color=np.random.uniform(0.0, 0.95, 3), alpha=1., linestyle='-', linewidth=0.01, edgecolor='w')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.savefig(os.path.join(saveDir, "regionImg.png"))
    plt.clf()
    plt.close()


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
    train()
    lab()
