import logging
import math
import os
from typing import Dict
import polytope as pc
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import *
from torch.utils.data import dataloader

from analysis_lib.utils import areaUtils
from nets.TestNet import TestTNetLinear

GPU_ID = 0
SEED = 5
# DATASET = f"random{SEED}"
DATASET = 'toy'
N_NUM = [5, 5, 5, 5]
N_SAMPLE = 1000
TAG = f"Linear-{N_NUM}".replace(' ', '')

MAX_EPOCH = 100
# SAVE_EPOCH = [0, 1, 5, 10, 30, 50, 80, 100, 200, 300, 400, 500, 800, 1000]
# SAVE_EPOCH = [0, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
SAVE_EPOCH = [0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 100]

BATCH_SIZE = 32
LR = 1e-3
ROOT_DIR = os.path.abspath("./")
SAVE_DIR = os.path.join(ROOT_DIR, 'cache', DATASET, TAG)
MODEL_DIR = os.path.join(SAVE_DIR, 'model')
LAB_DIR = os.path.join(SAVE_DIR, 'lab')
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(LAB_DIR):
    os.makedirs(LAB_DIR)

device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
COLOR = ('royalblue', 'limegreen', 'darkorchid', 'aqua', 'tomato', 'violet', 'teal')

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


class ToyDateBase(dataloader.Dataset):
    def __init__(self, x, y, isNorm=True) -> None:
        super().__init__()
        self.x = x
        if isNorm:
            self.x = (self.x - np.min(self.x)) / (np.max(self.x) - np.min(self.x))
            self.x = (self.x - self.x.mean(0, keepdims=True)) / ((self.x.std(0, keepdims=True) + 1e-16))
            self.x /= np.max(self.x)
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
            x, y = x.float().to(device), y.long().to(device)
            x = net(x)
            val_acc = accuracy(x, y)
            val_accuracy_sum += val_acc
        val_accuracy_sum /= len(val_dataloader.dataset)
    return val_accuracy_sum


def getDataSet(setName, n_sample, noise, random_state, data_path):
    isNorm = False
    savePath = os.path.join(data_path, 'dataset.pkl')
    if os.path.exists(savePath):
        dataDict = torch.load(savePath)
        x, y = dataDict['x'], dataDict['y']
        try:
            isNorm = dataDict['isNorm']
        except:
            pass
        return ToyDateBase(x, y, isNorm)
    if setName == "toy":
        x, y = make_moons(n_sample, noise=noise, random_state=random_state)
        isNorm = True
    if setName[:6] == "random":
        x = np.random.uniform(-1, 1, (n_sample, 2))
        y = np.sign(np.random.uniform(-1, 1, [n_sample, ]))
        y = (np.abs(y) + y) / 2
        isNorm = False
    dataset = ToyDateBase(x, y, isNorm)
    torch.save({'x': x, 'y': y, 'isNorm': isNorm}, savePath)
    return dataset


def train():
    dataset = getDataSet(DATASET, N_SAMPLE, 0.2, 5, SAVE_DIR)
    totalStep = math.ceil(len(dataset) / BATCH_SIZE)
    trainLoader = dataloader.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    net = TestTNetLinear((2,), N_NUM).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4, betas=[0.9, 0.999])
    ce = torch.nn.CrossEntropyLoss()

    save_step = [v for v in SAVE_EPOCH if v < 1]
    steps = [math.floor(v * totalStep) for v in save_step]

    torch.save(net.state_dict(), os.path.join(MODEL_DIR, f'net_0.pth'))
    for epoch in range(MAX_EPOCH):
        net.train()
        for j, (x, y) in enumerate(trainLoader, 1):
            x, y = x.float().to(device), y.long().to(device)
            x = net(x)
            loss = ce(x, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            acc = accuracy(x, y) / x.size(0)

            if (epoch+1) == 1 and (j in steps):
                net.eval()
                idx = steps.index(j)
                torch.save(net.state_dict(), os.path.join(MODEL_DIR, f'net_{save_step[idx]}.pth'))
            print(f"Epoch: {epoch+1} / {MAX_EPOCH}, Step: {j} / {totalStep}, Loss: {loss:.4f}, Acc: {acc:.4f}")

        print(f"Epoch: {epoch+1} / {MAX_EPOCH}")
        if (epoch + 1) in SAVE_EPOCH:
            print("Save net....")
            torch.save(net.state_dict(), os.path.join(MODEL_DIR, f'net_{epoch+1}.pth'))


def lab():
    dataset = getDataSet(DATASET, N_SAMPLE, 0.2, 5, SAVE_DIR)
    val_dataloader = dataloader.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    net = TestTNetLinear((2,), N_NUM)
    net.eval()
    au = areaUtils.AnalysisReLUNetUtils(device=device)
    modelList = os.listdir(MODEL_DIR)
    with torch.no_grad():
        for modelName in modelList:
            sign = float(modelName[4:-4])
            if sign not in SAVE_EPOCH:
                continue
            print(f"Solve fileName: {modelName} ....")
            saveDir = os.path.join(LAB_DIR, os.path.splitext(modelName)[0])
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            au.logger = getLogger(saveDir, f"region-{os.path.splitext(modelName)[0]}")
            modelPath = os.path.join(MODEL_DIR, modelName)
            net.load_state_dict(torch.load(modelPath,  map_location='cpu'))
            net = net.to(device)
            acc = val_net(net, val_dataloader).cpu().numpy()
            print(f'Accuracy: {acc:.4f}')
            regionNum = au.getAreaNum(net, 1, countLayers=net.reLUNum, saveArea=True)
            funcs, areas, points = au.getAreaData()
            # draw fig
            drawRegionImg(regionNum, funcs, areas, points, saveDir)
            drawRegionImgResult(regionNum, funcs, areas, points, saveDir, net)
            # Save data
            dataSaveDict = {
                "funcs": funcs,
                "areas": areas,
                "points": points,
                "regionNum": regionNum,
                "accuracy": acc,
            }
            torch.save(dataSaveDict, os.path.join(saveDir, "dataSave.pkl"))


def drawRegionImg(regionNum, funcs, areas, points, saveDir, minBound=-1, maxBound=1, fileName="regionImg.png"):
    fig = plt.figure(0, figsize=(8, 7), dpi=600)
    ax = fig.subplots()
    ax.cla()
    ax.tick_params(labelsize=15)
    for i in range(regionNum):
        func, area = funcs[i], areas[i]
        func = -area.view(-1, 1) * func
        func = func.numpy()
        A, B = func[:, :-1], -func[:, -1]
        p = pc.Polytope(A, B)
        p.plot(ax, color=np.random.uniform(0.0, 0.95, 3), alpha=1., linestyle='-', linewidth=0.01, edgecolor='w')
    ax.set_xlim(minBound, maxBound)
    ax.set_ylim(minBound, maxBound)
    plt.savefig(os.path.join(saveDir, fileName))
    plt.clf()
    plt.close()


def drawRegionImgResult(regionNum, funcs, areas, points, saveDir, net, minBound=-1, maxBound=1, fileName="regionImgResult.png"):
    net = net.to(device)
    fig = plt.figure(0, figsize=(8, 7), dpi=600)
    ax = fig.subplots()
    ax.cla()
    ax.tick_params(labelsize=15)
    for i in range(regionNum):
        func, area, point = funcs[i], areas[i], points[i]
        point = torch.from_numpy(point).float().to(device).unsqueeze(dim=0)
        output = net(point)[0]
        m = (output[0] - output[1]).cpu().numpy()
        m = np.clip(m, -12., 12.) / 12
        if m <= 0:
            m = np.array([0, -m*0.9, 0.5])
        elif m > 0:
            m = np.array([m*0.9, 0, 0.5])
        func = -area.view(-1, 1) * func
        func = func.numpy()
        A, B = func[:, :-1], -func[:, -1]
        p = pc.Polytope(A, B)
        p.plot(ax, color=m, alpha=1., linestyle='-', linewidth=0.3, edgecolor='w')
    ax.set_xlim(minBound, maxBound)
    ax.set_ylim(minBound, maxBound)
    plt.savefig(os.path.join(saveDir, fileName))
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
