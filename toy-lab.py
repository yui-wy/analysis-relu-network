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

SEED = 1658123
DATASET = f"random{SEED}"
N_NUM = [16, 32, 64]
TAG = f"Linear-{N_NUM}".replace(' ', '')
N_SAMPLE = 600
GPU_ID = 0
MAX_EPOCH = 1000
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


class default_plt:
    def __init__(self, savePath, xlabel='', ylabel='', mode='png', isGray=False, isLegend=True):
        self.savePath = savePath
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.mode = mode
        self.isGray = isGray
        self.isLegend = isLegend

    def __enter__(self):
        fig = plt.figure(0, figsize=(8, 7), dpi=600)
        self.ax = fig.subplots()
        self.ax.cla()
        if not self.isGray:
            self.ax.patch.set_facecolor("w")
        self.ax.tick_params(labelsize=15)
        self.ax.set_xlabel(self.xlabel, fontdict={'weight': 'normal', 'size': 20})
        self.ax.set_ylabel(self.ylabel, fontdict={'weight': 'normal', 'size': 20})
        self.ax.grid(color="#EAEAEA", linewidth=1)
        # self.ax.spines['right'].set_color('none')
        # self.ax.spines['top'].set_color('none')
        return self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.isLegend:
            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0, box.width, box.height*0.95])
            self.ax.legend(prop={'weight': 'normal', 'size': 14}, loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=3, mode="expand")

        plt.savefig(self.savePath, dpi=600, format=f'{self.mode}')
        plt.clf()
        plt.close()


def default(savePath, xlabel='', ylabel='', mode='png', isGray=False, isLegend=True):
    return default_plt(savePath, xlabel, ylabel, mode, isGray, isLegend)


class ToyDateBase(dataloader.Dataset):
    def __init__(self, x, y, isNorm=True) -> None:
        super().__init__()
        self.x = (x - np.min(x)) / (np.max(x) - np.min(x))
        if isNorm:
            self.x = (self.x - x.mean(0, keepdims=True)) / ((x.std(0, keepdims=True) + 1e-16) * 2)
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


def getRegion(net, name, logPath, au, countLayers):
    num = au.getAreaNum(net, 1, countLayers=countLayers)
    logPath.write(f"Key: {name}; RegionNum: {num}\n")
    logPath.flush()
    return num


def getDataSet(setName, n_sample, noise, random_state, data_path):
    savePath = os.path.join(data_path, 'dataset.pkl')
    if os.path.exists(savePath):
        dataDict = torch.load(savePath)
        x, y = dataDict['x'], dataDict['y']
        return ToyDateBase(x, y, False)
    if setName == "toy":
        x, y = make_moons(n_sample, noise=noise, random_state=random_state)
        dataset = ToyDateBase(x, y)
    if setName[:6] == "random":
        x = np.random.uniform(-1, 1, (n_sample, 2))
        y = np.sign(np.random.uniform(-1, 1, [n_sample, ]))
        y = (np.abs(y) + y) / 2
        dataset = ToyDateBase(x, y, False)
    saveDict = {
        'x': x,
        'y': y
    }
    torch.save(saveDict, savePath)
    return dataset


def train():
    dataset = getDataSet(DATASET, N_SAMPLE, 0.2, 5, SAVE_DIR)
    totalStep = math.ceil(len(dataset) / BATCH_SIZE)
    trainLoader = dataloader.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    net = TestTNetLinear((2,), N_NUM).to(device)
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
    dataset = getDataSet(DATASET, N_SAMPLE, 0.2, 5, SAVE_DIR)
    val_dataloader = dataloader.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    net = TestTNetLinear((2,), N_NUM)
    net.eval()
    au = areaUtils.AnalysisReLUNetUtils(device=device)
    # epoch = [0, 1, 5, 10, 30, 50, 80, 100, 200, 300, 400, 500, 800, 1000]
    epoch = [0, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # epoch = [0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 100]
    # epoch = [800]
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
            acc = val_net(net, val_dataloader).cpu().numpy()
            print(f'Accuracy: {acc:.4f}')
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
                "accuracy": acc,
            }
            torch.save(dataSaveDict, os.path.join(saveDir, "dataSave.pkl"))


def drawRegionImg(regionNum, funcs, areas, points, saveDir, minBound=-1, maxBound=1, fileName="regionImg.png"):
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


def lab2():
    labDict = {}
    DatasetDir = os.path.join(ROOT_DIR, 'cache', DATASET)
    saveDir = os.path.join(DatasetDir, "All")
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    for tag in os.listdir(DatasetDir):
        if tag == "All":
            continue
        labDict[tag] = {}
        datasetPath = os.path.join(DatasetDir, tag, 'dataset.pkl')
        dataset = torch.load(datasetPath)
        drawDataSet(dataset, os.path.join(DatasetDir, tag))
        labDir = os.path.join(DatasetDir, tag, 'lab')
        for epochFold in os.listdir(labDir):
            epoch = float(epochFold[4:])
            pklDict = torch.load(os.path.join(labDir, epochFold, 'dataSave.pkl'))
            labDict[tag][epoch] = pklDict
    drawRegionEpochPlot(labDict, saveDir)
    drawRegionAccPlot(labDict, saveDir)
    drawEpochAccPlot(labDict, saveDir)


def drawRegionEpochPlot(labDict: Dict, saveDir):
    savePath = os.path.join(saveDir, "regionEpoch.png")
    with default(savePath, 'Epoch', 'Number of Rgions') as ax:
        i = 0
        for tag, epochDict in labDict.items():
            tag1 = tag.split('-')[-1]
            dataList = []
            for epoch, fileDict in epochDict.items():
                data = [epoch, fileDict['regionNum']]
                dataList.append(data)
            dataList = np.array(dataList)
            a = dataList[:, 0]
            index = np.lexsort((a,))
            dataList = dataList[index]
            ax.plot(dataList[:, 0], dataList[:, 1], label=tag1, color=COLOR[i])
            i += 1


def drawRegionAccPlot(labDict: Dict, saveDir):
    savePath = os.path.join(saveDir, "regionAcc.png")
    with default(savePath, 'Accuracy', 'Number of Rgions') as ax:
        i = 0
        for tag, epochDict in labDict.items():
            tag1 = tag.split('-')[-1]
            dataList = []
            for _, fileDict in epochDict.items():
                acc = fileDict['accuracy']
                if isinstance(acc, torch.Tensor):
                    acc = acc.cpu().numpy()
                data = [acc, fileDict['regionNum']]
                dataList.append(data)
            dataList = np.array(dataList)
            a = dataList[:, 0]
            index = np.lexsort((a,))
            dataList = dataList[index]
            ax.plot(dataList[:, 0], dataList[:, 1], label=tag1, color=COLOR[i])
            i += 1


def drawEpochAccPlot(labDict: Dict, saveDir):
    savePath = os.path.join(saveDir, "EpochAcc.png")
    with default(savePath, 'Epoch', 'Accuracy') as ax:
        i = 0
        for tag, epochDict in labDict.items():
            tag1 = tag.split('-')[-1]
            dataList = []
            for epoch, fileDict in epochDict.items():
                acc = fileDict['accuracy']
                if isinstance(acc, torch.Tensor):
                    acc = acc.cpu().numpy()
                data = [epoch, acc]
                dataList.append(data)
            dataList = np.array(dataList)
            a = dataList[:, 0]
            index = np.lexsort((a,))
            dataList = dataList[index]
            ax.plot(dataList[:, 0], dataList[:, 1], label=tag1, color=COLOR[i])
            i += 1


def drawDataSet(dataset, saveDir):
    savePath = os.path.join(saveDir, "distribution.png")
    x, y = dataset['x'], dataset['y']
    with default(savePath, 'x1', 'x2', isLegend=False) as ax:
        ax.scatter(x[y == 0, 0], x[y == 0, 1], color=COLOR[0])
        ax.scatter(x[y == 1, 0], x[y == 1, 1], color=COLOR[1])


if __name__ == "__main__":
    train()
    lab()
    # lab2()
