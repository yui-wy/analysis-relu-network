import math
import os
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polytope as pc
import torch
from torch.utils import data

from dataset import GAUSSIAN_QUANTILES, MOON, RANDOM, simple_get_data
from torchays import nn
from torchays.analysis import ReLUNets
from torchays.graph import COLOR, color, plot_regions, plot_regions_3d
from torchays.models.testnet import TestTNetLinear
from torchays.nn import Module
from torchays.utils import get_logger

GPU_ID = 0
SEED = 5
DATASET = RANDOM
N_NUM = [16, 16, 16]
# Dataset
N_SAMPLES = 1000
DATASET_BIAS = 0
# only GAUSSIAN_QUANTILES
N_CLASSES = 3
# only RANDOM
IN_FEATURES = 2

# TAG
TAG = f"Linear-{N_NUM}-{DATASET}-{N_SAMPLES}-{SEED}".replace(' ', '')
# Training
MAX_EPOCH = 100
SAVE_EPOCH = [0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 100]
BATCH_SIZE = 32
LR = 1e-3

# Path
ROOT_DIR = os.path.abspath("./")
SAVE_DIR = os.path.join(ROOT_DIR, "cache", DATASET, TAG)
MODEL_DIR = os.path.join(SAVE_DIR, "model")
LAB_DIR = os.path.join(SAVE_DIR, "lab")
DATASET_PATH = os.path.join(SAVE_DIR, "dataset.pkl")


device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')


def init():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    for dir in [SAVE_DIR, MODEL_DIR, LAB_DIR]:
        os.makedirs(dir, exist_ok=True)


def get_data_set():
    return simple_get_data(
        DATASET,
        N_SAMPLES,
        0.2,
        5,
        DATASET_PATH,
        n_classes=N_CLASSES,
        in_features=IN_FEATURES,
        bias=DATASET_BIAS,
    )


def accuracy(x, classes):
    arg_max = torch.argmax(x, dim=1).long()
    eq = torch.eq(classes, arg_max)
    return torch.sum(eq).float()


def val_net(net: nn.Module, val_dataloader):
    net.eval()
    with torch.no_grad():
        val_accuracy_sum = 0
        for x, y in val_dataloader:
            x, y = x.float().to(device), y.long().to(device)
            x = net(x)
            val_acc = accuracy(x, y)
            val_accuracy_sum += val_acc
        val_accuracy_sum /= len(val_dataloader.dataset)
    return val_accuracy_sum


def init_net(n_classes: int):
    return TestTNetLinear(IN_FEATURES, N_NUM, n_classes).to(device)


class DrawRegionImage:
    def __init__(
        self,
        region_num,
        funcs,
        areas,
        points,
        save_dir,
        net: Module,
        n_classes=2,
        min_bound=-1,
        max_bound=1,
    ) -> None:
        self.region_num = region_num
        self.funcs = funcs
        self.areas = areas
        self.points = points
        self.save_dir = save_dir
        self.net = net.to(device).eval()
        self.n_classes = n_classes
        self.min_bound = min_bound
        self.max_bound = max_bound

    def draw(self):
        for draw_fun in [
            self.draw_region_img,
            self.draw_region_img_3d,
            self.draw_region_img_result,
        ]:
            draw_fun()

    def draw_region_img(self, fileName="region_img.png"):
        fig = plt.figure(0, figsize=(8, 7), dpi=600)
        ax = fig.subplots()
        ax.cla()
        ax.tick_params(labelsize=15)
        plot_regions(
            self.funcs,
            self.areas,
            ax=ax,
            xlim=[self.min_bound, self.max_bound],
            ylim=[self.min_bound, self.max_bound],
        )
        plt.savefig(os.path.join(self.save_dir, fileName))
        plt.clf()
        plt.close()

    def _z_fun(self, xy: np.ndarray) -> Tuple[np.ndarray, int]:
        xy = torch.from_numpy(xy).to(device).float()
        z: torch.Tensor = self.net(xy)
        return z.cpu().numpy(), range(self.n_classes)

    def draw_region_img_3d(self, fileName="region_img_3d.png"):
        fig = plt.figure(0)
        ax = fig.add_subplot(projection="3d")
        ax.cla()
        ax.tick_params(labelsize=15)
        plot_regions_3d(
            self.funcs,
            self.areas,
            z_fun=self._z_fun,
            ax=ax,
            alpha=0.9,
            color=color,
            edgecolor="grey",
            linewidth=0.1,
            xlim=[self.min_bound, self.max_bound],
            ylim=[self.min_bound, self.max_bound],
        )
        plt.savefig(os.path.join(self.save_dir, fileName))
        plt.clf()
        plt.close()

    def draw_region_img_result(self, color_bar: bool = False, fileName: str = "region_img_result.png"):
        fig = plt.figure(0, figsize=(8, 7), dpi=600)
        ax = fig.subplots()
        ax.cla()
        ax.tick_params(labelsize=15)
        img = self.__draw_hot(ax)
        for i in range(self.region_num):
            func, area = self.funcs[i], self.areas[i]
            func = -area.reshape(-1, 1) * func
            A, B = func[:, :-1], -func[:, -1]
            p = pc.Polytope(A, B)
            p.plot(
                ax,
                color="w",
                alpha=0.1,
                linestyle='-',
                linewidth=0.3,
                edgecolor='black',
            )
        ax.set_xlim(self.min_bound, self.max_bound)
        ax.set_ylim(self.min_bound, self.max_bound)
        if color_bar:
            fig.colorbar(img)
        plt.savefig(os.path.join(self.save_dir, fileName))
        plt.clf()
        plt.close()

    def __draw_hot(self, ax):
        num = 1000
        data = self.__hot_data(num).float()
        result = self.net(data).softmax(dim=1)
        result = (result - 1 / self.n_classes) / (1 - 1 / self.n_classes)
        result, maxIdx = torch.max(result, dim=1)
        result, maxIdx = result.cpu().numpy(), maxIdx.cpu().numpy()
        result_alpha, result_color = np.empty((num, num)), np.empty((num, num))
        for i in range(num):
            result_color[num - 1 - i] = maxIdx[i * num : (i + 1) * num]
            result_alpha[num - 1 - i] = result[i * num : (i + 1) * num]
        cmap = matplotlib.colors.ListedColormap(COLOR, name="Region")
        return ax.imshow(
            result_color,
            alpha=result_alpha,
            cmap=cmap,
            extent=(self.min_bound, self.max_bound, self.min_bound, self.max_bound),
            vmin=0,
            vmax=len(COLOR),
        )

    def __hot_data(self, num=1000):
        x1 = np.linspace(self.min_bound, self.max_bound, num)
        x2 = np.linspace(self.min_bound, self.max_bound, num)
        X1, X2 = np.meshgrid(x1, x2)
        X1, X2 = X1.flatten(), X2.flatten()
        data = np.vstack((X1, X2)).transpose()
        data = torch.from_numpy(data).to(device)
        return data


def train():
    dataset, n_classes = get_data_set()
    trainLoader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    totalStep = math.ceil(len(dataset) / BATCH_SIZE)

    net = init_net(n_classes)
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

            if (epoch + 1) == 1 and (j in steps):
                net.eval()
                idx = steps.index(j)
                torch.save(net.state_dict(), os.path.join(MODEL_DIR, f'net_{save_step[idx]}.pth'))
                net.train()
            print(f"Epoch: {epoch+1} / {MAX_EPOCH}, Step: {j} / {totalStep}, Loss: {loss:.4f}, Acc: {acc:.4f}")

        print(f"Epoch: {epoch+1} / {MAX_EPOCH}")
        if (epoch + 1) in SAVE_EPOCH:
            print(f"Save net: net_{epoch+1}.pth")
            net.eval()
            torch.save(net.state_dict(), os.path.join(MODEL_DIR, f'net_{epoch+1}.pth'))

    acc = val_net(net, trainLoader).cpu().numpy()
    print(f'Accuracy: {acc:.4f}')


def get_region(is_draw: bool = False, lower: int = -1, upper: int = 1):
    dataset, n_classes = get_data_set()
    val_dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    net = init_net(n_classes)
    au = ReLUNets(device=device)
    model_list = os.listdir(MODEL_DIR)
    with torch.no_grad():
        for model_name in model_list:
            sign = float(model_name[4:-4])
            if sign not in SAVE_EPOCH:
                continue
            print(f"Solve fileName: {model_name} ....")
            save_dir = os.path.join(LAB_DIR, os.path.splitext(model_name)[0])
            os.makedirs(save_dir, exist_ok=True)
            au.logger = get_logger(f"region-{os.path.splitext(model_name)[0]}", os.path.join(save_dir, "region.log"))
            model_path = os.path.join(MODEL_DIR, model_name)
            net.load_state_dict(torch.load(model_path))
            acc = val_net(net, val_dataloader).cpu().numpy()
            print(f"Accuracy: {acc:.4f}")

            funcs, areas, points = [], [], []

            def handler(point, functions, region):
                points.append(point)
                funcs.append(functions.numpy())
                areas.append(region.numpy())

            region_num = au.get_region_counts(
                net,
                bounds=(lower, upper),
                input_size=(IN_FEATURES,),
                depth=net.n_relu,
                region_handler=handler,
            )
            print(f"Region counts: {region_num}")
            if is_draw:
                # draw fig
                drawReginImage = DrawRegionImage(
                    region_num,
                    funcs,
                    areas,
                    points,
                    save_dir,
                    net,
                    n_classes,
                    max_bound=upper,
                    min_bound=lower,
                )
                drawReginImage.draw()
            dataSaveDict = {
                "funcs": funcs,
                "areas": areas,
                "points": points,
                "regionNum": region_num,
                "accuracy": acc,
            }
            torch.save(dataSaveDict, os.path.join(save_dir, "data_save.pkl"))


def main(*, is_train: bool = True, is_draw: bool = False, lower: int = -1, upper: int = 1):
    init()
    if is_train:
        train()
    get_region(is_draw, lower=lower, upper=upper)


if __name__ == "__main__":
    main(
        is_train=True,
        is_draw=True,
        lower=-1,  # 绘图的下界
        upper=1,  # 绘图的上界
    )
