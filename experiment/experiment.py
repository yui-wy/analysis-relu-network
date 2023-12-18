import math
import os
from typing import Any, Callable, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polytope as pc
import torch
from torch._C import device
from torch.utils import data

from dataset import Dataset
from torchays import nn
from torchays.analysis import BaseHandler, Model, ReLUNets
from torchays.graph import COLOR, color, plot_regions, plot_regions_3d
from torchays.models.testnet import TestTNetLinear
from torchays.utils import get_logger


def accuracy(x, classes):
    arg_max = torch.argmax(x, dim=1).long()
    eq = torch.eq(classes, arg_max)
    return torch.sum(eq).float()


class _base:
    def __init__(
        self,
        save_dir: str,
        *,
        dataset: Callable[..., Tuple[Dataset, int]] = None,
        n_layers: List[int] = [16, 16, 16],
        in_features: int = 2,
        save_epoch: List[int] = [0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 100],
        device: torch.device = torch.device('cpu'),
    ) -> None:
        self.save_dir = save_dir
        self.n_layers = n_layers
        self.in_features = in_features
        self.save_epoch = save_epoch
        self.device = device
        self.dataset = dataset

    def _init(self, dataset: Dataset):
        self.tag = f"Linear-{self.n_layers}-{dataset.name}".replace(' ', '')
        self.root_dir = os.path.join(self.save_dir, self.tag)
        self.model_dir = os.path.join(self.root_dir, "model")
        self.experiment_dir = os.path.join(self.root_dir, "experiment")
        for dir in [
            self.root_dir,
            self.model_dir,
            self.experiment_dir,
        ]:
            os.makedirs(dir, exist_ok=True)

    def init_net(self, n_classes: int) -> nn.Module:
        return TestTNetLinear(self.in_features, self.n_layers, n_classes).to(self.device)

    def val_net(self, net: nn.Module, val_dataloader: data.DataLoader) -> torch.Tensor:
        net.eval()
        val_accuracy_sum = 0
        for x, y in val_dataloader:
            x, y = x.float().to(self.device), y.long().to(self.device)
            x = net(x)
            val_acc = accuracy(x, y)
            val_accuracy_sum += val_acc
        val_accuracy_sum /= len(val_dataloader.dataset)
        return val_accuracy_sum


class TrainToy(_base):
    def __init__(
        self,
        save_dir: str,
        dataset: Callable[..., Tuple[Dataset, int]],
        *,
        n_layers: List[int] = [16, 16, 16],
        in_features: int = 2,
        save_epoch: List[int] = [0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 100],
        max_epoch: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        super().__init__(
            save_dir,
            dataset=dataset,
            n_layers=n_layers,
            in_features=in_features,
            save_epoch=save_epoch,
            device=device,
        )
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.lr = lr

    def train(self):
        dataset, n_classes = self.dataset()
        self._init(dataset=dataset)
        trainLoader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        totalStep = math.ceil(len(dataset) / self.batch_size)

        net: nn.Module = self.init_net(n_classes)
        optim = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4, betas=[0.9, 0.999])
        ce = torch.nn.CrossEntropyLoss()

        save_step = [v for v in self.save_epoch if v < 1]
        steps = [math.floor(v * totalStep) for v in save_step]
        torch.save(net.state_dict(), os.path.join(self.model_dir, f'net_0.pth'))
        for epoch in range(self.max_epoch):
            net.train()
            for j, (x, y) in enumerate(trainLoader, 1):
                x: torch.Tensor = x.float().to(self.device)
                y: torch.Tensor = y.long().to(self.device)
                x = net(x)
                loss = ce(x, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                acc = accuracy(x, y) / x.size(0)

                if (epoch + 1) == 1 and (j in steps):
                    net.eval()
                    idx = steps.index(j)
                    torch.save(net.state_dict(), os.path.join(self.model_dir, f'net_{save_step[idx]}.pth'))
                    net.train()
                print(f"Epoch: {epoch+1} / {self.max_epoch}, Step: {j} / {totalStep}, Loss: {loss:.4f}, Acc: {acc:.4f}")

            print(f"Epoch: {epoch+1} / {self.max_epoch}")
            if (epoch + 1) in self.save_epoch:
                print(f"Save net: net_{epoch+1}.pth")
                net.eval()
                torch.save(net.state_dict(), os.path.join(self.model_dir, f'net_{epoch+1}.pth'))
        with torch.no_grad():
            acc = self.val_net(net, trainLoader).cpu().numpy()
            print(f'Accuracy: {acc:.4f}')


class DrawRegionImage:
    def __init__(
        self,
        region_num,
        funcs,
        regions,
        points,
        save_dir,
        net: nn.Module,
        n_classes=2,
        bounds=(-1, 1),
    ) -> None:
        self.region_num = region_num
        self.funcs = funcs
        self.regions = regions
        self.points = points
        self.save_dir = save_dir
        self.net = net
        self.n_classes = n_classes
        self.min_bound, self.max_bound = bounds

    def draw(self, img_3d: bool = False):
        draw_funs = [self.draw_region_img, self.draw_region_img_result]
        if img_3d:
            draw_funs.append(self.draw_region_img_3d)
        for draw_fun in draw_funs:
            draw_fun()

    def draw_region_img(self, fileName="region_img.png"):
        fig = plt.figure(0, figsize=(8, 7), dpi=600)
        ax = fig.subplots()
        ax.cla()
        ax.tick_params(labelsize=15)
        plot_regions(
            self.funcs,
            self.regions,
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
            self.regions,
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
            func, area = self.funcs[i], self.regions[i]
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


class HyperplaneArrangement:
    def __init__(
        self,
        p_funs: torch.Tensor,
        p_regions: torch.Tensor,
        c_funs: torch.Tensor,
        intersect_funs: torch.Tensor | None,
        n_regions: int,
        depth: int,
    ) -> None:
        self.p_funs = p_funs
        self.p_regions = p_regions
        self.c_funs = c_funs
        self.intersect_funs = intersect_funs
        self.depth = depth
        self.n_regions = n_regions
        self.n_intersect_funcs = 0 if intersect_funs is None else intersect_funs.size(0)


class Handler(BaseHandler):
    def __init__(self) -> None:
        self._init_region_handler()._init_inner_hyperplanes_handler()

    def _init_region_handler(self):
        self.funs = list()
        self.regions = list()
        self.points = list()
        return self

    def region_handler(
        self,
        fun: torch.Tensor,
        region: torch.Tensor,
        point: np.ndarray,
    ) -> None:
        self.funs.append(fun.numpy())
        self.regions.append(region.numpy())
        self.points.append(point)

    def _init_inner_hyperplanes_handler(self):
        return self

    def inner_hyperplanes_handler(
        self,
        p_funs: torch.Tensor,
        p_regions: torch.Tensor,
        c_funs: torch.Tensor,
        intersect_funs: torch.Tensor | None,
        n_regions: int,
        depth: int,
    ) -> None:
        return


class LinearRegion(_base):
    def __init__(
        self,
        save_dir: str,
        dataset: Callable[..., Tuple[Dataset, int]],
        *,
        n_layers: List[int] = [16, 16, 16],
        in_features: int = 2,
        save_epoch: List[int] = [0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 100],
        bounds: Tuple[float] = (-1, 1),
        is_draw: bool = True,
        device: device = torch.device('cpu'),
    ) -> None:
        super().__init__(
            save_dir,
            dataset=dataset,
            n_layers=n_layers,
            in_features=in_features,
            save_epoch=save_epoch,
            device=device,
        )
        self.bounds = bounds
        self.is_draw = is_draw

    def get_region(self):
        dataset, n_classes = self.dataset()
        self._init(dataset=dataset)
        val_dataloader = data.DataLoader(dataset, shuffle=True, pin_memory=True)
        net: Model = self.init_net(n_classes)
        au = ReLUNets(device=self.device)
        model_list = os.listdir(self.model_dir)
        with torch.no_grad():
            for model_name in model_list:
                epoch = float(model_name[4:-4])
                if epoch not in self.save_epoch:
                    continue
                print(f"Solve fileName: {model_name} ....")
                save_dir = os.path.join(self.experiment_dir, os.path.splitext(model_name)[0])
                os.makedirs(save_dir, exist_ok=True)
                au.logger = get_logger(f"region-{os.path.splitext(model_name)[0]}", os.path.join(save_dir, "region.log"))
                model_path = os.path.join(self.model_dir, model_name)
                net.load_state_dict(torch.load(model_path))
                acc = self.val_net(net, val_dataloader).cpu().numpy()
                print(f"Accuracy: {acc:.4f}")
                handler = Handler()
                region_num = au.get_region_counts(
                    net,
                    bounds=self.bounds,
                    input_size=(self.in_features,),
                    depth=net.n_relu,
                    handler=handler,
                )
                print(f"Region counts: {region_num}")
                if self.is_draw:
                    # draw fig
                    drawReginImage = DrawRegionImage(
                        region_num,
                        handler.funs,
                        handler.regions,
                        handler.points,
                        save_dir,
                        net,
                        n_classes,
                        bounds=self.bounds,
                    )
                    drawReginImage.draw()

                dataSaveDict = {
                    "funcs": handler.funs,
                    "regions": handler.regions,
                    "points": handler.points,
                    "regionNum": region_num,
                    "accuracy": acc,
                }
                torch.save(dataSaveDict, os.path.join(save_dir, "data_save.pkl"))


class Experiment(_base):
    def __init__(
        self,
        save_dir: str,
        init_fun: Callable[..., None],
        *,
        n_layers: List[int] = [16, 16, 16],
        in_features: int = 2,
        save_epoch: List[int] = [0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 100],
        device: device = torch.device('cpu'),
    ) -> None:
        super().__init__(
            save_dir,
            n_layers=n_layers,
            in_features=in_features,
            save_epoch=save_epoch,
            device=device,
        )
        self.init_fun = init_fun
        self.run_funs = list()

    def train(
        self,
        dataset: Callable[..., Tuple[Dataset, int]],
        max_epoch: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
    ):
        toy_train = TrainToy(
            save_dir=self.save_dir,
            dataset=dataset,
            n_layers=self.n_layers,
            in_features=self.in_features,
            save_epoch=self.save_epoch,
            max_epoch=max_epoch,
            batch_size=batch_size,
            lr=lr,
            device=torch.device('cpu'),
        )
        self.append(toy_train.train)
        return self

    def linear_region(
        self,
        dataset: Callable[..., Tuple[Dataset, int]],
        is_draw: bool = True,
        bounds: Tuple[float] = (-1, 1),
    ):
        linear_region = LinearRegion(
            save_dir=self.save_dir,
            dataset=dataset,
            n_layers=self.n_layers,
            in_features=self.in_features,
            save_epoch=self.save_epoch,
            is_draw=is_draw,
            bounds=bounds,
            device=torch.device('cpu'),
        )
        self.append(linear_region.get_region)

    def append(self, fun: Callable[..., None]):
        self.run_funs.append(self.init_fun)
        self.run_funs.append(fun)

    def run(self):
        for fun in self.run_funs:
            fun()

    def __call__(self, *args: Any, **kwds: Any):
        self.run(*args, **kwds)
