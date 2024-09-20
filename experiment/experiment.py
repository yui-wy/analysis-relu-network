from cProfile import label
from copy import deepcopy
import math
import os
from typing import Any, Callable, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polytope as pc
from sympy import count_ops
import torch
from torch.utils import data

from dataset import Dataset
from torchays import nn
from torchays.analysis import BaseHandler, Model, ReLUNets
from torchays.graph import (
    COLOR,
    color,
    plot_region,
    plot_regions,
    plot_regions_3d,
)
from torchays.utils import get_logger

EPSILON = 1e-16
STATISTIC_COUNT = "statistic_count"
STATISTIC_SCALE = "statistic_scale"
NEURAL_NUM = "neural_num"


def save_file(buff, file_path):
    with open(file_path, 'w') as w:
        w.write(buff)
        w.close()


def accuracy(x, classes):
    arg_max = torch.argmax(x, dim=1).long()
    eq = torch.eq(classes, arg_max)
    return torch.sum(eq).float()


class _base:
    def __init__(
        self,
        save_dir: str,
        *,
        net: Callable[[int], Model] = None,
        dataset: Callable[..., Tuple[Dataset, int]] = None,
        save_epoch: List[int] = [0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 100],
        device: torch.device = torch.device('cpu'),
    ) -> None:
        self.save_dir = save_dir
        self.net = net
        self.dataset = dataset
        self.save_epoch = save_epoch
        self.device = device
        self.root_dir = None

    def get_root(self):
        if self.root_dir is None:
            self.root_dir = os.path.join(self.save_dir, self.net(0).name)
        return self.root_dir

    def _init_model(self):
        dataset, n_classes = self.dataset()
        net = self.net(n_classes).to(self.device)
        self._init_dir(net.name)
        return net, dataset, n_classes

    def _init_dir(self, tag):
        self.root_dir = os.path.join(self.save_dir, tag)
        self.model_dir = os.path.join(self.root_dir, "model")
        self.experiment_dir = os.path.join(self.root_dir, "experiment")
        for dir in [
            self.root_dir,
            self.model_dir,
            self.experiment_dir,
        ]:
            os.makedirs(dir, exist_ok=True)

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
        net: Callable[[int], Model],
        dataset: Callable[..., Tuple[Dataset, int]],
        *,
        save_epoch: List[int] = [0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 100],
        max_epoch: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        train_handler: Callable[[nn.Module, int, int, int, torch.Tensor, torch.Tensor, str], None] = None,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        super().__init__(
            save_dir,
            net=net,
            dataset=dataset,
            save_epoch=save_epoch,
            device=device,
        )
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.train_handler = train_handler

    def train(self):
        net, dataset, _ = self._init_model()
        train_loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        total_step = math.ceil(len(dataset) / self.batch_size)

        optim = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4, betas=[0.9, 0.999])
        ce = torch.nn.CrossEntropyLoss()

        save_step = [v for v in self.save_epoch if v < 1]
        steps = [math.floor(v * total_step) for v in save_step]
        torch.save(net.state_dict(), os.path.join(self.model_dir, f'net_0.pth'))
        best_acc, best_dict, best_epoch = 0, {}, 0
        for epoch in range(self.max_epoch):
            net.train()
            loss_sum = 0
            for j, (x, y) in enumerate(train_loader, 1):
                x: torch.Tensor = x.float().to(self.device)
                y: torch.Tensor = y.long().to(self.device)
                x = net(x)
                loss: torch.Tensor = ce(x, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                acc = accuracy(x, y) / x.size(0)
                loss_sum += loss

                if (epoch + 1) == 1 and (j in steps):
                    net.eval()
                    idx = steps.index(j)
                    torch.save(net.state_dict(), os.path.join(self.model_dir, f'net_{save_step[idx]}.pth'))
                    net.train()
                if self.train_handler is not None:
                    self.train_handler(net, epoch, j, total_step, loss, acc, self.model_dir)
                # print(f"Epoch: {epoch+1} / {self.max_epoch}, Step: {j} / {total_step}, Loss: {loss:.4f}, Acc: {acc:.4f}")
            net.eval()
            if (epoch + 1) in self.save_epoch:
                print(f"Save net: net_{epoch+1}.pth")
                torch.save(net.state_dict(), os.path.join(self.model_dir, f'net_{epoch+1}.pth'))
            with torch.no_grad():
                loss_sum = loss_sum / total_step
                acc = self.val_net(net, train_loader).cpu().numpy()
                print(f'Epoch: {epoch+1} / {self.max_epoch}, Loss: {loss_sum:.4f}, Accuracy: {acc:.4f}')
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    best_dict = deepcopy(net.state_dict())
        torch.save(best_dict, os.path.join(self.model_dir, f'net_best_{best_epoch+1}.pth'))
        print(f'Best_Epoch: {best_epoch+1} / {self.max_epoch}, Accuracy: {best_acc:.4f}')


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
        device=torch.device("cpu"),
    ) -> None:
        self.region_num = region_num
        self.funcs = funcs
        self.regions = regions
        self.points = points
        self.save_dir = save_dir
        self.net = net.to(device).eval()
        self.n_classes = n_classes
        self.bounds = bounds
        self.min_bound, self.max_bound = bounds
        self.device = device

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
            xlim=self.bounds,
            ylim=self.bounds,
        )
        plt.savefig(os.path.join(self.save_dir, fileName))
        plt.clf()
        plt.close()

    def _z_fun(self, xy: np.ndarray) -> Tuple[np.ndarray, int]:
        xy = torch.from_numpy(xy).to(self.device).float()
        z: torch.Tensor = self.net(xy)
        return z.cpu().numpy(), range(self.n_classes)

    def draw_region_img_3d(self, fileName="region_img_3d.png"):
        fig = plt.figure(0)
        ax: plt.Axes = fig.add_subplot(projection="3d")
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
            xlim=self.bounds,
            ylim=self.bounds,
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
        ax.set_xlim(*self.bounds)
        ax.set_ylim(*self.bounds)
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
        data = torch.from_numpy(data).to(self.device)
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
        self.n_regions = n_regions
        self.depth = depth
        self.n_intersect_funcs = 0 if intersect_funs is None else intersect_funs.size(0)


class HyperplaneArrangements:
    def __init__(
        self,
        root_dir,
        hyperplane_arrangements: Dict[int, List[HyperplaneArrangement]],
        bounds: Tuple[int, int] = (-1, 1),
    ) -> None:
        self.root_dir = root_dir
        self.hyperplane_arrangements = hyperplane_arrangements
        self.bounds = bounds

    def _check_dim(self, hpa: HyperplaneArrangement):
        return hpa.p_funs.size(1) - 1 == 2

    def statistics_intersect(self):
        # 统计每一层与父区域相交的神经元数量。
        # 在不同父区域下，同一个神经元可能会出现中间态与非中间态
        p_dir = os.path.join(self.root_dir, "hpa_statistics")
        os.makedirs(p_dir, exist_ok=True)
        counts_str, scales_str = "", ""
        # MAP: [depth, Dict[name, [List[int]]]]
        statistic_dict: Dict[int, Dict[str, List[int]]] = dict()

        def csv_str(buff: str, num_list: List[int | float]):
            str_list = list(map(str, num_list))
            str_buff = ",".join(str_list)
            return buff + str_buff + "\r\n"

        def statistic_fun(neural_num: int, num_list: List[int | float]):
            statistic_list = [0] * (neural_num + 1)
            for num in num_list:
                statistic_list[num] += 1
            return statistic_list

        statistics = self._get_statistics()
        for depth, statistic in statistics.items():
            neural_num: int = statistic.get(NEURAL_NUM)
            intersect_counts: List[int] = statistic.get("intersect_counts")
            intersect_scales: List[float] = statistic.get("intersect_scales")
            counts_str = csv_str(counts_str, intersect_counts)
            scales_str = csv_str(scales_str, intersect_scales)
            # 统计期望与概率
            depth_statistic: Dict[str, List[int | float]] = dict()
            depth_statistic[NEURAL_NUM] = neural_num
            depth_statistic[STATISTIC_COUNT] = statistic_fun(neural_num, intersect_counts)
            # depth_statistic[STATISTIC_SCALE] = statistic_fun(neural_num, intersect_scales)
            statistic_dict[depth] = depth_statistic

        # csv
        save_file(counts_str, os.path.join(p_dir, "counts.csv"))
        save_file(scales_str, os.path.join(p_dir, "scales.csv"))
        # plot
        self._draw_statistic(p_dir, statistic_dict, STATISTIC_COUNT, "statistic counts", "counts")
        # self._draw_statistic(p_dir, statistic_dict, STATISTIC_COUNT, "statistic scales")

    def _get_statistics(self) -> Dict[str, Dict[str, int | List]]:
        statistics = dict()
        for depth, hpas in self.hyperplane_arrangements.items():
            intersect_counts = []
            intersect_scales = []
            neural_num = -1
            for hpa in hpas:
                if neural_num == -1:
                    neural_num = len(hpa.c_funs)
                intersects = 0 if hpa.intersect_funs is None else len(hpa.intersect_funs)
                intersect_counts.append(intersects)
                intersect_scales.append(intersects / neural_num)
            statistic = dict()
            statistic[NEURAL_NUM] = neural_num
            statistic["intersect_counts"] = intersect_counts
            statistic["intersect_scales"] = intersect_scales
            statistics[depth] = statistic
        return statistics

    def _draw_statistic(
        self,
        dir: str,
        statistic_dict: Dict[int, Dict[str, List[int]]],
        key: str,
        name: str = "",
        y_label: str = "",
    ):
        # 统计在不同深度下，有多少区域里面存在中间神经元，以及中间神经元的个数
        if name == "":
            name = key
        fig = plt.figure(0, figsize=(8, 7), dpi=600)
        ax = fig.subplots()
        ax.cla()
        ax.tick_params(labelsize=15)
        max_neural_num = 0
        legend_list: List[str] = list()
        for depth, depth_statistic in statistic_dict.items():
            neural_num = depth_statistic.get(NEURAL_NUM)
            if max_neural_num < neural_num:
                max_neural_num = neural_num
            x_list = depth_statistic.get(key)
            ax.plot(np.arange(neural_num + 1), x_list, label=name, color=color(depth))
            legend_list.append(f"depth-{depth}")
        ax.legend(legend_list)
        ax.set_xlabel("Neurals")
        ax.set_ylabel(y_label)
        ax.set_xticks(np.arange(max_neural_num + 1))
        fig_file = os.path.join(dir, f"{name}.jpg")
        plt.savefig(fig_file, format="jpg")
        plt.clf()
        plt.close()

    def draw_hyperplane_arrangments(self):
        p_dir = os.path.join(self.root_dir, "hyperplane_arrangments")
        for depth, hpas in self.hyperplane_arrangements.items():
            save_dir = os.path.join(p_dir, f"depth_{depth+1}")
            for idx, hpa in enumerate(hpas):
                pic_dir = os.path.join(save_dir, f"hpa_{idx}")
                os.makedirs(pic_dir, exist_ok=True)
                self._draw_hyperplane_arrangment(hpa, pic_dir, f"arrangement.jpg")
                self._draw_weights_scatter(hpa, pic_dir, f"weights_scatter.jpg")

    def _draw_weights_scatter(
        self,
        hpa: HyperplaneArrangement,
        pic_dir: str,
        fileName: str = "weights_scatter.jpg",
    ):
        if not self._check_dim(hpa):
            raise NotImplementedError("draw weight scatter can only be used on 2 dim.")

        # 绘制权重向量的散点 3d 散点图
        # weight of AX+b > 0
        def draw_weight(ax: plt.Axes, funs: torch.Tensor, color: Callable[[int], str], *args, **kwds) -> np.ndarray:
            scale_rate = (funs[:, 0].square() + funs[:, 1].square()).sqrt().unsqueeze(1)
            f_weight = (funs / scale_rate).numpy()
            for i in range(f_weight.shape[0]):
                ax.scatter(f_weight[i, 0], f_weight[i, 1], f_weight[i, 2], c=color(i), *args, **kwds)

        fig = plt.figure(0, figsize=(8, 7), dpi=600)
        # fig = plt.figure(0)
        ax: plt.Axes = fig.add_subplot(projection="3d")
        ax.cla()
        ax.tick_params(labelsize=15)
        # AX+B > 0
        p_funs = hpa.p_funs * hpa.p_regions.unsqueeze(1)
        draw_weight(ax, p_funs, lambda _: color(0), marker="x")
        draw_weight(ax, hpa.c_funs, lambda _: color(2), marker="v")
        draw_weight(ax, -hpa.c_funs, lambda _: color(2), marker="^")
        if hpa.intersect_funs is not None:
            draw_weight(ax, hpa.intersect_funs, lambda _: color(1), marker="v")
            draw_weight(ax, -hpa.intersect_funs, lambda _: color(1), marker="^")
        # plt.show()
        plt.savefig(os.path.join(pic_dir, fileName))
        plt.clf()
        plt.close()

    def _draw_hyperplane_arrangment(
        self,
        hpa: HyperplaneArrangement,
        pic_dir: str,
        fileName: str = "arrangement.jpg",
    ):
        if not self._check_dim(hpa):
            raise NotImplementedError("draw weight scatter can only be used on 2 dim.")
        fig = plt.figure(0, figsize=(8, 7), dpi=600)
        ax = fig.subplots()
        ax.cla()
        ax.tick_params(labelsize=15)
        # 绘制区域
        plot_region(
            hpa.p_funs.numpy(),
            hpa.p_regions.numpy(),
            color='tomato',
            alpha=0.5,
            ax=ax,
        )
        self.__plot(ax, hpa.c_funs, color=color(2), linewidth=0.3, linestyle="--")
        self.__plot(ax, hpa.intersect_funs, color=color(1), linewidth=0.4)
        ax.set_xlim(*self.bounds)
        ax.set_ylim(*self.bounds)
        ax.set_title(f"Counts of the regions: {hpa.n_regions}/{hpa.n_intersect_funcs}/{hpa.c_funs.size(0)}")
        plt.savefig(os.path.join(pic_dir, fileName))
        plt.clf()
        plt.close()

    def __plot(self, ax: plt.Axes, funcs: torch.Tensor, *args, **kwds):
        if funcs is None or len(funcs) == 0:
            return
        np_funcs: np.ndarray = funcs.numpy()
        x = np.linspace(self.bounds[0], self.bounds[1], num=3)
        for i in range(np_funcs.shape[0]):
            c_fun = np_funcs[i]
            w_y = EPSILON if c_fun[1] == 0 else c_fun[1]
            y = -(c_fun[0] * x + c_fun[2]) / w_y
            ax.plot(x, y, *args, **kwds)

    def save(self, name: str = "hyps.pkl"):
        save_path = os.path.join(self.root_dir, name)
        data = {
            "hpas": self.hyperplane_arrangements,
            "bounds": self.bounds,
        }
        torch.save(data, save_path)

    def run(self, is_draw=False, is_statistic=True):
        funs = [self.save]
        if is_draw:
            funs.append(self.draw_hyperplane_arrangments)
        if is_statistic:
            funs.append(self.statistics_intersect)
        for fun in funs:
            fun()


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
        self.hyperplane_arrangements: Dict[int, List[HyperplaneArrangement]] = dict()
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
        hp_arr = HyperplaneArrangement(p_funs, p_regions, c_funs, intersect_funs, n_regions, depth)
        depth_hp_arrs = self.hyperplane_arrangements.get(depth, list())
        depth_hp_arrs.append(hp_arr)
        self.hyperplane_arrangements[depth] = depth_hp_arrs
        return


class LinearRegion(_base):
    def __init__(
        self,
        save_dir: str,
        net: Callable[[int], Model],
        dataset: Callable[..., Tuple[Dataset, int]],
        *,
        save_epoch: List[int] = [0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 100],
        bounds: Tuple[float] = (-1, 1),
        is_draw: bool = True,
        is_draw_3d: bool = False,
        is_draw_hpas: bool = False,
        is_statistic_hpas: bool = True,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        super().__init__(
            save_dir,
            net=net,
            dataset=dataset,
            save_epoch=save_epoch,
            device=device,
        )
        self.bounds = bounds
        self.is_draw = is_draw
        self.is_draw_3d = is_draw_3d
        self.is_draw_hpas = is_draw_hpas
        self.is_statistic_hpas = is_statistic_hpas
        self.is_hpas = is_draw_hpas or is_statistic_hpas

    def get_region(self):
        net, dataset, n_classes = self._init_model()
        val_dataloader = data.DataLoader(dataset, shuffle=True, pin_memory=True)
        au = ReLUNets(device=self.device)
        model_list = os.listdir(self.model_dir)
        with torch.no_grad():
            for model_name in model_list:
                epoch = float(model_name.split("_")[-1][:-4])
                if epoch not in self.save_epoch and "best" not in model_name:
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
                    input_size=dataset.input_size,
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
                        device=self.device,
                    )
                    drawReginImage.draw(self.is_draw_3d)
                if self.is_hpas:
                    hpas = HyperplaneArrangements(
                        save_dir,
                        handler.hyperplane_arrangements,
                        self.bounds,
                    )
                    hpas.run(
                        is_draw=self.is_draw_hpas,
                        is_statistic=self.is_statistic_hpas,
                    )

                dataSaveDict = {
                    "funcs": handler.funs,
                    "regions": handler.regions,
                    "points": handler.points,
                    "regionNum": region_num,
                    "accuracy": acc,
                }
                torch.save(dataSaveDict, os.path.join(save_dir, "net_regions.pkl"))


class Experiment(_base):
    def __init__(
        self,
        net: Callable[[int], Model],
        dataset: Callable[..., Tuple[Dataset, int]],
        save_dir: str,
        init_fun: Callable[..., None],
        *,
        save_epoch: List[int] = [0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 100],
        device: torch.device = torch.device('cpu'),
    ) -> None:
        super().__init__(
            save_dir,
            net=net,
            dataset=dataset,
            save_epoch=save_epoch,
            device=device,
        )
        self.init_fun = init_fun
        self.run_funs = list()

    def train(
        self,
        max_epoch: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        train_handler: Callable[[nn.Module, int, int, int, torch.Tensor, torch.Tensor, str], None] = None,
    ):
        toy_train = TrainToy(
            save_dir=self.save_dir,
            net=self.net,
            dataset=self.dataset,
            save_epoch=self.save_epoch,
            max_epoch=max_epoch,
            batch_size=batch_size,
            lr=lr,
            train_handler=train_handler,
            device=self.device,
        )
        self.append(toy_train.train)
        return self

    def linear_region(
        self,
        is_draw: bool = True,
        is_draw_3d: bool = False,
        is_draw_hpas: bool = False,
        is_statistic_hpas: bool = True,
        bounds: Tuple[float] = (-1, 1),
    ):
        linear_region = LinearRegion(
            save_dir=self.save_dir,
            net=self.net,
            dataset=self.dataset,
            save_epoch=self.save_epoch,
            is_draw=is_draw,
            is_draw_3d=is_draw_3d,
            is_draw_hpas=is_draw_hpas,
            is_statistic_hpas=is_statistic_hpas,
            bounds=bounds,
            device=self.device,
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
