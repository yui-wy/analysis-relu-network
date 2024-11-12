import os
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polytope as pc
import torch

from torchays import nn
from torchays.graph import COLOR, color, plot_regions, plot_regions_3d


class DrawRegionImage:
    def __init__(
        self,
        region_num: int,
        funcs: np.ndarray,
        regions: np.ndarray,
        points: np.ndarray,
        save_dir: str,
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
            edgecolor="gray",
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
        fig = plt.figure(0, figsize=(8, 7), dpi=600)
        ax: plt.Axes = fig.add_subplot(projection="3d")
        ax.cla()
        ax.tick_params(labelsize=15)
        plot_regions_3d(
            self.funcs,
            self.regions,
            z_fun=self._z_fun,
            ax=ax,
            alpha=0.8,
            color=color,
            edgecolor="grey",
            linewidth=0.2,
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
