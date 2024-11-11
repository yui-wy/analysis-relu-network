import os
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchays.graph import color, plot_region
from torchays.utils import CSV

EPSILON = 1e-16
STATISTIC_COUNT = "statistic_count"
STATISTIC_SCALE = "statistic_scale"
NEURAL_NUM = "neural_num"


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
        p_dir = os.path.join(self.root_dir, "hpa_statistics")
        os.makedirs(p_dir, exist_ok=True)
        counts_csv = CSV(os.path.join(p_dir, "counts.csv"))
        counts_statistic_csv = CSV(os.path.join(p_dir, "counts_statistic.csv"))
        counts_scales_csv = CSV(os.path.join(p_dir, "counts_scales.csv"))
        avg_csv = CSV(os.path.join(p_dir, "hpa_avg_regions.csv"))
        # MAP: [depth, Dict[name, [List[int]]]]
        statistic_dict: Dict[int, Dict[str, List[int] | int | Dict[int, List[int]]]] = dict()

        def statistic_fun(neural_num: int, num_list: List[int | float]):
            statistic_list = [0] * (neural_num + 1)
            for num in num_list:
                statistic_list[num] += 1
            return statistic_list

        statistics = self._get_statistics()
        max_neural_num = -1
        for depth, statistic in statistics.items():
            neural_num: int = statistic.get(NEURAL_NUM)
            if max_neural_num < neural_num:
                max_neural_num = neural_num
            intersect_counts: List[int] = statistic.get("intersect_counts")

            counts_list = statistic_fun(neural_num, intersect_counts)
            counts_array = np.array(counts_list)
            counts_sum: np.ndarray = np.sum(counts_array)
            tag = f"{depth}/{neural_num}/{counts_sum.item()}"
            #
            counts_csv.add_row(tag, intersect_counts)
            # 统计每个区域中相交的超平面的数量
            counts_statistic_csv.add_row(tag, counts_list)
            # 统计每个区域中相交超平面的数量占比
            counts_scales: np.ndarray = counts_array / counts_sum
            counts_scales_csv.add_row(tag, counts_scales.tolist())
            # n个超平面能分割区域的平均数量
            sub_hpa_region_counts: Dict[int, List[int]] = statistic.get("sub_hpa_region_counts")
            avg_list = self._get_average(neural_num, sub_hpa_region_counts)
            avg_csv.add_row(tag, avg_list)
            # save statistic
            depth_statistic: Dict[str, List[int | float]] = dict()
            depth_statistic[NEURAL_NUM] = neural_num
            depth_statistic[STATISTIC_COUNT] = counts_list
            statistic_dict[depth] = depth_statistic

        def save_csvs(*csvs: CSV):
            header_tag = "tag/neurals"
            header = [i for i in range(max_neural_num + 1)]
            for csv in csvs:
                csv.set_header(header_tag, header)
                csv.save()

        # save csv
        counts_csv.save()
        save_csvs(counts_statistic_csv, counts_scales_csv, avg_csv)
        # plot, 考虑概率和期望模型
        self._draw_statistic(p_dir, statistic_dict, max_neural_num, STATISTIC_COUNT, "statistic counts", "counts")

    def _get_statistics(self) -> Dict[str, Dict[str, int | List]]:
        statistics = dict()
        for depth, hpas in self.hyperplane_arrangements.items():
            intersect_counts = []
            sub_hpa_region_counts: Dict[int, List[int]] = dict()
            neural_num = -1
            for hpa in hpas:
                if neural_num == -1:
                    neural_num = len(hpa.c_funs)
                n_intersects = hpa.n_intersect_funcs
                intersect_counts.append(n_intersects)
                n_regions_list: List[int] = sub_hpa_region_counts.get(n_intersects, list())
                n_regions_list.append(hpa.n_regions)
                sub_hpa_region_counts[n_intersects] = n_regions_list
            statistic = dict()
            statistic[NEURAL_NUM] = neural_num
            statistic["intersect_counts"] = intersect_counts
            statistic["sub_hpa_region_counts"] = sub_hpa_region_counts
            statistics[depth] = statistic
        return statistics

    def _draw_statistic(
        self,
        dir: str,
        statistic_dict: Dict[int, Dict[str, List[int]]],
        max_neural_num: int,
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
        legend_list: List[str] = list()
        for depth, depth_statistic in statistic_dict.items():
            neural_num = depth_statistic.get(NEURAL_NUM)
            x_list = depth_statistic.get(key)
            ax.plot(np.arange(neural_num + 1), x_list, label=name, color=color(depth))
            legend_list.append(f"depth-{depth}/{neural_num}")
        ax.legend(legend_list)
        ax.set_xlabel("Neurals")
        ax.set_ylabel(y_label)
        ax.set_xticks(np.arange(max_neural_num + 1))
        fig_file = os.path.join(dir, f"{name}.jpg")
        plt.savefig(fig_file, format="jpg")
        plt.clf()
        plt.close()

    def _get_average(self, neural_num: int, sub_hpa_region_counts: Dict[int, List[int]]) -> List[float]:
        avg_list: List[float] = [0] * (neural_num + 1)
        for n_intersects, n_region_list in sub_hpa_region_counts.items():
            n_region_array = np.array(n_region_list)
            avg_list[n_intersects] = np.average(n_region_array).item()
        return avg_list

    def draw_hyperplane_arrangments(self):
        p_dir = os.path.join(self.root_dir, "hyperplane_arrangments")
        for depth, hpas in self.hyperplane_arrangements.items():
            save_dir = os.path.join(p_dir, f"depth_{depth+1}")
            for idx, hpa in enumerate(hpas):
                pic_dir = os.path.join(save_dir, f"hpa_{idx}")
                os.makedirs(pic_dir, exist_ok=True)
                self._draw_hyperplane_arrangment(hpa, pic_dir, f"arrangement.jpg")
                # self._draw_weights_scatter(hpa, pic_dir, f"weights_scatter.jpg")

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
