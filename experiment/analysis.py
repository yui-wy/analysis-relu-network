import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from torchays.graph import color


class default_plt:
    def __init__(self, savePath, xlabel='', ylabel='', mode='png', isGray=False, isLegend=True, isGrid=True):
        self.savePath = savePath
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.mode = mode
        self.isGray = isGray
        self.isLegend = isLegend
        self.isGrid = isGrid

    def __enter__(self):
        fig = plt.figure(0, figsize=(8, 7), dpi=600)
        self.ax = fig.subplots()
        self.ax.cla()
        if not self.isGray:
            self.ax.patch.set_facecolor("w")
        self.ax.tick_params(labelsize=15)
        self.ax.set_xlabel(self.xlabel, fontdict={'weight': 'normal', 'size': 15})
        self.ax.set_ylabel(self.ylabel, fontdict={'weight': 'normal', 'size': 15})
        if self.isGrid:
            self.ax.grid(color="#EAEAEA", linewidth=1)
        return self.ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.isLegend:
            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])
            self.ax.legend(prop={'weight': 'normal', 'size': 14}, loc='lower left', bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=3, mode="expand")

        plt.savefig(self.savePath, dpi=600, format=f'{self.mode}')
        plt.clf()
        plt.close()


def default(savePath, xlabel='', ylabel='', mode='png', isGray=False, isLegend=True, isGrid=True):
    return default_plt(savePath, xlabel, ylabel, mode, isGray, isLegend, isGrid)


class Analysis:
    def __init__(
        self,
        root_dir,
        with_dataset: bool = False,
        with_bn: bool = False,
    ) -> None:
        self.root_dir = root_dir
        self.with_dataset = with_dataset
        self.with_bn = with_bn

    def analysis(self) -> None:
        # draw dataset distribution
        self.common()
        # get data
        experiment_dict = {}
        for tag in os.listdir(self.root_dir):
            if tag in ["bn_exp", "analysis"]:
                continue
            tag_dir = os.path.join(self.root_dir, tag)
            if not os.path.isdir(tag_dir):
                continue
            tag_dict = {}
            experiment_dir = os.path.join(tag_dir, "experiment")
            for epoch_fold in os.listdir(experiment_dir):
                epoch = float(epoch_fold[4:])
                net_reigions = torch.load(os.path.join(experiment_dir, epoch_fold, 'net_regions.pkl'))
                tag_dict[epoch] = net_reigions
            experiment_dict[tag] = tag_dict
        # save dir
        save_dir = os.path.join(self.root_dir, "analysis")
        os.makedirs(save_dir, exist_ok=True)
        # draw picture
        self.draw(experiment_dict, save_dir)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.analysis(*args, **kwds)

    def common(self):
        funs = []
        if self.with_dataset:
            funs.append(self.draw_dataset)
        if self.with_bn:
            funs.append(self.analysis_bn)
        for fun in funs:
            fun()

    def draw_dataset(self):
        dataset_path = os.path.join(self.root_dir, 'dataset.pkl')
        dataset = torch.load(dataset_path)
        save_path = os.path.join(self.root_dir, "distribution.png")
        x, y, n_classes = dataset['data'], dataset['classes'], dataset['n_classes']
        with default(save_path, 'x1', 'x2', isLegend=False, isGrid=False) as ax:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            for i in range(n_classes):
                ax.scatter(x[y == i, 0], x[y == i, 1], color=color(i))

    def analysis_bn(self):
        bn_path = os.path.join(self.root_dir, 'batch_norm.pkl')
        bn_data: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = torch.load(bn_path)
        save_dict: Dict[str, Dict[int, Dict[str, List[float]]] | List[str]] = dict()
        step_list = list()
        for step_name, step_data in bn_data.items():
            step_list.append(step_name)
            for layer_name, layer_data in step_data.items():
                for name, data in layer_data.items():
                    if name in ["num_batches_tracked"]:
                        continue
                    for i in range(len(data)):
                        neruals = save_dict.get(layer_name, dict())
                        values = neruals.get(i, dict())
                        value = values.get(name, list())
                        value.append(data[i])
                        values[name] = value
                        neruals[i] = values
                        save_dict[layer_name] = neruals
        save_dict["steps"] = step_list
        self._draw_bn_parameters(save_dict)

    def _draw_bn_parameters(self, save_dict: Dict[str, Dict[int, Dict[str, List[float]]]]):
        save_dir = os.path.join(self.root_dir, "bn_exp")
        os.makedirs(save_dir, exist_ok=True)
        step_list = save_dict.pop("steps", list())
        for layer_name, neruals in save_dict.items():
            for j, values in neruals.items():
                layer_dir = os.path.join(save_dir, layer_name)
                os.makedirs(layer_dir, exist_ok=True)
                save_path = os.path.join(layer_dir, f"nerual_{j}.png")
                with default(save_path, 'steps', 'values') as ax:
                    i = 0
                    for name, value in values.items():
                        ax.plot(range(len(step_list)), value, label=name, color=color(i))
                        i += 1

    def draw(self, experiment_dict: Dict, save_dir: str):
        funs = [
            self.save_region_epoch_tabel,
            self.draw_region_epoch_plot,
            self.draw_region_acc_plot,
            self.draw_epoch_acc_plot,
        ]
        for fun in funs:
            fun(experiment_dict, save_dir)

    def save_region_epoch_tabel(self, experiment_dict: Dict, save_dir):
        savePath = os.path.join(save_dir, "regionEpoch.csv")
        strBuff = ''
        head = None
        for tag, epochDict in experiment_dict.items():
            tag1 = tag.split('-')[-1].replace(',', '-')
            body = [tag1]
            dataList = []
            for epoch, fileDict in epochDict.items():
                data = [epoch, fileDict['regionNum']]
                dataList.append(data)
            dataList = np.array(dataList)
            a = dataList[:, 0]
            index = np.lexsort((a,))
            dataList = dataList[index]
            regionList = list(map(str, dataList[:, 1].astype(np.int16).tolist()))
            body.extend(regionList)
            bodyStr = ','.join(body)
            if head is None:
                epochList = list(map(str, dataList[:, 0].tolist()))
                head = [
                    'model/epoch',
                ]
                head.extend(epochList)
                headStr = ','.join(head)
                strBuff = strBuff + headStr + '\r\n'
            strBuff = strBuff + bodyStr + '\r\n'
        with open(savePath, 'w') as w:
            w.write(strBuff)
            w.close()

    def draw_region_epoch_plot(self, experiment_dict: Dict, save_dir):
        savePath = os.path.join(save_dir, "regionEpoch.png")
        with default(savePath, 'Epoch', 'Number of Rgions') as ax:
            i = 0
            for tag, epochDict in experiment_dict.items():
                tag1 = tag.split('-')[-1]
                dataList = []
                for epoch, fileDict in epochDict.items():
                    data = [epoch, fileDict['regionNum']]
                    dataList.append(data)
                dataList = np.array(dataList)
                a = dataList[:, 0]
                index = np.lexsort((a,))
                dataList = dataList[index]
                ax.plot(dataList[:, 0], dataList[:, 1], label=tag1, color=color(i))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                i += 1

    def draw_region_acc_plot(self, experiment_dict: Dict, save_dir):
        savePath = os.path.join(save_dir, "regionAcc.png")
        with default(savePath, 'Accuracy', 'Number of Rgions') as ax:
            i = 0
            for tag, epochDict in experiment_dict.items():
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
                ax.plot(dataList[:, 0], dataList[:, 1], label=tag1, color=color(i))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                # ax.set_xlim(0.965, 0.98)
                i += 1

    def draw_epoch_acc_plot(self, experiment_dict: Dict, save_dir):
        savePath = os.path.join(save_dir, "EpochAcc.png")
        with default(savePath, 'Epoch', 'Accuracy') as ax:
            i = 0
            for tag, epochDict in experiment_dict.items():
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
                ax.plot(dataList[:, 0], dataList[:, 1], label=tag1, color=color(i))
                i += 1
