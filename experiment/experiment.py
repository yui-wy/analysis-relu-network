import math
import os
from copy import deepcopy
from typing import Any, Callable, List, Tuple

import torch
from torch.utils import data

from dataset import Dataset
from torchays import nn
from torchays.cpa import CPA, Model
from torchays.utils import get_logger

from .draw import DrawRegionImage
from .handler import get_handler
from .hpa import HyperplaneArrangements


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

    def run(self):
        raise NotImplementedError()


class Train(_base):
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

    def run(self):
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


class CPAs(_base):
    def __init__(
        self,
        save_dir: str,
        net: Callable[[int], Model],
        dataset: Callable[..., Tuple[Dataset, int]],
        *,
        workers: int = 1,
        save_epoch: List[int] = [100],
        best_epoch: bool = False,
        bounds: Tuple[float] = (-1, 1),
        depth: int = -1,
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
        self.workers, self.multi = self._works(workers)
        self.bounds = bounds
        self.is_draw = is_draw
        self.is_draw_3d = is_draw_3d
        self.is_draw_hpas = is_draw_hpas
        self.is_statistic_hpas = is_statistic_hpas
        self.is_hpas = is_draw_hpas or is_statistic_hpas
        self.best_epoch = best_epoch
        self.depth = depth

    def _works(self, workers: int):
        workers = math.ceil(workers)
        if workers <= 1:
            return 1, False
        return workers, True

    def run(self):
        net, dataset, n_classes = self._init_model()
        depth = self.depth if self.depth >= 0 else net.n_relu
        val_dataloader = data.DataLoader(dataset, shuffle=True, pin_memory=True)
        cpa = CPA(device=self.device, workers=self.workers)
        model_list = os.listdir(self.model_dir)
        with torch.no_grad():
            for model_name in model_list:
                epoch = float(model_name.split("_")[-1][:-4])
                if epoch not in self.save_epoch:
                    if self.best_epoch and "best" not in model_name:
                        continue
                    else:
                        continue
                print(f"Solve fileName: {model_name} ....")
                save_dir = os.path.join(self.experiment_dir, os.path.splitext(model_name)[0])
                os.makedirs(save_dir, exist_ok=True)
                net.load_state_dict(torch.load(os.path.join(self.model_dir, model_name), weights_only=False))
                acc = self.val_net(net, val_dataloader).cpu().numpy()
                print(f"Accuracy: {acc:.4f}")
                handler = get_handler(self.multi)
                logger = get_logger(
                    f"region-{os.path.splitext(model_name)[0]}",
                    os.path.join(save_dir, "region.log"),
                    multi=self.multi,
                )
                region_num = cpa.start(
                    net,
                    bounds=self.bounds,
                    input_size=dataset.input_size,
                    depth=depth,
                    # Warning: if multiprocessing is used, handler will do nothing.
                    handler=handler,
                    logger=logger,
                )
                print(f"Region counts: {region_num}")
                if self.is_draw:
                    draw_dir = os.path.join(save_dir, f"draw-region-{depth}")
                    os.makedirs(draw_dir, exist_ok=True)
                    dri = DrawRegionImage(
                        region_num,
                        handler.funs,
                        handler.regions,
                        handler.points,
                        draw_dir,
                        net,
                        n_classes,
                        bounds=self.bounds,
                        device=self.device,
                    )
                    dri.draw(self.is_draw_3d)
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
        self.runs = list()

    def train(
        self,
        max_epoch: int = 100,
        batch_size: int = 32,
        lr: float = 0.001,
        train_handler: Callable[[nn.Module, int, int, int, torch.Tensor, torch.Tensor, str], None] = None,
    ):
        _train = Train(
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
        self.append(_train.run)
        return self

    def cpas(
        self,
        workers: int = 1,
        best_epoch: bool = False,
        bounds: Tuple[float] = (-1, 1),
        depth: int = -1,
        is_draw: bool = False,
        is_draw_3d: bool = False,
        is_draw_hpas: bool = False,
        is_statistic_hpas: bool = False,
    ):
        cpas = CPAs(
            save_dir=self.save_dir,
            net=self.net,
            dataset=self.dataset,
            save_epoch=self.save_epoch,
            workers=workers,
            best_epoch=best_epoch,
            bounds=bounds,
            depth=depth,
            is_draw=is_draw,
            is_draw_3d=is_draw_3d,
            is_draw_hpas=is_draw_hpas,
            is_statistic_hpas=is_statistic_hpas,
            device=self.device,
        )
        self.append(cpas.run)

    def append(self, fun: Callable[..., None]):
        self.runs.append(self.init_fun)
        self.runs.append(fun)

    def run(self):
        for run in self.runs:
            run()

    def __call__(self, *args: Any, **kwds: Any):
        self.run(*args, **kwds)
