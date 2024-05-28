import os
from typing import Dict

import numpy as np
import torch

from dataset import GAUSSIAN_QUANTILES, MOON, RANDOM, simple_get_data
from experiment import Analysis, Experiment
from torchays import nn
from torchays.analysis import Model
from torchays.models import TestTNetLinear
from torchays.nn.modules.batchnorm import BatchNorm1d, BatchNormNone
from torchays.nn.modules.norm import Norm1d

GPU_ID = 0
SEED = 5
NAME = "Linear-single-bn"
DATASET = GAUSSIAN_QUANTILES
N_LAYERS = [32]
# Dataset
N_SAMPLES = 1000
DATASET_BIAS = 1
# only GAUSSIAN_QUANTILES
N_CLASSES = 5
# only RANDOM
IN_FEATURES = 2
# Training
MAX_EPOCH = 5000
SAVE_EPOCH = [0, 10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000]
BATCH_SIZE = 32
LR = 1e-3
BOUND = (0, 2)

# Experiment
IS_EXPERIMENT = True
# is use batchnorm
IS_BN = True
# is training the network.
IS_TRAIN = True
# is drawing the region picture. Only for 2d input.
IS_DRAW = True
# is drawing the 3d region picture.
IS_DRAW_3D = False
# is handlering the hyperplanes arrangement.
IS_HPAS = False

# Analysis
IS_ANALYSIS = True
# draw the dataset distribution
WITH_DATASET = True
# analysis the batch norm
WITH_BN = True


def init_fun():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)


def norm(num_features):
    is_norm = False
    if not is_norm:
        return BatchNormNone(num_features)
    # freeze parameters
    freeze = True
    # set init parameters
    set_parameters = None
    return Norm1d(num_features, freeze, set_parameters)


def _norm(is_bn: bool = True):
    if is_bn:
        return BatchNorm1d
    return norm


def net(n_classes: int) -> Model:
    return TestTNetLinear(
        in_features=IN_FEATURES,
        layers=N_LAYERS,
        name=NAME,
        n_classes=n_classes,
        norm_layer=_norm(IS_BN),
    )


def dataset(save_dir: str, name: str = "dataset.pkl"):
    def make_dataset():
        return simple_get_data(
            dataset=DATASET,
            n_samples=N_SAMPLES,
            noise=0.09,
            random_state=5,
            data_path=os.path.join(save_dir, name),
            n_classes=N_CLASSES,
            in_features=IN_FEATURES,
            bias=DATASET_BIAS,
        )

    return make_dataset


# 当前步数下的, 某一层的bn数据
batch_norm_data: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = dict()


def train_handler(
    net: nn.Module,
    epoch: int,
    step: int,
    total_step: int,
    loss: torch.Tensor,
    acc: torch.Tensor,
    save_dir: str,
):
    step_name = f"{epoch}/{step}"
    current_bn_data = dict()
    for layer_name, module in net._modules.items():
        if "_norm" not in layer_name:
            continue
        # 存储每一个batch下的bn的参数
        module: nn.BatchNorm1d
        parameters: Dict[str, torch.Tensor] = module.state_dict()
        weight = parameters.get("weight").cpu()
        bias = parameters.get("bias").cpu()
        running_mean = parameters.get("running_mean").cpu()
        running_var = parameters.get("running_var").cpu()
        num_batches_tracked = parameters.get("num_batches_tracked").cpu()
        # 计算对应的A_bn和B_bn
        p = torch.sqrt(running_var + module.eps)
        # weight_bn = w/√(var)
        weight_bn = weight / p
        # bias_bn = b - w*mean/√(var)
        bias_bn = bias - weight_bn * running_mean
        save_dict = {
            "weight": weight,
            "bias": bias,
            "running_mean": running_mean,
            "running_var": running_var,
            "num_batches_tracked": num_batches_tracked,
            "weight_bn": weight_bn,
            "bias_bn": bias_bn,
        }
        current_bn_data[layer_name] = save_dict
    batch_norm_data[step_name] = current_bn_data


if __name__ == "__main__":
    root_dir = os.path.abspath("./")
    save_dir = os.path.join(root_dir, "cache", f"{DATASET}-{N_SAMPLES}-{SEED}")
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
    if IS_EXPERIMENT:
        exp = Experiment(
            save_dir=save_dir,
            net=net,
            dataset=dataset(save_dir),
            init_fun=init_fun,
            save_epoch=SAVE_EPOCH,
            device=device,
        )
        if IS_TRAIN:
            default_handler = train_handler if IS_BN else None
            exp.train(
                max_epoch=MAX_EPOCH,
                batch_size=BATCH_SIZE,
                train_handler=default_handler,
                lr=LR,
            )
        exp.linear_region(
            bounds=BOUND,
            is_draw=IS_DRAW,
            is_hpas=IS_HPAS,
            is_draw_3d=IS_DRAW_3D,
        )
        exp()
        if IS_BN:
            # 保存batch_norm
            batch_norm_path = os.path.join(exp.get_root(), f"batch_norm.pkl")
            torch.save(batch_norm_data, batch_norm_path)
    if IS_ANALYSIS:
        analysis = Analysis(
            root_dir=save_dir,
            with_dataset=WITH_DATASET,
            with_bn=WITH_BN,
        )
        analysis()
