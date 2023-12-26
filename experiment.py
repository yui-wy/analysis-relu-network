import os

import numpy as np
import torch

from dataset import (
    GAUSSIAN_QUANTILES,
    MOON,
    RANDOM,
    simple_get_data,
)
from experiment import Experiment
from torchays import nn
from torchays.models import TestTNetLinear

GPU_ID = 0
SEED = 5
NAME = "Linear"
DATASET = MOON
N_LAYERS = [16, 16, 16]
# Dataset
N_SAMPLES = 1000
DATASET_BIAS = 0
# only GAUSSIAN_QUANTILES
N_CLASSES = 2
# only RANDOM
IN_FEATURES = 2
# Training
MAX_EPOCH = 5000
SAVE_EPOCH = [100, 200, 500, 1000, 2000, 4000, 5000]
BATCH_SIZE = 32
LR = 1e-3
BOUND = (-1, 1)

# is training the network.
IS_TRAIN = True
# is drawing the region picture. Only for 2d input.
IS_DRAW = True
# is handlering the hyperplanes arrangement.
IS_HPAS = False


def init_fun():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)


def net(n_classes: int) -> nn.Module:
    return TestTNetLinear(
        in_features=IN_FEATURES,
        layers=N_LAYERS,
        name=NAME,
        n_classes=n_classes,
        norm_layer=nn.BatchNorm1d,
    )


def dataset(save_dir: str, name: str = "dataset.pkl"):
    def fun():
        return simple_get_data(
            DATASET,
            N_SAMPLES,
            0.2,
            5,
            os.path.join(save_dir, name),
            n_classes=N_CLASSES,
            in_features=IN_FEATURES,
            bias=DATASET_BIAS,
        )

    return fun


if __name__ == "__main__":
    root_dir = os.path.abspath("./")
    save_dir = os.path.join(root_dir, "cache", f"{DATASET}-{N_SAMPLES}-{SEED}")
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
    exp = Experiment(
        save_dir=save_dir,
        net=net,
        dataset=dataset(save_dir),
        init_fun=init_fun,
        save_epoch=SAVE_EPOCH,
        device=device,
    )
    if IS_TRAIN:
        exp.train(
            max_epoch=MAX_EPOCH,
            batch_size=BATCH_SIZE,
            lr=LR,
        )
    exp.linear_region(
        bounds=BOUND,
        is_draw=IS_DRAW,
        is_hpas=IS_HPAS,
    )
    exp()
