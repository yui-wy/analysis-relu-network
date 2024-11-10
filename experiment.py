import os

import numpy as np
import torch

from dataset import (
    GAUSSIAN_QUANTILES,
    MNIST,
    MNIST_TYPE,
    MOON,
    RANDOM,
    simple_get_data,
)
from experiment import Analysis, Experiment
from torchays import nn
from torchays.models import LeNet, TestTNetLinear, TestResNet

GPU_ID = 5
SEED = 5
NAME = "Linear"
# ===========================================
TYPE = RANDOM
# ===========================================
# Test-Net
N_LAYERS = [32, 32, 32]
# ===========================================
# Dataset
N_SAMPLES = 10000
DATASET_BIAS = 0
# only GAUSSIAN_QUANTILES
N_CLASSES = 2
# only RANDOM
IN_FEATURES = 4
# is download for mnist
DOWNLOAD = False
# ===========================================
# Training
MAX_EPOCH = 10000
SAVE_EPOCH = [10000]
BATCH_SIZE = 64
LR = 1e-3
# is training the network.
IS_TRAIN = True
# ===========================================
# Experiment
IS_EXPERIMENT = True
BOUND = (-1, 1)
# the number of the workers
WORKERS = 1
# ===========================================
# Drawing
# is drawing the region picture. Only for 2d input.
IS_DRAW = False
# the depth of the NN to draw
DRAW_DEPTH = -1
# is drawing the 3d region picture.
IS_DRAW_3D = False
# is handlering the hyperplanes arrangement.
IS_DRAW_HPAS = False
IS_STATISTIC_HPAS = False
# ===========================================
# Analysis
IS_ANALYSIS = True
# draw the dataset distribution
WITH_DATASET = True
# ===========================================
# path
root_dir = os.path.abspath("./")
cache_dir = os.path.join(root_dir, "cache")
save_dir = os.path.join(cache_dir, f"{TYPE}-{N_SAMPLES}-{IN_FEATURES}-{SEED}")


def init_fun():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)


def net(type: str = MOON):

    def make_net(n_classes: int):
        if type == MNIST_TYPE:
            return LeNet()
        return TestTNetLinear(
            in_features=IN_FEATURES,
            layers=N_LAYERS,
            name=NAME,
            n_classes=n_classes,
            norm_layer=nn.BatchNormNone,
        )

    return make_net


def dataset(
    save_dir: str,
    type: str = MOON,
    name: str = "dataset.pkl",
):
    def make_dataset():
        if type == MNIST_TYPE:
            mnist = MNIST(root=os.path.join(save_dir, "mnist"), download=DOWNLOAD)
            return mnist, len(mnist.classes)
        return simple_get_data(dataset=type, n_samples=N_SAMPLES, noise=0.2, random_state=5, data_path=os.path.join(save_dir, name), n_classes=N_CLASSES, in_features=IN_FEATURES, bias=DATASET_BIAS)

    return make_dataset


if __name__ == "__main__":
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda', GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
    if IS_EXPERIMENT:
        exp = Experiment(
            save_dir=save_dir,
            net=net(type=TYPE),
            dataset=dataset(save_dir, type=TYPE),
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
            workers=WORKERS,
            bounds=BOUND,
            is_draw=IS_DRAW,
            is_draw_3d=IS_DRAW_3D,
            is_draw_hpas=IS_DRAW_HPAS,
            is_statistic_hpas=IS_STATISTIC_HPAS,
            draw_depth=DRAW_DEPTH,
        )
        exp()
    if IS_ANALYSIS:
        analysis = Analysis(
            root_dir=save_dir,
            with_dataset=WITH_DATASET,
        )
        analysis()
