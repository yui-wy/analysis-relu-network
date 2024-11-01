import os

import numpy as np
import torch

from dataset import GAUSSIAN_QUANTILES, MOON, RANDOM, simple_get_data
from experiment import Analysis, Experiment
from torchays import nn
from torchays.analysis import Model
from torchays.models import TestTNetLinear

GPU_ID = 0
SEED = 5
NAME = "Linear"
DATASET = MOON
N_LAYERS = [32, 32, 32]
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
# Experiment
IS_EXPERIMENT = True
# is training the network.
IS_TRAIN = False
# the number of the workers
WORKERS = 1
# is drawing the region picture. Only for 2d input.
IS_DRAW = True
# the depth of the NN to draw
DRAW_DEPTH = -1
# is drawing the 3d region picture.
IS_DRAW_3D = False
# is handlering the hyperplanes arrangement.
IS_DRAW_HPAS = False
IS_STATISTIC_HPAS = True

# Analysis
IS_ANALYSIS = True
# draw the dataset distribution
WITH_DATASET = True

root_dir = os.path.abspath("./")
save_dir = os.path.join(root_dir, "cache", f"{DATASET}-{N_SAMPLES}-{SEED}")


def init_fun():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)


def net(n_classes: int) -> Model:
    return TestTNetLinear(
        in_features=IN_FEATURES,
        layers=N_LAYERS,
        name=NAME,
        n_classes=n_classes,
        norm_layer=nn.BatchNorm1d,
    )


def dataset(save_dir: str, name: str = "dataset.pkl"):
    def make_dataset():
        return simple_get_data(
            dataset=DATASET,
            n_samples=N_SAMPLES,
            noise=0.2,
            random_state=5,
            data_path=os.path.join(save_dir, name),
            n_classes=N_CLASSES,
            in_features=IN_FEATURES,
            bias=DATASET_BIAS,
        )

    return make_dataset


if __name__ == "__main__":
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
