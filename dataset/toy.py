import os
from typing import Callable, Dict, Tuple, TypeAlias

import numpy as np
import torch
from sklearn.datasets import make_gaussian_quantiles, make_moons, make_classification
from torch.utils import data

DataFunc: TypeAlias = Callable[[], Tuple[np.ndarray, np.ndarray]]

DataType: TypeAlias = str

MOON: DataType = "moon"
GAUSSIAN_QUANTILES: DataType = "gaussian quantiles"
RANDOM: DataType = "random"
CLASSIFICATION: DataType = "classification"


class Dataset(data.Dataset):
    def __init__(self, name: DataType, data_fun: DataFunc) -> None:
        super().__init__()
        self.name = name
        self.data, self.classes = data_fun()
        self.input_size = self.data.shape[1:]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        x, target = torch.from_numpy(self.data[index]), self.classes[index]
        return x, target

    def __len__(self):
        return self.data.shape[0]


def _norm(data: np.ndarray) -> np.ndarray:
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = (data - data.mean(0, keepdims=True)) / ((data.std(0, keepdims=True) + 1e-16))
    data /= np.max(np.abs(data))
    return data


def moon(
    n_samples: int | Tuple[int, int] = 1000,
    *,
    noise: float = None,
    random_state: int = None,
    bias: int = 0,
    norm_func: Callable[[np.ndarray], np.ndarray] = _norm,
) -> Tuple[DataFunc, int]:
    def data_fun() -> Tuple[np.ndarray, np.ndarray]:
        data, classes = make_moons(n_samples, noise=noise, random_state=random_state)
        if norm_func is not None:
            data = norm_func(data)
        return data + bias, classes

    return data_fun, 2


def classification(
    n_samples: int | Tuple[int, int] = 1000,
    *,
    in_features: int = 2,
    n_classes: int = 3,
    bias: int = 0,
    random_state: int | None = None,
    norm_func: Callable[[np.ndarray], np.ndarray] = _norm,
) -> Tuple[DataFunc, int]:
    def data_fun() -> Tuple[np.ndarray, np.ndarray]:
        data, classes = make_classification(
            n_samples,
            n_features=in_features,
            n_informative=in_features,
            n_clusters_per_class=1,
            n_redundant=0,
            n_classes=n_classes,
            class_sep=10,
            random_state=random_state,
            hypercube=True,
        )
        if norm_func is not None:
            data = norm_func(data)
        return data + bias, classes

    return data_fun, n_classes


def gaussian_quantiles(
    n_samples: int = 1000,
    *,
    mean: np.ndarray | None = None,
    cov: float = 1,
    n_features: int = 2,
    n_classes: int = 3,
    shuffle: bool = True,
    random_state: int | None = None,
    bias: int = 0,
    norm_func: Callable[[np.ndarray], np.ndarray] = _norm,
) -> Tuple[DataFunc, int]:
    def data_fun() -> Tuple[np.ndarray, np.ndarray]:
        data, classes = make_gaussian_quantiles(mean=mean, cov=cov, n_samples=n_samples, n_features=n_features, n_classes=n_classes, shuffle=shuffle, random_state=random_state)
        if norm_func is not None:
            data = norm_func(data)
        return data + bias, classes

    return data_fun, n_classes


def random(
    n_samples: int = 1000,
    in_features: int = 2,
    bias: int = 0,
) -> Tuple[DataFunc, int]:
    def data_fun() -> Tuple[np.ndarray, np.ndarray]:
        data = np.random.uniform(-1, 1, (n_samples, in_features))
        classes = np.sign(np.random.uniform(-1, 1, [n_samples]))
        classes = np.where(classes > 0, 1, 0)
        return data + bias, classes

    return data_fun, 2


def from_path(data_path: str) -> Tuple[DataFunc, int]:
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"cannot find the dataset file [{data_path}]")
    data_dict: Dict = torch.load(data_path, weights_only=False)
    data: np.ndarray = data_dict.get("data")
    classes: np.ndarray = data_dict.get("classes")
    n_classes: int = data_dict.get("n_classes")

    def data_fun() -> Tuple[np.ndarray, np.ndarray]:
        return data, classes

    return data_fun, n_classes


def save_data(
    data_func: DataFunc,
    n_classes: int,
    *,
    save_path: str,
) -> Tuple[DataFunc, int]:
    def wrapper() -> Tuple[np.ndarray, np.ndarray]:
        data, classes = data_func()
        torch.save(
            {
                "data": data,
                "classes": classes,
                "n_classes": n_classes,
            },
            save_path,
        )
        return data, classes

    return wrapper, n_classes


def simple_get_data(
    dataset: DataType,
    n_samples: int,
    noise: int,
    random_state: int,
    data_path: str,
    n_classes: int = 2,
    in_features: int = 2,
    bias: int = 0,
) -> Tuple[Dataset, int]:
    if os.path.exists(data_path):
        data_fun, n_classes = from_path(data_path)
    if dataset == MOON:
        data_fun, n_classes = save_data(*moon(n_samples, noise=noise, random_state=random_state, bias=bias), save_path=data_path)
    if dataset == GAUSSIAN_QUANTILES:
        data_fun, n_classes = save_data(*gaussian_quantiles(n_samples, n_classes=n_classes, bias=bias), save_path=data_path)
    if dataset == RANDOM:
        data_fun, n_classes = save_data(*random(n_samples, in_features, bias=bias), save_path=data_path)
    if dataset == CLASSIFICATION:
        data_fun, n_classes = save_data(*classification(n_samples, in_features=in_features, n_classes=n_classes, bias=bias, random_state=random_state), save_path=data_path)
    if (data_fun is None) or (n_classes is None):
        raise NotImplementedError(f"cannot find the dataset [{dataset}]")
    return Dataset(dataset, data_fun), n_classes
