from dataset.cifar import CIFAR10
from dataset.mnist import MNIST, MNIST_TYPE
from dataset.toy import (
    GAUSSIAN_QUANTILES,
    MOON,
    RANDOM,
    CLASSIFICATION,
    Dataset,
    from_path,
    gaussian_quantiles,
    moon,
    random,
    simple_get_data,
)

__all__ = [
    "simple_get_data",
    "moon",
    "gaussian_quantiles",
    "random",
    "from_path",
    "MNIST",
    "CIFAR10",
    "MOON",
    "GAUSSIAN_QUANTILES",
    "RANDOM",
    "Dataset",
    "MNIST_TYPE",
    "CLASSIFICATION",
]
