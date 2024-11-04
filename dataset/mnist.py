from typing import Any, Tuple

from PIL import Image
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor, Compose

MNIST_TYPE: str = "mnist"


def _transform(img: Image.Image):
    trans = Compose([ToTensor()])
    return trans(img)


class MNIST(mnist.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        if self.transform is None:
            self.transform = _transform
        self.input_size = (1, 28, 28)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
