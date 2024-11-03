from typing import Any, Dict, Tuple

import torch

from ..nn import Module


class Model(Module):
    n_relu: int
    name: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = __class__.__name__

    def forward_layer(self, x, depth: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        raise NotImplementedError()
