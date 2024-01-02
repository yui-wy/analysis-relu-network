from typing import Any, Dict, Tuple

import torch

from ..nn import Module


class Model(Module):
    n_relu: int
    name: str

    def forward_layer(x, depth: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        raise NotImplementedError()
