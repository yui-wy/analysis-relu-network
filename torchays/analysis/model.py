from typing import Any, Dict, Tuple

import torch

from ..nn import Module


class Model(Module):
    n_relu: int

    def forward_graph_Layer(x, depth: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        NotImplementedError
