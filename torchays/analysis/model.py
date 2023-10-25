from typing import Any, Tuple, Dict

import torch
from torchays.nn import Module


class Model(Module):
    def forward_graph_Layer(x, depth: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        NotImplementedError
