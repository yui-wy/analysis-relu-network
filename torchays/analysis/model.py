from typing import Any, Tuple, Dict

import torch
from torchays.modules.base import BaseModule


class Model(BaseModule):
    def forward_graph_Layer(x, depth: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        NotImplementedError
