from typing import Any, Callable, Tuple

import torch
import torch.nn as nn
from torch import Tensor


WEIGHT_GRAPH = "weight_graph"
BIAS_GRAPH = "bias_graph"


def get_size_to_one(size: torch.Size):
    assert isinstance(size, torch.Size), 'Input must be a torch.Size'
    return torch.Size((1,) * len(size))


def get_origin_size(input: torch.Tensor, weight_graph: torch.Tensor):
    return weight_graph.size()[input.dim() :] if weight_graph is not None else input.size()[1:]


def get_input(input):
    # TODO: WEIGHT_GRAPH与BIAS_GRAPH单独剔除与处理加入
    if not isinstance(tuple(input)[-1], dict) or (WEIGHT_GRAPH not in tuple(input)[-1]) or (BIAS_GRAPH not in tuple(input)[-1]):
        input = input if isinstance(input, tuple) else (input,)
        return input, {
            WEIGHT_GRAPH: None,
            BIAS_GRAPH: None,
        }
    input = tuple(input)
    return input[:-1], input[-1]


class Module(nn.Module):
    """
    Getting weight_graph and bias_graph from network.

    Coding:
            >>> net.graph()
            >>> with torch.no_grad():
                    # out -> (output, graph)
                    # graph is a dict with "weight_graph", "bias_graph"
                    output, graph = net(input)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.graphing = False
        self.origin_size: torch.Size = None

    def _origin_size(self, input: torch.Tensor, weight_graph: torch.Tensor):
        if self.origin_size is None:
            self.origin_size = get_origin_size(input, weight_graph)
        return self.origin_size

    def forward_graph(self, *input, weight_graph: Tensor = None, bias_graph: Tensor = None) -> Tuple[Tensor, Tensor]:
        """
        forward_graph(Any):

        Return:
            weight_graph : A Tensor is the graph of the weight.
            bias_graph : A Tensor is the graph of the bias.

        Example:
            >>> def forward_graph(...):
            >>>     ....
            >>>     return weight_graph, bias_graph
        """
        raise NotImplementedError()

    def train(self, mode: bool = True):
        self.graphing = False
        return nn.Module.train(self, mode)

    def graph(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = False
        self.graphing = mode
        for module in self.children():
            if isinstance(module, Module):
                module.graph()
        return self

    def __forward_graph(self, graph_forward: Callable[..., Any], *args, **kwargs):
        """
        If the results of "forward_graph" is "weight_graph, bias_graph", you can use this function to wapper the graph to a 'dict'.

        return:
            graph: {
                "weight_graph": wg,
                "bias_graph": bg,
            }
        """
        output = super().forward(*args)
        wg, bg = graph_forward(*args, **kwargs)
        return output, {
            WEIGHT_GRAPH: wg,
            BIAS_GRAPH: bg,
        }

    def forward(self, input):
        if self.graphing:
            args, kwargs = get_input(input)
            return self.__forward_graph(self.forward_graph, *args, **kwargs)
        else:
            if not isinstance(input, tuple):
                input = (input,)
            return super().forward(*input)
