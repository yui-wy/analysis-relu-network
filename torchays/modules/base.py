from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor


class BaseModule(nn.Module):
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

    def _get_size_to_one(self, size):
        assert isinstance(size, torch.Size), 'Input must be a torch.Size'
        return torch.Size(map(lambda x: int(x / x), size))

    def _get_input_size(self, x, weight_graph):
        return x.size()[1:] if weight_graph is None else weight_graph.size()[len(x.size()) :]

    def _forward_graph_unimplemented(self, *input, weight_graph: Tensor = None, bias_graph: Tensor = None):
        raise NotImplementedError

    # forward_graph(Any):
    #
    # Return:
    #   weight_graph : A Tensor is the graph of the weight.
    #   bias_graph : A Tensor is the graph of the bias.
    #
    # Example:
    #   def forward_graph(...):
    #       ....
    #       return weight_graph, bias_graph
    forward_graph: Callable[..., Any] = _forward_graph_unimplemented

    def train(self, mode: bool = True):
        self.graphing = False
        return nn.Module.train(self, mode)

    def graph(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = False
        self.graphing = mode
        for module in self.children():
            if isinstance(module, BaseModule):
                module.graph()
        return self

    def _get_input(self, input):
        """
        If 'graphing' is True, using this function to get the input.
        """

        assert self.graphing, "This function is used when the parameter 'graphing' is 'True'."
        if not isinstance(tuple(input)[-1], dict) or ("weight_graph" not in tuple(input)[-1]) or ("bias_graph" not in tuple(input)[-1]):
            input = input if isinstance(input, tuple) else (input,)
            return input, {
                "weight_graph": None,
                "bias_graph": None,
            }
        input = tuple(input)
        return input[:-1], input[-1]

    def _forward_graph(self, graph_forward: Callable[..., Any], *args, **kwargs):
        """
        If the results of "forward_graph" is "weight_graph, bias_graph", you can use this function to wapper the graph to a 'dict'.

        return:
            graph: {
                "weight_graph": wg,
                "bias_graph": bg,
            }
        """
        wg, bg = graph_forward(*args, **kwargs)
        return {
            "weight_graph": wg,
            "bias_graph": bg,
        }

    def _forward(self, function: Callable[..., Any], input):
        """
        This function uses the "forward_graph", if self.graphing is True.

        args:
            function: a callable function.
            input: if there are many inputs, please use the 'tuple'. for example: (input_1,input_2,...)

        """
        if self.graphing:
            args, kwargs = self._get_input(input)
            output = (
                function(*args),
                self._forward_graph(self.forward_graph, *args, **kwargs),
            )
        else:
            if not isinstance(input, tuple):
                input = (input,)
            output = function(*input)
        return output
