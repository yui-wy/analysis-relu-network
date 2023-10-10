import torch
import torch.nn as nn

from torchays.modules import base


class _BatchNorm(nn.modules.batchnorm._BatchNorm, base.BaseModule):
    def forward(self, input):
        return self._forward(super().forward, input)

    def forward_graph(self, x, weight_graph=None, bias_graph=None):
        """
        Analyzing BatchNorm2d \n
        track_running_stats = True->Using saving var and mean.
        graph_size: ((*x.shape)), (*input_size))
        """
        assert self.track_running_stats, "Please set track_running_stats = True"
        assert weight_graph is not None, "BatchNorm2d is not a first layer!!"
        bias_graph = (
            torch.zeros_like(x, device=x.device) if bias_graph is None else bias_graph
        )
        input_size = self._get_input_size(x, weight_graph)
        bias_graph = super().forward(bias_graph)

        size = list(self._get_size_to_one(x.size()[1:]))
        size[0] = -1

        weight = (
            self.weight
            if self.affine
            else torch.ones(self.num_features, device=x.device)
        )
        real_weight = (weight / torch.sqrt(self.running_var + self.eps)).view(
            *size, *self._get_size_to_one(input_size)
        )
        weight_graph *= real_weight

        return weight_graph, bias_graph

    def train(self, mode: bool = True):
        return base.BaseModule.train(self, mode=mode)

    def eval(self):
        return base.BaseModule.eval(self)


class BatchNormNone(_BatchNorm):
    def forward(self, input):
        return input

    def forward_graph(self, x, weight_graph=None, bias_graph=None):
        return weight_graph, bias_graph


class BatchNorm1d(_BatchNorm):
    __doc__ = nn.BatchNorm1d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                'expected 2D or 3D input (got {}D input)'.format(input.dim())
            )


class BatchNorm2d(_BatchNorm):
    __doc__ = nn.BatchNorm2d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class BatchNorm3d(_BatchNorm):
    __doc__ = nn.BatchNorm3d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
