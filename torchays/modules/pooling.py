import torch
import torch.nn as nn
from torchays.modules import base


class AysAvgPool2d(nn.AvgPool2d, base.AysBaseModule):
    __doc__ = nn.AvgPool2d.__doc__

    def forward(self, input):
        return self.easy_forward(super().forward, input)

    def forward_graph(self, x, weight_graph=None,  bias_graph=None):
        assert self.ceil_mode, "'ceil_mode' must be True."
        assert self.count_include_pad, "'count_include_pad' must be True."
        # bias_graph
        bias_graph = torch.zeros_like(x, device=x.device) if bias_graph is None else bias_graph
        bias_graph = super().forward(bias_graph)
        # weight_graph
        input_size = self._get_input_size(x, weight_graph)
        weight_graph = self._conv2d_opt(x, input_size, bias_graph.size(), weight_graph)
        return weight_graph, bias_graph

    def _conv2d_opt(self, x, input_size, output_size, weight_graph):
        # for conv2d and Pooling2d(avg)
        channels, h_n, w_n = output_size[1], output_size[2], output_size[3]
        # create weight_graph(graph)
        # (n, c_out, h_out, w_out, *(input_size))
        wg = torch.zeros((*output_size, *input_size), device=x.device)
        ks = nn.modules.utils._pair(self.kernel_size)
        padding = nn.modules.utils._pair(self.padding)
        stride = nn.modules.utils._pair(self.stride)
        divisor = ks[0] * ks[1] if self.divisor_override is None else self.divisor_override
        kernel_weight = torch.eye(channels, device=x.device).view(channels, channels, 1, 1) / divisor
        # Conv2d opt
        # ===============================================================
        # hook_x :(n, c_out, c_in, h_in, w_in)
        if weight_graph is None:
            hook_x = torch.zeros(x.size(0), channels, channels, x.size(2)+padding[0]*2, x.size(3)+padding[1]*2, device=x.device)
            # origin
            for h in range(h_n):
                for w in range(w_n):
                    hook_x[:, :, :, h*stride[0]: h*stride[0]+ks[0], w*stride[1]: w*stride[1]+ks[1]] += kernel_weight
                    wg[:, :, h, w] = hook_x[:, :, :, padding[0]: hook_x.shape[3]-padding[0], padding[1]: hook_x.shape[4]-padding[1]]
                    hook_x.zero_()
        # ===============================================================
        # speed 1
        else:
            # hook_kernel_weight : (w_out, c_out, c_in, k , w_in+2padding)
            hook_kernel_weight = torch.zeros(w_n, channels, channels, ks[0], x.size(3)+padding[1]*2, device=x.device)
            for w in range(w_n):
                hook_kernel_weight[w, :, :, :, w*stride[1]: w*stride[1]+ks[1]] += kernel_weight
            # hook_kernel_weight : (w_out, c_out, c_in, k , w_in)
            hook_kernel_weight = hook_kernel_weight[:, :, :, :, padding[1]: hook_kernel_weight.size(4)-padding[1]]

            for h in range(h_n):
                # get pre_graph
                # pre_graph : (n, c_in, h_in, w_in, *(input_size))
                pos = h * stride[0] - padding[0]
                pos_end = pos + ks[0]

                pos_end = pos_end if pos_end < x.size(2) else x.size(2)
                pos_h = pos if pos > -1 else 0

                pos_kernal = pos_h - pos
                pos_end_kernal = pos_end - pos
                # hook_kernel_weight_s : (w_out, c_out, (c_in, h_pos-, w_in)) -> ((c_in, h_pos-, w_in), w_out, c_out)
                hook_kernel_weight_s = hook_kernel_weight[:, :, :, pos_kernal: pos_end_kernal, :].reshape(w_n * channels, -1).permute(1, 0)
                # graph_hook : (n, (c_in, h_pos-, w_in), (input_size)) -> (n, (input_size), (c_in, h_pos-, w_in))
                graph_hook = weight_graph[:, :, pos_h: pos_end, :, :, :, :].reshape(x.size(0), -1, input_size.numel()).permute(0, 2, 1)
                # weight_graph : (n, c_out, w_out, *(input_size))
                wg[:, :, h] = torch.matmul(graph_hook, hook_kernel_weight_s).reshape(x.size(0), *input_size, w_n, channels).permute(0, 5, 4, 1, 2, 3)
        return wg

    def train(self, mode: bool = True):
        return base.AysBaseModule.train(self, mode=mode)

    def eval(self):
        return base.AysBaseModule.eval(self)
