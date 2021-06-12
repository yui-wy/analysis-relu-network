import os
import gc
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from analysis_lib import timer


class AnalysisNet(nn.Module):
    """
    Getting weight_graph and bias_graph from network.

    args:
        input_size : cifar10 is (3, 32, 32) 
    """

    def __init__(self, input_size=None):
        super(AnalysisNet, self).__init__()
        if not isinstance(input_size, torch.Size):
            input_size = torch.Size(input_size)
        self._input_size = input_size
        self._size_prod = None
        self._size_one = None
        self._print_log = False
        self._layer_num = -1

    @property
    def size_prod(self):
        """ 
        if input size is (3,32,32), this is 3*32*32
        """
        assert self._input_size is not None, 'Input_size is None'
        if self._size_prod is None:
            self._size_prod = self._get_size_prod(self._input_size)
        return self._size_prod

    @property
    def size_one(self):
        """ 
        if input size is (3,32,32), this is (1,1,1)
        """
        assert self._input_size is not None, 'Input_size is None'
        if self._size_one is None:
            self._size_one = self._get_size_to_one(self._input_size)
        return self._size_one

    def _get_size_to_one(self, size):
        assert isinstance(size, torch.Size), 'Input must be a torch.Size'
        return torch.Size(map(lambda x: int(x / x), size))

    def _get_size_prod(self, size):
        assert isinstance(size, torch.Size), 'Input must be a torch.Size'
        return size.numel()

    def forward_graph(self, x, pre_weight_graph=None, pre_bias_graph=None):
        """
            >>> net.eval()
            >>> with torch.no_grad():
                    forward_graph(...)
        """
        raise NotImplementedError

    def forward_graph_Layer(self, x, layer=0, pre_weight_graph=None, pre_bias_graph=None):
        """
            >>> net.eval()
            >>> with torch.no_grad():
                    forward_graph_Layer(...)
        """
        raise NotImplementedError

    def analysis_module(self, x, module, pre_weight_graph=None, pre_bias_graph=None):
        """
        A tool function for analyzing basic module.(conv2d, linear, ReLU,...).
        'pre_weight_graph' and 'pre_bias_graph' is None when the layer is the first layer.

        args:
            x : input data;
            module : network module;
            pre_weight_graph : pre_weight_graph;
            pre_bias_graph : pre_bias_graph;
        """
        self._device = x.device

        # timer.timer.tic()

        if pre_weight_graph is None:
            pre_bias_graph = None

        if isinstance(module, nn.Conv2d):
            # conv2d unit
            output, weight_graph, bias_graph = self._analysis_conv2d(x, module, pre_weight_graph, pre_bias_graph)
        elif isinstance(module, nn.Linear):
            # linear unit
            output, weight_graph, bias_graph = self._analysis_linear(x, module, pre_weight_graph, pre_bias_graph)
        elif isinstance(module, nn.ReLU):
            # ReLU unit
            output, weight_graph, bias_graph = self._analysis_ReLU(x, module, pre_weight_graph, pre_bias_graph)
        elif isinstance(module, nn.AvgPool2d):
            # AvgPool2d unit
            output, weight_graph, bias_graph = self._analysis_avgPool2d(x, module, pre_weight_graph, pre_bias_graph)
        elif isinstance(module, nn.MaxPool2d):
            # MaxPool2d unit
            output, weight_graph, bias_graph = self._analysis_maxPool2d(x, module, pre_weight_graph, pre_bias_graph)
        elif isinstance(module, nn.BatchNorm2d):
            # MaxPool2d unit
            output, weight_graph, bias_graph = self._analysis_BatchNorm2d(x, module, pre_weight_graph, pre_bias_graph)
        elif isinstance(module, nn.BatchNorm1d):
            # MaxPool2d unit
            output, weight_graph, bias_graph = self._analysis_BatchNorm1d(x, module, pre_weight_graph, pre_bias_graph)
        elif isinstance(module, nn.Sequential):
            # Sequential unit
            output, weight_graph, bias_graph = self._analysis_Sequential(x, module, pre_weight_graph, pre_bias_graph)
        elif isinstance(module, AnalysisNet):
            # AnalysisNet unit
            output, weight_graph, bias_graph = self._analysis_child_class(x, module, pre_weight_graph, pre_bias_graph)
        else:
            raise "Sorry, this function is not Implemented or this class does not inherit 'AnalysisNet'. Please wait for updating. :<"

        # timer.timer.toc()
        if self._print_log and (not isinstance(module, nn.Sequential)) and (isinstance(module, nn.Conv2d)):
            print('Module : {}, completed!'.format(module.__repr__()))
            print('Time : {:.4f}s'.format(timer.timer.diff_time()))
        return output.detach(), weight_graph.detach(), bias_graph.detach()

    def _analysis_conv2d(self, x, module, pre_graph=None, pre_bias_graph=None):
        """
        Analyzing conv2d \n
        group = 1, dilation = 1\n
        graph_size : (n, c, h, w, (*input_size))
        """
        output = module(x)
        graph_size = (*output.size(), *self._input_size)

        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        padding = module.padding
        stride = module.stride
        kernel_weight = module.weight
        if module.bias is None:
            # (out_channels)
            bias = torch.zeros(out_channels, device=self._device)
        else:
            bias = module.bias

        weight_graph, bias_graph = self._conv2d_opt(x, in_channels, out_channels, kernel_size, padding, stride, kernel_weight, bias, graph_size, pre_graph, pre_bias_graph)
        return output, weight_graph, bias_graph

    def _analysis_avgPool2d(self, x, module, pre_graph=None, pre_bias_graph=None):
        """
        Analyzing AvgPool2d \n
        Don't support ceil_mode = True
        """
        assert not module.ceil_mode, "Please set ceil_mode = False"
        if not module.count_include_pad:
            assert module.padding != 0, "Please set padding = 0 !!!"
        output = module(x)
        graph_size = (*output.size(), *self._input_size)

        in_channels = graph_size[1]
        out_channels = graph_size[1]
        kernel_size = module.kernel_size
        padding = module.padding
        stride = module.stride
        kernel_weight = torch.eye(in_channels, device=self._device).view(in_channels, out_channels, 1, 1) / (kernel_size**2)
        bias = torch.zeros(out_channels, device=self._device)

        weight_graph, bias_graph = self._conv2d_opt(x, in_channels, out_channels, kernel_size, padding, stride, kernel_weight, bias, graph_size, pre_graph, pre_bias_graph)
        return output, weight_graph, bias_graph

    def _analysis_maxPool2d(self, x, module, pre_graph=None, pre_bias_graph=None):
        """
        Analyzing MaxPool2d \n
        Don't support ceil_mode = True and dilation != 1.
        """
        assert not module.ceil_mode, "Please set ceil_mode = False"
        assert module.dilation == 1, "Please set dilation = 1"
        is_ind = False
        if not module.return_indices:
            is_ind = True
            module.return_indices = True
        # indices :(n, out_channels, h, w)
        output, indices = module(x)
        graph_size = (*output.size(), *self._input_size)

        in_channels = graph_size[1]
        out_channels = graph_size[1]
        kernel_size = nn.modules.utils._pair(module.kernel_size)
        padding = nn.modules.utils._pair(module.padding)
        stride = nn.modules.utils._pair(module.stride)
        padding = (padding[1], padding[1], padding[0], padding[0])

        # create hook of x
        # hook_x :(n, c_out, c_in, h_in, w_in)
        pad_func = nn.ConstantPad2d(padding, 0)
        hook_x = torch.zeros_like(x, device=self._device)
        hook_x = pad_func(hook_x)
        hook_x = torch.zeros(hook_x.shape[0], out_channels, *hook_x.shape[1:], device=self._device)

        # create weight_graph(graph) and bias_graph
        graph = torch.zeros(graph_size, device=self._device)
        bias_graph = torch.zeros(graph_size, device=self._device)

        n_n = graph_size[0]
        h_n = graph_size[2]
        w_n = graph_size[3]

        for n in range(n_n):
            for h in range(h_n):
                for w in range(w_n):
                    # (c_out) with index.
                    d_indices = indices[n, :, h, w]
                    for c in range(d_indices.shape[0]):
                        # (h_x, w_x)
                        hook_x_size = hook_x[n, c, c, padding[0]:hook_x.shape[3]-padding[0], padding[1]:hook_x.shape[4]-padding[1]].size()
                        hook_x_ = hook_x[n, c, c, padding[0]:hook_x.shape[3]-padding[0], padding[1]:hook_x.shape[4]-padding[1]].reshape(-1)
                        hook_x_[d_indices[c]] = 1
                        hook_x[n, c, c, padding[0]:hook_x.shape[3]-padding[0], padding[1]:hook_x.shape[4]-padding[1]] = hook_x_.view(*hook_x_size)
                    if pre_graph is None:
                        graph[n, :, h, w] = hook_x[n, :, :, padding[0]:hook_x.shape[3]-padding[0], padding[1]:hook_x.shape[4]-padding[1]]
                    else:
                        hook_x_n = hook_x[n, :, :, padding[0]:hook_x.shape[3]-padding[0], padding[1]:hook_x.shape[4]-padding[1]].detach().clone()
                        pre_graph_n = pre_graph[n].detach().clone()
                        pre_bias_graph_n = pre_bias_graph[n].detach().clone()
                        # ========================================================
                        # matmul
                        hook_x_n = hook_x_n.view(out_channels, -1)
                        pre_graph_n = pre_graph_n.view(hook_x_n.shape[1], -1)
                        pre_bias_graph_n = pre_bias_graph_n.view(hook_x_n.shape[1], -1)
                        # (out_channels, *input_size)
                        hook_graph = torch.mm(hook_x_n, pre_graph_n).view(out_channels, *self._input_size)
                        hook_bias_graph = torch.mm(hook_x_n, pre_bias_graph_n).view(out_channels, *self._input_size)

                        graph[n, :, h, w] = hook_graph
                        bias_graph[n, :, h, w] = hook_bias_graph
                    hook_x.zero_()

        if is_ind:
            module.return_indices = False

        return output, graph, bias_graph

    def _analysis_linear(self, x, module, pre_graph, pre_bias_graph=None):
        """
        Analyzing linear \n
        graph_size: (n, out_feature, (*input_size)))
        """
        output = module(x)
        graph_size = (*output.size(), *self._input_size)

        in_features = module.in_features
        out_feature = module.out_features
        # (out_feature, in_features)
        weight = module.weight
        if module.bias is None:
            # (out_channels)
            bias = torch.zeros(out_feature, device=self._device)
        else:
            bias = module.bias

        # create hook of x
        # (n, out_features, in_features)
        hook_x = torch.zeros(x.shape[0], out_feature, *x.shape[1:], device=self._device)

        # create weight_graph and bias_graph
        # (n, out_features, (*input_size))
        graph = torch.zeros(graph_size, device=self._device)
        bias_graph = torch.zeros(graph_size, device=self._device)

        hook_x += weight
        if pre_graph is None:
            graph += hook_x
        else:
            pre_graph_n = pre_graph.view(pre_graph.size(0), in_features, -1)
            pre_bias_graph_n = pre_bias_graph.view(pre_bias_graph.size(0), in_features, -1)
            graph = torch.matmul(hook_x, pre_graph_n).view(-1, out_feature, *self._input_size)
            bias_graph = torch.matmul(hook_x, pre_bias_graph_n).view(-1, out_feature, *self._input_size)

        bias_graph += (bias.view(-1, *self.size_one) / self.size_prod)

        return output, graph, bias_graph

    def _analysis_ReLU(self, x, module, pre_graph=None, pre_bias_graph=None):
        """
        Analyzing ReLU \n
        Using inplace = False \n
        graph_size: ((*x.shape)), (*input_size))
        """
        if self._print_log and module.inplace:
            print('WARNING: \'inplace\' of ReLU is True!!!')
        output = module(x)
        graph_size = (*output.size(), *self._input_size)
        one_tensor = torch.ones((1), device=self._device)
        zero_tensor = torch.zeros((1), device=self._device)
        # ((*x.shape))
        x_relu_hot = torch.where(x > 0, one_tensor, zero_tensor)
        # ((*x.shape)), (*input_size))
        graph = torch.zeros(graph_size, device=self._device)
        bias_graph = torch.zeros(graph_size, device=self._device)
        graph += x_relu_hot.view(*x_relu_hot.size(), *self.size_one)
        bias_graph += x_relu_hot.view(*x_relu_hot.size(), *self.size_one)
        if pre_graph is not None:
            graph *= pre_graph
            bias_graph *= pre_bias_graph

        return output, graph, bias_graph

    def _analysis_BatchNorm2d(self, x, module, pre_graph=None, pre_bias_graph=None):
        """
        Analyzing BatchNorm2d \n
        track_running_stats = True->Using saving var and mean.
        graph_size: ((*x.shape)), (*input_size))
        """
        assert module.track_running_stats, "Please set track_running_stats = True"
        assert pre_graph is not None, "BatchNorm2d is not a first layer!!"
        output = module(x)

        graph_size = (*output.size(), *self._input_size)

        # num_features = in_channels = out_channels
        num_features = module.num_features
        # (num_features)
        running_var = module.running_var
        running_mean = module.running_mean
        eps = module.eps
        if module.affine:
            weight = module.weight
            bias = module.bias
        else:
            weight = torch.ones(num_features, device=self._device)
            bias = torch.zeros(num_features, device=self._device)

        # create weight_graph and bias_graph
        # (n, out_features, (*input_size))
        graph = torch.zeros(graph_size, device=self._device)
        bias_graph = torch.zeros(graph_size, device=self._device)

        real_weight = (weight / torch.sqrt(running_var + eps)).view(-1, 1, 1, *self.size_one)
        real_bias = bias - ((weight * running_mean) / torch.sqrt(running_var + eps))

        # TODO: Modify this way to get the bias
        real_bias = real_bias.view(-1, 1, 1, *self.size_one) / self.size_prod

        graph = pre_graph * real_weight
        bias_graph = pre_bias_graph * real_weight + real_bias

        return output, graph, bias_graph

    def _analysis_BatchNorm1d(self, x, module, pre_graph=None, pre_bias_graph=None):
        """
        Analyzing BatchNorm1d \n
        track_running_stats = True->Using saving var and mean.
        graph_size: ((*x.shape)), (*input_size))
        """
        assert module.track_running_stats, "Please set track_running_stats = True"
        assert pre_graph is not None, "BatchNorm1d is not a first layer!!"
        output = module(x)

        graph_size = (*output.size(), *self._input_size)

        # num_features = in_channels = out_channels
        num_features = module.num_features
        # (num_features)
        running_var = module.running_var
        running_mean = module.running_mean
        eps = module.eps
        if module.affine:
            weight = module.weight
            bias = module.bias
        else:
            weight = torch.ones(num_features, device=self._device)
            bias = torch.zeros(num_features, device=self._device)

        # create weight_graph and bias_graph
        # (n, out_features, (*input_size))
        graph = torch.zeros(graph_size, device=self._device)
        bias_graph = torch.zeros(graph_size, device=self._device)

        real_weight = (weight / torch.sqrt(running_var + eps)).view(-1, *self.size_one)
        real_bias = bias - ((weight * running_mean) / torch.sqrt(running_var + eps))

        # TODO: Modify this way to get the bias
        real_bias = real_bias.view(-1, *self.size_one) / self.size_prod

        graph = pre_graph * real_weight
        bias_graph = pre_bias_graph * real_weight + real_bias

        return output, graph, bias_graph

    def _analysis_Sequential(self, x, module, pre_weight_graph=None, pre_bias_graph=None):
        weight_graph, bias_graph, output = pre_weight_graph, pre_bias_graph, x
        for child_modules in module._modules.values():
            if isinstance(child_modules, AnalysisNet):
                output, weight_graph, bias_graph = child_modules.forward_graph(output, weight_graph, bias_graph)
            else:
                output, weight_graph, bias_graph = self.analysis_module(output, child_modules, weight_graph, bias_graph)
        return output, weight_graph, bias_graph

    def _analysis_child_class(self, x, module, pre_weight_graph=None, pre_bias_graph=None):
        output, weight_graph, bias_graph = module.forward_graph(x, pre_weight_graph, pre_bias_graph)
        return output, weight_graph, bias_graph

    def _conv2d_opt(self, x, in_channels, out_channels, kernel_size, padding, stride, kernel_weight, bias, graph_size, pre_graph, pre_bias_graph):
        # for conv2d and Pooling2d(avg)
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        padding = nn.modules.utils._pair(padding)
        padding = (padding[1], padding[1], padding[0], padding[0])

        h_n = graph_size[2]
        w_n = graph_size[3]

        # create weight_graph(graph) and bias_graph
        # (n, c_out, h_out, w_out, *(input_size))
        graph = torch.zeros(graph_size, device=self._device)
        bias_graph = torch.zeros(graph_size, device=self._device)
        # Conv2d opt
        # ===============================================================
        # slower!! for first layer
        # create hook of x
        # hook_x :(n, c_out, c_in, h_in, w_in)
        if pre_graph is None:
            pad_func = nn.ConstantPad2d(padding, 0)
            hook_x = torch.zeros_like(x, device=self._device)
            hook_x = pad_func(hook_x)
            hook_x = torch.zeros(hook_x.shape[0], out_channels, *hook_x.shape[1:], device=self._device)
            # origin
            for h in range(h_n):
                for w in range(w_n):
                    hook_x[:, :, :, h*stride[0]:h*stride[0]+kernel_size[0], w*stride[1]: w*stride[1]+kernel_size[1]] += kernel_weight
                    graph[:, :, h, w] = hook_x[:, :, :, padding[0]:hook_x.shape[3]-padding[0], padding[1]:hook_x.shape[4]-padding[1]]
                    hook_x.zero_()
        # ===============================================================
        # speed 1
        # rm w loop
        else:
            # hook_kernel_weight : (w_out, c_out, c_in, k , w_in+2padding)
            hook_kernel_weight = torch.zeros(w_n, out_channels, in_channels, kernel_size[0], x.size(3)+padding[1]*2, device=self._device)
            for w in range(w_n):
                hook_kernel_weight[w, :, :, :, w*stride[1]: w*stride[1]+kernel_size[1]] += kernel_weight
            # hook_kernel_weight : (w_out, c_out, c_in, k , w_in)
            hook_kernel_weight = hook_kernel_weight[:, :, :, :, padding[1]:hook_kernel_weight.size(4)-padding[1]]

            for h in range(h_n):
                # get pre_graph
                # pre_graph : (n, c_in, h_in, w_in, *(input_size))
                pos = h * stride[0] - padding[0]
                pos_end = pos + kernel_size[0]

                pos_end = pos_end if pos_end < x.size(2) else x.size(2)
                pos_h = pos if pos > -1 else 0

                pos_kernal = pos_h - pos
                pos_end_kernal = pos_end - pos
                # hook_kernel_weight_hook : (w_out, c_out, (c_in, h_pos-, w_in)) -> ((c_in, h_pos-, w_in), w_out, c_out)
                hook_kernel_weight_hook = hook_kernel_weight[:, :, :, pos_kernal:pos_end_kernal, :].reshape(w_n * out_channels, -1).permute(1, 0)
                # pre_graph_hook : (n, (c_in, h_pos-, w_in), *(input_size)) -> (n, *(input_size), (c_in, h_pos-, w_in))
                pre_graph_hook = pre_graph[:, :, pos_h:pos_end, :, :, :, :].reshape(x.size(0), -1, self.size_prod).permute(0, 2, 1)
                pre_bias_graph_hook = pre_bias_graph[:, :, pos_h:pos_end, :, :, :, :].reshape(x.size(0), -1, self.size_prod).permute(0, 2, 1)

                # graph_hook : (n, c_out, w_out, *(input_size))
                graph[:, :, h] = torch.matmul(pre_graph_hook, hook_kernel_weight_hook).reshape(x.size(0), *self._input_size, w_n, out_channels).permute(0, 5, 4, 1, 2, 3)
                bias_graph[:, :, h] = torch.matmul(pre_bias_graph_hook, hook_kernel_weight_hook).reshape(x.size(0), *self._input_size, w_n, out_channels).permute(0, 5, 4, 1, 2, 3)
        # ===============================================================
        # speed 2
        # rm w and h loop
        # ===============================================================
        # bias 采用平均值
        bias_graph += (bias.view(-1, 1, 1, *self.size_one) / self.size_prod)
        return graph, bias_graph
