from typing import Tuple

import torch
import torch.nn.functional as F
from torch.types import _dtype


def __get_h_idx(h, h_stride, h_pad, h_k, h_in):
    idx = h * h_stride - h_pad
    idx_start = idx if idx > -1 else 0
    idx_end = idx + h_k
    idx_end = idx_end if idx_end < h_in else h_in
    idx_start_kernal, idx_end_kernal = idx_start - idx, idx_end - idx
    return idx_start, idx_end, idx_start_kernal, idx_end_kernal


def _2d_opr_weight_none(
    kernel_weight: torch.Tensor,
    kernel_size: torch.Size | Tuple,
    origin_size: torch.Size | Tuple,
    input_size: torch.Size | Tuple,
    output_size: torch.Size | Tuple,
    out_channels: int,
    in_channels: int,
    padding: Tuple[int, int] = (0, 0),
    stride: Tuple[int, int] = (1, 1),
    dilation: Tuple[int, int] = (1, 1),
    device: torch.device = torch.device("cpu"),
    dtype: _dtype = torch.FloatTensor(),
) -> torch.Tensor:
    n, _, h_in, w_in = input_size
    _, _, h_out, w_out = output_size
    h_pad, w_pad = padding
    h_stride, w_stride = stride
    h_k, w_k = kernel_size
    # output_weight_graph: (n, c_out, h_out, w_out, *origin_size)
    output_weight_graph = torch.zeros((*output_size, *origin_size), device=device, dtype=dtype)
    # hook_x: (n, c_out, c_in, h_in, w_in)
    hook_x = torch.zeros(n, out_channels, in_channels, h_in + h_pad * 2, w_in + w_pad * 2, device=device, dtype=dtype)
    # origin
    for h in range(h_out):
        for w in range(w_out):
            hook_x[:, :, :, h * h_stride : h * h_stride + h_k, w * w_stride : w * w_stride + w_k] += kernel_weight
            output_weight_graph[:, :, h, w] = hook_x[:, :, :, h_pad : hook_x.size(3) - h_pad, w_pad : hook_x.size(4) - w_pad]
            hook_x.zero_()
    return output_weight_graph


def _2d_opr_weight(
    weight_graph: torch.Tensor,
    kernel_weight: torch.Tensor,
    kernel_size: torch.Size | Tuple,
    origin_size: torch.Size | Tuple,
    input_size: torch.Size | Tuple,
    output_size: torch.Size | Tuple,
    out_channels: int,
    in_channels: int,
    padding: Tuple[int, int] = (0, 0),
    stride: Tuple[int, int] = (1, 1),
    dilation: Tuple[int, int] = (1, 1),
    device: torch.device = torch.device("cpu"),
    dtype: _dtype = torch.FloatTensor(),
) -> torch.Tensor:
    n, _, h_in, w_in = input_size
    _, _, h_out, w_out = output_size
    h_pad, w_pad = padding
    h_stride, w_stride = stride
    h_k, w_k = kernel_size
    # output_weight_graph: (n, c_out, h_out, w_out, (*origin_size))
    output_weight_graph = torch.zeros((*output_size, *origin_size), device=device, dtype=dtype)
    # hook_kernel_weight: (w_out, c_out, c_in, k , w_in+2*padding)
    hook_kernel_weight = torch.zeros(w_out, out_channels, in_channels, h_k, w_in + w_pad * 2, device=device, dtype=dtype)
    for w in range(w_out):
        hook_kernel_weight[w, :, :, :, w * w_stride : w * w_stride + w_k] += kernel_weight
    # hook_kernel_weight : (w_out, c_out, c_in, k, w_in)
    hook_kernel_weight = hook_kernel_weight[:, :, :, :, w_pad : hook_kernel_weight.size(-1) - w_pad]
    for h in range(h_out):
        idx_start, idx_end, idx_start_kernal, idx_end_kernal = __get_h_idx(h, h_stride, h_pad, h_k, h_in)
        # hook_kernel_weight_s: (w_out, c_out, (c_in, h_pos-, w_in)) -> ((c_in, h_pos-, w_in), c_out, w_out)
        hook_kernel_weight_s = hook_kernel_weight[:, :, :, idx_start_kernal:idx_end_kernal].permute(2, 3, 4, 1, 0).reshape(-1, w_out * out_channels)
        # pre_graph: (n, c_in, h_in, w_in, *origin_size)
        # graph_hook: (n, (c_in, h_pos-, w_in), (origin_size)) -> (n, (origin_size), (c_in, h_pos-, w_in))
        hook_graph = weight_graph[:, :, idx_start:idx_end].reshape(n, -1, origin_size.numel()).permute(0, 2, 1)
        # output_weight_graph:  (n, (origin_size), c_out, w_out) -> (n, c_out, w_out, *origin_size)
        output_weight_graph[:, :, h] = torch.matmul(hook_graph, hook_kernel_weight_s).permute(0, 2, 1).reshape(n, out_channels, w_out, *origin_size)
        del hook_graph, hook_kernel_weight_s
    del hook_kernel_weight
    return output_weight_graph


def conv2d(
    weight_graph: torch.Tensor,
    kernel_weight: torch.Tensor,
    kernel_size: torch.Size | Tuple,
    origin_size: torch.Size | Tuple,
    input_size: torch.Size | Tuple,
    output_size: torch.Size | Tuple,
    out_channels: int,
    in_channels: int,
    padding: Tuple[int, int] = (0, 0),
    stride: Tuple[int, int] = (1, 1),
    dilation: Tuple[int, int] = (1, 1),
    device: torch.device = torch.device("cpu"),
    dtype: _dtype = torch.FloatTensor(),
):
    if weight_graph is None:
        return _2d_opr_weight_none(kernel_weight, kernel_size, origin_size, input_size, output_size, out_channels, in_channels, padding, stride, dilation, device, dtype)
    return _2d_opr_weight(weight_graph, kernel_weight, kernel_size, origin_size, input_size, output_size, out_channels, in_channels, padding, stride, dilation, device, dtype)


def avg_pool_2d(
    weight_graph: torch.Tensor,
    kernel_weight: torch.Tensor,
    kernel_size: torch.Size | Tuple,
    origin_size: torch.Size | Tuple,
    input_size: torch.Size | Tuple,
    output_size: torch.Size | Tuple,
    padding: Tuple[int, int] = (0, 0),
    stride: Tuple[int, int] = (1, 1),
    device: torch.device = torch.device("cpu"),
    dtype: _dtype = torch.FloatTensor(),
):
    n, channels, h_in, w_in = input_size
    _, _, h_out, w_out = output_size
    h_pad, w_pad = padding
    h_stride, w_stride = stride
    h_k, w_k = kernel_size
    # output_weight_graph: (n, c, h_out, w_out, *origin_size)
    output_weight_graph = torch.zeros((*output_size, *origin_size), device=device, dtype=dtype)
    # hook_kernel_weight: (w_out, k, w_in+2*padding)
    hook_kernel_weight = torch.zeros(w_out, w_k, w_in + w_pad * 2, device=device, dtype=dtype)
    for w in range(w_out):
        hook_kernel_weight[w, :, w * w_stride : w * w_stride + w_k] += kernel_weight
    # hook_kernel_weight : (w_out, k, w_in)
    hook_kernel_weight = hook_kernel_weight[:, :, w_pad : hook_kernel_weight.size(-1) - w_pad]
    for h in range(h_out):
        idx_start, idx_end, idx_start_kernal, idx_end_kernal = __get_h_idx(h, h_stride, h_pad, h_k, h_in)
        # hook_kernel_weight: (w_out, h_pos-, w_in) -> ((h_pos-, w_in), w_out)
        hook_kernel_weight_s = hook_kernel_weight[:, idx_start_kernal:idx_end_kernal].reshape(w_out, -1).permute(1, 0)
        # pre_graph: (n, c, h_in, w_in, *origin_size)
        # graph_hook: (n, c, (h_pos-, w_in), (origin_size)) -> (n, c, (origin_size), (h_pos-, w_in))
        hook_graph = weight_graph[:, :, idx_start:idx_end].reshape(n, channels, -1, origin_size.numel()).permute(0, 1, 3, 2)
        # output_weight_graph:(n, c, *(origin_size), w_out) -> (n, c, w_out, *origin_size)
        output_weight_graph[:, :, h] = torch.matmul(hook_graph, hook_kernel_weight_s).permute(0, 1, 3, 2).reshape(n, channels, w_out, *origin_size)
        del hook_kernel_weight_s, hook_graph
    del hook_kernel_weight
    return output_weight_graph


def max_pool_2d(
    indices: torch.Tensor,
    weight_graph: torch.Tensor,
    bias_graph: torch.Tensor,
    origin_size: torch.Size | Tuple,
    output_size: torch.Size | Tuple,
    device: torch.device = torch.device("cpu"),
    dtype: _dtype = torch.FloatTensor(),
):
    n, channels, h_out, w_out = output_size
    nc = n * channels
    # indices: (n, c, h_o, w_o) -> (n*c, h_o*w_o)
    indices = indices.reshape(nc, -1)
    # weight_graph: (n, c, h_in, w_in, *origin_size) -> (n*c, h_in*w_in, (origin_size))
    weight_graph = weight_graph.reshape(nc, -1, origin_size.numel())
    # bias_graph: (n, c, h_in, w_in) -> (n*c, h_in*w_in)
    bias_graph = bias_graph.reshape(nc, -1)
    # output: (n, c, h_o, w_o, *origin_size) -> (n*c, h_o*w_o, (origin_size))
    wg = torch.zeros((n, channels, h_out, w_out, *origin_size), device=device, dtype=dtype).reshape(nc, -1, origin_size.numel())
    bg = torch.zeros((n, channels, h_out, w_out), device=device, dtype=dtype).reshape(nc, -1)
    for nc_i in range(nc):
        # n_indices: (h_o*w_o)
        nc_indices = indices[nc_i]
        wg[nc_i] = weight_graph[nc_i, nc_indices]
        bg[nc_i] = bias_graph[nc_i, nc_indices]
    wg = wg.reshape(n, channels, h_out, w_out, *origin_size)
    bg = bg.reshape(n, channels, h_out, w_out)
    return wg, bg
