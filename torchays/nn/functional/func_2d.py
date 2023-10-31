from typing import Tuple

import torch
import torch.nn.functional as F
from torch.types import _dtype


def __get_h_idx(h, stride_h, padding_h, kernel_h, input_h):
    idx = h * stride_h - padding_h
    idx_start = idx if idx > -1 else 0
    idx_end = idx + kernel_h
    idx_end = idx_end if idx_end < input_h else input_h
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
    stride: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (0, 0),
    device: torch.device = torch.device("cpu"),
    dtype: _dtype = torch.FloatTensor(),
) -> torch.Tensor:
    n, _, input_h, input_w = input_size
    _, _, output_h, output_w = output_size
    padding_h, padding_w = padding
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_size
    # output_weight_graph: (n, c_out, h_out, w_out, *origin_size)
    output_weight_graph = torch.zeros((*output_size, *origin_size), device=device, dtype=dtype)
    # hook_x: (n, c_out, c_in, h_in, w_in)
    hook_x = torch.zeros(n, out_channels, in_channels, input_h + padding_h * 2, input_w + padding_w * 2, device=device, dtype=dtype)
    # origin
    for h in range(output_h):
        for w in range(output_w):
            hook_x[:, :, :, h * stride_h : h * stride_h + kernel_h, w * stride_w : w * stride_w + kernel_w] += kernel_weight
            output_weight_graph[:, :, h, w] = hook_x[:, :, :, padding_h : hook_x.size(3) - padding_h, padding_w : hook_x.size(4) - padding_w]
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
    stride: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (0, 0),
    device: torch.device = torch.device("cpu"),
    dtype: _dtype = torch.FloatTensor(),
) -> torch.Tensor:
    n, _, input_h, input_w = input_size
    _, _, output_h, output_w = output_size
    padding_h, padding_w = padding
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_size
    # output_weight_graph: (n, c_out, h_out, w_out, (*origin_size))
    output_weight_graph = torch.zeros((*output_size, *origin_size), device=device, dtype=dtype)
    # hook_kernel_weight: (w_out, c_out, c_in, k , w_in+2*padding)
    hook_kernel_weight = torch.zeros(output_w, out_channels, in_channels, kernel_h, input_w + padding_w * 2, device=device, dtype=dtype)
    for w in range(output_w):
        hook_kernel_weight[w, :, :, :, w * stride_w : w * stride_w + kernel_w] += kernel_weight
    # hook_kernel_weight : (w_out, c_out, c_in, k, w_in)
    hook_kernel_weight = hook_kernel_weight[:, :, :, :, padding_w : hook_kernel_weight.size(-1) - padding_w]
    for h in range(output_h):
        idx_start, idx_end, idx_start_kernal, idx_end_kernal = __get_h_idx(h, stride_h, padding_h, kernel_h, input_h)
        # hook_kernel_weight_s: (w_out, c_out, (c_in, h_pos-, w_in)) -> ((c_in, h_pos-, w_in), c_out, w_out)
        hook_kernel_weight_s = hook_kernel_weight[:, :, :, idx_start_kernal:idx_end_kernal].permute(2, 3, 4, 1, 0).reshape(-1, output_w * out_channels)
        # pre_graph: (n, c_in, h_in, w_in, *origin_size)
        # graph_hook: (n, (c_in, h_pos-, w_in), (origin_size)) -> (n, (origin_size), (c_in, h_pos-, w_in))
        hook_graph = weight_graph[:, :, idx_start:idx_end].reshape(n, -1, origin_size.numel()).permute(0, 2, 1)
        # output_weight_graph:  (n, (origin_size), c_out, w_out) -> (n, c_out, w_out, *origin_size)
        output_weight_graph[:, :, h] = torch.matmul(hook_graph, hook_kernel_weight_s).permute(0, 2, 1).reshape(n, out_channels, output_w, *origin_size)
        del hook_graph, hook_kernel_weight_s
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
    stride: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (0, 0),
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
    stride: Tuple[int, int] = (0, 0),
    device: torch.device = torch.device("cpu"),
    dtype: _dtype = torch.FloatTensor(),
):
    n, channels, input_h, input_w = input_size
    _, _, output_h, output_w = output_size
    padding_h, padding_w = padding
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_size
    # output_weight_graph: (n, c, h_out, w_out, *origin_size)
    output_weight_graph = torch.zeros((*output_size, *origin_size), device=device, dtype=dtype)
    # hook_kernel_weight: (w_out, k, w_in+2*padding)
    hook_kernel_weight = torch.zeros(output_w, kernel_w, input_w + padding_w * 2, device=device, dtype=dtype)
    for w in range(output_w):
        hook_kernel_weight[w, :, w * stride_w : w * stride_w + kernel_w] += kernel_weight
    # hook_kernel_weight : (w_out, k, w_in)
    hook_kernel_weight = hook_kernel_weight[:, :, padding_w : hook_kernel_weight.size(-1) - padding_w]
    for h in range(output_h):
        idx_start, idx_end, idx_start_kernal, idx_end_kernal = __get_h_idx(h, stride_h, padding_h, kernel_h, input_h)
        # hook_kernel_weight: (w_out, h_pos-, w_in) -> ((h_pos-, w_in), w_out)
        hook_kernel_weight_s = hook_kernel_weight[:, idx_start_kernal:idx_end_kernal].reshape(output_w, -1).permute(1, 0)
        # pre_graph: (n, c, h_in, w_in, *origin_size)
        # graph_hook: (n, c, (h_pos-, w_in), (origin_size)) -> (n, c, (origin_size), (h_pos-, w_in))
        hook_graph = weight_graph[:, :, idx_start:idx_end].reshape(n, channels, -1, origin_size.numel()).permute(0, 1, 3, 2)
        # output_weight_graph:(n, c, *(origin_size), w_out) -> (n, c, w_out, *origin_size)
        output_weight_graph[:, :, h] = torch.matmul(hook_graph, hook_kernel_weight_s).permute(0, 1, 3, 2).reshape(n, channels, output_w, *origin_size)
        del hook_kernel_weight_s, hook_graph
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
    n, channels, output_h, output_w = output_size
    nc = n * channels
    # indices: (n, c, h_o, w_o) -> (n*c, h_o*w_o)
    indices = indices.reshape(nc, -1)
    # weight_graph: (n, c, h_in, w_in, *origin_size) -> (n*c, h_in*w_in, (origin_size))
    weight_graph = weight_graph.reshape(nc, -1, origin_size.numel())
    # bias_graph: (n, c, h_in, w_in) -> (n*c, h_in*w_in)
    bias_graph = bias_graph.reshape(nc, -1)
    # output: (n, c, h_o, w_o, *origin_size) -> (n*c, h_o*w_o, (origin_size))
    wg = torch.zeros((n, channels, output_h, output_w, *origin_size), device=device, dtype=dtype).reshape(nc, -1, origin_size.numel())
    bg = torch.zeros((n, channels, output_h, output_w), device=device, dtype=dtype).reshape(nc, -1)
    for nc_i in range(nc):
        # n_indices: (h_o*w_o)
        nc_indices = indices[nc_i]
        wg[nc_i] = weight_graph[nc_i, nc_indices]
        bg[nc_i] = bias_graph[nc_i, nc_indices]
    wg = wg.reshape(n, channels, output_h, output_w, *origin_size)
    bg = bg.reshape(n, channels, output_h, output_w)
    return wg, bg
