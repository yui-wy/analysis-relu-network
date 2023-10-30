from typing import Tuple

import torch
import torch.nn.functional as F


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
) -> torch.Tensor:
    n, _, input_h, input_w = input_size
    _, _, output_h, output_w = output_size
    padding_h, padding_w = padding
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_size
    # output_weight_graph: (n, c_out, h_out, w_out, (*origin_size))
    output_weight_graph = torch.zeros((*output_size, *origin_size), device=device)
    # hook_x: (n, c_out, c_in, h_in, w_in)
    hook_x = torch.zeros(n, out_channels, in_channels, input_h + padding_h * 2, input_w + padding_w * 2, device=device)
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
) -> torch.Tensor:
    n, _, input_h, input_w = input_size
    _, _, output_h, output_w = output_size
    padding_h, padding_w = padding
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_size
    # output_weight_graph: (n, c_out, h_out, w_out, (*origin_size))
    output_weight_graph = torch.zeros((*output_size, *origin_size), device=device)
    # hook_kernel_weight: (w_out, c_out, c_in, k , w_in+2*padding)
    hook_kernel_weight = torch.zeros(output_w, out_channels, in_channels, kernel_h, input_w + padding_w * 2, device=device)
    for w in range(output_w):
        hook_kernel_weight[w, :, :, :, w * stride_w : w * stride_w + kernel_w] += kernel_weight
    # hook_kernel_weight : (w_out, c_out, c_in, k, w_in)
    hook_kernel_weight = hook_kernel_weight[:, :, :, :, padding_w : hook_kernel_weight.size(-1) - padding_w]
    for h in range(output_h):
        idx_start, idx_end, idx_start_kernal, idx_end_kernal = __get_h_idx(h, stride_h, padding_h, kernel_h, input_h)
        # hook_kernel_weight_s: (w_out, c_out, (c_in, h_pos-, w_in)) -> ((c_in, h_pos-, w_in), w_out, c_out)
        hook_kernel_weight_s = hook_kernel_weight[:, :, :, idx_start_kernal:idx_end_kernal].reshape(output_w * out_channels, -1).permute(1, 0)
        # pre_graph: (n, c_in, h_in, w_in, (*origin_size))
        # graph_hook: (n, (c_in, h_pos-, w_in), (origin_size)) -> (n, (origin_size), (c_in, h_pos-, w_in))
        hook_graph = weight_graph[:, :, idx_start:idx_end].reshape(n, -1, origin_size.numel()).permute(0, 2, 1)
        # output_weight_graph: (n, c_out, w_out, *(origin_size))
        output_weight_graph[:, :, h] = torch.matmul(hook_graph, hook_kernel_weight_s).reshape(n, *origin_size, output_w, out_channels).permute(0, 5, 4, 1, 2, 3)
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
):
    if weight_graph is None:
        return _2d_opr_weight_none(kernel_weight, kernel_size, origin_size, input_size, output_size, out_channels, in_channels, padding, stride, dilation, device)
    return _2d_opr_weight(weight_graph, kernel_weight, kernel_size, origin_size, input_size, output_size, out_channels, in_channels, padding, stride, dilation, device)


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
):
    n, channels, input_h, input_w = input_size
    _, _, output_h, output_w = output_size
    padding_h, padding_w = padding
    stride_h, stride_w = stride
    kernel_h, kernel_w = kernel_size
    # output_weight_graph: (n, c, h_out, w_out, (*origin_size))
    output_weight_graph = torch.zeros((*output_size, *origin_size), device=device)
    # hook_kernel_weight: (w_out, k, w_in+2*padding)
    hook_kernel_weight = torch.zeros(output_w, kernel_w, input_w + padding_w * 2, device=device)
    for w in range(output_w):
        hook_kernel_weight[w, :, w * stride_w : w * stride_w + kernel_w] += kernel_weight
    # hook_kernel_weight : (w_out, k, w_in)
    hook_kernel_weight = hook_kernel_weight[:, :, padding_w : hook_kernel_weight.size(-1) - padding_w]
    for h in range(output_h):
        idx_start, idx_end, idx_start_kernal, idx_end_kernal = __get_h_idx(h, stride_h, padding_h, kernel_h, input_h)
        # hook_kernel_weight: (w_out, h_pos-, w_in)) -> ((h_pos-, w_in), w_out)
        hook_kernel_weight_s = hook_kernel_weight[:, :, idx_start_kernal:idx_end_kernal].reshape(output_w, -1).permute(1, 0)
        # pre_graph: (n, c, h_in, w_in, (*origin_size))
        # graph_hook: (n, c, (h_pos-, w_in), (origin_size)) -> (n, c, (origin_size), (h_pos-, w_in))
        hook_graph = weight_graph[:, :, idx_start:idx_end].reshape(n, channels, -1, origin_size.numel()).permute(0, 1, 3, 2)
        # output_weight_graph:(n, c, *(origin_size), w_out) -> (n, c, w_out, *(origin_size))
        output_weight_graph[:, :, h] = torch.matmul(hook_graph, hook_kernel_weight_s).reshape(n, channels, *origin_size, output_w).permute(0, 1, 5, 2, 3, 4)
    return output_weight_graph


def max_pool_2d(
    indices: torch.Tensor,
    weight_graph: torch.Tensor,
    bias_graph: torch.Tensor,
    kernel_size: torch.Size | Tuple,
    origin_size: torch.Size | Tuple,
    output_size: torch.Size | Tuple,
    padding: Tuple[int, int] = (0, 0),
    stride: Tuple[int, int] = (0, 0),
    dilation: int = 0,
    device: torch.device = torch.device("cpu"),
):
    # 根据idx创建掩码 (n, c, h*w)
    output_one = torch.ones(*output_size, device=device)
    mask = F.max_unpool2d(output_one, indices, kernel_size=kernel_size, stride=stride, padding=padding).eq(1).reshape(-1)
    out_weight_graph = weight_graph.reshape(-1, *origin_size)
    out_bias_graph = bias_graph.reshape(-1)
    out_weight_graph = out_weight_graph[mask,].reshape(*output_size, *origin_size)
    out_bias_graph = out_bias_graph[mask].reshape(*output_size)
    return out_weight_graph, out_bias_graph
