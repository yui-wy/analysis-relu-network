from typing import Callable, Tuple
import torch


def _2d_opr_weight_none(
    kernel_weight: torch.Tensor,
    kernel_size: torch.Size | Tuple | int,
    origin_size: torch.Size | Tuple,
    input_size: torch.Size | Tuple,
    output_size: torch.Size | Tuple,
    out_channels: int,
    in_channels: int,
    padding: Tuple[int, int] = 0,
    stride: Tuple[int, int] = 0,
    dilation: Tuple[int, int] = 0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    output_h, output_w = output_size[2], output_size[3]
    # output_weight_graph: (n, c_out, h_out, w_out, (*origin_size))
    output_weight_graph = torch.zeros((*output_size, *origin_size), device=device)
    # hook_x: (n, c_out, c_in, h_in, w_in)
    hook_x = torch.zeros(input_size[0], out_channels, in_channels, input_size[2] + padding[0] * 2, input_size[3] + padding[1] * 2, device=device)
    # origin
    for h in range(output_h):
        for w in range(output_w):
            hook_x[:, :, :, h * stride[0] : h * stride[0] + kernel_size[0], w * stride[1] : w * stride[1] + kernel_size[1]] += kernel_weight
            output_weight_graph[:, :, h, w] = hook_x[:, :, :, padding[0] : hook_x.size(3) - padding[0], padding[1] : hook_x.size(4) - padding[1]]
            hook_x.zero_()
    return output_weight_graph


def _2d_opr_weight(
    weight_graph: torch.Tensor,
    kernel_weight: torch.Tensor,
    kernel_size: torch.Size | Tuple | int,
    origin_size: torch.Size | Tuple,
    input_size: torch.Size | Tuple,
    output_size: torch.Size | Tuple,
    out_channels: int,
    in_channels: int,
    padding: Tuple[int, int] = 0,
    stride: Tuple[int, int] = 0,
    dilation: Tuple[int, int] = 0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    output_h, output_w = output_size[2], output_size[3]
    # output_weight_graph: (n, c_out, h_out, w_out, (*origin_size))
    output_weight_graph = torch.zeros((*output_size, *origin_size), device=device)
    # hook_kernel_weight: (w_out, c_out, c_in, k , w_in+2padding)
    hook_kernel_weight = torch.zeros(output_w, out_channels, in_channels, kernel_size[0], input_size[3] + padding[1] * 2, device=device)
    for w in range(output_w):
        hook_kernel_weight[w, :, :, :, w * stride[1] : w * stride[1] + kernel_size[1]] += kernel_weight
    # hook_kernel_weight : (w_out, c_out, c_in, k , w_in)
    hook_kernel_weight = hook_kernel_weight[:, :, :, :, padding[1] : hook_kernel_weight.size(4) - padding[1]]
    for h in range(output_h):
        # pre_graph: (n, c_in, h_in, w_in, (*origin_size))
        pos = h * stride[0] - padding[0]
        pos_end = pos + kernel_size[0]

        pos_end = pos_end if pos_end < input_size[2] else input_size[2]
        pos_h = pos if pos > -1 else 0

        pos_kernal = pos_h - pos
        pos_end_kernal = pos_end - pos
        # hook_kernel_weight_s: (w_out, c_out, (c_in, h_pos-, w_in)) -> ((c_in, h_pos-, w_in), w_out, c_out)
        hook_kernel_weight_s = hook_kernel_weight[:, :, :, pos_kernal:pos_end_kernal].reshape(output_w * out_channels, -1).permute(1, 0)
        # graph_hook: (n, (c_in, h_pos-, w_in), (origin_size)) -> (n, (origin_size), (c_in, h_pos-, w_in))
        hook_graph = weight_graph[:, :, pos_h:pos_end].reshape(input_size[0], -1, origin_size.numel()).permute(0, 2, 1)
        # output_weight_graph: (n, c_out, w_out, *(origin_size))
        output_weight_graph[:, :, h] = torch.matmul(hook_graph, hook_kernel_weight_s).reshape(input_size[0], *origin_size, output_w, out_channels).permute(0, 5, 4, 1, 2, 3)
    return output_weight_graph


conv2d = _2d_opr_weight


def conv2d(
    weight_graph: torch.Tensor,
    kernel_weight: torch.Tensor,
    kernel_size: torch.Size | Tuple | int,
    origin_size: torch.Size | Tuple,
    input_size: torch.Size | Tuple,
    output_size: torch.Size | Tuple,
    out_channels: int,
    in_channels: int,
    padding: Tuple[int, int] = 0,
    stride: Tuple[int, int] = 0,
    dilation: Tuple[int, int] = 0,
    device: torch.device = torch.device("cpu"),
):
    if weight_graph is None:
        return _2d_opr_weight_none(kernel_weight, kernel_size, origin_size, input_size, output_size, out_channels, in_channels, padding, stride, dilation, device)
    return _2d_opr_weight(weight_graph, kernel_weight, kernel_size, origin_size, input_size, output_size, out_channels, in_channels, padding, stride, dilation, device)


def avg_pool_2d(
    weight_graph: torch.Tensor,
    kernel_weight: torch.Tensor,
    kernel_size: torch.Size | Tuple | int,
    origin_size: torch.Size | Tuple,
    input_size: torch.Size | Tuple,
    output_size: torch.Size | Tuple,
    channels: int,
    padding: Tuple[int, int] | int = 0,
    stride: Tuple[int, int] | int = 0,
    device: torch.device = torch.device("cpu"),
):
    return _2d_opr_weight(weight_graph, kernel_weight, kernel_size, origin_size, input_size, output_size, channels, channels, padding, stride, 1, device)


def _common_2d_opr_weight(
    weight_graph: torch.Tensor,
    kernel_weight: Callable[[int, int], torch.Tensor],
    kernel_size: torch.Size | Tuple | int,
    origin_size: torch.Size | Tuple,
    input_size: torch.Size | Tuple,
    output_size: torch.Size | Tuple,
    out_channels: int,
    in_channels: int,
    padding: Tuple[int, int] = 0,
    stride: Tuple[int, int] = 0,
    dilation: Tuple[int, int] = 0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    output_h, output_w = output_size[2], output_size[3]
    # output_weight_graph: (n, c_out, h_out, w_out, (*origin_size))
    output_weight_graph = torch.zeros((*output_size, *origin_size), device=device)
    # hook_x: (n, c_out, c_in, h_in+2*padding, w_in+2*padding)
    hook_weight = torch.zeros(input_size[0], out_channels, in_channels, input_size[2] + padding[0] * 2, input_size[3] + padding[1] * 2, device=device)
    # origin
    for h in range(output_h):
        for w in range(output_w):
            h_start, w_start = h * stride[0], w * stride[1]
            h_end, w_end = h * stride[0] + kernel_size[0], w * stride[1] + kernel_size[1]
            # kernel_weight: (n, c_out, c_in, (*kernel_size))
            kernel_weight = kernel_weight(h, w)
            hook_weight[:, :, :, h_start:h_end, w_start:w_end] = kernel_weight(h, w)
            # hook_weight: (n, c_out, c_in, h_in, w_in) -> (n, c_out, (c_in, h_in, w_in)) -> (n, c_out, xx)
            hook_weight = hook_weight.reshape(input_size[0], out_channels, -1)
            # (h-, w-) = kernel_size
            # hook_graph: (n, c_in, h_in, w_in, (*origin_size)) -> (n, (c_in, h_in, w_in), (*origin_size)) -> (n, xx, (*origin_size))
            hook_graph = weight_graph.reshape(input_size[0], -1, origin_size.numel())
            # bmm -> (n, c_out, (*origin_size))
            # output_weight_graph: (n, c_out, h_out, w_out, *(origin_size))
            output_weight_graph[:, :, h, w] = torch.bmm(hook_weight, hook_graph).reshape(input_size[0], out_channels, *origin_size)
            hook_weight.zero_()
    return output_weight_graph


def max_pool_2d(
    max_pool_indices: torch.Tensor,
    weight_graph: torch.Tensor,
    kernel_size: torch.Size | Tuple | int,
    origin_size: torch.Size | Tuple,
    input_size: torch.Size | Tuple,
    output_size: torch.Size | Tuple,
    channels: int,
    padding: Tuple[int, int] | int = 0,
    stride: Tuple[int, int] | int = 0,
    dilation: Tuple[int, int] = 0,
    device: torch.device = torch.device("cpu"),
):
    def kernel_weight(h: int, w: int):
        # kernel_weight: (n, c_out, c_in, (*kernel_size))
        kw = torch.zeros(input_size[0], channels, channels, *kernel_size, device=device)
        # max_pool_indices: (n, c_out, h_out, w_out)
        # d_indices = (n, c_out)
        d_indices = max_pool_indices[:, :, h, w]
        kw_ = kw.reshape(input_size[0], -1)
        for i in range(input_size[0]):
            # kw_[] = d_indices[i]
            pass
        return kw

    return _common_2d_opr_weight(weight_graph, kernel_weight, kernel_size, origin_size, input_size, output_size, channels, channels, padding, stride, dilation, device)
