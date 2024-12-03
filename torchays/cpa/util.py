from logging import Logger
import time
from typing import Callable, List, Tuple, TypeAlias

import torch

one = torch.ones(1)


def _get_regions(x: torch.Tensor, functions: torch.Tensor) -> torch.Tensor:
    W, B = functions[:, :-1], functions[:, -1]
    regions = torch.sign(x @ W.T + B)
    return regions


def get_regions(x: torch.Tensor, functions: torch.Tensor) -> torch.Tensor:
    regions = _get_regions(x, functions)
    regions = torch.where(regions == 0, -one, regions).type(torch.int8)
    return regions


def vertify(x: torch.Tensor, functions: torch.Tensor, region: torch.Tensor) -> bool:
    # Verify that the current point x is in the region
    point = _get_regions(x, functions)
    return -1 not in region * point


def find_projection(x: torch.Tensor, hyperplanes: torch.Tensor) -> torch.Tensor:
    # Find a point on the hyperplane that passes through x and is perpendicular to the hyperplane.
    # x: [d]
    # hyperplane: [n, d+1] (w: [n, d], b: [n])
    W, B = hyperplanes[:, :-1], hyperplanes[:, -1]
    # d: [n]
    d = (x @ W.T + B) / torch.sum(torch.square(W), dim=1)
    # p_point: [n, d]
    p_point = x - W * d.view(-1, 1)
    return p_point


BoundTypes: TypeAlias = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]


def generate_bound_regions(
    bounds: float | int | Tuple[float, float] | List[Tuple[float, float]],
    dim: int = 2,
) -> BoundTypes:
    assert dim > 0, f"dim must be more than 0. [dim = {dim}]"
    handler = _number_bound
    if isinstance(bounds, tuple):
        if isinstance(bounds[0], tuple):
            handler = _tuple_bounds
        else:
            handler = _tuple_bound
    return handler(bounds, dim)


def _tuple_bound(bounds: Tuple[float, float], dim: int = 2) -> BoundTypes:
    low, upper = _bound(bounds)
    inner_point = torch.ones(dim) * (low + upper) / 2
    lows, uppers = torch.zeros(dim) + low, torch.zeros(dim) + upper
    o_bound = [(low, upper) for _ in range(dim)]
    # for x_bias
    o_bound.append((None, None))
    return *_bound_regions(lows, uppers, dim), inner_point, tuple(o_bound)


def _tuple_bounds(bounds: Tuple[Tuple[float, float]], dim: int = 2) -> BoundTypes:
    assert len(bounds) == dim, f"length of the bounds must match the dim [dim = {dim}]."
    inner_point = torch.zeros(dim)
    lows, uppers = torch.zeros(dim), torch.zeros(dim)
    o_bound = []
    for i in range(dim):
        low, upper = _bound(bounds[i])
        lows[i], uppers[i] = low, upper
        inner_point[i] = (low + upper) / 2
        o_bound.append((low, upper))
    # for x_bias
    o_bound.append((None, None))
    return *_bound_regions(lows, uppers, dim), inner_point, tuple(o_bound)


def _number_bound(bound: float | int, dim: int = 2) -> BoundTypes:
    assert len(bound) == 2, f"length of the bounds must be 2. len(bounds) = {len(bound)}."
    bounds = (-bound, bound)
    return _tuple_bound(bounds, dim)


def _bound(bound: Tuple[float, float]) -> Tuple[float, float]:
    low, upper = bound
    if upper < low:
        low, upper = upper, low
    return low, upper


def _bound_regions(lows: torch.Tensor, uppers: torch.Tensor, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    low_bound_functions = torch.cat([torch.eye(dim), -lows.unsqueeze(1)], dim=1)
    upper_bound_functions = torch.cat([-torch.eye(dim), uppers.unsqueeze(1)], dim=1)
    funcs = torch.cat([low_bound_functions, upper_bound_functions], dim=0)
    region = torch.ones(dim * 2, dtype=torch.int8)
    return funcs, region


class LogClz:
    logging: bool
    logger: Logger


def log_time(fun_name: str, indent: int = 0, logging: bool = True):
    def wapper(fun: Callable):
        def new_func(self: LogClz, *args, **kwargs):
            if not self.logging:
                return result
            start = time.time()
            result = fun(self, *args, **kwargs)
            t = time.time() - start
            self.logger.info(f"{' '*indent}[{fun_name}] took time: {t}s.")
            return result

        if logging:
            return new_func
        return fun

    return wapper
