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
