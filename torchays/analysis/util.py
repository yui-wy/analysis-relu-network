import numpy as np
import torch

one = torch.ones(1).double()


def get_regions(x: torch.Tensor, functions: torch.Tensor) -> torch.Tensor:
    W, B = functions[:, :-1].double(), functions[:, -1].double()
    regions = torch.sign(x @ W.T + B)
    regions = torch.where(regions == 0, -one, regions).type(torch.int8)
    return regions


def projection_point(x: torch.Tensor, hyperplane: torch.Tensor) -> torch.Tensor:
    # Find a point on the hyperplane that passes through x and is perpendicular to the hyperplane.
    # x: [d]
    # hyperplane: [n, d+1] (w: [n, d], b: [n])
    W, B = hyperplane[:, :-1], hyperplane[:, -1]
    # dis: [n]
    dis = (x @ W.T + B) / torch.sum(torch.square(W),dim=1)
    p_point = x
    return p_point


def vertify(x: torch.Tensor, functions: torch.Tensor, region: torch.Tensor) -> bool:
    # Verify that the current point x is in the region
    point = get_regions(x.double(), functions)
    return not (False in np.equal(point, region))
