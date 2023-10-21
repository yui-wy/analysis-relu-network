from typing import Any, Callable, List, Tuple

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from polytope import polytope

COLOR = (
    'lightcoral',
    'royalblue',
    'limegreen',
    'gold',
    'darkorchid',
    'aqua',
    'tomato',
    'deeppink',
    'teal',
)


def color(idx: int):
    return COLOR[idx]


def _color(_: int):
    return np.random.uniform(0.0, 0.95, 3)


def plot_regions(
    functions_list: List[np.ndarray],
    regions: List[np.ndarray],
    ax: plt.Axes,
    color: Callable[[int], Any] = _color,
    alpha=1.0,
    edgecolor="w",
    linewidth=0.01,
    xlim=(-1, 1),
    ylim=(-1, 1),
):
    for i in range(len(functions_list)):
        functions: np.ndarray = functions_list[i]
        region: np.ndarray = regions[i]
        functions = -region.reshape(-1, 1) * functions
        A, b = functions[:, :-1], -functions[:, -1]
        poly = polytope.Polytope(A, b)
        poly.plot(
            ax=ax,
            color=color(i),
            alpha=alpha,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    xlim and ax.set_xlim(*xlim)
    ylim and ax.set_ylim(*ylim)
    return ax


def plot_regions_3d(
    functions_list: List[np.ndarray],
    regions: List[np.ndarray],
    z_fun: Callable[[np.ndarray], Tuple[np.ndarray, List[int]]],
    ax: axes3d.Axes3D,
    color: Callable[[int], Any] = _color,
    alpha=1.0,
    edgecolor="w",
    linewidth=0.01,
    xlim: List | Tuple = None,
    ylim: List | Tuple = None,
    zlim: List | Tuple = None,
):
    # 包含按序顶点idx的list
    regions_idxs: List[List[int]] = list()
    # 输出顶点的集合, 并且标记顶点属于的面的idx
    # 所有regions的顶点(x, y)
    xy: np.ndarray = np.empty(shape=(0, 2))
    for idx in range(len(functions_list)):
        functions: np.ndarray = functions_list[idx]
        region: np.ndarray = regions[idx]
        functions = -region.reshape(-1, 1) * functions
        A, b = functions[:, :-1], -functions[:, -1]
        poly = polytope.Polytope(A, b)
        verts = _sort_xy(poly)
        region_idxs: List[int] = list()
        for i in range(verts.shape[0]):
            vert = verts[i]
            xy_idxs = np.where((xy == vert).all(1))[0]
            if len(xy_idxs) == 0:
                xy = np.concatenate((xy, vert[np.newaxis, :]), axis=0)
                xy_idx = xy.shape[0] - 1
            else:
                xy_idx = xy_idxs[0]
            region_idxs.append(xy_idx)
        regions_idxs.append(region_idxs)
    # z: 所有regions的顶点的z坐标
    # classes: 需要画图的标签列表
    z, classes = z_fun(xy)
    ploy_sets, ploy_colors = list(), list()
    for clz in classes:
        # [[(vert1),(vert2),(vert3),...],...]
        ploy_set = [[_vert(xy[idx], z[idx][clz]) for idx in idxs] for idxs in regions_idxs]
        ploy_sets.extend(ploy_set)
        ploy_colors.extend([color(clz)] * len(regions_idxs))
    ploy = art3d.Poly3DCollection(
        ploy_sets,
        facecolor=ploy_colors,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
    )
    ax.add_collection3d(ploy)

    xlim and ax.set_xlim(*xlim)
    ylim and ax.set_ylim(*ylim)
    zlim = zlim or [np.min(z), np.max(z)]
    ax.set_zlim(*zlim)
    return ax


def _vert(xy: np.ndarray, z: np.ndarray):
    return np.append(xy, z)


def _sort_xy(poly: polytope.Polytope) -> np.ndarray:
    verts = polytope.extreme(poly)
    _, xc = polytope.cheby_ball(poly)
    x = verts[:, 1] - xc[1]
    y = verts[:, 0] - xc[0]
    mult = np.sqrt(x**2 + y**2)
    x = x / mult
    angle = np.arccos(x)
    corr = np.ones(y.size) - 2 * (y < 0)
    angle = angle * corr
    idx = np.argsort(angle)
    return verts[idx, :]
