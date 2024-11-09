from typing import Any, Dict, Tuple

import numpy as np
from scipy import optimize


def lineprog(
    c: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    a_eq: np.ndarray = None,
    b_eq: np.ndarray = None,
    x0: np.ndarray = None,
    method: str = "highs",
    bounds: Any = (None, None),
    options: Dict = {
        "maxiter": 100,
        # "disp": True,
    },
):
    return optimize.linprog(c, a_ub, b_ub, a_eq, b_eq, method=method, options=options, bounds=bounds, x0=x0)


def cheby_ball(funcs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    *   min_{x, r} -r
    *   st. -ax+r||a|| <= b,
    *       -ax <= b
    """
    a, b_ub = funcs[:, :-1], funcs[:, -1]
    c = np.negative(np.r_[np.zeros(np.shape(a)[1]), 1])
    norm = np.sqrt(np.sum(a * a, axis=1))
    # -ax+r||a|| <= b
    r_ub = np.c_[-a, norm]
    # -ax <= b
    a_ub = np.c_[-a, np.zeros_like(b_ub)]
    # -------------
    a_ub = np.concatenate([r_ub, a_ub])
    b_ub = np.concatenate([b_ub, b_ub])
    sol = lineprog(c, a_ub, b_ub)
    if sol.success:
        x, r = sol.x[:-1], sol.x[-1]
        if r > 0:
            return x, r, sol.success
    return None, None, sol.success


def lineprog_intersect(
    func: np.ndarray,
    pn_funcs: np.ndarray,
    x0: np.ndarray,
    bounds: Any = None,
) -> bool:
    """
    *   min c @ x
    *   st. A_ub @ x <= b_ub,
    *       x[-1] = 1
    """
    c, b = func[:-1], func[-1]
    x0_r = c @ x0 + b
    if x0_r == 0:
        return True
    sign = 1 if x0_r > 0 else -1
    a_eq = np.r_[np.zeros_like(c), 1]
    a_eq = np.expand_dims(a_eq, axis=0)
    b_eq = np.ones(1)
    a_ub, b_ub = -pn_funcs, np.zeros(pn_funcs.shape[0])
    sol = lineprog(
        func * sign,
        a_ub,
        b_ub,
        a_eq,
        b_eq,
        bounds=bounds,
    )
    if sol.fun <= 0 and sol.slack.all() >= 0:
        return True
    return False
