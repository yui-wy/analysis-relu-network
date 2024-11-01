from typing import Tuple
import numpy as np
from scipy import optimize

# ================================================================
# Minimize function.


def square(function):
    """(aX+b)^2"""

    def fun(x):
        return np.square(np.matmul(x, function[:-1]) + function[-1])

    return fun


def jac_square(function):
    """{2a_i(aX+b);i in (0, n)}"""

    def jac(x):
        return np.array([((np.matmul(x, function[:-1]) + function[-1]) * function[i] * 2) for i in range(function.shape[0] - 1)])

    return jac


def linear(function):
    """aX+b"""

    def fun(x):
        return np.matmul(x, function[:-1]) + function[-1]

    return fun


def linear_error(function):
    """aX+b"""

    def fun(x):
        return np.matmul(x, function[:-1]) + function[-1] - 1e-10

    return fun


def jac_linear(function):
    """{a_i;i in (0, n)}"""

    def jac(x):
        return function[:-1]

    return jac


def radius_constraint(function, norm):
    def fun(x):
        return np.matmul(x[:-1], function[:-1]) + function[-1] - norm * x[-1]

    return fun


def jac_radius_constraint(function, norm):
    def jac(x):
        return np.append(function[:-1], -norm)

    return jac


def radius():
    def fun(x):
        return -x[-1]

    return fun


def jac_radius(function):
    def jac(x):
        output = np.zeros_like(function)
        output[-1] -= 1
        return output

    return jac


def fun_bound(x):
    return np.sum(np.square(x)) - 1e-32


def jac_bound(x):
    return np.array([2 * x[i] for i in range(x.shape[0])])


def constraint(fun, jac, tpye: str = "ineq"):
    return {
        "type": tpye,
        "fun": fun,
        "jac": jac,
    }


def minimize(
    function,
    x0,
    constraints,
    jac,
    method="SLSQP",
    tol=1e-16,
    options={
        "maxiter": 100,
        "ftol": 1e-16,
    },
) -> Tuple[float, np.ndarray]:
    result = optimize.minimize(function, x0, method=method, constraints=constraints, jac=jac, tol=tol, options=options)
    return result.fun, result.x
