import numpy as np

# ================================================================
# Minimize function.


def func(function):
    """  (aX+b)^2 """
    def xx(x):
        return np.square(np.matmul(x, function[:-1]) + function[-1])
    return xx


def funcJac(function):
    """ {2a_i(aX+b);i in (0, n)} """
    def xx(x):
        return np.array([((np.matmul(x, function[:-1]) + function[-1]) * function[i] * 2) for i in range(function.shape[0]-1)])
    return xx


def conFunc(function):
    """ aX+b """
    def xx(x):
        return np.matmul(x, function[:-1]) + function[-1]
    return xx


def conFunc1(function):
    """ aX+b """
    def xx(x):
        return np.matmul(x, function[:-1]) + function[-1] - 1e-10
    return xx


def conJac(function):
    """ {a_i;i in (0, n)} """
    def xx(x):
        return function[:-1]
    return xx


def boundFun(x):
    return np.sum(np.square(x)) - 1e-32


def boundJac(x):
    return np.array([2*x[i] for i in range(x.shape[0])])


def cfunc(function, normA):
    def xx(x):
        return np.matmul(x[:-1], function[:-1]) + function[-1] - normA * x[-1]
    return xx


def jacCfunc(function, normA):
    def xx(x):
        return np.append(function[:-1], -normA)
    return xx


def ccfunc(funcion):
    def xx(x):
        return -x[-1]
    return xx


def jacCcfunc(function):
    def xx(x):
        output = np.zeros_like(function)
        output[-1] -= 1
        return output
    return xx
