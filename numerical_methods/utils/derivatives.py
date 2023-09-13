import numpy as np
from inspect import getfullargspec

def grad(f, h=1e-2):
    num_arguments = len(getfullargspec(f).args)
    I = np.identity(num_arguments)
    def df(x):
        return np.array([
            (f(*(x + h * I[i])) - f(*x)) / h
            for i in range(num_arguments)
        ])
    return df

def hess(f, h=1e-2):
    num_arguments = len(getfullargspec(f).args)
    gradient = grad(f, h)
    I = np.identity(num_arguments)
    def d2f(x):
        return np.array([
            [(gradient(x + h * I[j])[i] - gradient(x)[i]) / h for j in range(num_arguments)]
            for i in range(num_arguments)
        ])
    return d2f

__all__ = ['grad', 'hess']
