import numpy as np
from dataclasses import dataclass
from numpy import ndarray

from .utils.derivatives import grad, hess

@dataclass
class NewtonMultivariableIteration:
    iteration: int
    x: ndarray
    step_size: ndarray
    f_x: float

@dataclass
class NewtonMultivariableResult:
    max_iteration: int
    iterations: list[NewtonMultivariableIteration]
    solution: ndarray
    f_solution: float

def optimize_newton_multivariable(f, x0, gradient=None, hessian=None, max_iter=100, eps=1e-5):
    if not gradient:
        gradient = grad(f)
    if not hessian:
        hessian = hess(f)
    x = x0
    newton_iterations = []
    for iteration in range(max_iter):
        step_size = np.linalg.solve(hessian(x), gradient(x))
        x1 = x - step_size
        newton_iterations.append(NewtonMultivariableIteration(
            iteration=iteration+1,
            step_size=-step_size,
            x=x1,
            f_x=f(*x1)
        ))
        x = x1
        if np.linalg.norm(gradient(x)) <= eps:
            break
    return NewtonMultivariableResult(
        max_iteration=iteration+1,
        iterations=newton_iterations,
        solution=x,
        f_solution=f(*x)
    )

__all__ = ['optimize_newton_multivariable']
