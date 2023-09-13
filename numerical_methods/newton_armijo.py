import numpy as np
from dataclasses import dataclass
from numpy import ndarray

from .utils.derivatives import grad, hess

@dataclass
class NewtonArmijoIteration:
    iteration: int
    x: ndarray
    step_direction: ndarray
    alpha: float
    f_x: float

@dataclass
class NewtonArmijoResult:
    max_iteration: int
    iterations: list[NewtonArmijoIteration]
    solution: ndarray
    f_solution: float

def armijo_line_search(f, x, gradient, step_direction, alpha_0=2, sigma=0.48, beta=0.95, max_iter=20):
    alpha = alpha_0
    for _ in range(max_iter):
        alpha *= beta
        if f(*(x + alpha * step_direction)) <= f(*x) + alpha * sigma * float(gradient(x).reshape((1, -1)) @ step_direction.reshape(-1, 1)):
            break
    return alpha

def optimize_newton_armijo(f, x0, gradient=None, hessian=None, max_iter=100, eps=1e-5):
    if not gradient:
        gradient = grad(f)
    if not hessian:
        hessian = hess(f)
    x = x0
    newton_iterations = []
    for iteration in range(max_iter):
        step_direction = np.linalg.solve(hessian(x), gradient(x))
        # find alpha
        alpha = armijo_line_search(f, x, gradient, -step_direction)
        x1 = x - alpha * step_direction
        newton_iterations.append(NewtonArmijoIteration(
            iteration=iteration+1,
            step_direction=-step_direction,
            alpha=alpha,
            x=x1,
            f_x=f(*x1)
        ))
        x = x1
        if np.linalg.norm(gradient(x)) <= eps:
            break
    return NewtonArmijoResult(
        max_iteration=iteration+1,
        iterations=newton_iterations,
        solution=x,
        f_solution=f(*x)
    )

__all__ = ['optimize_newton_armijo']
