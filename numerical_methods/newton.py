import numpy as np
from dataclasses import dataclass

@dataclass
class NewtonIteration:
    iteration: int
    step_size: float
    x: float
    f_x: float

@dataclass
class NewtonResult:
    max_iteration: int
    iterations: list[NewtonIteration]
    solution: float
    f_solution: float

def newton_1_dim(f, f_prime, x0, max_iter=100, eps=1e-5):
    newton_iterations = []
    x = x0
    for iteration in range(max_iter):
        step_size = f(x) / f_prime(x)
        x1 = x - step_size
        newton_iterations.append(NewtonIteration(
            iteration=iteration+1,
            step_size=step_size,
            x=x1,
            f_x=f(x1)
        ))
        x = x1
        if abs(f(x1)) <= eps:
            break
    return NewtonResult(
        max_iteration=iteration+1,
        iterations=newton_iterations,
        solution=x,
        f_solution=f(x)
    )

def optimize_newton(grad, hess, x0, max_iter=100, eps=1e-5):
    return newton_1_dim(grad, hess, x0, max_iter, eps)

__all__ = ['optimize_newton']
