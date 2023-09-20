import numpy as np
from numpy import ndarray
from dataclasses import dataclass
from .newton import optimize_newton
from .utils.derivatives import grad

@dataclass
class SteepestDescentIteration:
    iteration: int
    x: ndarray
    t: float
    step_direction: ndarray
    f_x: float

@dataclass
class SteepestDescentResult:
    max_iteration: int
    iterations: list[SteepestDescentIteration]
    solution: ndarray
    f_solution: float

def steepest_descent(f, x0, gradient=None, max_iter=100, eps=1e-5):
    if not gradient:
        gradient = grad(f)
    
    x = x0
    steepest_iterations = []
    for iteration in range(max_iter):
        step_direction = -gradient(x)
        ft = lambda t: f(*(x + t * step_direction))
        dft = lambda t: (ft(t + 1e-2) - ft(t)) / 1e-2
        d2ft = lambda t: (dft(t + 1e-2) - dft(t)) / 1e-2
        step_size = optimize_newton(dft, d2ft, 1)
        step_size = step_size.solution
        x1 = x + float(step_size) * step_direction
        steepest_iterations.append(
            SteepestDescentIteration(
                iteration=iteration+1,
                x=x1,
                t=step_size,
                step_direction=step_direction,
                f_x=f(*x1)
            )
        )
        x = x1
        if np.linalg.norm(step_direction) <= eps:
            break
    return SteepestDescentResult(
        max_iteration=iteration + 1,
        iterations=steepest_iterations,
        solution=x,
        f_solution=f(*x)
    )

__all__ = ['steepest_descent']
