from dataclasses import dataclass
import numpy as np
from numpy import ndarray

@dataclass
class ConjugateGradientIteration:
    iteration: int
    x: ndarray
    step_size: float
    residual: ndarray
    step_direction: ndarray
    f_x: float

@dataclass
class ConjugateGradientResult:
    max_iteration: int
    iterations: list[ConjugateGradientIteration]
    solution: float
    f_solution: float

def conjugate_gradient(A: ndarray, b: ndarray, x0: ndarray, eps=1e-5):
    x = x0
    r = (A @ x - b).reshape(-1, 1)
    step_direction = -r
    max_iteration = len(x)
    iterations = []
    for iteration in range(max_iteration):
        step_size = - (r.T @ step_direction) / (step_direction.T @ A @ step_direction)
        x1 = x + float(step_size) * step_direction.flatten()
        fx = 0.5 * float(x1.reshape(1,-1) @ A @ x1.reshape(-1,1)) - np.dot(b, x1)
        r1 = (A @ x1 - b).reshape(-1, 1)
        direction_step_change = float(r1.T @ A @ step_direction) / float(step_direction.T @ A @ step_direction)
        step_direction1 = -r1 + direction_step_change * step_direction
        iterations.append(ConjugateGradientIteration(
            iteration=iteration+1,
            x=x1,
            step_size=step_size,
            residual=r1,
            step_direction=step_direction1,
            f_x=fx
        ))
        x = x1
        r = r1
        step_direction = step_direction1
        if np.linalg.norm(r) <= eps:
            break
    return ConjugateGradientResult(
        max_iteration=iteration+1,
        iterations=iterations,
        solution=x,
        f_solution=fx
    )

__all__ = ['conjugate_gradient']
