import numpy as np
from dataclasses import dataclass

@dataclass
class BisectionIteration:
    iteration: int
    left: float
    right: float
    mid: float
    f_left: float
    f_right: float
    f_mid: float

@dataclass
class BisectionResult:
    max_iteration: int
    iterations: list[BisectionIteration]
    solution: float
    f_solution: float

def bisection(f, a, b, max_iter=100, eps=1e-5):
    """Perform the bisection method to find the solution to f(x) = 0 with two points a and b"""
    # swap the variables if a > b
    if a > b:
        a, b = b, a
    # perform the bisection method
    assert f(a) * f(b) < 0; "Cannot perform bisection method due to f(a) and f(b) has the same sign"
    bisection_iterations = []
    for iteration in range(max_iter):
        c = (a + b) / 2
        bisection_iterations.append(BisectionIteration(
            iteration=iteration+1,
            left=a,
            right=b,
            mid=c,
            f_left=f(a),
            f_right=f(b),
            f_mid=f(c)
        ))
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
        if np.abs(a - b) <= 2 * eps:
            break
    return BisectionResult(
        max_iteration=iteration,
        iterations=bisection_iterations,
        solution=c,
        f_solution=f(c)
    )

def optimize_bisection(f, a, b, max_iter=100, eps=1e-5, h=1e-2):
    f_prime = lambda x: (f(x + h) - f(x)) / h
    return bisection(f_prime, a, b, max_iter, eps)

__all__ = ['optimize_bisection']
