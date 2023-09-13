from dataclasses import dataclass
from math import sqrt

phi = (sqrt(5) - 1) / 2

@dataclass
class GoldenSectionIteration:
    iteration: int
    a: float
    b: float
    lambda_: float
    mu: float
    f_lambda: float
    f_mu: float

@dataclass
class GoldenSectionResult:
    max_iteration: int
    iterations: list[GoldenSectionIteration]
    solution: float
    f_solution: float

def golden_section(f, a, b, max_iter=100, eps=1e-5):
    if a > b: a, b = b, a
    golden_iterations = []
    solution = a
    for iteration in range(max_iter):
        lambda_i = b - phi * (b - a)
        mu_i = a + phi * (b - a)
        golden_iterations.append(GoldenSectionIteration(
            iteration=iteration+1,
            a=a,
            b=b,
            lambda_=lambda_i,
            mu=mu_i,
            f_lambda=f(lambda_i),
            f_mu=f(mu_i)
        ))
        if f(lambda_i) <= f(mu_i):
            b = mu_i
            solution = a
        else:
            a = lambda_i
            solution = b
        if b - a < eps:
            break

    return GoldenSectionResult(
        max_iteration=iteration + 1,
        iterations=golden_iterations,
        solution=solution,
        f_solution=f(solution)
    )


__all__ = ['golden_section']
