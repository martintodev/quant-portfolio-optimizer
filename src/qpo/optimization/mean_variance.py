import cvxpy as cp
import numpy as np


def min_variance_weights(cov_matrix: pd.DataFrame):
    n = len(cov_matrix)
    w = cp.Variable(n)

    portfolio_variance = cp.quad_form(w, cov_matrix.values)

    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    problem.solve()

    return w.value
