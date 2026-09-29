import cvxpy as cp
import numpy as np


# Function to calculate minimum variance weights given a covariance matrix
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


# Function to calculate minimum variance weights for a target return
def min_variance_weights_for_target_return(cov_matrix, expected_returns, target_return):
    n = len(cov_matrix)
    w = cp.Variable(n)

    portfolio_variance = cp.quad_form(w, cov_matrix.values)
    portfolio_return = w @ expected_returns.values

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        portfolio_return == target_return
    ]

    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    problem.solve()

    return w.value
