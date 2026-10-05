import cvxpy as cp
import numpy as np
import pandas as pd


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


# Function to calculate portfolio return given weights and expected returns
def portfolio_return(weights, expected_returns):
    return weights @ expected_returns.values


# Function to calculate portfolio volatility given weights and covariance matrix
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights @ cov_matrix.values @ weights)


# Function to calculate the efficient frontier given a covariance matrix and expected returns
def efficient_frontier(cov_matrix, expected_returns, n_points=20):
    min_var_weights = min_variance_weights(cov_matrix)
    min_return = portfolio_return(min_var_weights, expected_returns)
    max_return = expected_returns.max()

    target_returns = np.linspace(min_return, max_return, n_points)

    results = []
    for target in target_returns:
        weights = min_variance_weights_for_target_return(
            cov_matrix, expected_returns, target)
        if weights is None:
            continue
        vol = portfolio_volatility(weights, cov_matrix)
        ret = portfolio_return(weights, expected_returns)
        results.append({"target_return": target, "return": ret,
                       "volatility": vol, "weights": weights})

    return pd.DataFrame(results)
