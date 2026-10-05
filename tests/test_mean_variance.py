from qpo.optimization.mean_variance import efficient_frontier, min_variance_weights, min_variance_weights_for_target_return
import pandas as pd
import numpy as np
import cvxpy as cp


# Test cases for the mean_variance module
def test_min_variance_weights_sums_to_one():
    # Create a simple covariance matrix for testing
    cov_matrix = pd.DataFrame({
        'A': [1.0, 0.5],
        'B': [0.5, 1.0]
    })
    weights = min_variance_weights(cov_matrix)
    assert np.isclose(np.sum(weights), 1.0)


def test_min_variance_weights_are_non_negative():
    # Create a simple covariance matrix for testing
    cov_matrix = pd.DataFrame({
        'A': [1.0, 0.5],
        'B': [0.5, 1.0]
    })
    weights = min_variance_weights(cov_matrix)
    assert np.all(weights >= 0)


def test_min_variance_weights_for_target_return_sum_to_one():
    # Create a simple covariance matrix and expected returns for testing
    cov_matrix = pd.DataFrame({
        'A': [1.0, 0.5],
        'B': [0.5, 1.0]
    })
    expected_returns = pd.Series({'A': 0.1, 'B': 0.2})
    target_return = 0.15
    weights = min_variance_weights_for_target_return(
        cov_matrix, expected_returns, target_return)
    assert np.isclose(np.sum(weights), 1.0)

    assert np.isclose(weights[0], 0.5, atol=0.01)
    assert np.isclose(weights[1], 0.5, atol=0.01)


def test_min_variance_weights_for_target_return_non_negative():
    # Create a simple covariance matrix and expected returns for testing
    cov_matrix = pd.DataFrame({
        'A': [1.0, 0.5],
        'B': [0.5, 1.0]
    })
    expected_returns = pd.Series({'A': 0.1, 'B': 0.2})
    target_return = 0.15
    weights = min_variance_weights_for_target_return(
        cov_matrix, expected_returns, target_return)
    assert np.all(weights >= 0)


def test_min_variance_weights_for_target_return_acheives_target_return():
    # Create a simple covariance matrix and expected returns for testing
    cov_matrix = pd.DataFrame({
        'A': [1.0, 0.5],
        'B': [0.5, 1.0]
    })
    expected_returns = pd.Series({'A': 0.1, 'B': 0.2})
    target_return = 0.15
    weights = min_variance_weights_for_target_return(
        cov_matrix, expected_returns, target_return)
    portfolio_return = weights @ expected_returns.values

    assert np.isclose(portfolio_return, target_return)


def test_efficient_frontier_returns_increasing():
    # Create a simple covariance matrix and expected returns for testing
    cov_matrix = pd.DataFrame({
        'A': [1.0, 0.5],
        'B': [0.5, 1.0]
    })
    expected_returns = pd.Series({'A': 0.1, 'B': 0.2})
    frontier = efficient_frontier(cov_matrix, expected_returns, n_points=10)

    # Check that the returns are increasing
    assert frontier['return'].is_monotonic_increasing


def test_efficient_frontier_number_of_rows():
    # Create a simple covariance matrix and expected returns for testing
    cov_matrix = pd.DataFrame({
        'A': [1.0, 0.5],
        'B': [0.5, 1.0]
    })
    expected_returns = pd.Series({'A': 0.1, 'B': 0.2})
    n_points = 10
    frontier = efficient_frontier(
        cov_matrix, expected_returns, n_points=n_points)

    # Check that the number of rows in the frontier matches n_points
    assert len(frontier) == n_points


def test_efficient_frontier_first_row_weights():
    # Create a simple covariance matrix and expected returns for testing
    cov_matrix = pd.DataFrame({
        'A': [1.0, 0.5],
        'B': [0.5, 1.0]
    })
    expected_returns = pd.Series({'A': 0.1, 'B': 0.2})
    n_points = 10
    frontier = efficient_frontier(
        cov_matrix, expected_returns, n_points=n_points)
    min_weights = min_variance_weights(cov_matrix)

    # Check the weights of the first row
    first_row_weights = frontier.iloc[0]['weights']
    assert np.allclose(first_row_weights, min_weights)
