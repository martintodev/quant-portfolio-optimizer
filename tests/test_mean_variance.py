from qpo.optimization.mean_variance import min_variance_weights
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
