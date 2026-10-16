from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from qpo.api.service import STRATEGIES, run_optimization
from qpo.optimization.mean_variance import SolverError

RETURNS = pd.DataFrame({"A": [0.01, -0.01, 0.01, -0.01],
                        "B": [0.01, 0.01, -0.01, -0.01]})

FLAT_RETURNS = pd.DataFrame({"A": [0.01] * 4, "B": [0.01] * 4})


def test_run_optimization_computes_metrics_for_fixed_weights():
    with patch.dict(STRATEGIES, {"min_variance": lambda window: np.array([0.25, 0.75])}):
        response = run_optimization(RETURNS, "min_variance")

    assert response.weights == {"A": 0.25, "B": 0.75}
    assert np.isclose(response.expected_return, 0)
    assert np.isclose(response.volatility, np.sqrt(0.021))
    assert np.isclose(response.sharpe, -0.03 / np.sqrt(0.021))


def test_run_optimization_unidentified_sharpe():
    with patch.dict(STRATEGIES, {"min_variance": lambda window: np.array([0.25, 0.75])}):
        response = run_optimization(FLAT_RETURNS, "min_variance")

    assert response.sharpe is None


def test_run_optimization_no_solution():
    with patch.dict(STRATEGIES, {"min_variance": lambda window: None}):
        with pytest.raises(SolverError, match="no solution"):
            run_optimization(RETURNS, "min_variance")
