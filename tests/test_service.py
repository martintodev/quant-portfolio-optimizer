from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from qpo.api.service import STRATEGIES, run_optimization, run_frontier
from qpo.optimization.mean_variance import SolverError

RETURNS = pd.DataFrame({"A": [0.01, -0.01, 0.01, -0.01],
                        "B": [0.01, 0.01, -0.01, -0.01]})

FLAT_RETURNS = pd.DataFrame({"A": [0.01] * 4, "B": [0.01] * 4})

DRIFT_RETURNS = pd.DataFrame({
    "A": np.array([0.01, -0.01] * 4) + 0.0005,
    "B": np.array([0.01, 0.01, -0.01, -0.01] * 2) + 0.0020,
})


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


def test_run_frontier_first_point_is_min_variance():
    first = run_frontier(DRIFT_RETURNS, 10).points[0]
    assert first.weights["A"] == pytest.approx(0.5, abs=1e-3)
    assert first.weights["B"] == pytest.approx(0.5, abs=1e-3)
    assert first.expected_return == pytest.approx(0.315, abs=1e-3)
    assert first.volatility == pytest.approx(0.12, abs=1e-3)


def test_run_frontier_properties():
    frontier = run_frontier(DRIFT_RETURNS, 10)
    assert 1 <= len(frontier.points) <= 10
    assert frontier.requested_points == 10

    for point in frontier.points:
        assert sum(point.weights.values()) == pytest.approx(1.0, abs=1e-6)

    # Check that the frontier is sorted by volatility
    volatilities = [point.volatility for point in frontier.points]
    assert all(b >= a - 1e-9 for a, b in zip(volatilities, volatilities[1:]))

    # Check that the expected return increases with volatility
    expected_returns = [point.expected_return for point in frontier.points]
    assert all(b >= a - 1e-9 for a,
               b in zip(expected_returns, expected_returns[1:]))


def test_run_frontier_empty():
    with patch("qpo.api.service.efficient_frontier", return_value=pd.DataFrame()):
        with pytest.raises(SolverError, match="no solution"):
            run_frontier(DRIFT_RETURNS, 10)
