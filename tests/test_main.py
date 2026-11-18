from qpo.pipeline import DataError
from qpo.optimization.mean_variance import SolverError

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from qpo.api.main import app, get_returns_provider
from qpo.api.service import STRATEGIES, run_risk_analysis

client = TestClient(app)


FAKE_RETURNS = pd.DataFrame({"A": [0.01, -0.01, 0.01, -0.01],
                             "B": [0.01, 0.01, -0.01, -0.01]})

BODY = {"tickers": ["A", "B"], "start": "2023-01-01",
        "end": "2023-06-30", "strategy": "min_variance"}


DRIFT_RETURNS = pd.DataFrame({
    "A": np.array([0.01, -0.01] * 4) + 0.0005,
    "B": np.array([0.01, 0.01, -0.01, -0.01] * 2) + 0.0020,
})


def test_optimize_returns_422_for_data_error(make_client):
    def failing_provider(tickers, start, end):
        raise DataError("No price data returned for ticker ZZZ")

    response = make_client(failing_provider).post("/optimize", json=BODY)

    assert response.status_code == 422
    assert response.json()["error"] == "data_error"
    assert "ZZZ" in response.json()["detail"]


def test_optimize_returns_422_for_solver_error(make_client):
    def failing_provider(tickers, start, end):
        raise SolverError("Solver failed to converge")

    response = make_client(failing_provider).post("/optimize", json=BODY)

    assert response.status_code == 422
    assert response.json()["error"] == "solver_error"
    assert "converge" in response.json()["detail"]


def test_optimize_returns_422_when_optimizer_finds_no_solution(client_and_calls):
    client, calls = client_and_calls
    with patch.dict(STRATEGIES, {"min_variance": lambda window: None}):
        response = client.post("/optimize", json=BODY)

    assert response.status_code == 422
    assert response.json()["error"] == "solver_error"
    assert "no solution" in response.json()["detail"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_frontier_endpoint(make_client):
    client = make_client(lambda tickers, start, end: DRIFT_RETURNS)
    body = {"tickers": ["A", "B"], "start": "2023-01-01",
            "end": "2023-06-30", "n_points": 10}
    response = client.post("/frontier", json=body)

    assert response.status_code == 200
    result = response.json()
    assert result["requested_points"] == 10
    assert 1 <= len(result["points"]) <= 10
    assert set(result["points"][0]) == {
        "expected_return", "volatility", "weights"}


@pytest.fixture
def client_and_calls():
    calls = []

    def fake_provider(tickers, start, end):
        calls.append((tickers, start, end))
        return FAKE_RETURNS

    app.dependency_overrides[get_returns_provider] = lambda: fake_provider
    yield TestClient(app), calls
    app.dependency_overrides.clear()


def test_optimize_endpoint(client_and_calls):
    client, calls = client_and_calls
    body = {"tickers": ["A", "B"], "start": "2023-01-01", "end": "2023-06-30",
            "strategy": "min_variance"}
    with patch.dict(STRATEGIES, {"min_variance": lambda window: np.array([0.25, 0.75])}):
        response = client.post("/optimize", json=body)

    assert response.status_code == 200
    result = response.json()
    assert result["weights"] == {"A": 0.25, "B": 0.75}
    assert np.isclose(result["sharpe"], -0.03 / np.sqrt(0.021))
    assert calls == [(["A", "B"], "2023-01-01", "2023-06-30")]


def test_optimize_single_ticker(client_and_calls):
    client, calls = client_and_calls
    body = {"tickers": ["A"], "start": "2023-01-01", "end": "2023-06-30",
            "strategy": "min_variance"}
    response = client.post("/optimize", json=body)

    assert response.status_code == 422
    assert calls == []


@pytest.fixture
def make_client():
    def _make(provider):
        app.dependency_overrides[get_returns_provider] = lambda: provider
        return TestClient(app)
    yield _make
    app.dependency_overrides.clear()


def test_frontier_incorrect_n_points(client_and_calls):
    client, calls = client_and_calls
    body = {"tickers": ["A", "B"], "start": "2023-01-01", "end": "2023-06-30",
            "n_points": 100}  # Invalid n_points > 50
    response = client.post("/frontier", json=body)

    assert response.status_code == 422
    assert calls == []


def test_frontier_failing_provider(make_client):
    def failing_provider(tickers, start, end):
        raise DataError("No price data returned for ticker ZZZ")

    client = make_client(failing_provider)
    body = {"tickers": ["A", "B"], "start": "2023-01-01",
            "end": "2023-06-30", "n_points": 10}
    response = client.post("/frontier", json=body)

    assert response.status_code == 422
    assert response.json()["error"] == "data_error"
    assert "ZZZ" in response.json()["detail"]


def test_risk_happy_path(make_client):
    def fake_provider(tickers, start, end):
        return FAKE_RETURNS

    client = make_client(fake_provider)
    body = {
        "tickers": ["A", "B"],
        "start": "2023-01-01",
        "end": "2023-06-30",
        "weights": {"A": 0.5, "B": 0.5},
        "confidence_level": 0.95,
        "seed": 42
    }
    response = client.post("/risk", json=body)

    assert response.status_code == 200
    result = response.json()
    assert set(result.keys()) == {
        "historical_var", "parametric_var", "monte_carlo_var", "expected_shortfall"}


def test_risk_correct_provider(make_client):
    calls = []

    def fake_provider(tickers, start, end):
        calls.append((tickers, start, end))
        return FAKE_RETURNS

    client = make_client(fake_provider)
    body = {
        "tickers": ["A", "B"],
        "start": "2023-01-01",
        "end": "2023-06-30",
        "weights": {"A": 0.5, "B": 0.5},
        "confidence_level": 0.95,
        "seed": 42
    }
    response = client.post("/risk", json=body)

    assert response.status_code == 200
    assert calls == [(["A", "B"], "2023-01-01", "2023-06-30")]


def test_risk_endpoint_matches_service(make_client):
    weights = {"A": 0.3, "B": 0.7}
    confidence_level = 0.97
    seed = 7

    client = make_client(lambda tickers, start, end: FAKE_RETURNS)
    body = {
        "tickers": ["A", "B"],
        "start": "2023-01-01",
        "end": "2023-06-30",
        "weights": weights,
        "confidence_level": confidence_level,
        "seed": seed,
    }
    response = client.post("/risk", json=body)
    assert response.status_code == 200
    result = response.json()

    expected = run_risk_analysis(
        FAKE_RETURNS, weights, confidence_level, seed=seed)

    assert result["historical_var"] == pytest.approx(expected.historical_var)
    assert result["parametric_var"] == pytest.approx(expected.parametric_var)
    assert result["monte_carlo_var"] == pytest.approx(expected.monte_carlo_var)
    assert result["expected_shortfall"] == pytest.approx(
        expected.expected_shortfall)


def test_risk_weight_mismatch_ticker(client_and_calls):
    client, calls = client_and_calls
    body = {
        "tickers": ["A", "B"],
        "start": "2023-01-01",
        "end": "2023-06-30",
        "weights": {"A": 0.5, "C": 0.5},  # C is not in tickers
        "confidence_level": 0.95,
        "seed": 42
    }
    response = client.post("/risk", json=body)

    assert response.status_code == 422
    assert "match" in response.json()["detail"][0]["msg"]
    assert calls == []


def test_risk_low_confidence_level(make_client):
    def fake_provider(tickers, start, end):
        return FAKE_RETURNS

    client = make_client(fake_provider)
    body = {
        "tickers": ["A", "B"],
        "start": "2023-01-01",
        "end": "2023-06-30",
        "weights": {"A": 0.5, "B": 0.5},
        "confidence_level": 0.50,  # Below lower bound
        "seed": 42
    }
    response = client.post("/risk", json=body)

    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["loc"][-1] == "confidence_level"
    assert error["type"] == "greater_than_equal"


def test_risk_negative_weights_sum_to_one(make_client):
    def fake_provider(tickers, start, end):
        return FAKE_RETURNS

    client = make_client(fake_provider)
    body = {
        "tickers": ["A", "B"],
        "start": "2023-01-01",
        "end": "2023-06-30",
        "weights": {"A": -0.5, "B": 1.5},  # Sum is 1.0 but A is negative
        "confidence_level": 0.95,
        "seed": 42
    }
    response = client.post("/risk", json=body)

    assert response.status_code == 200
    result = response.json()
    assert set(result.keys()) == {
        "historical_var", "parametric_var", "monte_carlo_var", "expected_shortfall"}


def test_risk_provider_raises_data_error(make_client):
    def failing_provider(tickers, start, end):
        raise DataError("No price data returned for ticker ZZZ")

    client = make_client(failing_provider)
    body = {
        "tickers": ["A", "B"],
        "start": "2023-01-01",
        "end": "2023-06-30",
        "weights": {"A": 0.5, "B": 0.5},
        "confidence_level": 0.95,
        "seed": 42
    }
    response = client.post("/risk", json=body)

    assert response.status_code == 422
    assert response.json()["error"] == "data_error"
    assert "ZZZ" in response.json()["detail"]


def test_risk_same_request_twice(make_client):
    def fake_provider(tickers, start, end):
        return FAKE_RETURNS

    client = make_client(fake_provider)
    body = {
        "tickers": ["A", "B"],
        "start": "2023-01-01",
        "end": "2023-06-30",
        "weights": {"A": 0.5, "B": 0.5},
        "confidence_level": 0.95,
        "seed": 42
    }
    response1 = client.post("/risk", json=body)
    response2 = client.post("/risk", json=body)

    assert response1.status_code == 200
    assert response2.status_code == 200
    assert response1.json() == response2.json()
