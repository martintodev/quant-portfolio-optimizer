from qpo.pipeline import DataError
from qpo.optimization.mean_variance import SolverError

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from qpo.api.main import app, get_returns_provider
from qpo.api.service import STRATEGIES

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
