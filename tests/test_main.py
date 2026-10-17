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


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
