from qpo.backtest.performance import wealth_index, max_drawdown, rolling_sharpe
import numpy as np
import pandas as pd


def test_wealth_index():
    returns = pd.Series(np.log([1.10, 1.10]))
    expected_wealth_index = pd.Series([1.10, 1.21])

    wealth_index_result = wealth_index(returns)
    assert np.allclose(wealth_index_result.values,
                       expected_wealth_index.values)


def test_max_drawdown_positive_drawdown():
    returns = pd.Series(np.log([1.01, 1.02, 1.03]))
    expected_max_drawdown = 0

    max_drawdown_result = max_drawdown(returns)
    assert np.isclose(max_drawdown_result, expected_max_drawdown)


def test_max_drawdown_known_drawdown():
    returns = pd.Series(np.log([1.10, 0.90, 1.20, 0.60]))
    expected_max_drawdown = 0.4

    max_drawdown_result = max_drawdown(returns)
    assert np.isclose(max_drawdown_result, expected_max_drawdown)


def test_max_drawdown_start_with_loss():
    returns = pd.Series(np.log([0.8, 1.1]))
    expected_max_drawdown = 0.2

    max_drawdown_result = max_drawdown(returns)
    assert np.isclose(max_drawdown_result, expected_max_drawdown)


def test_rolling_sharpe_length():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 100))
    window = 20

    rolling_sharpe_result = rolling_sharpe(returns, window=window)
    assert len(rolling_sharpe_result) == 100
    assert rolling_sharpe_result.iloc[:window - 1].isna().all()
    assert rolling_sharpe_result.iloc[window - 1:].notna().all()


def test_rolling_sharpe_full_window():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 100))
    result = rolling_sharpe(returns, window=100)
    expected = (returns.mean() * 252 - 0.03) / (returns.std() * np.sqrt(252))
    assert np.isclose(result.iloc[-1], expected)


def test_rolling_sharpe_constant_returns():
    returns = pd.Series(np.full(100, 0.01))
    window = 20

    rolling_sharpe_result = rolling_sharpe(returns, window=window)
    expected_sharpe = (returns.rolling(window).mean() * 252 - 0.03) / (
        returns.rolling(window).std() * np.sqrt(252))

    assert isinstance(rolling_sharpe_result, pd.Series)
    assert len(rolling_sharpe_result) == 100
    assert rolling_sharpe_result.isna().all()
