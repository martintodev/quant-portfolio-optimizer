from qpo.backtest.performance import wealth_index, max_drawdown
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
