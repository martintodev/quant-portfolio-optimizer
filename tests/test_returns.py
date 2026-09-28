from qpo.analytics.returns import simple_returns, log_returns
import pandas as pd
import numpy as np


# Test cases for the returns module
def test_simple_returns():
    prices = pd.Series([100, 110, 121])
    expected_returns = pd.Series([0.1, 0.1], index=[1, 2])
    calculated_returns = simple_returns(prices)
    pd.testing.assert_series_equal(calculated_returns, expected_returns)


def test_log_returns():
    prices = pd.Series([100, 110, 121])
    expected_returns = pd.Series(
        [np.log(110 / 100), np.log(121 / 110)], index=[1, 2])
    calculated_returns = log_returns(prices)
    pd.testing.assert_series_equal(calculated_returns, expected_returns)
