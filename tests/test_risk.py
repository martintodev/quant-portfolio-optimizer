from scipy.stats import norm
from qpo.analytics.risk import annualized_volatility, covariance_matrix, capm_beta, expected_shortfall, historical_var, monte_carlo_var, parametric_var
import pandas as pd
import numpy as np


# Test cases for the risk module
def test_annualized_volatility():
    returns = pd.Series([0.01, -0.01, 0.01, -0.01])
    # std of this series computed independently: mean = 0, so variance = mean of squared deviations
    # values: 0.01,-0.01,0.01,-0.01 -> squared devs all 0.0001, sample std (ddof=1) = 0.011547...
    expected_daily_std = 0.011547005383792515
    expected_volatility = expected_daily_std * (252 ** 0.5)
    calculated_volatility = annualized_volatility(returns)
    assert np.isclose(calculated_volatility, expected_volatility)


def test_covariance_matrix():
    returns_df = pd.DataFrame({
        'AAPL': [0.01, -0.02, 0.015, -0.005],
        'MSFT': [-0.01, 0.02, -0.015, 0.005]
    })
    expected_cov_matrix = returns_df.cov() * 252
    calculated_cov_matrix = covariance_matrix(returns_df)
    pd.testing.assert_frame_equal(calculated_cov_matrix, expected_cov_matrix)
    assert np.isclose(calculated_cov_matrix.loc['AAPL', 'AAPL'], annualized_volatility(
        returns_df['AAPL']) ** 2)


def test_capm_beta_of_market_against_itself_is_one():
    market_returns = pd.Series([0.01, -0.02, 0.03, 0.00])

    assert np.isclose(capm_beta(market_returns, market_returns), 1.0)


def test_capm_beta_for_distinct_returns():
    market_returns = pd.Series([0.01, 0.02, 0.03])
    stock_returns = pd.Series([0.01, 0.04, 0.05])

    # Centered values are [-1, 0, 1] and [-7/3, 2/3, 5/3] in percentage points.
    # Their sample covariance is 2, while the market sample variance is 1.
    expected_beta = 2.0
    calculated_beta = capm_beta(stock_returns, market_returns)
    assert np.isclose(calculated_beta, expected_beta)


def test_historical_var():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    confidence_level = 0.95
    expected_var = 0.018
    calculated_var = historical_var(returns, confidence_level)
    assert np.isclose(calculated_var, expected_var)


def test_parametric_var():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    confidence_level = 0.95
    z = norm.ppf(1 - confidence_level)
    meanReturns = returns.mean()
    sdReturns = returns.std()
    expected_var = -(meanReturns + z * sdReturns)
    calculated_var = parametric_var(returns, confidence_level)
    assert np.isclose(calculated_var, expected_var)


def test_monte_carlo_var():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    confidence_level = 0.95
    calculated_var = monte_carlo_var(returns, confidence_level)
    parametric_var_value = parametric_var(returns, confidence_level)
    # Monte Carlo VaR should be close to parametric VaR for a normal distribution
    assert np.isclose(calculated_var, parametric_var_value, atol=0.02)


def test_expected_shortfall():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    confidence_level = 0.95
    historical_var_value = historical_var(returns, confidence_level)
    calculated_es = expected_shortfall(returns, confidence_level)
    # ES should be greater than or equal to VaR
    assert calculated_es >= historical_var_value
