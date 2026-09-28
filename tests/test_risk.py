from qpo.analytics.risk import annualized_volatility, covariance_matrix
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
