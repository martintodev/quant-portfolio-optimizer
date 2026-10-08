import pandas as pd
from qpo.analytics.risk import covariance_matrix
from qpo.optimization.mean_variance import min_variance_weights
import numpy as np


# Function to calculate portfolio weights for a given window of returns
def weights_for_window(window_returns: pd.DataFrame):
    cov_matrix = covariance_matrix(window_returns)
    weights = min_variance_weights(cov_matrix)

    return weights


# Function to run a backtest on a DataFrame of returns
def run_backtest(returns_df, lookback=504, rebalance_frequency=21, cost_rate=0.001):
    rebalance_points = range(lookback, len(returns_df), rebalance_frequency)
    previous_weights = np.zeros(returns_df.shape[1])

    all_period_returns = []

    for i in rebalance_points:
        window = returns_df.iloc[i - lookback:i]
        holding_period = returns_df.iloc[i:i + rebalance_frequency]

        new_weights = weights_for_window(window)

        period_returns = holding_period @ new_weights

        turnover = np.abs(new_weights - previous_weights).sum()
        cost = turnover * cost_rate
        period_returns.iloc[0] -= cost

        all_period_returns.append(period_returns)

        previous_weights = new_weights

    portfolio_returns = pd.concat(all_period_returns)
    return portfolio_returns
