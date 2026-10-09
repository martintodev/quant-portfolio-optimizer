import pandas as pd
import warnings
from qpo.analytics.risk import annualized_expected_returns, covariance_matrix
from qpo.optimization.mean_variance import efficient_frontier, max_sharpe_portfolio, min_variance_weights
import numpy as np


# Function to calculate portfolio weights for a given window of returns
def weights_for_window(window_returns: pd.DataFrame):
    cov_matrix = covariance_matrix(window_returns)
    weights = min_variance_weights(cov_matrix)

    return weights


# Function to run a backtest on a DataFrame of returns.
#
# weight_fn takes a window of returns and returns an array of weights, or None
# to mean "no decision, keep what you hold". The engine owns that fallback
# because only it knows the previous weights.
#
# With return_weights=True it returns (returns, weights, fallback_count), where
# weights holds the weights actually used at each rebalance (after any
# fallback), indexed by the first day each set of weights was held.
def run_backtest(returns_df, lookback=504, rebalance_frequency=21, cost_rate=0.001,
                 weight_fn=weights_for_window, return_weights=False):
    if len(returns_df) <= lookback:
        raise ValueError(
            f"Need more than lookback={lookback} rows of returns, "
            f"got {len(returns_df)}.")

    rebalance_points = range(lookback, len(returns_df), rebalance_frequency)
    previous_weights = np.zeros(returns_df.shape[1])
    fallback_count = 0

    all_period_returns = []
    weights_used = []

    for i in rebalance_points:
        window = returns_df.iloc[i - lookback:i]
        holding_period = returns_df.iloc[i:i + rebalance_frequency]

        new_weights = weight_fn(window)
        if new_weights is None:
            fallback_count += 1
            if previous_weights.sum() == 0:
                # No position to keep yet, so fall back to min-variance
                new_weights = min_variance_weights(covariance_matrix(window))
            else:
                new_weights = previous_weights.copy()

        period_returns = holding_period @ new_weights

        turnover = np.abs(new_weights - previous_weights).sum()
        cost = turnover * cost_rate
        period_returns.iloc[0] -= cost

        all_period_returns.append(period_returns)
        weights_used.append(new_weights)

        previous_weights = new_weights

    portfolio_returns = pd.concat(all_period_returns)
    if fallback_count:
        warnings.warn(
            f"Optimizer fallback used on {fallback_count} of "
            f"{len(rebalance_points)} rebalances.",
            RuntimeWarning,
            stacklevel=2,
        )

    if return_weights:
        weights_df = pd.DataFrame(
            weights_used,
            index=returns_df.index[list(rebalance_points)],
            columns=returns_df.columns,
        )
        return portfolio_returns, weights_df, fallback_count

    return portfolio_returns


def max_sharpe_weights_for_window(window_returns):
    cov_matrix = covariance_matrix(window_returns)
    mean_returns = annualized_expected_returns(window_returns)

    frontier_df = efficient_frontier(cov_matrix, mean_returns)
    return max_sharpe_portfolio(frontier_df, mean_returns, cov_matrix)
