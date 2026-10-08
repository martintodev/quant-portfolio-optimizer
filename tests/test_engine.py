from qpo.backtest.engine import run_backtest
import numpy as np
import pandas as pd


# Create a simple synthetic dataset for testing
def create_fake_returns_data():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(100, 2)),
        columns=['A', 'B'],
        index=pd.date_range('2023-01-01', periods=100, freq='B'),
    )


# Test the run_backtest function with a simple synthetic dataset
def test_run_backtest_output():
    returns_df = create_fake_returns_data()

    result = run_backtest(returns_df, lookback=30,
                          rebalance_frequency=10, cost_rate=0.001)

    assert isinstance(result, pd.Series)
    assert len(result) == 70
    assert result.index[0] == returns_df.index[30]


# Test to ensure that the backtest function correctly applies costs to the portfolio returns
def test_run_backtest_costs():
    returns_df = create_fake_returns_data()

    result_no_cost = run_backtest(
        returns_df, lookback=30, rebalance_frequency=10, cost_rate=0.0)
    result_with_cost = run_backtest(
        returns_df, lookback=30, rebalance_frequency=10, cost_rate=0.001)

    rebalance_positions = list(range(0, 70, 10))
    diff = result_no_cost - result_with_cost

    # first purchase: turnover of 1
    assert np.isclose(diff.iloc[0], 0.001, rtol=1e-3)
    assert all(diff.iloc[rebalance_positions] >= -1e-12)
    assert np.allclose(diff.drop(diff.index[rebalance_positions]), 0)


# Test to ensure that the backtest function does not peek at future returns when calculating weights
def test_run_backtest_lookahead():
    returns_df = create_fake_returns_data()
    modified_returns_df = returns_df.copy()

    rng = np.random.default_rng(1)
    modified_returns_df.iloc[40:] = rng.normal(
        0.0005, 0.01, size=modified_returns_df.iloc[40:].shape)

    result = run_backtest(
        returns_df, lookback=30, rebalance_frequency=10, cost_rate=0.001)
    modified_result = run_backtest(
        modified_returns_df, lookback=30, rebalance_frequency=10, cost_rate=0.001)

    assert np.allclose(result.iloc[:10], modified_result.iloc[:10])


# Test to ensure that the backtest function does not peek at future returns when calculating weights, even if the first holding day is modified
def test_run_backtest_no_peek_at_first_holding_day():
    returns_df = create_fake_returns_data()
    modified_returns_df = returns_df.copy()
    # overwrite only the first holding day
    modified_returns_df.iloc[30] = [0.15, -0.15]

    result = run_backtest(returns_df, lookback=30,
                          rebalance_frequency=10, cost_rate=0.001)
    modified_result = run_backtest(
        modified_returns_df, lookback=30, rebalance_frequency=10, cost_rate=0.001)

    assert np.allclose(result.iloc[1:10], modified_result.iloc[1:10])
