from contextlib import contextmanager
from qpo.backtest.engine import (
    max_sharpe_weights_for_window,
    run_backtest,
    weights_for_window,
)
import numpy as np
import pandas as pd
import pytest
import warnings


# Create a simple synthetic dataset for testing
def create_fake_returns_data():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.0005, 0.01, size=(100, 2)),
        columns=['A', 'B'],
        index=pd.date_range('2023-01-01', periods=100, freq='B'),
    )


# Helper function to create a weight function that returns None on specific calls
def make_intermittent_weight_fn():
    calls = 0

    def intermittent_weight_fn(window):
        nonlocal calls
        calls += 1
        if calls in (2, 4):
            return None
        if calls == 1:
            return np.array([0.6, 0.4])
        return np.array([0.3, 0.7])

    return intermittent_weight_fn


@contextmanager
def no_optimizer_fallback():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error", message="Optimizer fallback", category=RuntimeWarning)
        yield


# Test the run_backtest function with a simple synthetic dataset
def test_run_backtest_output():
    returns_df = create_fake_returns_data()

    result = run_backtest(returns_df, lookback=30,
                          rebalance_frequency=10, cost_rate=0.001)

    assert isinstance(result, pd.Series)
    assert len(result) == 70
    assert result.index[0] == returns_df.index[30]


# Test that too little data raises a clear error instead of a confusing one from pd.concat
def test_run_backtest_rejects_short_data():
    returns_df = create_fake_returns_data()

    with pytest.raises(ValueError, match="more than lookback"):
        run_backtest(returns_df, lookback=100, rebalance_frequency=10)


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
@pytest.mark.parametrize(
    "weight_fn",
    [weights_for_window, max_sharpe_weights_for_window],
    ids=["min_variance", "max_sharpe"],
)
def test_run_backtest_lookahead(weight_fn):
    returns_df = create_fake_returns_data()
    modified_returns_df = returns_df.copy()

    rng = np.random.default_rng(1)
    modified_returns_df.iloc[40:] = rng.normal(
        0.0005, 0.01, size=modified_returns_df.iloc[40:].shape)

    with no_optimizer_fallback():
        result = run_backtest(
            returns_df, lookback=30, rebalance_frequency=10, cost_rate=0.001,
            weight_fn=weight_fn)
        modified_result = run_backtest(
            modified_returns_df, lookback=30, rebalance_frequency=10,
            cost_rate=0.001, weight_fn=weight_fn)
    assert np.allclose(result.iloc[:10], modified_result.iloc[:10])


# Test to ensure that the first holding day is not peeked at, which catches an off-by-one in the window
@pytest.mark.parametrize(
    "weight_fn",
    [weights_for_window, max_sharpe_weights_for_window],
    ids=["min_variance", "max_sharpe"],
)
def test_run_backtest_no_peek_at_first_holding_day(weight_fn):
    returns_df = create_fake_returns_data()
    modified_returns_df = returns_df.copy()

    # Overwrite only the first holding day with a large shock
    modified_returns_df.iloc[30] = [0.15, -0.15]

    with no_optimizer_fallback():
        result = run_backtest(
            returns_df, lookback=30, rebalance_frequency=10, cost_rate=0.001,
            weight_fn=weight_fn)
        modified_result = run_backtest(
            modified_returns_df, lookback=30, rebalance_frequency=10,
            cost_rate=0.001, weight_fn=weight_fn)

    # Position 0 is the shocked day itself, so only positions 1-9 isolate the weights
    assert np.allclose(result.iloc[1:10], modified_result.iloc[1:10])


# Test to ensure that the backtest function correctly handles cases where the optimizer fails to find a solution
def test_run_backtest_warns_when_optimizer_fallback_is_used():
    returns_df = create_fake_returns_data()

    with pytest.warns(RuntimeWarning, match="2 of 7 rebalances"):
        result = run_backtest(
            returns_df,
            lookback=30,
            rebalance_frequency=10,
            weight_fn=make_intermittent_weight_fn(),
        )

    assert len(result) == 70
    assert np.allclose(
        result.iloc[10:20], returns_df.iloc[40:50] @ [0.6, 0.4])
    assert np.allclose(
        result.iloc[30:40], returns_df.iloc[60:70] @ [0.3, 0.7])

    with pytest.warns(RuntimeWarning, match="2 of 7 rebalances"):
        result_no_cost = run_backtest(
            returns_df,
            lookback=30,
            rebalance_frequency=10,
            cost_rate=0,
            weight_fn=make_intermittent_weight_fn(),
        )

    # Costs: 1.0 turnover at position 0, 0.6 at position 20, zero on reused weights
    diff = result_no_cost - result
    expected_diff = np.zeros(len(diff))
    expected_diff[[0, 10, 20, 30]] = [0.001, 0, 0.0006, 0]
    assert np.allclose(diff, expected_diff)


# Test the case where there is no previous position to keep on the very first rebalance
def test_run_backtest_warns_when_first_optimizer_call_fails():
    returns_df = create_fake_returns_data()

    def no_weights(window):
        return None

    with pytest.warns(RuntimeWarning, match="7 of 7 rebalances"):
        result = run_backtest(
            returns_df,
            lookback=30,
            rebalance_frequency=10,
            weight_fn=no_weights,
        )

    # The first period falls back to min-variance, same as the default strategy
    default_result = run_backtest(
        returns_df, lookback=30, rebalance_frequency=10)
    assert np.allclose(result.iloc[:10], default_result.iloc[:10])

    with pytest.warns(RuntimeWarning, match="7 of 7 rebalances"):
        result_no_cost = run_backtest(
            returns_df,
            lookback=30,
            rebalance_frequency=10,
            cost_rate=0,
            weight_fn=no_weights,
        )

    # Only the first purchase costs anything, because every later rebalance reuses it
    diff = result_no_cost - result
    assert np.isclose(diff.iloc[0], 0.001)
    assert np.allclose(diff.iloc[1:], 0)


# Test that return_weights gives back the weights used at each rebalance without changing the returns
def test_run_backtest_return_weights():
    returns_df = create_fake_returns_data()

    result, weights, fallbacks = run_backtest(
        returns_df, lookback=30, rebalance_frequency=10, return_weights=True)
    plain_result = run_backtest(
        returns_df, lookback=30, rebalance_frequency=10)

    assert fallbacks == 0
    assert np.allclose(result, plain_result)
    assert list(weights.columns) == ['A', 'B']
    assert len(weights) == 7
    assert weights.index[0] == returns_df.index[30]
    assert np.allclose(weights.sum(axis=1), 1.0)


# Test that the returned weights are the weights actually used, not None, on fallback rebalances
def test_run_backtest_return_weights_records_fallback_weights():
    returns_df = create_fake_returns_data()

    with pytest.warns(RuntimeWarning, match="2 of 7 rebalances"):
        _, weights, fallbacks = run_backtest(
            returns_df,
            lookback=30,
            rebalance_frequency=10,
            weight_fn=make_intermittent_weight_fn(),
            return_weights=True,
        )

    assert fallbacks == 2
    assert np.allclose(weights.iloc[1], weights.iloc[0])
    assert np.allclose(weights.iloc[3], weights.iloc[2])
    assert not np.allclose(weights.iloc[2], weights.iloc[1])
