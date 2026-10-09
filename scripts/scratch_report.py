import sys
import time
import warnings
from qpo.pipeline import get_close_prices
from qpo.analytics.returns import log_returns
from qpo.backtest.engine import (
    max_sharpe_weights_for_window,
    run_backtest,
    weights_for_window,
)
from qpo.backtest.performance import max_drawdown, rolling_sharpe


LOOKBACK = 504
REBALANCE_FREQUENCY = 21
COST_RATE = 0.001


def report_strategy(name, returns, weight_fn):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        strategy_returns, weight_history, fallbacks = run_backtest(
            returns,
            lookback=LOOKBACK,
            rebalance_frequency=REBALANCE_FREQUENCY,
            cost_rate=COST_RATE,
            weight_fn=weight_fn,
            return_weights=True,
        )
    inaccurate = sum("inaccurate" in str(w.message) for w in caught)

    elapsed = time.perf_counter() - start

    print(f"\n{name}")
    print(f"Run time: {elapsed:.1f}s")
    print("Inaccurate solves:", inaccurate)
    print("Sharpe:", rolling_sharpe(
        strategy_returns, window=len(strategy_returns)).iloc[-1])
    print("Max drawdown:", max_drawdown(strategy_returns))
    print("Average turnover:",
          weight_history.diff().abs().sum(axis=1).iloc[1:].mean())
    print("Average weight per ticker:")
    average_weights = weight_history.mean()
    average_weights = average_weights.mask(average_weights.abs() < 1e-8, 0.0)
    print(average_weights.sort_values(ascending=False))
    print(f"Fallbacks: {fallbacks} of {len(weight_history)} rebalances")

    largest = weight_history.max(axis=1)
    print("Mean largest single weight:", largest.mean())
    print("Median largest single weight:", largest.median())
    top = weight_history.idxmax(axis=1)
    print("Top-ticker changes:", (top != top.shift()).sum() - 1)
    print(top.value_counts())


def main():
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG"]
    prices = get_close_prices(tickers, "2019-01-01", "2026-01-01")
    returns = log_returns(prices).dropna()
    mode = sys.argv[1] if len(sys.argv) > 1 else "slice"

    if mode == "slice":
        sample = returns.iloc[:LOOKBACK + 3 * REBALANCE_FREQUENCY]
    elif mode == "full":
        sample = returns
    else:
        raise SystemExit(
            "Usage: python scripts/scratch_report.py [slice|full]")

    print(f"Running {mode} sample: {sample.shape[0]} return rows")
    report_strategy("Min-variance", sample, weights_for_window)
    report_strategy("Max-Sharpe", sample, max_sharpe_weights_for_window)

    equal_weight = sample.iloc[LOOKBACK:].mean(axis=1)
    print("\nEqual-weight")
    print("Sharpe:", rolling_sharpe(
        equal_weight, window=len(equal_weight)).iloc[-1])
    print("Max drawdown:", max_drawdown(equal_weight))


if __name__ == "__main__":
    main()
