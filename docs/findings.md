# Backtest findings

## Setup

- **Universe:** AAPL, MSFT, JPM, XOM, JNJ, PG (six large caps, chosen by hand)
- **Data:** daily closes from yfinance, 2019-01-01 to 2026-01-01 requested, 1,759 return rows, log returns
- **Method:** walk-forward. 504-day lookback, rebalance every 21 trading days, long-only, fully invested
- **Out-of-sample:** 60 rebalances covering 1,255 trading days
- **Costs:** 10 bps per unit of turnover, including the initial purchase
- **Sharpe:** annualized over the whole out-of-sample series, 3% risk-free rate
- **Benchmark:** equal-weight, rebalanced daily with no costs, over the same dates
- **Scorecard:** Sharpe, max drawdown, average turnover, mean weight per ticker, fallback count

## Results

| Strategy     | Sharpe | Max drawdown | Avg turnover | Mean largest weight | Fallbacks |
| ------------ | ------ | ------------ | ------------ | ------------------- | --------- |
| Min-variance | 0.362  | 15.3%        | 0.051        | 0.43                | 0 / 60    |
| Max-Sharpe   | 0.449  | 20.2%        | 0.346        | 0.71                | 0 / 60    |
| Equal-weight | 0.835  | 16.0%        | n/a          | 0.17                | n/a       |

Turnover excludes the initial purchase. Ending value of $1: min-variance 1.47, equal-weight 2.11, max-Sharpe not measured.

Average weights:

- **Min-variance:** JNJ 0.42, PG 0.32, XOM 0.12, MSFT 0.08, JPM 0.05, AAPL 0.02
- **Max-Sharpe:** XOM 0.28, AAPL 0.27, JPM 0.22, MSFT 0.11, PG 0.08, JNJ 0.04

## What the results support

1. **Min-variance underperformed because of where it put the money.** It averaged 74% in JNJ and PG, the two weakest performers over the period. Its weights cost 0.6% per rebalance against equal-weight on average and helped in 30% of rebalances. Over 60 rebalances that is about -0.36 in log return, which matches its actual gap to equal-weight.
2. **Max-Sharpe held concentrated positions that shifted every month.** Its top-weighted ticker changed only 6 times in 60 rebalances, but median turnover was 0.32 (maximum 1.09) and the largest single weight averaged 71%. Its weights cost 0.16% per rebalance on average before costs and helped in 55% of rebalances.
3. **The Sharpe gap between the two strategies is within noise.** A Sharpe ratio over about five years has a standard error of roughly 0.45, and the difference is 0.09.
4. **Both lost to equal-weight on Sharpe.** This is consistent with the finding that estimated expected returns hurt out of sample. It does not prove it, given the sample below.

## Robustness checks

- Of about 1,260 frontier solves in the max-Sharpe run, 2 were flagged inaccurate. The target-return solver now treats any non-optimal status as a failure. Those two windows lost a frontier point, and every metric was identical to the run before the check.
- Transaction costs were not the explanation. Min-variance cost drag was 0.006 in ending wealth against a gap of 0.64 to equal-weight.
- Lookahead tests cover both strategies. Introducing a one-row lookahead leak into the engine makes the first-holding-day test fail for both, so that test can detect a leak.

## Open question (hypothesis, not measured)

Max-Sharpe's weights cost little against equal-weight, yet its Sharpe is far lower. A back-of-envelope estimate from the figures above suggests its volatility is around 21% against 14% for equal-weight, which would make concentration the cause and not poor stock selection. Max-Sharpe's total return and volatility were not measured in the final run, so this needs a rerun to confirm or reject.

## Limitations

- Six tickers chosen by hand, and several of them were strong performers over this window.
- One five-year window.
- The equal-weight benchmark rebalances daily at no cost, which flatters it slightly.
- A weighted sum of per-asset log returns only approximates the portfolio's log return. The error is small at daily frequency.
- Turnover is measured against previous target weights, not weights that have drifted with prices.
- Expected returns are historical means over a two-year window, which is a noisy estimate.
- No weight cap was applied.
