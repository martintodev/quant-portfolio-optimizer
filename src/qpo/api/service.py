from collections.abc import Callable

import pandas as pd
import math
from qpo.analytics.risk import annualized_expected_returns, covariance_matrix, expected_shortfall, historical_var, monte_carlo_var, parametric_var
from qpo.api.schemas import OptimizeResponse, FrontierPoint, FrontierResponse, RiskResponse
from qpo.backtest.engine import WeightFn, max_sharpe_weights_for_window, weights_for_window
from qpo.optimization.mean_variance import SolverError, portfolio_return, portfolio_volatility, sharpe_ratio, efficient_frontier

STRATEGIES: dict[str, WeightFn] = {
    "min_variance": weights_for_window,
    "max_sharpe": max_sharpe_weights_for_window,
}


def run_optimization(returns: pd.DataFrame, strategy: str) -> OptimizeResponse:
    weights = STRATEGIES[strategy](returns)
    if weights is None:
        raise SolverError(
            f"The {strategy} optimizer found no solution for this data.")

    cov_matrix = covariance_matrix(returns)
    expected_returns = annualized_expected_returns(returns)
    raw_sharpe = sharpe_ratio(weights, expected_returns, cov_matrix)

    sharpe: float | None = None if math.isnan(raw_sharpe) else raw_sharpe

    return OptimizeResponse(
        weights=dict(zip(returns.columns, weights.tolist())),
        expected_return=portfolio_return(weights, expected_returns),
        volatility=portfolio_volatility(weights, cov_matrix),
        sharpe=sharpe,
    )


def run_frontier(returns: pd.DataFrame, n_points: int) -> FrontierResponse:
    cov_matrix = covariance_matrix(returns)
    expected_returns = annualized_expected_returns(returns)
    frontier = efficient_frontier(cov_matrix, expected_returns, n_points)
    if frontier.empty:
        raise SolverError(
            "The efficient frontier solver found no solution for this data.")

    points = [
        FrontierPoint(
            expected_return=row['return'],
            volatility=row['volatility'],
            weights=dict(zip(returns.columns, row['weights'].tolist()))
        )
        for _, row in frontier.iterrows()
    ]
    return FrontierResponse(points=points, requested_points=n_points)


def run_risk_analysis(returns: pd.DataFrame, weights: dict[str, float], confidence_level: float, seed: int) -> RiskResponse:
    unknown = sorted(set(weights) - set(returns.columns))
    if unknown:
        raise ValueError(f"weights reference unknown tickers: {unknown}")
    portfolio = returns.dot(pd.Series(weights))
    return RiskResponse(
        historical_var=float(historical_var(portfolio, confidence_level)),
        parametric_var=float(parametric_var(portfolio, confidence_level)),
        monte_carlo_var=float(monte_carlo_var(
            portfolio, confidence_level, seed=seed)),
        expected_shortfall=float(expected_shortfall(
            portfolio, confidence_level)),
    )
