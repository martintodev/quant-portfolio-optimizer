import pandas as pd
import math
from qpo.analytics.risk import annualized_expected_returns, covariance_matrix
from qpo.api.schemas import OptimizeResponse
from qpo.backtest.engine import WeightFn, max_sharpe_weights_for_window, weights_for_window
from qpo.optimization.mean_variance import SolverError, portfolio_return, portfolio_volatility, sharpe_ratio

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
