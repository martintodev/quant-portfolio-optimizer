import numpy as np
import pandas as pd
from typing import cast


# Calculate the wealth index from a series of returns
def wealth_index(returns: pd.Series) -> pd.Series:
    return cast(pd.Series, np.exp(returns.cumsum()))


# Calculate the maximum drawdown from a series of returns
def max_drawdown(returns: pd.Series) -> float:
    wealth = wealth_index(returns)

    # Ensure that the starting capital is a real peak
    previous_peaks = wealth.cummax().clip(lower=1)
    drawdowns = (wealth - previous_peaks) / previous_peaks

    # Return the maximum drawdown as a positive value to match VaR and ES conventions
    return float(-drawdowns.min())


# Calculate the rolling Sharpe ratio from a series of returns
def rolling_sharpe(returns: pd.Series, window: int = 63, risk_free_rate: float = 0.03, periods_per_year: int = 252) -> pd.Series:
    mean_returns = returns.rolling(window).mean() * periods_per_year
    std_returns = returns.rolling(window).std() * np.sqrt(periods_per_year)
    sharpe = (mean_returns - risk_free_rate) / std_returns
    return sharpe.replace([np.inf, -np.inf], np.nan)
