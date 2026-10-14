import pandas as pd
import numpy as np
from scipy.stats import norm


# Calculates the annualized volatility of a series of returns
def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std() * np.sqrt(periods_per_year))


# Calculates the covariance matrix of a DataFrame of returns
def covariance_matrix(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    return returns.cov() * periods_per_year


# Calculates the CAPM beta of a stock relative to the market
def capm_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    df = pd.concat([stock_returns, market_returns],
                   axis=1, join="inner").dropna()
    stock = df.iloc[:, 0].to_numpy()
    market = df.iloc[:, 1].to_numpy()

    beta = np.cov(stock, market)[0, 1] / np.var(market, ddof=1)
    return float(beta)


# Calculates the annualized expected returns of a DataFrame of returns
def annualized_expected_returns(returns_df: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    return returns_df.mean() * periods_per_year


# Calculates the historical Value at Risk (VaR) of a series of returns at a given confidence level
def historical_var(returns: pd.Series, confidence_level: float) -> float:
    return float(-returns.quantile(1 - confidence_level))


# Calculates the parametric Value at Risk (VaR) of a series of returns at a given confidence level
def parametric_var(returns: pd.Series, confidence_level: float) -> float:
    z = norm.ppf(1 - confidence_level)
    return float(-(returns.mean() + z * returns.std()))


# Calculates the Monte Carlo Value at Risk (VaR) of a series of returns at a given confidence level
def monte_carlo_var(returns: pd.Series, confidence_level: float, num_simulations: int = 10000) -> float:
    simMean = returns.mean()
    simStd = returns.std()
    simReturns = pd.Series(np.random.normal(simMean, simStd, num_simulations))
    return historical_var(simReturns, confidence_level)


# Calculates the Expected Shortfall (ES) of a series of returns at a given confidence level
def expected_shortfall(returns: pd.Series, confidence_level: float) -> float:
    var = historical_var(returns, confidence_level)
    return float(-returns[returns <= -var].mean())
