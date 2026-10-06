import pandas as pd
import numpy as np
from scipy.stats import norm

# Calculates the annualized volatility of a series of returns


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252):
    return returns.std() * np.sqrt(periods_per_year)


# Calculates the covariance matrix of a DataFrame of returns
def covariance_matrix(returns: pd.DataFrame, periods_per_year: int = 252):
    return returns.cov() * periods_per_year


# Calculates the CAPM beta of a stock relative to the market
def capm_beta(stock_returns: pd.Series, market_returns: pd.Series):
    beta = pd.DataFrame(
        {'stock': stock_returns, 'market': market_returns}).cov().loc['stock', 'market'] / market_returns.var()
    return beta


# Calculates the annualized expected returns of a DataFrame of returns
def annualized_expected_returns(returns_df: pd.DataFrame, periods_per_year: int = 252):
    return returns_df.mean() * periods_per_year


# Calculates the historical Value at Risk (VaR) of a series of returns at a given confidence level
def historical_var(returns: pd.Series, confidence_level: float):
    return -returns.quantile(1 - confidence_level)


# Calculates the parametric Value at Risk (VaR) of a series of returns at a given confidence level
def parametric_var(returns: pd.Series, confidence_level: float):
    z = norm.ppf(1 - confidence_level)
    return -(returns.mean() + z * returns.std())


# Calculates the Monte Carlo Value at Risk (VaR) of a series of returns at a given confidence level
def monte_carlo_var(returns: pd.Series, confidence_level: float, num_simulations: int = 10000):
    simMean = returns.mean()
    simStd = returns.std()
    simReturns = pd.Series(np.random.normal(simMean, simStd, num_simulations))
    return historical_var(simReturns, confidence_level)


# Calculates the Expected Shortfall (ES) of a series of returns at a given confidence level
def expected_shortfall(returns: pd.Series, confidence_level: float):
    var = historical_var(returns, confidence_level)
    return -returns[returns <= -var].mean()
