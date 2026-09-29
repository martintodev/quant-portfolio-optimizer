import pandas as pd
import numpy as np


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
