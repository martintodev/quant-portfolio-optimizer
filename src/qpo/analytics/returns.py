import pandas as pd
import numpy as np


# Calculates simple returns for a given series of prices
def simple_returns(prices: pd.Series):
    return prices.pct_change().dropna()


# Calculates log returns for a given series of prices
def log_returns(prices: pd.Series):
    return np.log(prices / prices.shift(1)).dropna()
