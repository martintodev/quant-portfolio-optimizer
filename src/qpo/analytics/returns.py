import pandas as pd
import numpy as np
from typing import TypeVar, cast

T = TypeVar("T", pd.Series, pd.DataFrame)


# Calculates simple returns for a given series of prices
def simple_returns(prices: pd.Series):
    return prices.pct_change().dropna()


# Calculates log returns for a given series of prices
def log_returns(prices: T) -> T:
    return cast(T, np.log(prices / prices.shift(1))).dropna()
