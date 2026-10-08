import numpy as np


# Calculate the wealth index from a series of returns
def wealth_index(returns):
    return np.exp(returns.cumsum())


# Calculate the maximum drawdown from a series of returns
def max_drawdown(returns):
    wealth = wealth_index(returns)

    # Ensure that the starting capital is a real
    previous_peaks = wealth.cummax().clip(lower=1)
    drawdowns = (wealth - previous_peaks) / previous_peaks

    # Return the maximum drawdown as a positive value to match VaR and ES conventions
    return -drawdowns.min()
