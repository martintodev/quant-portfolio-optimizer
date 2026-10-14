import yfinance as yf
import pandas as pd


# Function to fetch price history for a list of tickers
def get_price_history(ticker_list: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}

    for ticker in ticker_list:
        try:
            result[ticker] = yf.Ticker(ticker).history(
                start=start, end=end, interval="1d")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return result
