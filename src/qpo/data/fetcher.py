import yfinance as yf


def get_price_history(ticker_list, start, end):
    result = {}

    for ticker in ticker_list:
        try:
            result[ticker] = yf.Ticker(ticker).history(
                start=start, end=end, interval="1d")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return result


data = get_price_history(["AAPL", "MSFT"], "2023-01-01", "2023-12-31")
print(data["AAPL"].head())
