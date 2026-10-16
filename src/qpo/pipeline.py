from qpo.data.storage import load_prices, save_prices, init_db
from qpo.data.fetcher import get_price_history
from qpo.analytics.returns import log_returns
import pandas as pd
from pathlib import Path


class DataError(ValueError):
    """Raised when price data is missing or unusable for the requested range."""


def cache_covers_range(cached_prices: pd.DataFrame, start: str, end: str, tolerance_days: int = 4) -> bool:
    if cached_prices.empty:
        return False

    dates = pd.to_datetime(cached_prices['date'])
    min_date = dates.min()
    max_date = dates.max()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    return (
        min_date <= start_ts + pd.Timedelta(days=tolerance_days)
        and max_date >= end_ts - pd.Timedelta(days=tolerance_days)
    )


def get_close_prices(tickers: list[str], start: str, end: str, db_path: str | Path = 'data/prices.db', max_missing_fraction: float = 0.05) -> pd.DataFrame:
    init_db(db_path)  # Ensure the database is initialized
    ticker_data: dict[str, pd.Series] = {}

    for ticker in tickers:
        # Load any cached data for the requested range
        cached = load_prices(ticker, start, end, db_path)

        if not cache_covers_range(cached, start, end):
            # Fetch from Yahoo Finance (end is exclusive)
            price_data = get_price_history([ticker], start, end)

            if ticker not in price_data or price_data[ticker].empty:
                raise DataError(
                    f"No price data returned for ticker {ticker}"
                )

            # Save, ignoring duplicate records
            save_prices(ticker, price_data[ticker], db_path=db_path)

            # Reload the combined data from the database
            cached = load_prices(ticker, start, end, db_path)

        if cached.empty:
            raise DataError(
                f"No cached price data available for ticker {ticker}"
            )

        ticker_data[ticker] = (
            cached.assign(date=pd.to_datetime(cached['date']))
            .set_index('date')['close']
        )

    close_prices = pd.DataFrame(ticker_data)
    close_prices.index.name = 'date'

    before = len(close_prices)

    missing_counts = close_prices.isna().sum()

    after = close_prices.dropna()
    dropped = before - len(after)

    if before == 0:
        raise DataError(
            "No price dates available for the requested range."
        )

    missing_fraction = dropped / before

    if missing_fraction > max_missing_fraction:
        counts = ", ".join(
            f"{ticker}: {count} missing"
            for ticker, count in missing_counts.items()
        )

        raise DataError(
            f"Dropped {missing_fraction:.1%} of dates "
            f"({dropped}/{before}) due to missing prices. "
            f"Missing prices by ticker: {counts}. "
            "Check ticker histories before optimizing."
        )

    if after.empty:
        raise DataError(
            "No dates have complete closing prices for all tickers."
        )

    return after


def get_returns(tickers: list[str], start: str, end: str, db_path: str | Path = 'data/prices.db', max_missing_fraction: float = 0.05) -> pd.DataFrame:
    prices = get_close_prices(
        tickers, start, end, db_path, max_missing_fraction)
    return log_returns(prices)
