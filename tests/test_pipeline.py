import pandas as pd
import numpy as np
from unittest.mock import patch
import pytest
from qpo.pipeline import get_close_prices, cache_covers_range


# Helper function to create a fake price DataFrame for testing
def make_fake_prices(start, end):
    index = pd.bdate_range(start, end)
    n = len(index)
    return pd.DataFrame({
        "Close": np.linspace(100, 110, n),
        "Open": np.linspace(99, 109, n),
        "High": np.linspace(101, 111, n),
        "Low": np.linspace(98, 108, n),
        "Volume": np.random.default_rng(0).integers(1000, 10000, n)
    }, index=index)


# Test that get_close_prices uses cached data when available and does not call the fetch function again
def test_get_close_prices_uses_cache(tmp_path):
    db_path = tmp_path / "prices.db"
    fake_df = make_fake_prices("2023-01-03", "2023-01-31")

    with patch("qpo.pipeline.get_price_history") as mock_fetch:
        mock_fetch.return_value = {"AAPL": fake_df}
        first = get_close_prices(["AAPL"], "2023-01-01", "2023-02-01", db_path)
        second = get_close_prices(
            ["AAPL"], "2023-01-01", "2023-02-01", db_path)

    assert first.equals(second)
    assert mock_fetch.call_count == 1


# Test that get_close_prices raises an error when the missing fraction exceeds the default threshold
def test_get_close_prices_default_threshold_raises(tmp_path):
    frames = {
        "A": make_fake_prices("2023-01-02", "2023-01-31"),
        "B": make_fake_prices("2023-01-05", "2023-01-31"),
    }
    with patch("qpo.pipeline.get_price_history") as mock_fetch:
        mock_fetch.side_effect = lambda tickers, start, end: {
            tickers[0]: frames[tickers[0]]}
        with pytest.raises(ValueError, match="B: 3 missing"):
            get_close_prices(["A", "B"], "2023-01-02",
                             "2023-02-01", tmp_path / "prices.db")


# Test that a looser threshold lets the misaligned data through
def test_get_close_prices_looser_threshold_passes(tmp_path):
    frames = {
        "A": make_fake_prices("2023-01-02", "2023-01-31"),
        "B": make_fake_prices("2023-01-05", "2023-01-31"),
    }
    with patch("qpo.pipeline.get_price_history") as mock_fetch:
        mock_fetch.side_effect = lambda tickers, start, end: {
            tickers[0]: frames[tickers[0]]}

        result = get_close_prices(["A", "B"], "2023-01-02", "2023-02-01",
                                  tmp_path / "prices.db", max_missing_fraction=0.5)
    assert len(result) == 19
    assert not result.isna().any().any()
    assert list(result.columns) == ["A", "B"]


# Test that aligned ticker data retains every requested date
def test_get_close_prices_no_drop_when_no_missing(tmp_path):
    db_path = tmp_path / "prices.db"
    ticker_A = make_fake_prices("2023-01-02", "2023-01-31")
    ticker_B = make_fake_prices("2023-01-02", "2023-01-31")
    frames = {"AAPL": ticker_A, "MSFT": ticker_B}

    with patch("qpo.pipeline.get_price_history") as mock_fetch:
        mock_fetch.side_effect = lambda tickers, start, end: {
            ticker: frames[ticker] for ticker in tickers}
        result = get_close_prices(
            ["AAPL", "MSFT"], "2023-01-02", "2023-02-01", db_path)

    assert len(result) == len(ticker_A)


# Test that the cache_covers_range function correctly identifies when the cached data does not cover the requested range
def test_cache_covers_range_empty_frame():
    empty_df = pd.DataFrame(columns=["date", "close"])
    assert not cache_covers_range(empty_df, "2023-01-01", "2023-02-01")


# Test that the cache_covers_range function correctly identifies when the cached data fully covers the requested range
def test_cache_covers_fully_covered_range():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", "2023-01-31"),
        "close": np.linspace(100, 110, 31)
    })
    assert cache_covers_range(df, "2023-01-01", "2023-02-01")


# Test that the cache_covers_range function tolerates weekend starts but rejects later cache starts
def test_cache_covers_start_on_weekend():
    dates = pd.bdate_range("2023-01-03", "2023-01-31")
    df = pd.DataFrame({
        "date": dates,
        "close": np.linspace(100, 110, len(dates))
    })
    assert cache_covers_range(df, "2023-01-01", "2023-02-01")

    later_dates = pd.bdate_range("2023-01-10", "2023-01-31")
    later_df = pd.DataFrame({
        "date": later_dates,
        "close": np.linspace(100, 110, len(later_dates))
    })
    assert not cache_covers_range(later_df, "2023-01-01", "2023-02-01")


# Test that the cache_covers_range function correctly identifies when the cached data does not cover the requested end date
def test_cache_covers_missing_end_date():
    dates = pd.date_range("2023-01-01", "2023-01-20")
    df = pd.DataFrame(
        {"date": dates, "close": np.linspace(100, 110, len(dates))})
    assert not cache_covers_range(df, "2023-01-01", "2023-02-01")
