from pydantic import ValidationError
from qpo.api.schemas import OptimizeRequest
from datetime import date
import pytest


class TestOptimizeRequest:
    def test_OptimizeRequest_valid_request(self):
        # Test a perfectly valid request payload
        request = OptimizeRequest(
            tickers=["AAPL", "MSFT"],
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            strategy="max_sharpe"
        )
        assert request.tickers == ["AAPL", "MSFT"]
        assert request.strategy == "max_sharpe"

    def test_OptimizeRequest_single_ticker_rejected(self):
        # min_length=2 should reject 1 ticker
        with pytest.raises(ValidationError) as exc_info:
            OptimizeRequest(
                tickers=["AAPL"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31)
            )
        assert exc_info.value.errors()[0]["type"] == "too_short"

    def test_OptimizeRequest_eleven_tickers_rejected(self):
        # max_length=10 should reject the fixture's 11 tickers
        with pytest.raises(ValidationError) as exc_info:
            OptimizeRequest(
                tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                         "TSLA", "NVDA", "NFLX", "AMD", "INTC", "QCOM"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31)
            )
        assert exc_info.value.errors()[0]["type"] == "too_long"

    def test_OptimizeRequest_unknown_strategy_rejected(self):
        # Literal constraints should reject invalid strategy types
        with pytest.raises(ValidationError) as exc_info:
            OptimizeRequest(
                tickers=["AAPL", "MSFT"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                strategy="invalid_strategy"  # type: ignore
            )
        assert exc_info.value.errors()[0]["type"] == "literal_error"

    def test_OptimizeRequest_end_before_start_rejected(self):
        # The custom @model_validator should trigger when end <= start
        with pytest.raises(ValidationError) as exc_info:
            OptimizeRequest(
                tickers=["AAPL", "MSFT"],
                start=date(2023, 12, 31),
                end=date(2023, 1, 1)
            )
        assert exc_info.value.errors()[0]["type"] == "value_error"

    def test_OptimizeRequest_end_equal_start_rejected(self):
        # The custom @model_validator should trigger when end == start
        with pytest.raises(ValidationError) as exc_info:
            OptimizeRequest(
                tickers=["AAPL", "MSFT"],
                start=date(2023, 12, 31),
                end=date(2023, 12, 31)
            )
        assert exc_info.value.errors()[0]["type"] == "value_error"
