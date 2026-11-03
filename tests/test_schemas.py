from pydantic import ValidationError
from qpo.api.schemas import OptimizeRequest, FrontierRequest
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


class TestFrontierRequest:
    # Test a perfectly valid request payload with n_points above upper bound
    def test_FrontierRequest_valid_upper_bound(self):
        request = FrontierRequest(
            tickers=["AAPL", "MSFT"],
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            n_points=50
        )
        assert request.n_points == 50

    # Test a perfectly valid request payload with n_points below lower bound
    def test_FrontierRequest_valid_lower_bound(self):

        request = FrontierRequest(
            tickers=["AAPL", "MSFT"],
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            n_points=5
        )
        assert request.n_points == 5

    # Test the default value for n_points
    def test_FrontierRequest_default_n_points(self):

        request = FrontierRequest(
            tickers=["AAPL", "MSFT"],
            start=date(2023, 1, 1),
            end=date(2023, 12, 31)
        )
        assert request.n_points == 20

    def test_FrontierRequest_start_equal_end_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            FrontierRequest(
                tickers=["AAPL", "MSFT"],
                start=date(2023, 12, 31),
                end=date(2023, 12, 31)
            )
        assert exc_info.value.errors()[0]["type"] == "value_error"

    def test_FrontierRequest_above_upper_bound_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            FrontierRequest(
                tickers=["AAPL", "MSFT"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                n_points=51
            )
        assert exc_info.value.errors()[0]["type"] == "less_than_equal"

    def test_FrontierRequest_below_lower_bound_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            FrontierRequest(
                tickers=["AAPL", "MSFT"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                n_points=4
            )
        assert exc_info.value.errors()[0]["type"] == "greater_than_equal"
