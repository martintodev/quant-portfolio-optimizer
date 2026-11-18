from pydantic import ValidationError
from qpo.api.schemas import OptimizeRequest, FrontierRequest, RiskRequest
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

    def test_duplicate_tickers_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            FrontierRequest(tickers=["AAPL", "AAPL"], start=date(2023, 1, 1),
                            end=date(2023, 12, 31), n_points=10)
        assert exc_info.value.errors()[0]["type"] == "value_error"
        assert "unique" in str(exc_info.value)

    def test_tickers_case_insensitive_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            FrontierRequest(tickers=["AAPL", "aapl"], start=date(2023, 1, 1),
                            end=date(2023, 12, 31), n_points=10)
        assert exc_info.value.errors()[0]["type"] == "value_error"
        assert "unique" in str(exc_info.value)


class TestRiskRequest:
    def test_RiskRequest_valid_request(self):
        request = RiskRequest(
            tickers=["AAPL", "MSFT"],
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            weights={"AAPL": 0.6, "MSFT": 0.4},
            confidence_level=0.95,
            seed=0
        )
        assert request.weights == {"AAPL": 0.6, "MSFT": 0.4}
        assert request.confidence_level == 0.95
        assert request.seed == 0

    def test_RiskRequest_negative_weights_accepted(self):
        request = RiskRequest(
            tickers=["AAPL", "MSFT"],
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            weights={"AAPL": -0.6, "MSFT": 1.6},
        )
        assert request.weights == {"AAPL": -0.6, "MSFT": 1.6}

    def test_RiskRequest_weights_mismatch_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            RiskRequest(
                tickers=["AAPL", "MSFT"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                weights={"AAPL": 0.6, "MSFT": 0.4},
                confidence_level=0.95,
                seed=0
            )
        assert exc_info.value.errors()[0]["type"] == "value_error"
        assert "match" in str(exc_info.value)

    def test_weights_sum_not_close_to_1_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            RiskRequest(
                tickers=["AAPL", "MSFT"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                weights={"AAPL": 0.5, "MSFT": 0.4},  # Sum is 0.9
                confidence_level=0.95,
                seed=0
            )
        assert exc_info.value.errors()[0]["type"] == "value_error"
        assert "sum" in str(exc_info.value)

    def test_RiskRequest_confidence_level_bounds(self):
        # Test lower bound
        with pytest.raises(ValidationError) as exc_info:
            RiskRequest(
                tickers=["AAPL", "MSFT"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                weights={"AAPL": 0.6, "MSFT": 0.4},
                confidence_level=0.89,  # Below lower bound
                seed=0
            )
        assert exc_info.value.errors()[0]["type"] == "greater_than_equal"

        # Test upper bound
        with pytest.raises(ValidationError) as exc_info:
            RiskRequest(
                tickers=["AAPL", "MSFT"],
                start=date(2023, 1, 1),
                end=date(2023, 12, 31),
                weights={"AAPL": 0.6, "MSFT": 0.4},
                confidence_level=0.991,  # Above upper bound
                seed=0
            )
        assert exc_info.value.errors()[0]["type"] == "less_than_equal"
