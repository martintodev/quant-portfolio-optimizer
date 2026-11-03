from datetime import date
from typing import Literal, Self
from pydantic import BaseModel, Field, model_validator


class DateRangeRequest(BaseModel):
    tickers: list[str] = Field(min_length=2, max_length=10)
    start: date
    end: date

    @model_validator(mode="after")
    def validate_time_order(self) -> Self:
        if self.end <= self.start:
            raise ValueError(
                "The 'end' date must come strictly after the 'start' date.")
        return self


class OptimizeResponse(BaseModel):
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float | None


class OptimizeRequest(DateRangeRequest):
    strategy: Literal["min_variance", "max_sharpe"] = "min_variance"


class FrontierRequest(DateRangeRequest):
    n_points: int = Field(default=20, ge=5, le=50)


class FrontierPoint(BaseModel):
    expected_return: float
    volatility: float
    weights: dict[str, float]


class FrontierResponse(BaseModel):
    points: list[FrontierPoint]
    requested_points: int
