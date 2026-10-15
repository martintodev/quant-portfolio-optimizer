from datetime import date
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(min_length=2, max_length=10)
    start: date
    end: date
    strategy: Literal["min_variance", "max_sharpe"] = "min_variance"

    @model_validator(mode="after")
    def validate_time_order(self) -> "OptimizeRequest":
        if self.end <= self.start:
            raise ValueError(
                "The 'end' date must come strictly after the 'start' date.")
        return self


class OptimizeResponse(BaseModel):
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float | None
