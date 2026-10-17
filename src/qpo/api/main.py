from collections.abc import Callable
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Depends

from qpo.api.schemas import OptimizeRequest, OptimizeResponse
from qpo.api.service import run_optimization
from qpo.pipeline import get_returns

ReturnsProvider = Callable[[list[str], str, str], pd.DataFrame]

app = FastAPI(title="Quant Portfolio Optimizer", version="0.1.0")


def get_returns_provider() -> ReturnsProvider:
    return get_returns


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize")
def optimize(request: OptimizeRequest, provider: Annotated[ReturnsProvider, Depends(get_returns_provider)],) -> OptimizeResponse:
    returns = provider(
        request.tickers, request.start.isoformat(), request.end.isoformat())
    return run_optimization(returns, request.strategy)
