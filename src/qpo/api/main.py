from collections.abc import Callable
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse

from qpo.api.schemas import OptimizeRequest, OptimizeResponse, FrontierRequest, FrontierResponse
from qpo.api.service import run_optimization, run_frontier
from qpo.optimization.mean_variance import SolverError
from qpo.pipeline import get_returns, DataError

ReturnsProvider = Callable[[list[str], str, str], pd.DataFrame]

app = FastAPI(title="Quant Portfolio Optimizer", version="0.1.0")


@app.exception_handler(DataError)
def data_error_handler(request: Request, exc: DataError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "data_error", "detail": str(exc)})


@app.exception_handler(SolverError)
def solver_error_handler(request: Request, exc: SolverError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "solver_error", "detail": str(exc)})


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


@app.post("/frontier")
def frontier(
    request: FrontierRequest,
    provider: Annotated[ReturnsProvider, Depends(get_returns_provider)],
) -> FrontierResponse:
    returns = provider(
        request.tickers, request.start.isoformat(), request.end.isoformat())
    return run_frontier(returns, request.n_points)
