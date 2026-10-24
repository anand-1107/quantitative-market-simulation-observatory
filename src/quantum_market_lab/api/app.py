"""FastAPI application for the Quantitative Market Simulation Observatory."""

from math import exp, sqrt
from datetime import date
import hashlib
import json
import logging
import os
import platform
import uuid

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Query
from pydantic import BaseModel, Field

from quantum_market_lab.data.features import compute_log_returns, drawdown, summarize_returns
from quantum_market_lab.data.market_data import MarketDataError, MarketDataLoader, MarketDataProviderError, validate_market_data
from quantum_market_lab.data.storage import MarketDataStore
from quantum_market_lab.data.experiments import ExperimentStore
from quantum_market_lab.data.events import EventAnalysisCache, EventThresholds, analyze_historical_events
from quantum_market_lab.data.tournament import run_model_tournament
from quantum_market_lab.data.news import FinnhubNewsProvider, NewsProviderError, NewsQuery, NewsResponseCache, fetch_historical_news, rank_news_for_events

from quantum_market_lab.gbm import simulate_gbm
from quantum_market_lab.api.schemas import (
    GARCHCalibrationRequest,
    GARCHCalibrationResponse,
    GBMCalibrationRequest,
    GBMCalibrationResponse,
    GBMEvaluationRequest,
    GBMEvaluationResponse,
    ModelTournamentRequest,
    ModelTournamentResponse,
    MertonCalibrationRequest,
    MertonCalibrationResponse,
    OUCalibrationRequest,
    OUCalibrationResponse,
    GARCHTrajectoryRequest,
    MarketSummaryResponse,
    GBMTrajectoryRequest,
    JumpDiffusionTrajectoryRequest,
    OUTrajectoryRequest,
    OHLCVSeriesResponse,
    SimulationTrajectoriesResponse,
    PathForensicsResponse,
    PathCounterfactualRequest,
    PathCounterfactualResponse,
    RetainedForensicsRunsResponse,
    ScenarioMatchingRequest,
    ScenarioMatchingResponse,
    SavedTournamentsResponse,
    ExperimentCreateRequest,
    ExperimentRecord,
    ExperimentListResponse,
    ExperimentComparisonRequest,
    ExperimentComparisonResponse,
    ExperimentRunResponse,
    MarketEventAnalysisRequest,
    MarketEventNewsRequest,
    MarketEventNewsResponse,
    NewsProviderStatusResponse,
)
from quantum_market_lab.models.diagnostics import empirical_var_es, path_diagnostics, return_diagnostics, terminal_return_loss_risk, terminal_return_risk, trajectory_summary
from quantum_market_lab.models.garch import garch_returns_to_prices, simulate_garch
from quantum_market_lab.models.forensics import reconstruct_path
from quantum_market_lab.models.scenario_matching import match_objective, normalized_path_rmse, simple_return_and_drawdown
from quantum_market_lab.models.jump_diffusion import simulate_jump_diffusion
from quantum_market_lab.models.ou import simulate_ou
from quantum_market_lab.models.registry import list_models
from quantum_market_lab.models.calibration import (
    calibrate_garch,
    calibrate_merton,
    calibrate_ou,
    evaluate_gbm,
    estimate_gbm,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quantitative Market Simulation & Prediction Observatory",
    version="0.1.0",
    description="API for reproducible quantitative market simulations.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class GBMSimulationRequest(BaseModel):
    initial_price: float = Field(100.0, gt=0, le=1_000_000_000)
    drift: float = Field(0.08, ge=-2, le=2)
    volatility: float = Field(0.2, ge=0, le=5)
    years: float = Field(1.0, gt=0, le=100)
    steps: int = Field(252, ge=1, le=5000)
    paths: int = Field(10000, ge=1, le=100000)
    seed: int = Field(42, ge=0, le=2**32 - 1)


class MarketDataDownloadRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=32, pattern=r"^[A-Za-z0-9^=._-]+$")
    start: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    interval: str = Field("1d", pattern=r"^(1m|2m|5m|15m|30m|60m|90m|1h|1d|5d|1wk|1mo|3mo)$")


class OURequest(BaseModel):
    initial_value: float = Field(0, ge=-1e9, le=1e9)
    mean_level: float = Field(0, ge=-1e9, le=1e9)
    mean_reversion_speed: float = Field(1, ge=0, le=1000)
    volatility: float = Field(1, ge=0, le=1e6)
    years: float = Field(1, gt=0, le=100)
    steps: int = Field(252, ge=1, le=10000)
    paths: int = Field(1000, ge=1, le=10000)
    seed: int | None = Field(42, ge=0, le=2**32 - 1)


class JumpDiffusionRequest(BaseModel):
    initial_price: float = Field(100, gt=0, le=1e9)
    drift: float = Field(.08, ge=-10, le=10)
    volatility: float = Field(.2, ge=0, le=5)
    jump_intensity: float = Field(.1, ge=0, le=100)
    mean_log_jump: float = Field(-.1, ge=-10, le=10)
    jump_volatility: float = Field(.2, ge=0, le=5)
    years: float = Field(1, gt=0, le=100)
    steps: int = Field(252, ge=1, le=10000)
    paths: int = Field(1000, ge=1, le=10000)
    seed: int | None = Field(42, ge=0, le=2**32 - 1)


class GARCHRequest(BaseModel):
    omega: float = Field(1e-6, gt=0, le=1)
    alpha: float = Field(.05, ge=0, le=10)
    beta: float = Field(.9, ge=0, le=10)
    steps: int = Field(252, ge=1, le=10000)
    paths: int = Field(1000, ge=1, le=10000)
    mean: float = Field(0, ge=-10, le=10)
    initial_variance: float | None = Field(None, gt=0, le=100)
    seed: int | None = Field(42, ge=0, le=2**32 - 1)
    return_convention: str = Field("log", pattern=r"^(log|simple)$")
    initial_price: float | None = Field(None, gt=0, le=1e9)


# Keep the pre-existing GBM defaults (10,000 paths x 253 points) valid while
# bounding requests that could otherwise allocate hundreds of millions of values.
MAX_SIMULATION_CELLS = 5_000_000


def _check_simulation_size(paths: int, steps: int, *, price_path: bool = True) -> None:
    cells = paths * (steps + 1 if price_path else steps)
    if cells > MAX_SIMULATION_CELLS:
        raise HTTPException(status_code=422, detail=f"simulation exceeds resource limit ({MAX_SIMULATION_CELLS:,} path values)")


def _path_summary(paths: np.ndarray) -> dict:
    diagnostics = path_diagnostics(paths)
    return {"terminal_mean": diagnostics["terminal_mean"],
            "terminal_variance": diagnostics["terminal_variance"],
            "terminal_quantiles": diagnostics["terminal_quantiles"]}


def _trajectory_response(model: str, request: BaseModel, values: np.ndarray,
                         times: np.ndarray, quantity_kind: str, quantity_label: str,
                         time_label: str, conditional_variances: np.ndarray | None = None):
    try:
        summary = trajectory_summary(values, times)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if conditional_variances is not None and not np.isfinite(conditional_variances).all():
        raise HTTPException(status_code=422, detail="simulated conditional variances exceeded finite numeric range")
    risk = _horizon_risk(terminal_return_risk(values, quantity_kind, getattr(request, "return_convention", None)), request)
    return {"model": model, "parameters": request.model_dump(exclude={"retain_forensics"}), "quantity_kind": quantity_kind,
            "quantity_label": quantity_label, "time_label": time_label,
            "times": times.tolist(), "trajectories": values.tolist(),
            "conditional_variances": conditional_variances.tolist() if conditional_variances is not None else None,
            "summary": summary, "terminal_risk": risk}


def _persist_path_forensics(model: str, request: BaseModel, times: np.ndarray, trajectories: np.ndarray,
                            innovations: dict[str, np.ndarray], conditional_variances: np.ndarray | None,
                            terminal_risk: dict) -> str:
    parameters = request.model_dump(exclude={"retain_forensics"}, mode="json")
    if model == "garch" and conditional_variances is not None:
        parameters["initial_variance"] = float(conditional_variances[0, 0])
    payload = {
        "model": model,
        "model_version": f"{app.version}:{model}:1",
        "parameters": parameters,
        "times": times.astype(float).tolist(),
        "trajectories": trajectories.astype(float).tolist(),
        "innovations": {key: values.astype(float).tolist() for key, values in innovations.items()},
        "conditional_variances": conditional_variances.astype(float).tolist() if conditional_variances is not None else None,
        "terminal_risk": terminal_risk,
    }
    return _experiment_store().save_path_forensics(payload)


def _horizon_risk(risk: dict, request: BaseModel) -> dict:
    years = getattr(request, "years", None)
    risk["horizon_steps"] = int(request.steps)
    risk["horizon_years"] = float(years) if years is not None else None
    risk["horizon_unit"] = "years" if years is not None else "observation_periods"
    return risk


def _terminal_risk_from_returns(total_returns: np.ndarray, request: BaseModel) -> dict:
    return _horizon_risk(terminal_return_loss_risk(total_returns), request)


def _check_calibration_parameters(request: BaseModel, calibration: BaseModel | None,
                                  names: tuple[str, ...]) -> None:
    if calibration is None:
        return
    for name in names:
        expected = getattr(calibration, name)
        actual = getattr(request, name)
        if actual is None or not np.isclose(float(actual), float(expected), rtol=0, atol=1e-10):
            raise HTTPException(status_code=422, detail=f"{name} must match the attached model-specific calibration provenance")


def _market_download(ticker: str, start: str | None = None, end: str | None = None, interval: str = "1d"):
    try:
        loader = MarketDataLoader(ticker, start, end, interval)
        frame = loader.download()
        return loader, frame
    except ValueError as exc:
        logger.warning("Historical market-data request rejected: ticker=%s start=%s end=%s interval=%s category=validation",
                       ticker, start, end, interval)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MarketDataError as exc:
        logger.warning("Historical market-data provider failed: ticker=%s start=%s end=%s interval=%s exception_type=%s",
                       ticker, start, end, interval, type(exc).__name__)
        if isinstance(exc, MarketDataProviderError):
            detail = f"Yahoo Finance could not retrieve historical data for {ticker}. Check the symbol, date coverage, and provider availability."
        else:
            detail = str(exc)
        raise HTTPException(status_code=502, detail=detail) from None
    except Exception as exc:
        # Avoid exception text/tracebacks in case an upstream error includes request details.
        logger.error("Unexpected historical market-data failure: ticker=%s start=%s end=%s interval=%s exception_type=%s",
                     ticker, start, end, interval, type(exc).__name__)
        raise HTTPException(status_code=502, detail="Historical market-data retrieval failed unexpectedly. Check the backend log for the request diagnostic.") from None


@app.get("/health", tags=["system"])
def health():
    return {
        "status": "ok",
        "service": "quantitative-market-simulation-observatory",
    }


@app.get("/models", tags=["simulation"], summary="List available stochastic models")
def available_models():
    return {"models": list_models()}


@app.post("/simulate-gbm", tags=["simulation"])
def simulate_gbm_endpoint(request: GBMSimulationRequest):
    """Simulate GBM paths and return summary statistics."""

    _check_simulation_size(request.paths, request.steps)

    times, prices = simulate_gbm(
        initial_price=request.initial_price,
        drift=request.drift,
        volatility=request.volatility,
        years=request.years,
        steps=request.steps,
        paths=request.paths,
        seed=request.seed,
    )

    terminal = prices[:, -1]

    return {
        "model": "Geometric Brownian Motion",
        "parameters": request.model_dump(),
        "time_points": len(times),
        "paths_simulated": prices.shape[0],
        "terminal_price": {
            "mean": float(terminal.mean()),
            "median": float(np.median(terminal)),
            "std": float(terminal.std(ddof=1)) if len(terminal) > 1 else 0.0,
            "min": float(terminal.min()),
            "max": float(terminal.max()),
        },
        "theoretical_terminal_mean": (
            request.initial_price * exp(request.drift * request.years)
        ),
        "mean_standard_error": (
            float(terminal.std(ddof=1) / sqrt(len(terminal)))
            if len(terminal) > 1
            else None
        ),
        "terminal_risk": _horizon_risk(
            terminal_return_loss_risk(terminal / request.initial_price - 1.0), request
        ),
        "seed": request.seed,
    }


@app.post("/simulate-gbm/paths", response_model=SimulationTrajectoriesResponse,
          tags=["simulation"], summary="Return bounded GBM trajectories")
def simulate_gbm_paths(request: GBMTrajectoryRequest):
    """Return genuine GBM paths; capped at 128 paths and 512 steps for visualization."""
    parameters = request.model_dump(exclude={"calibration", "retain_forensics"})
    if request.calibration is not None:
        if not (np.isclose(request.drift, request.calibration.drift, rtol=0, atol=1e-12)
                and np.isclose(request.volatility, request.calibration.volatility, rtol=0, atol=1e-12)):
            raise HTTPException(status_code=422, detail="drift and volatility must match the supplied calibration provenance")
        if request.calibration.estimation_start >= request.calibration.estimation_end:
            raise HTTPException(status_code=422, detail="calibration estimation_start must be earlier than estimation_end")
    if request.retain_forensics:
        times, prices, innovations = simulate_gbm(**parameters, return_innovations=True)
    else:
        times, prices = simulate_gbm(**parameters)
    # The engine exponentiates log(initial_price); restore the exact supplied S0
    # in this new endpoint's metadata-facing first column against round-off.
    prices[:, 0] = request.initial_price
    response = _trajectory_response("gbm", request, prices, times, "price", "Simulated price", "Time (years)")
    response["parameters"] = parameters | {"calibration": request.calibration.model_dump() if request.calibration else None}
    if request.retain_forensics:
        response["forensics_run_id"] = _persist_path_forensics("gbm", request, times, prices, innovations, None, response["terminal_risk"])
    return response


@app.post("/calibration/gbm", response_model=GBMCalibrationResponse, tags=["calibration"],
          summary="Estimate GBM parameters from validated historical prices")
def calibrate_gbm_endpoint(request: GBMCalibrationRequest):
    if request.start >= request.end:
        raise HTTPException(status_code=422, detail="start must be earlier than end")
    if request.interval not in {"1d", "5d", "1wk", "1mo", "3mo"}:
        raise HTTPException(status_code=422, detail="GBM calibration supports intervals 1d, 5d, 1wk, 1mo, and 3mo only")
    loader, frame = _market_download(request.ticker, request.start, request.end, request.interval)
    try:
        estimate = estimate_gbm(loader.ticker, frame, request.interval, request.start, request.end)
    except MarketDataError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return estimate.as_dict()


@app.post("/calibration/ou", response_model=OUCalibrationResponse, tags=["calibration"],
          summary="Fit an OU process to an explicitly selected historical state series")
def calibrate_ou_endpoint(request: OUCalibrationRequest):
    if request.start >= request.end:
        raise HTTPException(status_code=422, detail="start must be earlier than end")
    _, frame = _market_download(request.ticker, request.start, request.end, request.interval)
    try:
        return calibrate_ou(request.ticker, frame, series=request.input_series,
                            price_field=request.price_field, interval=request.interval,
                            start=request.start, end=request.end,
                            allow_price_series=request.confirm_mean_reverting_price_series)
    except (MarketDataError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/calibration/jump-diffusion", response_model=MertonCalibrationResponse, tags=["calibration"],
          summary="Fit the Merton compound-Poisson return mixture by maximum likelihood")
def calibrate_merton_endpoint(request: MertonCalibrationRequest):
    if request.start >= request.end:
        raise HTTPException(status_code=422, detail="start must be earlier than end")
    _, frame = _market_download(request.ticker, request.start, request.end, request.interval)
    try:
        return calibrate_merton(request.ticker, frame, price_field=request.price_field,
                                interval=request.interval, start=request.start, end=request.end)
    except (MarketDataError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/calibration/garch", response_model=GARCHCalibrationResponse, tags=["calibration"],
          summary="Fit stationary GARCH(1,1) by Gaussian conditional maximum likelihood")
def calibrate_garch_endpoint(request: GARCHCalibrationRequest):
    if request.start >= request.end:
        raise HTTPException(status_code=422, detail="start must be earlier than end")
    _, frame = _market_download(request.ticker, request.start, request.end, request.interval)
    try:
        return calibrate_garch(request.ticker, frame, price_field=request.price_field,
                               interval=request.interval, start=request.start, end=request.end)
    except (MarketDataError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/evaluation/gbm", response_model=GBMEvaluationResponse, tags=["evaluation"],
          summary="Chronologically evaluate calibrated GBM on a later historical window")
def evaluate_gbm_endpoint(request: GBMEvaluationRequest):
    if request.estimation_start >= request.estimation_end:
        raise HTTPException(status_code=422, detail="estimation_start must be earlier than estimation_end")
    if request.evaluation_start >= request.evaluation_end:
        raise HTTPException(status_code=422, detail="evaluation_start must be earlier than evaluation_end")
    if request.evaluation_start <= request.estimation_end:
        raise HTTPException(status_code=422, detail="evaluation window must start after the estimation window ends")
    if request.interval not in {"1d", "5d", "1wk", "1mo", "3mo"}:
        raise HTTPException(status_code=422, detail="GBM evaluation supports intervals 1d, 5d, 1wk, 1mo, and 3mo only")
    loader, estimation_frame = _market_download(request.ticker, request.estimation_start, request.estimation_end, request.interval)
    try:
        calibration = estimate_gbm(loader.ticker, estimation_frame, request.interval,
                                   request.estimation_start, request.estimation_end)
    except (MarketDataError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _, evaluation_frame = _market_download(request.ticker, request.evaluation_start,
                                           request.evaluation_end, request.interval)
    if evaluation_frame.index.min().date().isoformat() <= estimation_frame.index.max().date().isoformat():
        raise HTTPException(status_code=422, detail="provider observations overlap or reverse the requested chronological windows")
    steps = len(evaluation_frame) - 1
    if steps < 2:
        raise HTTPException(status_code=422, detail="evaluation window must contain at least three prices")
    years = steps / calibration.annualization_periods
    try:
        evaluation_price_field = "adj_close" if "adj_close" in evaluation_frame.columns else "close"
        _, simulated_prices = simulate_gbm(float(evaluation_frame[evaluation_price_field].iloc[0]),
                                           calibration.drift, calibration.volatility, years,
                                           steps, request.paths, request.seed)
        result = evaluate_gbm(calibration, evaluation_frame, simulated_prices)
    except (MarketDataError, ValueError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"ticker": loader.ticker, "calibration": calibration.as_dict(), "evaluation": result}


@app.post("/model-tournaments", response_model=ModelTournamentResponse, tags=["evaluation"],
          summary="Calibrate selected models on training data and evaluate chronologically held-out observations")
def model_tournament_endpoint(request: ModelTournamentRequest):
    if len(set(request.models)) != len(request.models):
        raise HTTPException(status_code=422, detail="models must be unique")
    try:
        dates = [date.fromisoformat(value) for value in (request.train_start, request.train_end, request.test_start, request.test_end)]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="tournament dates must be valid ISO calendar dates") from exc
    train_start, train_end, test_start, test_end = dates
    if train_start >= train_end or test_start >= test_end:
        raise HTTPException(status_code=422, detail="each tournament period must have start before its exclusive end")
    if test_start <= train_end:
        raise HTTPException(status_code=422, detail="test period must start strictly after the training period ends")
    train_loader, train = _market_download(request.ticker, request.train_start, request.train_end, request.interval)
    _, test = _market_download(request.ticker, request.test_start, request.test_end, request.interval)
    try:
        result = run_model_tournament(ticker=train_loader.ticker, train=train, test=test,
            train_start=request.train_start, train_end=request.train_end,
            test_start=request.test_start, test_end=request.test_end, interval=request.interval,
            models=request.models, paths=request.paths, seed=request.seed)
    except (MarketDataError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["model_version"] = app.version
    if request.save and result["results"]:
        tournament_id = str(uuid.uuid4())
        result["tournament_id"] = tournament_id
        store = _experiment_store()
        params_by_model = {
            "gbm": ("drift", "volatility", "mean_period_log_return", "sample_std_period_log_return"),
            "ou": ("mean_level", "mean_reversion_speed", "volatility", "initial_value"),
            "jump_diffusion": ("drift", "volatility", "jump_intensity", "mean_log_jump", "jump_volatility"),
            "garch": ("mean", "omega", "alpha", "beta", "initial_variance"),
        }
        for item in result["results"]:
            fit = item.get("parameters") or {}
            model_parameters = {key: fit[key] for key in params_by_model[item["model"]] if key in fit}
            metadata = {
                "schema_version": 1, "application_version": app.version, "model_version": app.version,
                "experiment_type": "model_tournament", "tournament_id": tournament_id,
                "ticker": result["ticker"], "interval": request.interval,
                "training_period": result["training_period"], "test_period": result["test_period"],
                "parameter_source": "historical_calibration_training_only", "calibration_method": fit.get("method"),
                "assumptions": item["assumptions"], "warnings": item["warnings"], "limitation": item["limitation"],
                "paths": request.paths, "random_seed": request.seed, "reproducible_with_same_seed": True,
                "comparison_method": result["comparison_method"],
                "model_result_status": item["status"],
            }
            experiment_id = store.create(name=f"{(request.name or 'Tournament').strip()} · {item['model']}",
                description="Saved chronological model-tournament result.", model=item["model"],
                parameters=model_parameters, metadata=metadata)
            store.set_running(experiment_id)
            if item["status"] == "completed":
                saved = store.complete(experiment_id, {"tournament_metrics": item["metrics"],
                    "forecast_distribution_surface": item.get("forecast_distribution_surface"),
                    "forecast_risk_surface": item.get("forecast_risk_surface"),
                    "research_path": item.get("research_path")})
            else:
                store.fail(experiment_id, str(item.get("error") or "Model unavailable for this training and test configuration."))
                saved = store.get(experiment_id)
            item["experiment_id"] = saved["id"]
    return result


@app.get("/model-tournaments", response_model=SavedTournamentsResponse, tags=["evaluation"], summary="List saved model tournaments")
def list_model_tournaments():
    records = _experiment_store().list(500)
    grouped: dict[str, dict] = {}
    for record in records:
        metadata = record["metadata"]
        if metadata.get("experiment_type") != "model_tournament":
            continue
        tournament_id = metadata.get("tournament_id")
        if not isinstance(tournament_id, str):
            continue
        group = grouped.setdefault(tournament_id, {
            "tournament_id": tournament_id, "name": record["name"].rsplit(" · ", 1)[0],
            "ticker": metadata.get("ticker"), "interval": metadata.get("interval"),
            "training_period": metadata.get("training_period"), "test_period": metadata.get("test_period"),
            "created_at": record["created_at"], "models": [],
        })
        group["models"].append({"model": record["model"], "status": "unavailable" if record["status"] == "failed" else record["status"],
            "parameters": record["parameters"], "metrics": (record["metrics"] or {}).get("tournament_metrics"),
            "forecast_distribution_surface": (record["metrics"] or {}).get("forecast_distribution_surface"),
            "forecast_risk_surface": (record["metrics"] or {}).get("forecast_risk_surface"),
            "research_path": (record["metrics"] or {}).get("research_path"),
            "metadata": metadata, "experiment_id": record["id"], "error": record["error"]})
        if record["created_at"] < group["created_at"]:
            group["created_at"] = record["created_at"]
    return {"tournaments": sorted(grouped.values(), key=lambda item: item["created_at"], reverse=True)}


@app.post("/simulate-ou", tags=["simulation"], summary="Simulate an Ornstein-Uhlenbeck process")
def simulate_ou_endpoint(request: OURequest):
    """OU parameters are levels, annual reversion speed, and annual level volatility."""
    _check_simulation_size(request.paths, request.steps)
    try:
        times, levels = simulate_ou(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = _path_summary(levels)
    result.update({"model": "ornstein_uhlenbeck", "time_points": len(times),
                   "paths_simulated": request.paths, "seed": request.seed,
                   "terminal_risk": _horizon_risk({"available": False,
                       "unavailable_reason": "OU state-level paths do not define asset returns or losses."}, request),
                   "theoretical_terminal_mean": (request.mean_level + (request.initial_value - request.mean_level) * np.exp(-request.mean_reversion_speed * request.years)),
                   "theoretical_terminal_variance": (request.volatility**2 * request.years if request.mean_reversion_speed == 0 else request.volatility**2 * (1 - np.exp(-2 * request.mean_reversion_speed * request.years)) / (2 * request.mean_reversion_speed))})
    return result


@app.post("/simulate-ou/paths", response_model=SimulationTrajectoriesResponse,
          tags=["simulation"], summary="Return bounded OU state trajectories")
def simulate_ou_paths(request: OUTrajectoryRequest):
    _check_calibration_parameters(request, request.calibration,
                                  ("initial_value", "mean_level", "mean_reversion_speed", "volatility"))
    parameters = request.model_dump(exclude={"calibration", "retain_forensics"})
    if request.retain_forensics:
        times, levels, innovations = simulate_ou(**parameters, return_innovations=True)
    else:
        times, levels = simulate_ou(**parameters)
    response = _trajectory_response("ou", request, levels, times, "state", "OU state level", "Time (years)")
    if request.retain_forensics:
        response["forensics_run_id"] = _persist_path_forensics("ou", request, times, levels, innovations, None, response["terminal_risk"])
    return response


@app.post("/simulate-jump-diffusion", tags=["simulation"], summary="Simulate Merton jump-diffusion prices")
def simulate_jump_endpoint(request: JumpDiffusionRequest):
    """drift is compensated so expected price growth is exp(drift * years)."""
    _check_simulation_size(request.paths, request.steps)
    try:
        times, prices = simulate_jump_diffusion(**request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = _path_summary(prices)
    result.update({"model": "merton_jump_diffusion", "time_points": len(times),
                   "paths_simulated": request.paths, "seed": request.seed,
                   "terminal_risk": _horizon_risk(
                       terminal_return_loss_risk(prices[:, -1] / request.initial_price - 1.0), request),
                   "theoretical_terminal_mean": request.initial_price * np.exp(request.drift * request.years),
                   "diffusion_volatility": request.volatility})
    return result


@app.post("/simulate-jump-diffusion/paths", response_model=SimulationTrajectoriesResponse,
          tags=["simulation"], summary="Return bounded Merton price trajectories")
def simulate_jump_paths(request: JumpDiffusionTrajectoryRequest):
    _check_calibration_parameters(request, request.calibration,
                                  ("initial_price", "drift", "volatility", "jump_intensity", "mean_log_jump", "jump_volatility"))
    parameters = request.model_dump(exclude={"calibration", "retain_forensics"})
    if request.retain_forensics:
        times, prices, innovations = simulate_jump_diffusion(**parameters, return_innovations=True)
    else:
        times, prices = simulate_jump_diffusion(**parameters)
    response = _trajectory_response("jump_diffusion", request, prices, times, "price", "Simulated price", "Time (years)")
    if request.retain_forensics:
        response["forensics_run_id"] = _persist_path_forensics("jump_diffusion", request, times, prices, innovations, None, response["terminal_risk"])
    return response


@app.post("/simulate-garch", tags=["simulation"], summary="Simulate GARCH(1,1) returns")
def simulate_garch_endpoint(request: GARCHRequest):
    """GARCH parameters are per period; stationary variance requires alpha+beta < 1."""
    _check_simulation_size(request.paths, request.steps, price_path=request.initial_price is not None)
    try:
        returns, variances = simulate_garch(request.omega, request.alpha, request.beta, request.steps,
            request.paths, request.mean, request.initial_variance, request.seed, request.return_convention)
        prices = garch_returns_to_prices(returns, request.initial_price, request.return_convention) if request.initial_price is not None else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = return_diagnostics(returns, variances)
    result.update({"model": "garch_1_1", "paths_simulated": request.paths,
                   "steps": request.steps, "seed": request.seed,
                   "terminal_risk": _horizon_risk(
                       terminal_return_risk(returns, "return", request.return_convention), request),
                   "stationary_parameterization": request.alpha + request.beta < 1,
                   "return_convention": request.return_convention,
                   "return_mean": float(returns.mean()), "return_variance": float(returns.var()),
                   "mean_conditional_variance": float(variances.mean())})
    if prices is not None: result["price_terminal"] = _path_summary(prices)
    return result


@app.post("/simulate-garch/paths", response_model=SimulationTrajectoriesResponse,
          tags=["simulation"], summary="Return bounded GARCH return trajectories")
def simulate_garch_paths(request: GARCHTrajectoryRequest):
    _check_calibration_parameters(request, request.calibration,
                                  ("omega", "alpha", "beta", "mean", "initial_variance"))
    if request.retain_forensics:
        returns, variances, innovations = simulate_garch(request.omega, request.alpha, request.beta, request.steps,
            request.paths, request.mean, request.initial_variance, request.seed, request.return_convention, return_innovations=True)
    else:
        returns, variances = simulate_garch(request.omega, request.alpha, request.beta, request.steps,
            request.paths, request.mean, request.initial_variance, request.seed, request.return_convention)
    label = "Per-period log return" if request.return_convention == "log" else "Per-period simple return"
    times = np.arange(1, request.steps + 1, dtype=np.float64)
    response = _trajectory_response("garch", request, returns, times, "return", label,
                                     "Observation period", variances)
    if request.retain_forensics:
        response["forensics_run_id"] = _persist_path_forensics("garch", request, times, returns, innovations, variances, response["terminal_risk"])
    return response


@app.get("/path-forensics/{run_id}/paths/{path_id}", response_model=PathForensicsResponse,
         tags=["path forensics"], summary="Inspect and reconstruct one retained simulated path")
def get_path_forensics(run_id: str, path_id: int):
    try:
        uuid.UUID(run_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail="run_id must be a UUID") from exc
    try:
        record = _experiment_store().get_path_forensics(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="forensics run not found") from exc
    if path_id < 1:
        raise HTTPException(status_code=422, detail="path_id must be a positive 1-based path index")
    if record.get("model") not in {"gbm", "ou", "jump_diffusion", "garch"}:
        raise HTTPException(status_code=422, detail="stored forensics model is unsupported")
    trajectories = record.get("trajectories")
    innovations = record.get("innovations")
    times = record.get("times")
    if not isinstance(innovations, dict) or not innovations or not isinstance(times, list) or not isinstance(record.get("parameters"), dict) or not isinstance(record.get("model_version"), str):
        raise HTTPException(status_code=422, detail="stored path provenance is incomplete")
    if not isinstance(trajectories, list) or path_id > len(trajectories):
        raise HTTPException(status_code=404, detail="path_id is not present in this run")
    try:
        values = np.asarray(trajectories, dtype=np.float64)
        time_values = np.asarray(times, dtype=np.float64)
        innovation_arrays = {key: np.asarray(value, dtype=np.float64) for key, value in innovations.items()}
        if values.ndim != 2 or not np.isfinite(values).all() or time_values.ndim != 1 or not np.isfinite(time_values).all():
            raise ValueError("invalid stored trajectory dimensions or values")
        if any(array.ndim != 2 or array.shape[0] != values.shape[0] or not np.isfinite(array).all()
               for array in innovation_arrays.values()):
            raise ValueError("invalid stored innovation dimensions or values")
        stored_variances = record.get("conditional_variances")
        if stored_variances is not None:
            variance_values = np.asarray(stored_variances, dtype=np.float64)
            if variance_values.ndim != 2 or variance_values.shape[0] != values.shape[0] or not np.isfinite(variance_values).all():
                raise ValueError("invalid stored conditional-variance dimensions or values")
        path_index = path_id - 1
        reconstructed = reconstruct_path(record, path_index)
    except (TypeError, ValueError, KeyError, IndexError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=f"stored path provenance is invalid: {exc}") from exc
    model = record["model"]
    parameters = record["parameters"]
    initial_conditions = ({"initial_price": parameters["initial_price"]} if model in {"gbm", "jump_diffusion"}
                          else {"initial_value": parameters["initial_value"]} if model == "ou"
                          else {"initial_conditional_variance": parameters["initial_variance"], "initial_return_state": "none; returns are generated directly"})
    per_path_innovations = {key: array[path_index].astype(float).tolist() for key, array in innovation_arrays.items()}
    variances = record.get("conditional_variances")
    conditional_variance = variances[path_index] if variances is not None else None
    path_metrics = record.get("terminal_risk", {}).get("path_metrics") or []
    terminal_metrics = {"terminal_value": float(values[path_index, -1]),
                       "path_risk": path_metrics[path_index] if path_index < len(path_metrics) else None}
    return {"run_id": run_id, "path_id": path_id, "model": model, "model_version": record["model_version"],
            "created_at": record["created_at"], "parameters": parameters,
            "initial_conditions": initial_conditions, "times": time_values.astype(float).tolist(),
            "original_path": values[path_index].astype(float).tolist(), "innovations": per_path_innovations,
            "conditional_variance": conditional_variance,
            "transitions": reconstructed["transitions"], "terminal_metrics": terminal_metrics,
            "reconstruction": {"path": reconstructed["reconstructed_path"],
                               "maximum_absolute_error": reconstructed["maximum_absolute_error"],
                               "verification_tolerance": reconstructed["verification_tolerance"],
                               "verified": reconstructed["verified"],
                               "method": "Recomputed with the stored per-step innovations and the model transition equation."}}


@app.get("/path-forensics/runs", response_model=RetainedForensicsRunsResponse,
         tags=["path forensics"], summary="List recent retained runs available for path counterfactuals")
def list_path_forensics_runs():
    return {"runs": _experiment_store().list_path_forensics(25)}


@app.post("/path-forensics/counterfactual", response_model=PathCounterfactualResponse,
          tags=["path forensics"], summary="Replay one retained path under modified same-model parameters")
def run_path_counterfactual(request: PathCounterfactualRequest):
    try:
        uuid.UUID(request.run_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail="run_id must be a UUID") from exc
    try:
        record = _experiment_store().get_path_forensics(request.run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="reference forensics run not found") from exc
    model = record.get("model")
    schema_by_model = {"gbm": GBMTrajectoryRequest, "ou": OUTrajectoryRequest,
                       "jump_diffusion": JumpDiffusionTrajectoryRequest, "garch": GARCHTrajectoryRequest}
    parameters = record.get("parameters")
    trajectories = record.get("trajectories")
    innovation_record = record.get("innovations")
    if model not in schema_by_model or not isinstance(parameters, dict):
        raise HTTPException(status_code=422, detail="retained run has unsupported or incomplete model provenance")
    if not isinstance(trajectories, list) or request.path_id > len(trajectories):
        raise HTTPException(status_code=404, detail="reference path_id is not present in this run")
    if not isinstance(innovation_record, dict) or not innovation_record:
        raise HTTPException(status_code=422, detail="reference run did not retain the stochastic inputs required for counterfactual replay")
    mutable = {
        "gbm": {"initial_price", "drift", "volatility", "years", "steps"},
        "ou": {"initial_value", "mean_level", "mean_reversion_speed", "volatility", "years", "steps"},
        "jump_diffusion": {"initial_price", "drift", "volatility", "jump_intensity", "mean_log_jump", "jump_volatility", "years", "steps"},
        "garch": {"omega", "alpha", "beta", "mean", "initial_variance", "steps", "return_convention"},
    }[model]
    unknown = set(request.candidate_parameters) - mutable
    if unknown:
        raise HTTPException(status_code=422, detail=f"unsupported {model} counterfactual parameter(s): {', '.join(sorted(unknown))}")
    candidate_steps = request.candidate_parameters.get("steps", parameters.get("steps"))
    if candidate_steps != parameters.get("steps"):
        raise HTTPException(status_code=422, detail="step count must match the retained path; stored innovations cannot be split or merged without changing their distribution")
    source = {key: value for key, value in parameters.items() if key not in {"calibration", "retain_forensics"}}
    candidate_input = {**source, **request.candidate_parameters, "paths": 1,
                       "retain_forensics": False, "calibration": None}
    try:
        config = schema_by_model[model].model_validate(candidate_input)
        candidate_parameters = config.model_dump(mode="json", exclude={"retain_forensics", "calibration"})
        path_index = request.path_id - 1
        reference_path = np.asarray(trajectories[path_index], dtype=np.float64)
        if reference_path.ndim != 1 or len(reference_path) < 1 or not np.isfinite(reference_path).all():
            raise ValueError("stored reference path is invalid")
        stored = {key: np.asarray(value[path_index], dtype=np.float64) for key, value in innovation_record.items()}
        if not stored or any(values.ndim != 1 or not np.isfinite(values).all() for values in stored.values()):
            raise ValueError("stored path innovations are invalid")
    except (TypeError, ValueError, KeyError, IndexError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=f"counterfactual configuration or retained provenance is invalid: {exc}") from exc
    coupling_method = "Exact common random numbers: all stored per-step innovations are reused."
    warnings: list[str] = []
    exact_common_innovations = True
    if model == "jump_diffusion":
        baseline_intensity = float(parameters["jump_intensity"])
        candidate_intensity = float(candidate_parameters["jump_intensity"])
        baseline_years = float(parameters["years"])
        candidate_years = float(candidate_parameters["years"])
        baseline_dt = baseline_years / int(parameters["steps"])
        candidate_dt = candidate_years / int(candidate_parameters["steps"])
        counts = stored.get("poisson_jump_count")
        if counts is None:
            raise HTTPException(status_code=422, detail="Merton reference is missing stored Poisson jump counts")
        if candidate_intensity != baseline_intensity or candidate_dt != baseline_dt:
            if candidate_intensity * candidate_dt == baseline_intensity * baseline_dt:
                stored["poisson_jump_count"] = np.rint(counts).astype(float)
                coupling_method = "Exact per-step Merton count coupling: λcandidate·Δtcandidate equals λreference·Δtreference, so stored Poisson counts and both Gaussian innovation arrays are reused unchanged."
            else:
                exact_common_innovations = False
                canonical = json.dumps({"run_id": request.run_id, "path_id": request.path_id,
                    "candidate": candidate_parameters}, sort_keys=True, separators=(",", ":"))
                coupling_seed = int.from_bytes(hashlib.sha256(canonical.encode()).digest()[:8], "little")
                rng = np.random.default_rng(coupling_seed)
                if baseline_intensity > 0 and candidate_intensity * candidate_dt < baseline_intensity * baseline_dt:
                    probability = (candidate_intensity * candidate_dt) / (baseline_intensity * baseline_dt)
                    stored["poisson_jump_count"] = rng.binomial(np.rint(counts).astype(int), probability).astype(float)
                    coupling_method = "Merton event thinning: stored reference Poisson events are retained independently with probability λcandidate·Δtcandidate/(λreference·Δtreference); diffusion and aggregate jump-size normals are reused. Thinning draws are deterministic for this run, path, and configuration."
                elif baseline_intensity > 0:
                    extra_mean = candidate_intensity * candidate_dt - baseline_intensity * baseline_dt
                    stored["poisson_jump_count"] = (np.rint(counts).astype(int) + rng.poisson(extra_mean, size=len(counts))).astype(float)
                    coupling_method = "Merton event superposition: stored reference Poisson events are kept and independent excess Poisson events are added to reach the candidate λ·Δt; diffusion and aggregate jump-size normals are reused. Added-event draws are deterministic for this run, path, and configuration."
                else:
                    stored["poisson_jump_count"] = rng.poisson(candidate_intensity * candidate_dt, size=len(counts)).astype(float)
                    coupling_method = "Merton independent candidate arrivals: the reference intensity is zero, so candidate Poisson counts are drawn independently; stored diffusion and aggregate jump-size normals are reused. Counts are deterministic for this run, path, and configuration."
                warnings.append("Changing Merton jump intensity or time-step duration changes the event count. Poisson thinning/superposition preserves candidate count marginals, but the jump-event coupling is a counterfactual construction, not identical events or a forecast.")
        else:
            stored["poisson_jump_count"] = np.rint(counts).astype(float)
    try:
        candidate_times = (np.arange(int(candidate_parameters["steps"]), dtype=np.float64) + 1
                           if model == "garch" else np.linspace(0.0, float(candidate_parameters["years"]), int(candidate_parameters["steps"]) + 1))
        synthetic = {"model": model, "parameters": candidate_parameters, "times": candidate_times.tolist(),
                     "trajectories": [reference_path.tolist()], "innovations": {key: [value.tolist()] for key, value in stored.items()}}
        candidate_path = np.asarray(reconstruct_path(synthetic, 0)["reconstructed_path"], dtype=np.float64)
        if candidate_path.shape != reference_path.shape:
            raise ValueError("counterfactual and reference path dimensions differ")
        if not np.isfinite(candidate_path).all():
            raise ValueError("counterfactual path contains non-finite values")
        reference_return, reference_drawdown = simple_return_and_drawdown(model, reference_path, parameters.get("return_convention"))
        candidate_return, candidate_drawdown = simple_return_and_drawdown(model, candidate_path, candidate_parameters.get("return_convention"))
        reference_metrics = {"initial_value": float(reference_path[0]), "terminal_value": float(reference_path[-1]),
                             "total_simple_return": reference_return, "maximum_drawdown": reference_drawdown}
        candidate_metrics = {"initial_value": float(candidate_path[0]), "terminal_value": float(candidate_path[-1]),
                             "total_simple_return": candidate_return, "maximum_drawdown": candidate_drawdown}
    except (ValueError, KeyError, IndexError, OverflowError, FloatingPointError) as exc:
        raise HTTPException(status_code=422, detail=f"counterfactual could not be computed: {exc}") from exc
    if model == "ou":
        warnings.append("OU is a state process; total asset return and wealth drawdown are undefined for this path.")
    if model == "garch":
        warnings.append("GARCH output is per-period returns; drawdown and total return are derived only when the configured return convention yields positive cumulative wealth.")
    return {"run_id": request.run_id, "path_id": request.path_id, "model": model,
        "parameters": {key: candidate_parameters[key] for key in mutable if key in candidate_parameters},
        "times": candidate_times.astype(float).tolist(),
        "reference_path": reference_path.astype(float).tolist(), "candidate_path": candidate_path.astype(float).tolist(),
        "differences": (candidate_path - reference_path).astype(float).tolist(),
        "reference_metrics": reference_metrics, "candidate_metrics": candidate_metrics,
        "raw_path_rmse": float(np.sqrt(np.mean((candidate_path - reference_path) ** 2))),
        "exact_common_innovations": exact_common_innovations, "coupling_method": coupling_method,
        "warnings": warnings, "transitions": reconstruct_path(synthetic, 0)["transitions"],
        "innovations": {key: value.astype(float).tolist() for key, value in stored.items()}}


@app.post("/path-forensics/scenario-matches", response_model=ScenarioMatchingResponse,
          tags=["path forensics"], summary="Rank bounded parameter variations against a retained reference path")
def match_path_scenarios(request: ScenarioMatchingRequest):
    try:
        uuid.UUID(request.run_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail="run_id must be a UUID") from exc
    try:
        record = _experiment_store().get_path_forensics(request.run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="reference forensics run not found") from exc
    model = record.get("model")
    schema_by_model = {"gbm": GBMTrajectoryRequest, "ou": OUTrajectoryRequest,
                       "jump_diffusion": JumpDiffusionTrajectoryRequest, "garch": GARCHTrajectoryRequest}
    override_names = {
        "gbm": {"initial_price", "drift", "volatility"},
        "ou": {"initial_value", "mean_level", "mean_reversion_speed", "volatility"},
        "jump_diffusion": {"initial_price", "drift", "volatility", "jump_intensity", "mean_log_jump", "jump_volatility"},
        "garch": {"omega", "alpha", "beta", "mean", "initial_variance"},
    }
    if model not in schema_by_model:
        raise HTTPException(status_code=422, detail="reference model does not support scenario matching")
    trajectories = record.get("trajectories")
    if not isinstance(trajectories, list) or request.path_id > len(trajectories):
        raise HTTPException(status_code=404, detail="reference path_id is not present in this run")
    path_index = request.path_id - 1
    target = np.asarray(trajectories[path_index], dtype=np.float64)
    innovation_record = record.get("innovations", {})
    if not isinstance(innovation_record, dict) or not innovation_record:
        raise HTTPException(status_code=422, detail="reference run is missing usable path innovations")
    try:
        path_innovations = {key: [values[path_index]] for key, values in innovation_record.items()}
    except (TypeError, IndexError, KeyError) as exc:
        raise HTTPException(status_code=422, detail="reference run is missing usable path innovations") from exc
    source_parameters = dict(record["parameters"])
    source_parameters.pop("calibration", None)
    source_parameters["retain_forensics"] = False
    target_return, target_drawdown = simple_return_and_drawdown(
        model, target, source_parameters.get("return_convention"))
    ranked: list[dict] = []
    for candidate_index, overrides in enumerate(request.candidate_parameters):
        unknown = set(overrides) - override_names[model]
        if unknown:
            raise HTTPException(status_code=422, detail=f"unsupported {model} scenario parameter(s): {', '.join(sorted(unknown))}")
        candidate_request = {**source_parameters, **overrides, "calibration": None, "paths": 1,
                             "retain_forensics": False}
        try:
            config = schema_by_model[model].model_validate(candidate_request)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"invalid scenario candidate {candidate_index + 1}: {exc}") from exc
        candidate_parameters = config.model_dump(mode="json", exclude={"retain_forensics", "calibration"})
        synthetic_record = {"model": model, "parameters": candidate_parameters, "times": record["times"],
                            "trajectories": [target.tolist()], "innovations": path_innovations}
        try:
            reconstruction = reconstruct_path(synthetic_record, 0)
            candidate_path = reconstruction["reconstructed_path"]
            if len(candidate_path) != len(target):
                raise ValueError("candidate and reference path dimensions differ")
            path_rmse = normalized_path_rmse(target, candidate_path)
            candidate_return, candidate_drawdown = simple_return_and_drawdown(
                model, candidate_path, candidate_parameters.get("return_convention"))
            objective, return_difference, drawdown_difference = match_objective(
                path_rmse, target_return, candidate_return, target_drawdown, candidate_drawdown,
                request.terminal_return_weight, request.drawdown_weight)
        except (ValueError, KeyError, IndexError, OverflowError) as exc:
            raise HTTPException(status_code=422, detail=f"scenario candidate {candidate_index + 1} could not be evaluated: {exc}") from exc
        ranked.append({"rank": 0, "model": model, "parameters": candidate_parameters,
            "objective": objective, "normalized_path_rmse": path_rmse,
            "terminal_return_difference": return_difference, "drawdown_difference": drawdown_difference,
            "terminal_value": float(candidate_path[-1]), "total_simple_return": candidate_return,
            "maximum_drawdown": candidate_drawdown, "candidate_path": candidate_path,
            "exactly_replayable": bool(np.isfinite(candidate_path).all()),
            "replay_provenance": {"reference_run_id": request.run_id, "reference_path_id": request.path_id,
                "model_version": record["model_version"], "same_stored_innovations": True,
                "random_seed_alone_used": False, "reference_calibration": record["parameters"].get("calibration")}})
    ranked.sort(key=lambda item: item["objective"])
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return {"reference_run_id": request.run_id, "reference_path_id": request.path_id,
        "method": "Bounded same-innovation counterfactual search. Path RMSE is normalized by max(reference SD, 1% of reference scale, 1e-8); optional terminal simple-return and maximum-drawdown absolute differences are weighted and added.",
        "warnings": ["This is an in-sample match to the selected simulated reference path, not an out-of-sample forecast.",
                     "Candidates reuse the reference path's stored innovations to isolate parameter changes; matches are not unique causal explanations."],
        "candidates": ranked}


def _experiment_store() -> ExperimentStore:
    return ExperimentStore(os.environ.get("QML_EXPERIMENT_DB", "data/experiments.sqlite3"))


def _experiment_metrics(result: dict) -> dict:
    """Persist compact diagnostics while leaving large trajectories transient."""
    summary = result["summary"]
    compact_summary = {key: summary[key] for key in ("terminal_mean", "terminal_variance", "terminal_quantiles")
                       if key in summary}
    risk = dict(result["terminal_risk"])
    path_metrics = risk.pop("path_metrics", None)
    if path_metrics:
        total_returns = np.asarray([metric["total_simple_return"] for metric in path_metrics], dtype=float)
        risk["terminal_simple_return_std"] = float(total_returns.std(ddof=1) if len(total_returns) > 1 else 0.0)
    output: dict = {"quantity_kind": result["quantity_kind"], "summary": compact_summary, "terminal_risk": risk}
    if result.get("conditional_variances") is not None:
        variances = np.asarray(result["conditional_variances"], dtype=float)
        output["conditional_variance_summary"] = {
            "mean": float(variances.mean()), "terminal_mean": float(variances[:, -1].mean()),
            "terminal_quantiles": {str(q): float(np.quantile(variances[:, -1], q)) for q in (.05, .5, .95)},
        }
    return output


@app.post("/experiments", response_model=ExperimentRunResponse, status_code=201,
          tags=["experiments"], summary="Run and persist a reproducible simulation experiment")
def create_experiment(request: ExperimentCreateRequest):
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="experiment name cannot be blank")
    schemas = {"gbm": GBMTrajectoryRequest, "ou": OUTrajectoryRequest,
               "jump_diffusion": JumpDiffusionTrajectoryRequest, "garch": GARCHTrajectoryRequest}
    runners = {"gbm": simulate_gbm_paths, "ou": simulate_ou_paths,
               "jump_diffusion": simulate_jump_paths, "garch": simulate_garch_paths}
    schema = schemas[request.model]
    allowed = set(schema.model_fields) - {"retain_forensics"}
    unknown = set(request.parameters) - allowed
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown {request.model} parameter(s): {', '.join(sorted(unknown))}")
    try:
        configuration = schema.model_validate(request.parameters)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    effective_parameters = configuration.model_dump(mode="json", exclude={"retain_forensics"})
    specified_parameters = sorted(configuration.model_fields_set - {"retain_forensics"})
    derived_parameters: list[str] = []
    if request.model == "garch" and configuration.initial_variance is None:
        persistence = configuration.alpha + configuration.beta
        effective_parameters["initial_variance"] = (configuration.omega / (1 - persistence)
                                                       if persistence < 1 else configuration.omega)
        derived_parameters.append("initial_variance")
    defaulted_parameters = sorted(allowed - set(specified_parameters) - set(derived_parameters) - {"calibration"})
    calibration = getattr(configuration, "calibration", None)
    if calibration and request.ticker and request.ticker.upper() != calibration.ticker.upper():
        raise HTTPException(status_code=422, detail="experiment ticker must match its model-specific calibration provenance")
    ticker = request.ticker.upper() if request.ticker else (calibration.ticker.upper() if calibration else None)
    years = getattr(configuration, "years", None)
    metadata = {
        "schema_version": 1, "application_version": app.version, "model_version": app.version,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "parameter_source": "historical_calibration" if calibration else "manual_or_default",
        "user_specified_parameters": specified_parameters,
        "manually_specified_parameters": [key for key in specified_parameters if key != "calibration"],
        "defaulted_parameters": defaulted_parameters,
        "derived_parameters": derived_parameters,
        "calibration": calibration.model_dump(mode="json") if calibration else None,
        "ticker": ticker,
        "observation_frequency": getattr(calibration, "interval", None),
        "parameter_time_unit": "per_observation_period" if request.model == "garch" else "per_year",
        "annualization_convention": ("GARCH parameters and mean are per observation; no annualization is implied."
                                      if request.model == "garch" else
                                      "Continuous-time parameters are annualized; simulation dt = years / steps."),
        "horizon": {"steps": configuration.steps,
                    "unit": "years" if years is not None else "observation_periods",
                    "value": float(years) if years is not None else int(configuration.steps)},
        "path_count": configuration.paths,
        "random_seed": configuration.seed,
        "reproducible_with_same_seed": configuration.seed is not None,
        "reproducibility_scope": "Same model code, NumPy version, parameters, seed, and runtime assumptions." if configuration.seed is not None else "No seed supplied; this run is not guaranteed to reproduce.",
        "quantity_kind": "state" if request.model == "ou" else ("return" if request.model == "garch" else "price"),
        "quantity_unit": ((f"price:{ticker}" if calibration.input_series == "price" and ticker else calibration.input_series)
                          if request.model == "ou" and calibration else
                          ("unspecified_ou_state" if request.model == "ou" else "simple_return_fraction")),
    }
    store = _experiment_store()
    experiment_id = store.create(name=name, description=request.description,
                                 model=request.model, parameters=effective_parameters, metadata=metadata)
    store.set_running(experiment_id)
    try:
        result = runners[request.model](configuration)
        record = store.complete(experiment_id, _experiment_metrics(result))
        return {"experiment": record, "simulation": result}
    except HTTPException as exc:
        store.fail(experiment_id, str(exc.detail))
        raise
    except Exception as exc:
        store.fail(experiment_id, str(exc))
        raise HTTPException(status_code=500, detail=f"Experiment simulation failed: {exc}") from exc


@app.get("/experiments", response_model=ExperimentListResponse, tags=["experiments"],
         summary="List saved experiments")
def list_experiments(limit: int = Query(100, ge=1, le=500)):
    experiments = _experiment_store().list(limit)
    return {"experiments": experiments, "count": len(experiments)}


def _read_experiment(experiment_id: str) -> dict:
    try:
        uuid.UUID(experiment_id)
    except (ValueError, TypeError, AttributeError) as exc:
        raise HTTPException(status_code=422, detail="experiment_id must be a UUID") from exc
    try:
        return _experiment_store().get(experiment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="experiment not found") from exc


@app.get("/experiments/compare", response_model=ExperimentComparisonResponse,
         tags=["experiments"], summary="Compare saved experiments")
def compare_experiments(experiment_ids: list[str] = Query(..., min_length=2, max_length=10)):
    if len(set(experiment_ids)) != len(experiment_ids):
        raise HTTPException(status_code=422, detail="experiment_ids must be unique")
    records = [_read_experiment(experiment_id) for experiment_id in experiment_ids]
    completed = all(record["status"] == "completed" for record in records)
    first_meta = records[0]["metadata"]
    first_kind = first_meta.get("quantity_kind")
    first_horizon = first_meta.get("horizon")
    first_unit = first_meta.get("quantity_unit")
    compatible = all(record["metadata"].get("quantity_kind") == first_kind
                     and record["metadata"].get("horizon") == first_horizon
                     and record["metadata"].get("quantity_unit") == first_unit for record in records)
    reason = None
    common: dict[str, dict[str, float | None]] = {}
    if not completed:
        reason = "Only completed experiments can be compared."
    elif not compatible:
        reason = "Experiments have different quantities, units, or horizons; numerical metrics are not directly comparable."
    elif first_kind == "state" and first_unit == "unspecified_ou_state":
        compatible = False
        reason = "OU state units were not specified, so their numerical metrics cannot be compared safely."
    else:
        metric_values: dict[str, dict[str, float | None]] = {}
        for record in records:
            metrics = record["metrics"] or {}
            risk = metrics.get("terminal_risk", {})
            candidates: dict[str, Any] = {}
            summary = metrics.get("summary", {})
            if first_kind == "state":
                candidates = {"terminal_state_mean": summary.get("terminal_mean"),
                              "terminal_state_variance": summary.get("terminal_variance")}
            else:
                for key in ("mean_total_simple_return", "terminal_simple_return_std", "var",
                            "expected_shortfall"):
                    candidates[key] = risk.get(key)
                candidates.update({f"maximum_drawdown_p{label}": risk.get("maximum_drawdown_quantiles", {}).get(key)
                                   for label, key in ((5, "0.05"), (50, "0.5"), (95, "0.95"))})
            for key, value in candidates.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    metric_values.setdefault(key, {})[record["id"]] = float(value)
        common = {key: values for key, values in metric_values.items() if len(values) == len(records)}
    return {"experiments": records, "comparable": bool(completed and compatible),
            "comparison_reason": reason, "common_metrics": common}


@app.get("/experiments/{experiment_id}", response_model=ExperimentRecord,
         tags=["experiments"], summary="Retrieve an experiment")
def get_experiment(experiment_id: str):
    return _read_experiment(experiment_id)


@app.get("/experiments/{experiment_id}/summary", tags=["experiments"],
         summary="Retrieve saved experiment metrics")
def get_experiment_summary(experiment_id: str):
    record = _read_experiment(experiment_id)
    return {"id": record["id"], "status": record["status"], "metrics": record["metrics"], "error": record["error"]}


@app.delete("/experiments/{experiment_id}", status_code=204, tags=["experiments"],
            summary="Delete a saved experiment")
def delete_experiment(experiment_id: str):
    record = _read_experiment(experiment_id)
    if record["status"] in {"pending", "running"}:
        raise HTTPException(status_code=409, detail="running experiments cannot be deleted")
    _experiment_store().delete(experiment_id)
    return None


@app.get("/market-data/validate/{ticker}", tags=["market data"], summary="Download and validate historical data")
def market_data_validate(ticker: str, start: str | None = None, end: str | None = None, interval: str = "1d"):
    """Yahoo Finance access may require internet and is subject to provider limits."""
    _, frame = _market_download(ticker, start, end, interval)
    return validate_market_data(frame).__dict__


@app.get("/market-data/summary/{ticker}", response_model=MarketSummaryResponse, tags=["market data"], summary="Summarize historical market data")
def market_data_summary(ticker: str, start: str | None = None, end: str | None = None, interval: str = "1d"):
    """Download history and return compact descriptive risk statistics."""
    loader, frame = _market_download(ticker, start, end, interval)
    returns = compute_log_returns(frame)
    stats = summarize_returns(returns)
    adjusted = frame["adj_close"] if "adj_close" in frame else None
    if adjusted is None or adjusted.isna().any() or not np.isfinite(adjusted.to_numpy(dtype=float)).all() or (adjusted <= 0).any():
        adjusted = frame["close"]
        drawdown_price_field = "close"
    else:
        drawdown_price_field = "adj_close"
    _, maximum_dd = drawdown(adjusted)
    periods_per_year = {"1m": 98_280, "2m": 49_140, "5m": 19_656, "15m": 6_552,
                        "30m": 3_276, "60m": 1_638, "90m": 1_092, "1h": 1_638,
                        "1d": 252, "5d": 52, "1wk": 52, "1mo": 12, "3mo": 4}.get(interval, 252)
    with np.errstate(over="ignore", invalid="ignore"):
        simple_returns = np.expm1(returns.dropna().to_numpy(dtype=np.float64))
    historical_risk = None
    risk_unavailable_reason = None
    if len(simple_returns) and np.isfinite(simple_returns).all():
        empirical = empirical_var_es(-simple_returns, .95)
        historical_risk = {**empirical, "measure": "historical_one_period", "interval": interval,
                           "loss_definition": "L = -R, where R is the close-to-close simple return for one observation interval",
                           "warnings": ["Small historical sample; empirical tail estimate is coarse."] if len(simple_returns) < 20 else []}
    elif len(simple_returns):
        risk_unavailable_reason = "At least one historical simple return is outside finite numeric range; loss risk is unavailable."
    else:
        risk_unavailable_reason = "At least two valid historical close observations are required to estimate one-period risk."
    return {"ticker": loader.ticker, "row_count": len(frame), "first_date": str(frame.index.min().date()),
            "last_date": str(frame.index.max().date()), "latest_close": float(frame["close"].iloc[-1]),
            "mean_daily_log_return": stats["mean"], "daily_volatility": stats["standard_deviation"],
            "annualized_volatility": (float(stats["standard_deviation"] * np.sqrt(periods_per_year)) if stats["standard_deviation"] is not None else None),
            "interval": interval, "annualization_periods": periods_per_year,
            "latest_timestamp": frame.index[-1].isoformat(),
            "maximum_drawdown": maximum_dd, "historical_period_risk": historical_risk,
            "drawdown_price_field": drawdown_price_field,
            "historical_period_risk_unavailable_reason": risk_unavailable_reason}


@app.get("/market-data/prices/{ticker}", tags=["market data"], summary="Get bounded historical price observations")
def market_data_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    price_column: str = Query("adj_close", pattern=r"^(adj_close|close)$"),
    max_points: int = Query(1500, ge=2, le=5000),
):
    """Return at most max_points actual prices, evenly sampled from the requested range."""
    _, frame = _market_download(ticker, start, end, interval)
    selected_column = price_column
    if selected_column not in frame.columns and selected_column == "adj_close":
        selected_column = "close"
    if selected_column not in frame.columns:
        raise HTTPException(status_code=422, detail=f"requested price field '{price_column}' is unavailable")
    series = frame[selected_column].dropna()
    if len(series) > max_points:
        positions = np.linspace(0, len(series) - 1, max_points, dtype=int)
        series = series.iloc[np.unique(positions)]
    return {"ticker": ticker.upper(), "price_column": selected_column, "interval": interval,
            "observation_count": len(series), "source_observation_count": int(frame[selected_column].notna().sum()),
            "downsampled": bool(frame[selected_column].notna().sum() > len(series)),
            "observations": [{"date": index.date().isoformat() if index == index.normalize() else index.isoformat(), "price": float(value)} for index, value in series.items()]}


@app.get("/market-data/ohlcv/{ticker}", response_model=OHLCVSeriesResponse, tags=["market data"], summary="Get bounded historical OHLCV bars")
def market_data_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    max_points: int = Query(1500, ge=2, le=5000),
):
    """Return real OHLCV bars, aggregating consecutive source bars if bounded."""
    loader, frame = _market_download(ticker, start, end, interval)
    columns = ["open", "high", "low", "close", "volume"]
    available = frame[columns].dropna(subset=["open", "high", "low", "close"])
    source_count = len(available)
    downsampled = source_count > max_points
    if downsampled:
        # Aggregate contiguous time buckets so each candle preserves the actual
        # open, high, low, close, and summed volume of its source observations.
        bucket_ids = np.minimum(np.arange(source_count) * max_points // source_count, max_points - 1)
        grouped = []
        for bucket in np.unique(bucket_ids):
            part = available.iloc[bucket_ids == bucket]
            observed_volume = part["volume"].dropna()
            grouped.append({
                "date": part.index[-1].date().isoformat() if part.index[-1] == part.index[-1].normalize() else part.index[-1].isoformat(), "open": float(part["open"].iloc[0]),
                "high": float(part["high"].max()), "low": float(part["low"].min()),
                "close": float(part["close"].iloc[-1]),
                "volume": float(observed_volume.sum()) if len(observed_volume) else None,
            })
    else:
        grouped = [{"date": index.date().isoformat() if index == index.normalize() else index.isoformat(), "open": float(row.open), "high": float(row.high),
                    "low": float(row.low), "close": float(row.close),
                    "volume": float(row.volume) if pd.notna(row.volume) else None}
                   for index, row in available.iterrows()]
    return {"ticker": loader.ticker, "interval": interval, "observation_count": len(grouped),
            "source_observation_count": source_count, "downsampled": downsampled,
            "timestamp_convention": "UTC" if interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"} else "provider_session_date",
            "observations": grouped}


@app.post("/market-events/analyze", tags=["market event intelligence"],
          summary="Detect historical market movements and calculate descriptive impacts")
def market_event_analysis(request: MarketEventAnalysisRequest):
    if request.start >= request.end:
        raise HTTPException(status_code=422, detail="start must be earlier than end")
    cache = EventAnalysisCache(os.getenv("MARKET_EVENT_ANALYSIS_CACHE", "data/event_analysis_cache.sqlite3"))
    request_key = request.model_dump()
    cached = cache.get(request_key)
    if cached is not None:
        cached["analysis_cache"] = "hit"
        return cached
    loader, frame = _market_download(request.ticker, request.start, request.end, request.interval)
    thresholds = EventThresholds(
        absolute_log_return=request.absolute_log_return_threshold,
        volume_multiple=request.volume_multiple_threshold,
        opening_gap=request.opening_gap_threshold,
        volatility_multiple=request.volatility_multiple_threshold,
        sustained_log_return=request.sustained_log_return_threshold,
        sustained_periods=request.sustained_periods,
    )
    benchmark_frame = None
    benchmark_warning = None
    if request.benchmark:
        if request.benchmark.upper() == loader.ticker:
            benchmark_warning = "Benchmark matches the selected instrument; benchmark-relative analysis is not meaningful."
        else:
            try:
                _, benchmark_frame = _market_download(request.benchmark, request.start, request.end, request.interval)
            except (HTTPException, ValueError) as exc:
                benchmark_warning = f"Benchmark series unavailable; relative returns omitted: {getattr(exc, 'detail', str(exc))}"
    try:
        result = analyze_historical_events(frame, ticker=loader.ticker, start=request.start, end=request.end,
                                           interval=request.interval, thresholds=thresholds,
                                           impact_window=request.impact_window_observations, benchmark=benchmark_frame)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    source_count = result["source_observation_count"]
    bars = result["bars"]
    max_points = 2500
    if len(bars) > max_points:
        positions = np.linspace(0, len(bars) - 1, max_points, dtype=int)
        result["bars"] = [bars[index] for index in np.unique(positions)]
    result["source_observation_count"] = source_count
    result["downsampled"] = result["observation_count"] > len(result["bars"])
    result["chart_note"] = "Historical chart bars are evenly sampled for display when necessary; event detection and impact calculations use every valid source observation."
    result["benchmark_ticker"] = request.benchmark.upper() if request.benchmark else None
    if benchmark_warning:
        result["warnings"].append(benchmark_warning)
    result["analysis_cache"] = "miss"
    cache.put(request_key, result)
    return result


@app.get("/market-events/news/provider", response_model=NewsProviderStatusResponse, tags=["market event intelligence"],
         summary="Report historical-news provider configuration")
def market_event_news_provider():
    provider = FinnhubNewsProvider()
    return {"provider": provider.name, "configured": bool(provider.api_key),
            "setup_message": None if provider.api_key else "Historical news is not configured. Set FINNHUB_API_KEY on the backend and restart the API. No news search has been performed.",
            "credentials_location": "backend environment only", "historical_coverage": "Finnhub company-news supports North American companies; historical depth and access depend on the account plan and may not include all articles."}


@app.post("/market-events/news", response_model=MarketEventNewsResponse, tags=["market event intelligence"],
          summary="Retrieve timestamped historical news and rank it against candidate movements")
def market_event_news(request: MarketEventNewsRequest):
    if request.start >= request.end:
        raise HTTPException(status_code=422, detail="start must be earlier than end")
    try:
        date.fromisoformat(request.start)
        date.fromisoformat(request.end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="start and end must be valid ISO dates") from exc
    provider = FinnhubNewsProvider()
    if not provider.api_key:
        return {"provider": provider.name, "configured": False, "searched": False, "cached": False, "articles": [], "event_matches": [],
                "message": "Historical news is not configured. Set FINNHUB_API_KEY on the backend and restart the API. No news search was performed.",
                "retrieved_at": None}
    query = NewsQuery(request.ticker.upper(), request.start, request.end)
    try:
        cache = NewsResponseCache(os.getenv("MARKET_EVENT_NEWS_CACHE", "data/news_cache.sqlite3"))
        articles, provider_name, cached = fetch_historical_news(query, provider, cache)
    except NewsProviderError as exc:
        # Preserve rate limiting as such; credential/coverage and upstream errors remain gateway failures.
        status_code = 429 if exc.http_status == 429 else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    events = [event.model_dump() for event in request.events]
    matches = rank_news_for_events(articles, events, request.ticker.upper())
    diagnostics = getattr(provider, "last_diagnostics", {})
    message = None
    if not articles:
        raw_count = diagnostics.get("raw_article_count")
        if cached:
            message = "Finnhub returned no usable articles for this exact query earlier; this empty response came from the response cache."
        elif isinstance(raw_count, int) and raw_count > 0:
            message = (f"Finnhub returned HTTP {diagnostics.get('http_status')} with {raw_count} item(s), but none had the required headline and URL fields; "
                       f"{diagnostics.get('rejected_item_count')} item(s) were excluded during normalization.")
        else:
            message = (f"Finnhub returned an empty company-news list (HTTP {diagnostics.get('http_status', 'unknown')}) for {query.ticker} "
                       f"from {query.start} through {query.end} (application end is exclusive). Check symbol coverage, date coverage, and account plan.")
    return {"provider": provider_name, "configured": True, "searched": not cached, "cached": cached,
            "articles": articles, "event_matches": matches, "message": message,
            "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "publication_timestamps_are_distinct_from_retrieval_timestamps": True,
            "matching_note": "Ranking signals are heuristic evidence relevance scores; temporal association does not establish causation. Daily bars do not support reliable pre-market/intraday/after-market classification."}


@app.post("/market-data/download", tags=["market data"], summary="Download and persist market history")
def market_data_download(request: MarketDataDownloadRequest):
    """Download, validate, and save history; Yahoo Finance requires network access."""
    loader, frame = _market_download(request.ticker, request.start, request.end, request.interval)
    store = MarketDataStore()
    try:
        metadata = store.save(loader.ticker, frame, loader.metadata(frame), overwrite=False)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, MarketDataError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return metadata
    MertonCalibrationRequest,
    MertonCalibrationResponse,
    OUCalibrationRequest,
    OUCalibrationResponse,
