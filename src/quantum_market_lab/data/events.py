"""Historical market-event detection and descriptive event-window metrics."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import sqlite3
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from quantum_market_lab.data.market_data import MarketDataError
from quantum_market_lab.models.calibration import calibrate_garch, calibrate_merton, estimate_gbm


@dataclass(frozen=True)
class EventThresholds:
    absolute_log_return: float = 0.04
    volume_multiple: float = 2.5
    opening_gap: float = 0.03
    volatility_multiple: float = 2.0
    sustained_log_return: float = 0.08
    sustained_periods: int = 5
    volume_lookback: int = 20
    volatility_lookback: int = 20


class EventAnalysisCache:
    """Short-lived persistent cache for identical analysis requests."""
    def __init__(self, path: str | os.PathLike[str] = "data/event_analysis_cache.sqlite3", ttl_seconds: int = 900):
        self.path = os.fspath(path)
        self.ttl_seconds = ttl_seconds

    def _key(self, request: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def get(self, request: dict[str, Any]) -> dict[str, Any] | None:
        if not os.path.exists(self.path):
            return None
        try:
            with sqlite3.connect(self.path) as db:
                db.execute("CREATE TABLE IF NOT EXISTS event_analysis (cache_key TEXT PRIMARY KEY, stored_at REAL NOT NULL, payload TEXT NOT NULL)")
                row = db.execute("SELECT stored_at,payload FROM event_analysis WHERE cache_key=?", (self._key(request),)).fetchone()
        except sqlite3.Error:
            return None
        if row is None or time.time() - row[0] > self.ttl_seconds:
            return None
        try:
            value = json.loads(row[1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None

    def put(self, request: dict[str, Any], result: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        try:
            with sqlite3.connect(self.path) as db:
                db.execute("CREATE TABLE IF NOT EXISTS event_analysis (cache_key TEXT PRIMARY KEY, stored_at REAL NOT NULL, payload TEXT NOT NULL)")
                db.execute("INSERT OR REPLACE INTO event_analysis(cache_key,stored_at,payload) VALUES(?,?,?)",
                           (self._key(request), time.time(), json.dumps(result, separators=(",", ":"), allow_nan=False)))
                db.commit()
        except sqlite3.Error:
            return


def _valid_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.isna().any():
        raise ValueError("Historical event analysis requires a valid datetime index")
    missing = [name for name in ("open", "high", "low", "close", "volume") if name not in frame.columns]
    if missing:
        raise ValueError(f"Historical OHLCV is missing columns: {', '.join(missing)}")
    result = frame.copy().sort_index()
    source_observation_count = len(result)
    if result.index.has_duplicates:
        raise ValueError("Historical OHLCV contains duplicate timestamps")
    for name in ("open", "high", "low", "close", "volume"):
        result[name] = pd.to_numeric(result[name], errors="coerce")
    price_valid = np.isfinite(result[["open", "high", "low", "close"]]).all(axis=1) & (result[["open", "high", "low", "close"]] > 0).all(axis=1)
    valid = price_valid
    warnings: list[str] = []
    if (~valid).any():
        warnings.append(f"{int((~valid).sum())} observations with missing or invalid OHLC prices were excluded; returns crossing an excluded bar are suppressed.")
    original_positions = np.arange(len(result))
    kept_positions = original_positions[np.asarray(valid)]
    result = result.loc[valid].copy()
    result.attrs["event_source_observation_count"] = source_observation_count
    if len(result) < 3:
        raise ValueError("At least three valid OHLC observations are required for event analysis")
    if result.index.tz is not None:
        result.index = result.index.tz_convert("UTC")
    deltas = result.index.to_series().diff().dropna()
    expected = deltas.median() if len(deltas) else pd.Timedelta(0)
    gap_limit = max(expected * 4, pd.Timedelta(days=4)) if expected > pd.Timedelta(0) else pd.Timedelta(days=4)
    adjacent = np.r_[False, np.diff(kept_positions) == 1]
    timestamp_adjacent = np.r_[False, np.asarray(deltas <= gap_limit, dtype=bool)] if len(deltas) else np.zeros(len(result), dtype=bool)
    valid_edges = adjacent & timestamp_adjacent
    result.attrs["event_valid_return_edges"] = valid_edges
    result.attrs["event_had_invalid_ohlc"] = bool((~np.asarray(valid)).any())
    if len(deltas) > 2:
        typical = expected
        unusual = deltas > gap_limit
        if unusual.any():
            warnings.append(f"{int(unusual.sum())} timestamp gaps exceed the typical observation spacing; these may reflect missing bars or market closures.")
    if result["volume"].isna().any():
        warnings.append("Some volume observations are missing; volume-based triggers and impact metrics are unavailable at those points.")
    return result, warnings


def _drawdown(values: np.ndarray) -> float | None:
    if len(values) == 0 or np.any(~np.isfinite(values)) or np.any(values <= 0):
        return None
    peaks = np.maximum.accumulate(values)
    return float(np.min(values / peaks - 1.0))


def _period_return(values: np.ndarray, valid_edges: np.ndarray | None = None) -> float | None:
    if len(values) < 2 or not np.isfinite(values).all() or values[0] <= 0 or values[-1] <= 0 or (valid_edges is not None and not np.all(valid_edges)):
        return None
    return float(values[-1] / values[0] - 1.0)


def _window_metrics(frame: pd.DataFrame, index: int, window: int, benchmark: pd.DataFrame | None) -> dict[str, Any]:
    pre_start, post_end = max(0, index - window), min(len(frame) - 1, index + window)
    closes = frame["close"].to_numpy(dtype=float)
    valid_edges = np.asarray(frame.attrs.get("event_valid_return_edges", np.ones(len(frame), dtype=bool)))
    # Pre window ends at the prior close; post window starts at the event close.
    pre_closes, post_closes = closes[pre_start:index], closes[index:post_end + 1]
    pre_r = np.diff(np.log(pre_closes))
    post_r = np.diff(np.log(post_closes))
    pre_r = pre_r[valid_edges[pre_start + 1:index]]
    post_r = post_r[valid_edges[index + 1:post_end + 1]]
    pre_vol = float(np.std(pre_r, ddof=1)) if len(pre_r) > 1 else None
    post_vol = float(np.std(post_r, ddof=1)) if len(post_r) > 1 else None
    pre_volume = frame["volume"].iloc[pre_start:index].dropna().to_numpy(dtype=float)
    post_volume = frame["volume"].iloc[index + 1:post_end + 1].dropna().to_numpy(dtype=float)
    volume_change = None
    if len(pre_volume) and len(post_volume) and np.mean(pre_volume) > 0:
        volume_change = float(np.mean(post_volume) / np.mean(pre_volume) - 1.0)
    output: dict[str, Any] = {
        "window_observations": window, "available_pre_observations": int(index - pre_start),
        "available_post_observations": int(post_end - index),
        "pre_return": _period_return(pre_closes, valid_edges[pre_start + 1:index]), "post_return": _period_return(post_closes, valid_edges[index + 1:post_end + 1]),
        "pre_realized_volatility_log_return": pre_vol, "post_realized_volatility_log_return": post_vol,
        "pre_maximum_drawdown": _drawdown(pre_closes), "post_maximum_drawdown": _drawdown(post_closes),
        "volume_change_fraction": volume_change, "benchmark_relative_pre_return": None,
        "benchmark_relative_post_return": None,
        "return_convention": "simple close-to-close return across each available window; realized volatility is sample SD of close-to-close log returns per observation interval",
        "drawdown_convention": "minimum price / running peak - 1 within each window; each window starts at its first observed close",
        "causal_claim": False,
    }
    if benchmark is not None:
        aligned = benchmark["close"].reindex(frame.index).ffill(limit=1).to_numpy(dtype=float)
        pre_b, post_b = aligned[pre_start:index], aligned[index:post_end + 1]
        bp, bq = _period_return(pre_b), _period_return(post_b)
        rp, rq = output["pre_return"], output["post_return"]
        output["benchmark_relative_pre_return"] = rp - bp if rp is not None and bp is not None else None
        output["benchmark_relative_post_return"] = rq - bq if rq is not None and bq is not None else None
    return output


def _model_comparison(frame: pd.DataFrame, event_index: int, interval: str, ticker: str, start: str, end: str,
                      fit_cache: dict[str, Any]) -> dict[str, Any]:
    if frame.attrs.get("event_had_invalid_ohlc") or not np.all(np.asarray(frame.attrs.get("event_valid_return_edges", []), dtype=bool)[1:]):
        reason = "Invalid OHLC rows or irregular observation gaps prevent reliable one-period returns; model calibration is disabled to avoid bridging missing observations."
        return {"observed_log_return": None, "observed_return_convention": "close-to-close natural log return",
                "gbm": {"status": "unavailable", "reason": reason},
                "merton": {"status": "unavailable", "reason": reason},
                "garch": {"status": "unavailable", "reason": reason},
                "warning": "Model fits are omitted to avoid treating a multi-observation gap as a one-period return."}
    # This page detects events on charted OHLC close; fit the comparison baselines to that same field.
    if "fit_frame" not in fit_cache:
        fit_frame = frame.copy()
        fit_frame["adj_close"] = frame["close"]
        fit_cache["fit_frame"] = fit_frame
    fit_frame = fit_cache["fit_frame"]
    prices = frame["close"]
    returns = np.log(prices).diff().dropna().to_numpy(dtype=float)
    edge_valid = np.asarray(frame.attrs.get("event_valid_return_edges", np.ones(len(frame), dtype=bool)))[1:]
    returns = returns[edge_valid]
    event_return_index = event_index - 1
    result: dict[str, Any] = {"observed_log_return": float(returns[event_return_index]) if 0 <= event_return_index < len(returns) else None,
                              "observed_return_convention": "close-to-close natural log return",
                              "warning": "Calibrated statistical baselines compare distributional rarity; they do not explain causes or predict future prices."}
    if not len(returns) or result["observed_log_return"] is None:
        return {**result, "gbm": None, "merton": None, "garch": None}
    observed = result["observed_log_return"]
    if "gbm" not in fit_cache:
        try:
            fit_cache["gbm"] = {"fit": estimate_gbm(ticker, fit_frame, interval, start, end).as_dict()}
        except (ValueError, KeyError, MarketDataError) as exc:
            fit_cache["gbm"] = {"error": str(exc)}
    try:
        gbm = fit_cache["gbm"]["fit"]
        mu, sd = float(gbm["mean_period_log_return"]), float(gbm["sample_std_period_log_return"])
        result["gbm"] = {"status": "available" if sd > 0 else "unavailable", "method": gbm["method"],
                         "mean_log_return_per_interval": mu, "volatility_per_interval": sd,
                         "standardized_return": (observed - mu) / sd if sd > 0 else None,
                         "empirical_two_sided_tail_fraction": float(np.mean(np.abs(returns - mu) >= abs(observed - mu))),
                         "limitation": "Constant-volatility Gaussian increment baseline; tail fraction is in-sample."}
    except (ValueError, KeyError, MarketDataError) as exc:
        result["gbm"] = {"status": "unavailable", "reason": fit_cache["gbm"].get("error", str(exc)),
                         "limitation": "The existing GBM calibration supports daily, weekly, and monthly intervals only."}
    if "merton" not in fit_cache:
        try:
            fit_cache["merton"] = {"fit": calibrate_merton(ticker, fit_frame, price_field="auto", interval=interval, start=start, end=end)}
        except (ValueError, KeyError, ZeroDivisionError, RuntimeError, FloatingPointError, MarketDataError) as exc:
            fit_cache["merton"] = {"error": str(exc)}
    try:
        m = fit_cache["merton"]["fit"]
        dt = 1.0 / {"1d":252,"5d":52,"1wk":52,"1mo":12,"3mo":4}[interval]
        lam, jm, jv, sig, drift = (float(m[k]) for k in ("jump_intensity", "mean_log_jump", "jump_volatility", "volatility", "drift"))
        kappa = float(np.exp(jm + .5*jv*jv) - 1)
        max_k = int(min(100, poisson.ppf(.999999, lam * dt)))
        tail = 0.
        for k in range(max_k + 1):
            weight = float(poisson.pmf(k, lam * dt))
            mean = (drift - .5*sig*sig - lam*kappa)*dt + k*jm
            variance = sig*sig*dt + k*jv*jv
            if variance <= 0:
                tail += weight * float(abs(observed-mean) <= 1e-12)
            else:
                z = (observed-mean)/np.sqrt(variance)
                tail += weight * float(2*norm.sf(abs(z)))
        result["merton"] = {"status": "available", "method": m["method"], "parameters": {k:m[k] for k in ("drift","volatility","jump_intensity","mean_log_jump","jump_volatility")},
                            "fitted_two_sided_tail_probability": float(np.clip(tail,0,1)), "warnings": m["warnings"],
                            "limitation": "Compound-Poisson normal mixture fit may be weakly identified; tail probability is model based, not causal."}
    except (ValueError, KeyError, ZeroDivisionError, RuntimeError, FloatingPointError, MarketDataError) as exc:
        result["merton"] = {"status": "unavailable", "reason": str(exc), "limitation": "Requires at least 250 finite return observations for the existing Merton MLE."}
    if "garch" not in fit_cache:
        try:
            fit_cache["garch"] = {"fit": calibrate_garch(ticker, fit_frame, price_field="auto", interval=interval, start=start, end=end)}
        except (ValueError, KeyError, ZeroDivisionError, RuntimeError, FloatingPointError, MarketDataError) as exc:
            fit_cache["garch"] = {"error": str(exc)}
    try:
        g = fit_cache["garch"]["fit"]
        sigma = float(g["conditional_volatility"][event_return_index]["value"])
        z = (observed - float(g["mean"])) / sigma if sigma > 0 else None
        result["garch"] = {"status": "available", "method": g["method"], "mean": g["mean"], "omega": g["omega"], "alpha": g["alpha"], "beta": g["beta"],
                           "conditional_volatility": sigma, "standardized_residual": z,
                           "gaussian_two_sided_tail_probability": float(2*norm.sf(abs(z))) if z is not None else None,
                           "warnings": g["warnings"], "limitation": g["limitations"]}
    except (ValueError, KeyError, ZeroDivisionError, RuntimeError, FloatingPointError, MarketDataError) as exc:
        result["garch"] = {"status": "unavailable", "reason": str(exc), "limitation": "Requires at least 100 finite return observations for the existing GARCH MLE."}
    return result


def analyze_historical_events(frame: pd.DataFrame, *, ticker: str, start: str, end: str, interval: str = "1d",
                              thresholds: EventThresholds = EventThresholds(), impact_window: int = 10,
                              benchmark: pd.DataFrame | None = None) -> dict[str, Any]:
    frame, warnings = _valid_frame(frame)
    if len(frame) < max(6, thresholds.volume_lookback // 2 + 1):
        warnings.append("Insufficient prior observations for the configured rolling volume baseline; volume-spike detection may be unavailable.")
    if len(frame) < 80:
        warnings.append("Insufficient history for the 20-observation volatility measure and 60-observation prior baseline; volatility-spike detection may be unavailable.")
    if len(frame) <= thresholds.sustained_periods:
        warnings.append("Insufficient history for the configured sustained-movement horizon.")
    supports_session_gaps = interval in {"1d", "1wk", "1mo"}
    if not supports_session_gaps:
        warnings.append("Opening-gap detection is disabled for intraday bars because this application has no exchange-session calendar to distinguish session opens from ordinary bar-to-bar gaps.")
    close = frame["close"].astype(float)
    log_returns = np.log(close).diff()
    edge_valid = np.asarray(frame.attrs.get("event_valid_return_edges", np.ones(len(frame), dtype=bool)))
    log_returns.iloc[~edge_valid] = np.nan
    volume_base = frame["volume"].rolling(thresholds.volume_lookback, min_periods=max(5, thresholds.volume_lookback//2)).median().shift(1)
    rolling_vol = log_returns.rolling(thresholds.volatility_lookback, min_periods=max(5, thresholds.volatility_lookback//2)).std(ddof=1)
    vol_baseline = rolling_vol.shift(thresholds.volatility_lookback).rolling(60, min_periods=20).median()
    gap = frame["open"] / close.shift(1) - 1
    candidates: list[dict[str, Any]] = []
    def add(i: int, category: str, evidence: dict[str, Any], method: str) -> None:
        stamp = frame.index[i]
        candidates.append({"event_id": f"{ticker.upper()}:{stamp.isoformat()}:{category}", "ticker": ticker.upper(),
                           "timestamp": stamp.isoformat(), "date": stamp.date().isoformat(), "category": category,
                           "timestamp_resolution": "intraday_observation_utc" if interval in {"15m", "30m", "60m", "1h"} else "session_date",
                           "method": method, "evidence": evidence, "close": float(close.iloc[i]),
                           "event_log_return": float(log_returns.iloc[i]) if pd.notna(log_returns.iloc[i]) else None,
                           "impact": _window_metrics(frame, i, impact_window, benchmark),
                           "model_comparison": _model_comparison(frame, i, interval, ticker.upper(), start, end, model_fit_cache)})
    model_fit_cache: dict[str, Any] = {}
    for i in range(1, len(frame)):
        ret = log_returns.iloc[i]
        if pd.notna(ret) and abs(ret) >= thresholds.absolute_log_return:
            add(i, "large_absolute_return", {"log_return": float(ret), "simple_return": float(np.expm1(ret)), "threshold_absolute_log_return": thresholds.absolute_log_return},
                "Close-to-close log return exceeds configurable absolute threshold.")
        ratio = frame["volume"].iloc[i] / volume_base.iloc[i] if pd.notna(frame["volume"].iloc[i]) and pd.notna(volume_base.iloc[i]) and volume_base.iloc[i] > 0 else np.nan
        if np.isfinite(ratio) and ratio >= thresholds.volume_multiple:
            add(i, "unusual_volume", {"volume": float(frame["volume"].iloc[i]), "prior_rolling_median_volume": float(volume_base.iloc[i]), "volume_multiple": float(ratio), "threshold_multiple": thresholds.volume_multiple},
                f"Volume versus prior-only {thresholds.volume_lookback}-observation rolling median; baseline excludes event bar.")
        gap_value = gap.iloc[i]
        if supports_session_gaps and edge_valid[i] and pd.notna(gap_value) and abs(gap_value) >= thresholds.opening_gap:
            add(i, "opening_gap", {"open_to_previous_close_simple_return": float(gap_value), "threshold_absolute_gap": thresholds.opening_gap},
                "Open relative to previous valid close; only meaningful when provider bars include session opens.")
        current_vol, base_vol = rolling_vol.iloc[i], vol_baseline.iloc[i]
        vol_ratio = current_vol / base_vol if pd.notna(current_vol) and pd.notna(base_vol) and base_vol > 0 else np.nan
        previous_vol, previous_base = rolling_vol.iloc[i-1], vol_baseline.iloc[i-1]
        previous_vol_ratio = previous_vol / previous_base if pd.notna(previous_vol) and pd.notna(previous_base) and previous_base > 0 else np.nan
        if np.isfinite(vol_ratio) and vol_ratio >= thresholds.volatility_multiple and (not np.isfinite(previous_vol_ratio) or previous_vol_ratio < thresholds.volatility_multiple):
            add(i, "volatility_spike", {"rolling_log_return_volatility": float(current_vol), "prior_volatility_baseline": float(base_vol), "volatility_multiple": float(vol_ratio), "threshold_multiple": thresholds.volatility_multiple},
                f"{thresholds.volatility_lookback}-observation rolling sample SD versus prior 60-observation median volatility.")
        h = thresholds.sustained_periods
        if i >= h and np.all(edge_valid[i-h+1:i+1]):
            cumulative = float(np.log(close.iloc[i] / close.iloc[i-h]))
            previous_valid = i > h and np.all(edge_valid[i-h:i])
            previous_cumulative = float(np.log(close.iloc[i-1] / close.iloc[i-1-h])) if previous_valid else 0.0
            first_threshold_crossing = not previous_valid or abs(previous_cumulative) < thresholds.sustained_log_return or np.sign(previous_cumulative) != np.sign(cumulative)
            if abs(cumulative) >= thresholds.sustained_log_return and first_threshold_crossing:
                add(i, "sustained_movement", {"periods": h, "cumulative_log_return": cumulative, "cumulative_simple_return": float(np.expm1(cumulative)), "threshold_absolute_log_return": thresholds.sustained_log_return},
                    f"Trailing {h}-observation close-to-close cumulative log return; overlapping windows are allowed.")
    candidates.sort(key=lambda event: (event["timestamp"], event["category"]))
    # Retain OHLCV bars for charting as source observations; convert timestamps to ISO with timezone explicit when available.
    bars = [{"date": idx.isoformat(), **{key: (float(row[key]) if pd.notna(row[key]) else None) for key in ("open","high","low","close","volume")}}
            for idx, row in frame.iterrows()]
    return {"ticker": ticker.upper(), "interval": interval, "estimation_period": {"start": start, "end_exclusive": end},
            "observation_count": len(frame), "source_observation_count": int(frame.attrs.get("event_source_observation_count", len(frame))),
            "excluded_observation_count": int(frame.attrs.get("event_source_observation_count", len(frame))) - len(frame),
            "bars": bars, "candidate_events": candidates,
            "detection_configuration": thresholds.__dict__, "impact_window_observations_each_side": impact_window,
            "warnings": warnings, "data_convention": "Movement returns, volatility, event-window prices and model baselines use the provider's raw close; opening gaps use raw open versus prior raw close. Provider timestamps are normalized to UTC-naive by ingestion and are not guaranteed exchange-local publication times. Corporate actions can therefore appear as price events.",
            "causality_note": "Event-window comparisons are descriptive and do not establish causation."}
