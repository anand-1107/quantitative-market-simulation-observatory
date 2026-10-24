"""Reusable empirical diagnostics for simulated paths and returns."""

from __future__ import annotations

import numpy as np


def empirical_var_es(losses: np.ndarray, confidence_level: float = 0.95) -> dict:
    """Empirical loss VaR/ES using a linear quantile and inclusive threshold ties.

    VaR is the confidence-level quantile of losses. ES is the arithmetic mean
    of all observed losses greater than or equal to VaR; all ties are included.
    This is a transparent finite-sample convention (tail counts can exceed the
    nominal tail fraction when observations tie at the threshold).
    """
    values = np.asarray(losses, dtype=np.float64)
    try:
        confidence_level = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be finite and strictly between 0 and 1") from exc
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be finite and strictly between 0 and 1")
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("losses must be a non-empty finite one-dimensional array")
    var = float(np.quantile(values, confidence_level, method="linear"))
    tail = values[values >= var]
    return {"confidence_level": float(confidence_level), "var": var,
            "expected_shortfall": float(tail.mean()), "tail_observation_count": int(tail.size),
            "sample_size": int(values.size),
            "quantile_method": "linear interpolation of the empirical loss quantile",
            "tail_convention": "arithmetic mean of losses >= VaR; all threshold ties included"}


def path_return_drawdown(path: np.ndarray, quantity_kind: str,
                         return_convention: str | None = None) -> dict | None:
    """Return cumulative simple return and max drawdown for one simulated path.

    Price paths include their initial value. Return paths are converted to a
    wealth curve starting at one. State paths have no inherent return meaning.
    ``None`` indicates that positive finite wealth cannot be formed.
    """
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        return None
    if quantity_kind == "state":
        return None
    if quantity_kind == "price":
        if np.any(values <= 0):
            return None
        wealth = values
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            total_return = float(values[-1] / values[0] - 1.0)
    elif quantity_kind == "return":
        if return_convention == "log":
            with np.errstate(over="ignore", invalid="ignore"):
                log_wealth = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
            if not np.isfinite(log_wealth).all() or np.any(log_wealth > np.log(np.finfo(np.float64).max)):
                return None
            with np.errstate(under="ignore"):
                wealth = np.exp(log_wealth)
        elif return_convention == "simple":
            if np.any(values <= -1):
                return None
            with np.errstate(over="ignore", invalid="ignore"):
                wealth = np.concatenate(([1.0], np.cumprod(1.0 + values, dtype=np.float64)))
        else:
            return None
        if not np.isfinite(wealth).all() or np.any(wealth <= 0):
            return None
        total_return = float(wealth[-1] - 1.0)
    else:
        return None
    peaks = np.maximum.accumulate(wealth)
    max_drawdown = float(np.min(wealth / peaks - 1.0))
    if not np.isfinite(total_return) or not np.isfinite(max_drawdown):
        return None
    return {"total_simple_return": total_return, "maximum_drawdown": max_drawdown}


def terminal_return_loss_risk(total_returns: np.ndarray, confidence_level: float = 0.95) -> dict:
    """Summarize a sample of terminal cumulative simple returns as losses."""
    returns = np.asarray(total_returns, dtype=np.float64)
    if returns.ndim != 1 or returns.size == 0 or not np.isfinite(returns).all():
        raise ValueError("total_returns must be a non-empty finite one-dimensional array")
    empirical = empirical_var_es(-returns, confidence_level)
    return {"available": True, "measure": "terminal_horizon", "confidence_level": empirical["confidence_level"],
            "unit": "simple_return_fraction", "loss_definition": "L = -R; R is cumulative simple return from initial wealth to the simulation horizon",
            "scenario_count": int(returns.size), "mean_total_simple_return": float(returns.mean()),
            "total_simple_return_quantiles": {str(q): float(np.quantile(returns, q, method="linear")) for q in (.05, .5, .95)},
            "var": empirical["var"], "expected_shortfall": empirical["expected_shortfall"],
            "tail_scenario_count": empirical["tail_observation_count"],
            "quantile_method": empirical["quantile_method"], "tail_convention": empirical["tail_convention"],
            "warnings": (["Small scenario sample; empirical tail estimates are coarse."] if returns.size < 20 else [])}


def terminal_return_risk(paths: np.ndarray, quantity_kind: str,
                         return_convention: str | None = None,
                         confidence_level: float = 0.95) -> dict:
    """Aggregate terminal-horizon loss risk and path drawdowns for simulations."""
    values = np.asarray(paths, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("paths must be a non-empty finite two-dimensional array")
    try:
        confidence_level = float(confidence_level)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_level must be finite and strictly between 0 and 1") from exc
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be finite and strictly between 0 and 1")
    if quantity_kind == "state":
        return {"available": False, "unavailable_reason": "State-level paths do not define an asset-return or loss distribution.", "path_metrics": None}
    path_metrics = [path_return_drawdown(path, quantity_kind, return_convention) for path in values]
    if any(metric is None for metric in path_metrics):
        return {"available": False, "unavailable_reason": (
            "Risk is unavailable because paths do not define finite, strictly positive wealth "
            "(for example a simple return at or below -100%, underflow, or overflow)."),
            "path_metrics": None}
    returns = np.asarray([metric["total_simple_return"] for metric in path_metrics], dtype=np.float64)
    drawdowns = np.asarray([metric["maximum_drawdown"] for metric in path_metrics], dtype=np.float64)
    risk = terminal_return_loss_risk(returns, confidence_level)
    return {**risk,
            "maximum_drawdown_quantiles": {str(q): float(np.quantile(drawdowns, q, method="linear")) for q in (.05, .5, .95)},
            "path_metrics": path_metrics}


def path_diagnostics(paths: np.ndarray, quantiles: tuple[float, ...] = (0.01, 0.05, 0.5, 0.95, 0.99)) -> dict:
    """Summarize path-wise cross-sectional moments and terminal distribution."""
    values = np.asarray(paths, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("paths must be a finite (paths, time_points) array")
    if any(not 0 <= q <= 1 for q in quantiles): raise ValueError("quantiles must be in [0, 1]")
    terminal = values[:, -1]
    return {"time_mean": values.mean(axis=0).tolist(), "time_variance": values.var(axis=0, ddof=1 if len(values) > 1 else 0).tolist(),
            "terminal_mean": float(terminal.mean()), "terminal_variance": float(terminal.var(ddof=1) if len(terminal) > 1 else 0.0),
            "terminal_quantiles": {str(q): float(np.quantile(terminal, q)) for q in quantiles}}


def return_diagnostics(returns: np.ndarray, variances: np.ndarray | None = None) -> dict:
    """Compute mean lag-1 return autocorrelation and squared-return clustering."""
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all(): raise ValueError("returns must be a finite 2D array")
    def corr_lag1(sample: np.ndarray) -> float | None:
        if sample.shape[1] < 3: return None
        left, right = sample[:, :-1].ravel(), sample[:, 1:].ravel()
        if np.std(left) == 0 or np.std(right) == 0: return None
        return float(np.corrcoef(left, right)[0, 1])
    result = {"return_lag1_autocorrelation": corr_lag1(values),
              "squared_return_lag1_autocorrelation": corr_lag1(values**2)}
    if variances is not None:
        vols = np.asarray(variances, dtype=np.float64)
        if vols.shape != values.shape or not np.isfinite(vols).all() or np.any(vols <= 0):
            raise ValueError("variances must be positive finite values matching returns")
        result["mean_conditional_variance"] = float(vols.mean())
    return result


def trajectory_summary(paths: np.ndarray, times: np.ndarray) -> dict:
    """Summarize a finite (paths, time) matrix with per-time quantile bands."""
    values = np.asarray(paths, dtype=np.float64)
    coordinates = np.asarray(times, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("trajectories must be a non-empty two-dimensional array")
    if coordinates.ndim != 1 or len(coordinates) != values.shape[1]:
        raise ValueError("time coordinates must match trajectory length")
    if not np.isfinite(values).all() or not np.isfinite(coordinates).all():
        raise ValueError("simulated trajectories exceeded finite numeric range")
    percentiles = np.quantile(values, [.05, .5, .95], axis=0)
    bands = [{"time": float(coordinates[i]), "mean": float(values[:, i].mean()),
              "p05": float(percentiles[0, i]), "p50": float(percentiles[1, i]),
              "p95": float(percentiles[2, i])} for i in range(values.shape[1])]
    terminal = values[:, -1]
    return {"terminal_mean": float(terminal.mean()),
            "terminal_variance": float(terminal.var(ddof=1) if len(terminal) > 1 else 0.0),
            "terminal_quantiles": {"0.05": float(np.quantile(terminal, .05)),
                                   "0.5": float(np.quantile(terminal, .5)),
                                   "0.95": float(np.quantile(terminal, .95))},
            "bands": bands}
