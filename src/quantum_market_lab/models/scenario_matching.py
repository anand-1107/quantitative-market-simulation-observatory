"""Scoring helpers for bounded same-innovation path scenario comparisons."""
from __future__ import annotations

import numpy as np


def normalized_path_rmse(target: list[float] | np.ndarray, candidate: list[float] | np.ndarray) -> float:
    reference = np.asarray(target, dtype=np.float64)
    values = np.asarray(candidate, dtype=np.float64)
    if reference.ndim != 1 or values.ndim != 1 or len(reference) < 2 or len(values) < 2:
        raise ValueError("target and candidate paths must be one-dimensional with at least two values")
    if not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError("target and candidate paths must contain only finite values")
    aligned = np.interp(np.linspace(0.0, 1.0, len(reference)), np.linspace(0.0, 1.0, len(values)), values)
    scale = max(float(reference.std()), float(np.max(np.abs(reference))) * .01, 1e-8)
    return float(np.sqrt(np.mean(((aligned - reference) / scale) ** 2)))


def simple_return_and_drawdown(model: str, path: list[float] | np.ndarray,
                               return_convention: str | None = None) -> tuple[float | None, float | None]:
    values = np.asarray(path, dtype=np.float64)
    if model == "ou":
        return None, None
    if model == "garch":
        if return_convention == "log":
            wealth = np.concatenate(([1.0], np.exp(np.cumsum(values))))
        elif return_convention == "simple" and np.all(values > -1):
            wealth = np.concatenate(([1.0], np.cumprod(1.0 + values)))
        else:
            return None, None
    else:
        if len(values) < 1 or values[0] <= 0:
            return None, None
        wealth = values
    total_return = float(wealth[-1] / wealth[0] - 1.0) if wealth[0] > 0 else None
    peaks = np.maximum.accumulate(wealth)
    drawdown = float(np.min(wealth / peaks - 1.0)) if np.all(peaks > 0) else None
    return total_return, drawdown


def match_objective(path_rmse: float, target_return: float | None, candidate_return: float | None,
                    target_drawdown: float | None, candidate_drawdown: float | None,
                    return_weight: float, drawdown_weight: float) -> tuple[float, float | None, float | None]:
    return_difference = (abs(candidate_return - target_return)
                         if target_return is not None and candidate_return is not None else None)
    drawdown_difference = (abs(candidate_drawdown - target_drawdown)
                           if target_drawdown is not None and candidate_drawdown is not None else None)
    score = path_rmse
    if return_difference is not None:
        score += return_weight * return_difference
    if drawdown_difference is not None:
        score += drawdown_weight * drawdown_difference
    return float(score), return_difference, drawdown_difference
