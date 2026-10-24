"""Deterministic return, volatility, drawdown, and annualization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def compute_log_returns(df: pd.DataFrame, price_column: str = "adj_close") -> pd.Series:
    """Return close-to-close log returns, preferring adjusted close when available."""
    col = price_column if price_column in df.columns else ("adj_close" if "adj_close" in df.columns else "close")
    if col not in df.columns:
        raise ValueError("DataFrame must contain adj_close or close prices")
    prices = pd.to_numeric(df[col], errors="coerce")
    prices = prices.where(np.isfinite(prices) & (prices > 0))
    # Difference of logs is algebraically identical to log(P[t] / P[t-1]) but
    # avoids intermediate overflow/underflow for finite positive prices.
    result = np.log(prices).diff().replace([np.inf, -np.inf], np.nan)
    result.name = "log_return"
    return result


def summarize_returns(returns: pd.Series) -> dict[str, float | int | None]:
    """Summarize finite observations; undefined small-sample statistics are None."""
    series = pd.to_numeric(pd.Series(returns), errors="coerce")
    valid = series[np.isfinite(series)]
    n = len(valid)
    def val(x: float) -> float | None:
        return float(x) if np.isfinite(x) else None
    return {"observation_count": n, "mean": val(valid.mean()) if n else None,
            "standard_deviation": val(valid.std(ddof=1)) if n > 1 else None,
            "variance": val(valid.var(ddof=1)) if n > 1 else None,
            "skewness": val(skew(valid, bias=False)) if n > 2 else None,
            "excess_kurtosis": val(kurtosis(valid, fisher=True, bias=False)) if n > 3 else None,
            "minimum": val(valid.min()) if n else None, "maximum": val(valid.max()) if n else None,
            "median": val(valid.median()) if n else None,
            "lag1_autocorrelation": val(valid.autocorr(lag=1)) if n > 2 else None,
            "missing_fraction": float(series.isna().sum() / len(series)) if len(series) else 0.0}


def realized_volatility(returns: pd.Series, window: int = 21, annualization_factor: int = 252, annualized: bool = False) -> pd.Series:
    """Rolling sample standard deviation; output is daily unless annualized=True."""
    if window < 1 or annualization_factor < 1:
        raise ValueError("window and annualization_factor must be positive")
    series = pd.to_numeric(returns, errors="coerce")
    result = series.rolling(window=window, min_periods=window).std(ddof=1)
    result.name = "annualized_volatility" if annualized else "daily_volatility"
    return result * np.sqrt(annualization_factor) if annualized else result


def drawdown(prices_or_returns: pd.Series, input_type: str = "price") -> tuple[pd.Series, float]:
    """Compute peak-to-trough drawdown from positive prices or simple returns.

    Return inputs are simple per-period returns. Their wealth curve starts at
    one before the first observation, so a loss on the first period is counted.
    Missing, infinite, nonpositive wealth, and returns at or below -100% are
    rejected instead of silently changing the running peak.
    """
    values = pd.to_numeric(prices_or_returns, errors="coerce")
    if input_type not in {"price", "returns"}:
        raise ValueError("input_type must be 'price' or 'returns'")
    array = values.to_numpy(dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("prices/returns must be a non-empty finite series")
    if input_type == "price":
        prices = values
    else:
        if np.any(array <= -1):
            raise ValueError("simple returns must exceed -100% to define positive wealth")
        wealth = np.cumprod(1.0 + array)
        if not np.isfinite(wealth).all() or np.any(wealth <= 0):
            raise ValueError("simple returns produced invalid or nonpositive wealth")
        peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
        dd = pd.Series(wealth / peaks - 1.0, index=values.index, name="drawdown")
        return dd, float(dd.min())
    if (prices <= 0).any():
        raise ValueError("prices/equity curve must remain positive")
    peak = prices.cummax()
    dd = prices / peak - 1
    dd.name = "drawdown"
    maximum = float(dd.min()) if dd.notna().any() else float("nan")
    return dd, maximum


def annualize_mean_log_return(mean_daily_log_return: float, periods: int = 252) -> float:
    """Annualized continuously compounded log-return rate (not arithmetic return)."""
    if periods <= 0: raise ValueError("periods must be positive")
    return float(mean_daily_log_return * periods)


def annualize_arithmetic_return(mean_daily_arithmetic_return: float, periods: int = 252) -> float:
    """Geometric annualization of a constant mean daily arithmetic return."""
    if periods <= 0: raise ValueError("periods must be positive")
    if mean_daily_arithmetic_return <= -1: raise ValueError("daily arithmetic return must exceed -100%")
    return float((1 + mean_daily_arithmetic_return) ** periods - 1)


def annualize_volatility(daily_volatility: float, periods: int = 252) -> float:
    if periods <= 0 or daily_volatility < 0: raise ValueError("periods must be positive and volatility nonnegative")
    return float(daily_volatility * np.sqrt(periods))
