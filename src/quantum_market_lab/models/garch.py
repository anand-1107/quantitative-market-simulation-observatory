"""GARCH(1,1) model for discrete-period returns."""

from __future__ import annotations

import numpy as np

from .interface import FloatArray, validate_finite_parameters

ReturnConvention = str


def simulate_garch(
    omega: float,
    alpha: float,
    beta: float,
    steps: int,
    paths: int = 1,
    mean: float = 0.0,
    initial_variance: float | None = None,
    seed: int | None = 42,
    return_convention: ReturnConvention = "log",
    *,
    return_innovations: bool = False,
) -> tuple[FloatArray, FloatArray] | tuple[FloatArray, FloatArray, dict[str, FloatArray]]:
    """Simulate GARCH returns and conditional variances with shape (paths, steps).

    Periods are discrete; parameters are per period (not inherently daily or
    annual). ``epsilon[t]=sqrt(h[t])*z[t]`` and ``h[t+1]=omega+alpha*epsilon[t]²
    +beta*h[t]``. omega>0 and alpha,beta>=0 give positive conditional variance
    for any finite horizon. Covariance stationarity additionally requires
    alpha+beta<1. Initial variance defaults to omega/(1-alpha-beta) when
    stationary, or omega otherwise; explicitly pass another positive value as
    needed. The mean is per-period mean log return by default; ``simple`` uses
    the same recursion but does not make compounded simple returns lognormal.
    A fixed seed is reproducible.
    """
    validate_finite_parameters(omega=omega, alpha=alpha, beta=beta, mean=mean)
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if isinstance(paths, bool) or not isinstance(paths, (int, np.integer)) or paths <= 0:
        raise ValueError("paths must be a positive integer")
    if omega <= 0 or alpha < 0 or beta < 0:
        raise ValueError("omega must be positive and alpha/beta nonnegative")
    if return_convention not in {"log", "simple"}:
        raise ValueError("return_convention must be 'log' or 'simple'")
    stationary = alpha + beta < 1
    if initial_variance is None:
        initial_variance = omega / (1 - alpha - beta) if stationary else omega
    if not np.isfinite(initial_variance) or initial_variance <= 0:
        raise ValueError("initial_variance must be finite and positive")
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((paths, steps))
    returns = np.empty((paths, steps), dtype=np.float64)
    variance = np.empty_like(returns)
    residuals = np.empty_like(returns)
    variance[:, 0] = initial_variance
    for t in range(steps):
        residual = np.sqrt(variance[:, t]) * shocks[:, t]
        residuals[:, t] = residual
        returns[:, t] = mean + residual
        if t + 1 < steps:
            variance[:, t + 1] = omega + alpha * residual**2 + beta * variance[:, t]
    if return_innovations:
        return returns, variance, {"standard_normal": shocks, "innovation": residuals}
    return returns, variance


def garch_returns_to_prices(
    returns: np.ndarray,
    initial_price: float = 100.0,
    return_convention: ReturnConvention = "log",
) -> FloatArray:
    """Convert (paths, steps) log/simple returns to prices (paths, steps+1)."""
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("returns must be a finite two-dimensional array")
    if not np.isfinite(initial_price) or initial_price <= 0:
        raise ValueError("initial_price must be finite and positive")
    if return_convention == "log":
        growth = np.exp(values)
    elif return_convention == "simple":
        if np.any(values <= -1): raise ValueError("simple returns must exceed -100%")
        growth = 1 + values
    else:
        raise ValueError("return_convention must be 'log' or 'simple'")
    prices = np.empty((values.shape[0], values.shape[1] + 1), dtype=np.float64)
    prices[:, 0] = initial_price
    prices[:, 1:] = initial_price * np.cumprod(growth, axis=1)
    return prices
