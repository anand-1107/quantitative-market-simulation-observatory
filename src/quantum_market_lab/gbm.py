"""Geometric Brownian Motion (GBM) simulation utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def simulate_gbm(
    initial_price: float,
    drift: float,
    volatility: float,
    years: float,
    steps: int,
    paths: int,
    seed: int | None = 42,
    *,
    return_innovations: bool = False,
) -> tuple[FloatArray, FloatArray] | tuple[FloatArray, FloatArray, dict[str, FloatArray]]:
    """Simulate GBM paths using the exact time-step transition.

    Parameters
    ----------
    initial_price:
        Starting price S_0; must be positive.
    drift:
        Annualized GBM drift mu, in inverse years.
    volatility:
        Annualized volatility sigma; must be non-negative.
    years:
        Total simulated horizon in years; must be positive.
    steps:
        Number of time intervals; must be positive.
    paths:
        Number of independent trajectories; must be positive.
    seed:
        Random seed for reproducibility, or None for a non-fixed stream.

    Returns
    -------
    times:
        Shape (steps + 1,), from 0 to `years`.
    prices:
        Shape (paths, steps + 1); each row is one trajectory.
    """
    if initial_price <= 0:
        raise ValueError("initial_price must be positive")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if years <= 0:
        raise ValueError("years must be positive")
    if steps <= 0 or paths <= 0:
        raise ValueError("steps and paths must be positive integers")

    rng = np.random.default_rng(seed)
    dt = years / steps
    times = np.linspace(0.0, years, steps + 1)

    # One independent standard-normal shock for each path and time interval.
    z = rng.standard_normal(size=(paths, steps))

    # Exact GBM transition over each interval.
    log_returns = (
        (drift - 0.5 * volatility**2) * dt
        + volatility * np.sqrt(dt) * z
    )

    # Add increments in log-price space, then exponentiate to keep prices positive.
    log_prices = np.empty((paths, steps + 1), dtype=np.float64)
    log_prices[:, 0] = np.log(initial_price)
    log_prices[:, 1:] = np.log(initial_price) + np.cumsum(log_returns, axis=1)
    prices = np.exp(log_prices)

    if return_innovations:
        return times, prices, {"standard_normal": z, "log_return_increment": log_returns}
    return times, prices
