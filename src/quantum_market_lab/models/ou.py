"""Exact-transition Ornstein–Uhlenbeck process for spreads or transformed variables."""

from __future__ import annotations

import numpy as np

from .interface import FloatArray, validate_finite_parameters, validate_simulation_grid


def simulate_ou(
    initial_value: float,
    mean_level: float,
    mean_reversion_speed: float,
    volatility: float,
    years: float,
    steps: int,
    paths: int,
    seed: int | None = 42,
    *,
    return_innovations: bool = False,
) -> tuple[FloatArray, FloatArray] | tuple[FloatArray, FloatArray, dict[str, FloatArray]]:
    """Simulate OU levels using the exact conditional Gaussian transition.

    ``mean_reversion_speed`` is theta per year, ``volatility`` is level units
    per square-root year. Returns times (steps+1,) and levels (paths, steps+1).
    For theta>0 the stationary moments are mean_level and sigma²/(2 theta).
    At theta=0 the process is Brownian motion with driftless level increments.
    OU is commonly a model for spreads or transformed state variables, not a
    complete positive stock-price model. A seed repeats results exactly.
    """
    validate_finite_parameters(initial_value=initial_value, mean_level=mean_level,
                               mean_reversion_speed=mean_reversion_speed, volatility=volatility)
    validate_simulation_grid(years, steps, paths)
    if mean_reversion_speed < 0 or volatility < 0:
        raise ValueError("mean_reversion_speed and volatility must be nonnegative")
    theta, sigma, dt = mean_reversion_speed, volatility, years / steps
    decay = np.exp(-theta * dt)
    conditional_sd = sigma * np.sqrt(dt if theta == 0 else -np.expm1(-2 * theta * dt) / (2 * theta))
    rng = np.random.default_rng(seed)
    levels = np.empty((paths, steps + 1), dtype=np.float64)
    levels[:, 0] = initial_value
    shocks = rng.standard_normal((paths, steps))
    conditional_means = np.empty((paths, steps), dtype=np.float64)
    diffusion_terms = np.empty_like(conditional_means)
    for i in range(steps):
        conditional_means[:, i] = mean_level + (levels[:, i] - mean_level) * decay
        diffusion_terms[:, i] = conditional_sd * shocks[:, i]
        levels[:, i + 1] = conditional_means[:, i] + diffusion_terms[:, i]
    times = np.linspace(0.0, years, steps + 1)
    if return_innovations:
        return times, levels, {"standard_normal": shocks, "conditional_mean": conditional_means,
                               "diffusion_term": diffusion_terms}
    return times, levels
