"""Merton compound-Poisson jump diffusion for positive asset prices."""

from __future__ import annotations

import numpy as np

from .interface import FloatArray, validate_finite_parameters, validate_simulation_grid


def simulate_jump_diffusion(
    initial_price: float,
    drift: float,
    volatility: float,
    jump_intensity: float,
    mean_log_jump: float,
    jump_volatility: float,
    years: float,
    steps: int,
    paths: int,
    seed: int | None = 42,
    *,
    return_innovations: bool = False,
) -> tuple[FloatArray, FloatArray] | tuple[FloatArray, FloatArray, dict[str, FloatArray]]:
    """Simulate Merton prices with exact compound-Poisson jump sums per step.

    ``drift`` is the expected proportional asset growth rate per year: the
    compensator ``lambda * (E[J]-1)`` is subtracted in log-price dynamics, so
    E[S_t]=S_0 exp(drift*t). Diffusion volatility is sigma; total return
    variability also includes jump frequency and jump-size dispersion. The
    Gaussian log jump has mean ``mean_log_jump`` and sd ``jump_volatility``.
    Outputs have GBM shape: times (steps+1,), prices (paths, steps+1).
    Parameters are constant, jumps are independent compound Poisson, and jump
    sizes are lognormal multipliers; these are simplified model assumptions.
    """
    validate_finite_parameters(initial_price=initial_price, drift=drift, volatility=volatility,
                               jump_intensity=jump_intensity, mean_log_jump=mean_log_jump,
                               jump_volatility=jump_volatility)
    validate_simulation_grid(years, steps, paths)
    if initial_price <= 0: raise ValueError("initial_price must be positive")
    if volatility < 0 or jump_intensity < 0 or jump_volatility < 0:
        raise ValueError("volatility, jump_intensity, and jump_volatility must be nonnegative")
    dt = years / steps
    rng = np.random.default_rng(seed)
    counts = rng.poisson(jump_intensity * dt, size=(paths, steps))
    jump_normals = rng.standard_normal((paths, steps))
    jump_sums = counts * mean_log_jump + np.sqrt(counts) * jump_volatility * jump_normals
    compensator = jump_intensity * (np.exp(mean_log_jump + 0.5 * jump_volatility**2) - 1.0)
    diffusion_normals = rng.standard_normal((paths, steps))
    diffusion_terms = volatility * np.sqrt(dt) * diffusion_normals
    drift_terms = (drift - compensator - 0.5 * volatility**2) * dt
    increments = drift_terms + diffusion_terms + jump_sums
    log_prices = np.empty((paths, steps + 1), dtype=np.float64)
    log_prices[:, 0] = np.log(initial_price)
    log_prices[:, 1:] = np.log(initial_price) + np.cumsum(increments, axis=1)
    prices = np.exp(log_prices)
    prices[:, 0] = initial_price
    times = np.linspace(0.0, years, steps + 1)
    if return_innovations:
        return times, prices, {"diffusion_standard_normal": diffusion_normals,
                               "poisson_jump_count": counts.astype(np.float64),
                               "jump_aggregate_standard_normal": jump_normals,
                               "jump_log_sum": jump_sums,
                               "drift_log_term": np.broadcast_to(drift_terms, (paths, steps)).copy(),
                               "diffusion_term": diffusion_terms,
                               "log_price_increment": increments}
    return times, prices
