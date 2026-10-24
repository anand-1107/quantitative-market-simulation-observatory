"""Reconstruct model paths from the innovations actually consumed by each simulator."""
from __future__ import annotations

from typing import Any

import numpy as np


def reconstruct_path(record: dict[str, Any], path_index: int) -> dict[str, Any]:
    model = record["model"]
    parameters = record["parameters"]
    times = np.asarray(record["times"], dtype=np.float64)
    original = np.asarray(record["trajectories"][path_index], dtype=np.float64)
    innovations = {key: np.asarray(value[path_index], dtype=np.float64)
                   for key, value in record["innovations"].items()}
    steps = len(innovations[next(iter(innovations))])
    if len(times) != len(original) and model != "garch":
        raise ValueError("stored price/state path and time-grid dimensions do not agree")
    if model == "garch" and len(times) != steps:
        raise ValueError("stored GARCH return path and time-grid dimensions do not agree")
    if any(len(values) != steps for values in innovations.values()):
        raise ValueError("stored innovation arrays have inconsistent dimensions")

    transitions: list[dict[str, Any]] = []
    if model == "gbm":
        dt = float(parameters["years"]) / steps
        initial = float(parameters["initial_price"])
        reconstructed = np.empty(steps + 1, dtype=np.float64)
        reconstructed[0] = initial
        cumulative_log_return = 0.0
        for t, z in enumerate(innovations["standard_normal"]):
            drift = (float(parameters["drift"]) - .5 * float(parameters["volatility"])**2) * dt
            diffusion = float(parameters["volatility"]) * np.sqrt(dt) * z
            increment = drift + diffusion
            cumulative_log_return += increment
            reconstructed[t + 1] = np.exp(np.log(initial) + cumulative_log_return)
            transitions.append({"time": float(times[t + 1]), "standard_normal": float(z),
                                "drift_log_component": float(drift), "diffusion_component": float(diffusion),
                                "log_return_increment": float(increment), "value": float(reconstructed[t + 1])})
    elif model == "ou":
        dt = float(parameters["years"]) / steps
        theta = float(parameters["mean_reversion_speed"])
        sigma = float(parameters["volatility"])
        decay = np.exp(-theta * dt)
        conditional_sd = sigma * np.sqrt(dt if theta == 0 else -np.expm1(-2 * theta * dt) / (2 * theta))
        mean = float(parameters["mean_level"])
        reconstructed = np.empty(steps + 1, dtype=np.float64)
        reconstructed[0] = float(parameters["initial_value"])
        for t, z in enumerate(innovations["standard_normal"]):
            current = reconstructed[t]
            conditional_mean = mean + (current - mean) * decay
            reversion = conditional_mean - current
            diffusion = conditional_sd * z
            reconstructed[t + 1] = conditional_mean + diffusion
            transitions.append({"time": float(times[t + 1]), "current_value": float(current),
                                "conditional_mean": float(conditional_mean), "mean_reversion_component": float(reversion),
                                "standard_normal": float(z), "diffusion_component": float(diffusion),
                                "value": float(reconstructed[t + 1])})
    elif model == "jump_diffusion":
        dt = float(parameters["years"]) / steps
        drift_rate = float(parameters["drift"])
        sigma = float(parameters["volatility"])
        intensity = float(parameters["jump_intensity"])
        jump_mean = float(parameters["mean_log_jump"])
        jump_sd = float(parameters["jump_volatility"])
        compensator = intensity * (np.exp(jump_mean + .5 * jump_sd**2) - 1.0)
        drift = (drift_rate - compensator - .5 * sigma**2) * dt
        initial = float(parameters["initial_price"])
        reconstructed = np.empty(steps + 1, dtype=np.float64)
        reconstructed[0] = initial
        cumulative_log_return = 0.0
        for t in range(steps):
            count = int(innovations["poisson_jump_count"][t])
            jump_z = innovations["jump_aggregate_standard_normal"][t]
            diffusion_z = innovations["diffusion_standard_normal"][t]
            jump_sum = count * jump_mean + np.sqrt(count) * jump_sd * jump_z
            diffusion = sigma * np.sqrt(dt) * diffusion_z
            increment = drift + diffusion + jump_sum
            cumulative_log_return += increment
            reconstructed[t + 1] = np.exp(np.log(initial) + cumulative_log_return)
            transitions.append({"time": float(times[t + 1]), "diffusion_standard_normal": float(diffusion_z),
                                "poisson_jump_count": count, "jump_occurred": count > 0,
                                "jump_aggregate_standard_normal": float(jump_z), "jump_log_sum": float(jump_sum),
                                "drift_log_term": float(drift), "compensator_rate": float(compensator),
                                "diffusion_component": float(diffusion), "log_price_increment": float(increment),
                                "value": float(reconstructed[t + 1])})
    elif model == "garch":
        initial_variance = float(parameters["initial_variance"])
        omega, alpha, beta = (float(parameters[key]) for key in ("omega", "alpha", "beta"))
        mean = float(parameters["mean"])
        reconstructed = np.empty(steps, dtype=np.float64)
        conditional = np.empty(steps, dtype=np.float64)
        conditional[0] = initial_variance
        for t, z in enumerate(innovations["standard_normal"]):
            epsilon = np.sqrt(conditional[t]) * z
            value = mean + epsilon
            reconstructed[t] = value
            has_next_period = t + 1 < steps
            omega_term = omega if has_next_period else None
            arch_term = alpha * epsilon**2 if has_next_period else None
            garch_term = beta * conditional[t] if has_next_period else None
            next_variance = omega_term + arch_term + garch_term if has_next_period else None
            if next_variance is not None:
                conditional[t + 1] = next_variance
            transitions.append({"time": float(times[t]), "conditional_variance": float(conditional[t]),
                                "standard_normal": float(z), "innovation": float(epsilon),
                                "return": float(value), "variance_intercept_term": omega_term,
                                "arch_variance_term": arch_term, "garch_variance_term": garch_term,
                                "next_conditional_variance": next_variance})
    else:
        raise ValueError("unsupported forensic model")

    if not np.isfinite(reconstructed).all():
        raise ValueError("reconstruction produced non-finite values")
    difference = np.abs(reconstructed - original)
    max_error = float(difference.max(initial=0.0))
    tolerance = max(1e-12, float(np.max(np.abs(original), initial=0.0)) * 1e-12)
    return {"reconstructed_path": reconstructed.tolist(), "transitions": transitions,
            "maximum_absolute_error": max_error, "verification_tolerance": tolerance,
            "verified": bool(np.allclose(reconstructed, original, rtol=1e-12, atol=1e-12))}
