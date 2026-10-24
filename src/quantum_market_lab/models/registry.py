"""Discoverable stochastic-model metadata, without importing model implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelDescription:
    name: str
    description: str
    parameters: tuple[str, ...]
    output_convention: str
    calibration_status: str


MODEL_REGISTRY = {
    "gbm": ModelDescription("gbm", "Geometric Brownian Motion for positive prices.",
        ("initial_price", "drift", "volatility", "years", "steps", "paths", "seed"), "times + prices (paths, steps+1); μ and σ annualized, dt in years", "implemented_historical_moments"),
    "ou": ModelDescription("ou", "Ornstein-Uhlenbeck mean-reverting state process.",
        ("initial_value", "mean_level", "mean_reversion_speed", "volatility", "years", "steps", "paths", "seed"), "times + levels (paths, steps+1); θ per year, σ in level units/√year", "implemented_conditional_least_squares"),
    "jump_diffusion": ModelDescription("jump_diffusion", "Merton compound-Poisson jump diffusion for prices.",
        ("initial_price", "drift", "volatility", "jump_intensity", "mean_log_jump", "jump_volatility", "years", "steps", "paths", "seed"), "times + prices (paths, steps+1); annual drift/diffusion and jump intensity", "implemented_compound_poisson_mle"),
    "garch": ModelDescription("garch", "Discrete-period GARCH(1,1) return and variance simulation.",
        ("omega", "alpha", "beta", "steps", "paths", "mean", "initial_variance", "seed", "return_convention"), "returns + variances (paths, steps); all parameters per observation period", "implemented_conditional_mle"),
}


def list_models() -> list[dict]:
    return [asdict(description) for description in MODEL_REGISTRY.values()]


def get_model(name: str) -> ModelDescription:
    try: return MODEL_REGISTRY[name.lower()]
    except (KeyError, AttributeError): raise KeyError(f"unknown stochastic model: {name}") from None
