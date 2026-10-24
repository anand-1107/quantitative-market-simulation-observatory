"""Chronological, train-only model tournament evaluation for existing models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import kstest, norm, poisson

from quantum_market_lab.data.features import compute_log_returns, drawdown
from quantum_market_lab.models.calibration import (
    ANNUALIZATION_PERIODS, calibrate_garch, calibrate_merton, calibrate_ou, estimate_gbm,
)
from quantum_market_lab.models.diagnostics import empirical_var_es
from quantum_market_lab.models.garch import simulate_garch
from quantum_market_lab.models.jump_diffusion import simulate_jump_diffusion
from quantum_market_lab.models.ou import simulate_ou
from quantum_market_lab.gbm import simulate_gbm


def _normal_tail(mean: float, variance: float, confidence: float) -> tuple[float, float]:
    sd = max(float(np.sqrt(variance)), 1e-14)
    alpha = 1.0 - confidence
    q = float(norm.ppf(alpha, loc=mean, scale=sd))
    exp_lower = float(np.exp(mean + .5 * variance) * norm.cdf((q - mean - variance) / sd))
    return 1.0 - float(np.exp(q)), 1.0 - exp_lower / alpha


def _mixture_tail(weights: np.ndarray, means: np.ndarray, variances: np.ndarray,
                  confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    sds = np.sqrt(variances)
    def cdf(x: float) -> float:
        return float(np.sum(weights * norm.cdf((x - means) / sds)))
    lower = float(np.min(means - 12 * sds))
    upper = float(np.max(means + 12 * sds))
    q = float(brentq(lambda x: cdf(x) - alpha, lower, upper, xtol=1e-12))
    truncated_exp = np.exp(means + .5 * variances) * norm.cdf((q - means - variances) / sds)
    return 1.0 - float(np.exp(q)), 1.0 - float(weights @ truncated_exp) / alpha


def _score_forecasts(observed: np.ndarray, logpdf: np.ndarray, pits: np.ndarray,
                     means: np.ndarray, variances: np.ndarray, vars_: np.ndarray, es_: np.ndarray) -> dict:
    losses = -np.expm1(observed)
    historical = empirical_var_es(losses, .95)
    return {
        "sample_size": int(len(observed)),
        "mean_negative_log_score_nats_per_observation": float(np.mean(-logpdf)),
        "pit_ks_statistic": float(kstest(pits, "uniform").statistic),
        "volatility_qlike_mean": float(np.mean(np.log(variances) + (observed - means) ** 2 / variances)),
        "observed_test_period_log_return_mean": float(np.mean(observed)),
        "observed_test_period_log_return_volatility": float(np.std(observed, ddof=1)),
        "observed_one_period_95pct_var_loss_fraction": historical["var"],
        "observed_one_period_95pct_expected_shortfall_loss_fraction": historical["expected_shortfall"],
        "mean_predicted_one_period_95pct_var_loss_fraction": float(np.mean(vars_)),
        "mean_predicted_one_period_95pct_expected_shortfall_loss_fraction": float(np.mean(es_)),
        "one_period_95pct_var_breach_rate": float(np.mean(losses > vars_)),
        "historical_risk_tail_observations": historical["tail_observation_count"],
        "return_convention": "close-to-close natural-log returns for forecasts; losses use L=1-exp(log_return), the negative simple return",
        "risk_convention": "one-step 95% predictive loss VaR/ES; ES is conditional lower-tail log-return loss under the fitted predictive distribution",
        "volatility_score_convention": "mean log(predicted variance) + squared forecast error / predicted variance (QLIKE form); lower is better",
    }


def run_model_tournament(*, ticker: str, train: pd.DataFrame, test: pd.DataFrame,
                         train_start: str, train_end: str, test_start: str, test_end: str,
                         interval: str, models: list[str], paths: int, seed: int) -> dict:
    """Fit only to ``train`` and score conditional forecasts over ``test``.

    Test one-step forecasts are rolling-origin: each realized prior test return
    may update a model's state/variance only after its forecast is scored.
    """
    if interval not in ANNUALIZATION_PERIODS:
        raise ValueError("Tournament supports calibration intervals 1d, 5d, 1wk, 1mo, and 3mo only")
    try:
        requested = [pd.Timestamp(value).date() for value in (train_start, train_end, test_start, test_end)]
    except (TypeError, ValueError) as exc:
        raise ValueError("Tournament dates must be valid ISO calendar dates") from exc
    if not (requested[0] < requested[1] < requested[2] < requested[3]):
        raise ValueError("Tournament periods must satisfy train start < train end < test start < test end")
    if len(train) < 3 or len(test) < 3:
        raise ValueError("Training and test windows each require at least three valid prices")
    if not train.index.is_monotonic_increasing or not test.index.is_monotonic_increasing:
        raise ValueError("Tournament observations must be chronological")
    if train.index.max() >= test.index.min():
        raise ValueError("Training and test observations overlap or are out of order")
    if train.index.max().date() >= pd.Timestamp(test_start).date():
        raise ValueError("Test window must begin strictly after the final observed training date")
    if ((test.index.date < requested[2]).any() or (test.index.date >= requested[3]).any()):
        raise ValueError("Test observations fall outside the requested half-open test period")
    annual_periods = ANNUALIZATION_PERIODS[interval]
    dt = 1.0 / annual_periods
    train_prices = train["adj_close"] if "adj_close" in train else train["close"]
    test_prices = test["adj_close"] if "adj_close" in test else test["close"]
    if (not np.isfinite(train_prices.to_numpy(dtype=float)).all() or not np.isfinite(test_prices.to_numpy(dtype=float)).all()
            or (train_prices <= 0).any() or (test_prices <= 0).any()):
        raise ValueError("Training and test prices must be finite and strictly positive")
    observed = compute_log_returns(test, "adj_close" if "adj_close" in test else "close").dropna().to_numpy(dtype=float)
    if len(observed) < 2 or not np.isfinite(observed).all():
        raise ValueError("Test window must contain at least two finite within-window log returns")
    actual_simple_total = float(test_prices.iloc[-1] / test_prices.iloc[0] - 1)
    _, actual_drawdown = drawdown(test_prices.reset_index(drop=True), input_type="price")
    results = []
    forecast_specs: dict[str, dict] = {}

    for model in models:
        try:
            if model == "gbm":
                fit_obj = estimate_gbm(ticker, train, interval, train_start, train_end)
                fit = fit_obj.as_dict()
                price_field = fit["price_field"]
                mean = float(fit["mean_period_log_return"])
                variance = float(fit["sample_std_period_log_return"] ** 2)
                forecast_means = np.full(len(observed), mean)
                logpdf = norm.logpdf(observed, loc=mean, scale=np.sqrt(variance))
                pits = norm.cdf(observed, loc=mean, scale=np.sqrt(variance))
                v, e = _normal_tail(mean, variance, .95)
                variances = np.full(len(observed), variance)
                var_forecasts, es_forecasts = np.full(len(observed), v), np.full(len(observed), e)
                forecast_specs[model] = {"kind": "normal", "means": forecast_means, "variances": variances}
                research_path = {"initial_price": 100.0}
                horizon_years = len(observed) * dt
                _, sim = simulate_gbm(float(test_prices.iloc[0]), float(fit["drift"]), float(fit["volatility"]),
                                      horizon_years, len(observed), paths, seed)
                simulated_prices = sim
            elif model == "ou":
                fit = calibrate_ou(ticker, train, series="log_return", price_field="auto", interval=interval,
                                   start=train_start, end=train_end)
                if fit.get("optimizer_success") is False:
                    raise ValueError(f"OU calibration optimizer did not converge: {fit.get('optimizer_message', 'unknown optimizer status')}")
                price_field = fit["price_field"]
                # At the first within-test forecast origin, the first test close
                # has already been observed. Its incoming return is the current
                # OU state and does not enter the fitted parameter sample.
                state = float(np.log(test_prices.iloc[0] / train_prices.iloc[-1]))
                theta, level, sigma = float(fit["mean_reversion_speed"]), float(fit["mean_level"]), float(fit["volatility"])
                decay = np.exp(-theta * dt)
                conditional_variance = sigma**2 * (-np.expm1(-2 * theta * dt)) / (2 * theta)
                means = np.empty(len(observed)); variances = np.full(len(observed), conditional_variance)
                for i, value in enumerate(observed):
                    means[i] = level + (state - level) * decay
                    state = float(value)
                sd = np.sqrt(conditional_variance)
                forecast_means = means
                logpdf = norm.logpdf(observed, loc=means, scale=sd)
                pits = norm.cdf(observed, loc=means, scale=sd)
                tails = [_normal_tail(float(m), conditional_variance, .95) for m in means]
                var_forecasts = np.asarray([item[0] for item in tails]); es_forecasts = np.asarray([item[1] for item in tails])
                forecast_specs[model] = {"kind": "normal", "means": forecast_means, "variances": variances}
                initial_state = float(np.log(test_prices.iloc[0] / train_prices.iloc[-1]))
                research_path = {"initial_state": initial_state}
                _, states = simulate_ou(initial_state, level, theta, sigma, len(observed) * dt,
                                        len(observed), paths, seed)
                simulated_prices = float(test_prices.iloc[0]) * np.exp(np.cumsum(states[:, 1:], axis=1))
                simulated_prices = np.column_stack((np.full(paths, float(test_prices.iloc[0])), simulated_prices))
            elif model == "jump_diffusion":
                fit = calibrate_merton(ticker, train, price_field="auto", interval=interval,
                                       start=train_start, end=train_end)
                if fit.get("optimizer_success") is False:
                    raise ValueError(f"Merton calibration optimizer did not converge: {fit.get('optimizer_message', 'unknown optimizer status')}")
                price_field = fit["price_field"]
                lam, jm, js = float(fit["jump_intensity"]), float(fit["mean_log_jump"]), float(fit["jump_volatility"])
                sigma, drift = float(fit["volatility"]), float(fit["drift"])
                kappa = float(np.expm1(jm + .5 * js**2))
                max_count = max(1, min(100, int(poisson.ppf(.999999, max(lam * dt, 1e-12)))))
                counts = np.arange(max_count + 1)
                weights = poisson.pmf(counts, lam * dt)
                weights = weights / weights.sum()
                means = (drift - lam * kappa - .5 * sigma**2) * dt + counts * jm
                component_variances = sigma**2 * dt + counts * js**2
                sds = np.sqrt(component_variances)
                component_logpdf = norm.logpdf(observed[:, None], means[None, :], sds[None, :]) + np.log(np.maximum(weights, 1e-300))[None, :]
                logpdf = logsumexp(component_logpdf, axis=1)
                pits = np.sum(weights[None, :] * norm.cdf((observed[:, None] - means[None, :]) / sds[None, :]), axis=1)
                qvar, qes = _mixture_tail(weights, means, component_variances, .95)
                var_forecasts, es_forecasts = np.full(len(observed), qvar), np.full(len(observed), qes)
                mixture_mean = float(weights @ means)
                mixture_variance = float(weights @ (component_variances + means**2) - mixture_mean**2)
                variances = np.full(len(observed), mixture_variance)
                forecast_means = np.full(len(observed), mixture_mean)
                forecast_specs[model] = {"kind": "mixture", "weights": weights,
                    "component_means": means, "component_variances": component_variances}
                research_path = {"initial_price": 100.0}
                _, simulated_prices = simulate_jump_diffusion(float(test_prices.iloc[0]), drift, sigma, lam, jm, js,
                    len(observed) * dt, len(observed), paths, seed)
            elif model == "garch":
                fit = calibrate_garch(ticker, train, price_field="auto", interval=interval,
                                      start=train_start, end=train_end)
                if fit.get("optimizer_success") is False:
                    raise ValueError(f"GARCH calibration optimizer did not converge: {fit.get('optimizer_message', 'unknown optimizer status')}")
                price_field = fit["price_field"]
                mean, omega, alpha, beta = (float(fit[key]) for key in ("mean", "omega", "alpha", "beta"))
                fit_vols = np.asarray([point["value"] for point in fit["conditional_volatility"]], dtype=float)
                h = float(fit_vols[-1] ** 2)
                train_returns = compute_log_returns(train, price_field).dropna().to_numpy(dtype=float)
                h = omega + alpha * (train_returns[-1] - mean) ** 2 + beta * h
                initial_test_variance = h
                research_path = {"initial_variance": initial_test_variance}
                forecasts = np.empty(len(observed))
                for i, value in enumerate(observed):
                    forecasts[i] = h
                    h = omega + alpha * (value - mean) ** 2 + beta * h
                variances = forecasts
                forecast_means = np.full(len(observed), mean)
                sds = np.sqrt(forecasts)
                logpdf = norm.logpdf(observed, loc=mean, scale=sds)
                pits = norm.cdf(observed, loc=mean, scale=sds)
                tails = [_normal_tail(mean, float(v), .95) for v in forecasts]
                var_forecasts = np.asarray([item[0] for item in tails]); es_forecasts = np.asarray([item[1] for item in tails])
                forecast_specs[model] = {"kind": "normal", "means": forecast_means, "variances": variances}
                simulated_returns, _ = simulate_garch(omega, alpha, beta, len(observed), paths, mean,
                    initial_test_variance, seed, "log")
                simulated_prices = float(test_prices.iloc[0]) * np.exp(np.column_stack((np.zeros(paths), np.cumsum(simulated_returns, axis=1))))
            else:
                raise ValueError(f"Unsupported tournament model: {model}")

            sim_returns = simulated_prices[:, -1] / simulated_prices[:, 0] - 1.0
            risk = empirical_var_es(-sim_returns, .95)
            simulated_drawdowns = np.asarray([drawdown(pd.Series(row), input_type="price")[1] for row in simulated_prices])
            metrics = _score_forecasts(observed, np.asarray(logpdf), np.clip(np.asarray(pits), 1e-15, 1 - 1e-15),
                                      np.asarray(forecast_means), np.asarray(variances), np.asarray(var_forecasts), np.asarray(es_forecasts))
            metrics.update({
                "test_period_total_simple_return": actual_simple_total,
                "test_period_maximum_drawdown": float(actual_drawdown),
                "simulated_terminal_simple_return_p05_p50_p95": [float(x) for x in np.quantile(sim_returns, [.05, .5, .95])],
                "simulated_terminal_95pct_var_loss_fraction": float(risk["var"]),
                "simulated_terminal_95pct_expected_shortfall_loss_fraction": float(risk["expected_shortfall"]),
                "simulated_maximum_drawdown_p05_p50_p95": [float(x) for x in np.quantile(simulated_drawdowns, [.05, .5, .95])],
                "simulation_paths": int(paths), "seed": int(seed),
            })
            display_fit = {key: value for key, value in fit.items() if key != "conditional_volatility"}
            forecast_specs[model]["risk_var_95"] = np.asarray(var_forecasts)
            forecast_specs[model]["risk_es_95"] = np.asarray(es_forecasts)
            results.append({"model": model, "status": "completed", "parameters": display_fit,
                            "metrics": metrics, "warnings": list(fit.get("warnings", [])),
                            "assumptions": fit.get("assumptions", []),
                            "limitation": fit.get("limitations", ""), "price_field": price_field,
                            "research_path": research_path})
        except Exception as exc:  # Per-model sample requirements/fit failures should not hide valid models.
            results.append({"model": model, "status": "unavailable", "error": str(exc),
                            "parameters": None, "metrics": None, "warnings": [], "assumptions": [], "limitation": ""})

    # Compact, same-origin predictive densities let the research view inspect
    # genuine rolling-origin forecasts without exporting the entire test set.
    successful_specs = {key: value for key, value in forecast_specs.items()
                        if any(item["model"] == key and item["status"] == "completed" for item in results)}
    origin_count = min(64, len(observed))
    origin_indices = np.unique(np.linspace(0, len(observed) - 1, origin_count, dtype=int))
    support_low, support_high = [], []
    for spec in successful_specs.values():
        if spec["kind"] == "normal":
            means_array = np.asarray(spec["means"])[origin_indices]
            sds_array = np.sqrt(np.asarray(spec["variances"])[origin_indices])
            support_low.extend((means_array - 5 * sds_array).tolist())
            support_high.extend((means_array + 5 * sds_array).tolist())
        else:
            component_means, component_variances = spec["component_means"], spec["component_variances"]
            component_sds = np.sqrt(component_variances)
            support_low.extend((component_means - 5 * component_sds).tolist())
            support_high.extend((component_means + 5 * component_sds).tolist())
    return_grid = np.linspace(float(min(support_low)), float(max(support_high)), 49) if successful_specs else np.asarray([])
    confidence_levels = [.90, .95, .975]
    for item in results:
        if item["status"] != "completed":
            item["forecast_distribution_surface"] = None
            item["forecast_risk_surface"] = None
            continue
        spec = successful_specs[item["model"]]
        densities, probability_negative, var_by_confidence, es_95 = [], [], [], []
        for index in origin_indices:
            if spec["kind"] == "normal":
                mean_value = float(spec["means"][index])
                variance_value = float(spec["variances"][index])
                density = norm.pdf(return_grid, loc=mean_value, scale=np.sqrt(variance_value))
                probability_negative.append(float(norm.cdf(0, loc=mean_value, scale=np.sqrt(variance_value))))
                tails = [_normal_tail(mean_value, variance_value, confidence) for confidence in confidence_levels]
            else:
                weights_value = spec["weights"]
                means_value = spec["component_means"]
                variances_value = spec["component_variances"]
                sds_value = np.sqrt(variances_value)
                density = np.sum(weights_value[:, None] * norm.pdf(
                    (return_grid[None, :] - means_value[:, None]) / sds_value[:, None]) / sds_value[:, None], axis=0)
                probability_negative.append(float(np.sum(weights_value * norm.cdf((0 - means_value) / sds_value))))
                tails = [_mixture_tail(weights_value, means_value, variances_value, confidence) for confidence in confidence_levels]
            densities.append([float(value) for value in density])
            var_by_confidence.append([float(value[0]) for value in tails])
            es_95.append(float(tails[1][1]))
        item["forecast_distribution_surface"] = {
            "evaluation_dates": [test.index[index + 1].date().isoformat() for index in origin_indices],
            "origin_indices": [int(index + 1) for index in origin_indices],
            "log_return_grid": [float(value) for value in return_grid],
            "density_by_origin": densities,
            "density_method": "Analytical one-step conditional predictive density from the fitted model; no future realized returns are used to form density values.",
            "unit": "natural_log_return_per_observation_interval",
        }
        item["forecast_risk_surface"] = {
            "evaluation_dates": [test.index[index + 1].date().isoformat() for index in origin_indices],
            "origin_indices": [int(index + 1) for index in origin_indices],
            "confidence_levels": confidence_levels,
            "var_loss_fraction_by_origin": var_by_confidence,
            "expected_shortfall_95_loss_fraction_by_origin": es_95,
            "probability_negative_return_by_origin": probability_negative,
            "loss_definition": "L=1-exp(r), where r is the one-step natural-log return; values are predictive and conditional on information at each origin.",
        }

    return {
        "ticker": ticker.upper(), "interval": interval,
        "training_period": {"start": train_start, "end_exclusive": train_end,
                            "first_observation": train.index.min().date().isoformat(),
                            "last_observation": train.index.max().date().isoformat(), "observation_count": len(train)},
        "test_period": {"start": test_start, "end_exclusive": test_end,
                        "first_observation": test.index.min().date().isoformat(),
                        "last_observation": test.index.max().date().isoformat(), "observation_count": len(test),
                        "return_count": len(observed)},
        "comparison_method": "Train-only historical calibration; the first test close is the forecast origin and initializes model state; rolling-origin one-step predictive distributions are scored on subsequent within-test close-to-close log returns, with each realized test return updating state only after its forecast.",
        "directly_comparable_metrics": ["mean_negative_log_score_nats_per_observation", "pit_ks_statistic", "volatility_qlike_mean", "one_period_95pct_var_breach_rate"],
        "model_specific_metrics": ["test_period_total_simple_return", "test_period_maximum_drawdown", "simulated_terminal_simple_return_p05_p50_p95", "simulated_terminal_95pct_var_loss_fraction", "simulated_terminal_95pct_expected_shortfall_loss_fraction", "simulated_maximum_drawdown_p05_p50_p95"],
        "reproducibility": {"seed": int(seed), "paths": int(paths), "same_runtime_seeded_replay": True},
        "results": results,
    }
