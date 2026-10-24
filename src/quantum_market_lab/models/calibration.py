"""Historical parameter estimation and descriptive GBM evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import poisson

from quantum_market_lab.data.features import compute_log_returns, drawdown
from quantum_market_lab.data.market_data import MarketDataQualityError, validate_market_data
from quantum_market_lab.models.diagnostics import empirical_var_es


ANNUALIZATION_PERIODS = {"1d": 252, "5d": 52, "1wk": 52, "1mo": 12, "3mo": 4}
SUPPORTED_CALIBRATION_INTERVALS = frozenset(ANNUALIZATION_PERIODS)


def historical_series(frame: pd.DataFrame, price_field: str = "auto") -> tuple[str, pd.Series]:
    """Validate source observations and select adjusted/close prices explicitly."""
    report = validate_market_data(frame)
    if not report.valid:
        raise MarketDataQualityError("Historical observations failed market-data validation")
    if not frame.index.is_monotonic_increasing:
        raise MarketDataQualityError("Historical observations must be in strictly chronological order")
    field = ("adj_close" if "adj_close" in frame.columns else "close") if price_field == "auto" else price_field
    if field not in {"adj_close", "close"} or field not in frame.columns:
        raise MarketDataQualityError(f"Historical price field '{field}' is unavailable")
    prices = pd.to_numeric(frame[field], errors="coerce")
    if prices.isna().any() or not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any():
        raise MarketDataQualityError(f"Historical {field} prices contain missing, non-finite, or non-positive values")
    return field, prices


def _validate_window(frame: pd.DataFrame, start: str, end: str) -> None:
    try:
        start_date, end_date = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    except (TypeError, ValueError) as exc:
        raise ValueError("estimation dates must be valid ISO calendar dates") from exc
    if start_date >= end_date:
        raise ValueError("estimation start must be earlier than end")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.isna().any():
        raise MarketDataQualityError("Historical observations must have valid datetime indices")
    if not frame.index.is_monotonic_increasing:
        raise MarketDataQualityError("Historical observations must be in strictly chronological order")
    if len(frame) and ((frame.index.date < start_date).any() or (frame.index.date >= end_date).any()):
        raise MarketDataQualityError("Historical observations fall outside the requested half-open estimation date range")


def _historical_returns(frame: pd.DataFrame, price_field: str) -> tuple[pd.Series, pd.Series]:
    field, prices = historical_series(frame, price_field)
    returns = np.log(prices).diff().dropna()
    if len(returns) == 0 or not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("Historical price series did not produce finite returns")
    return prices, returns


def calibrate_ou(ticker: str, frame: pd.DataFrame, *, series: str, price_field: str,
                 interval: str, start: str, end: str, allow_price_series: bool = False) -> dict:
    """Fit exact-transition OU parameters by conditional least squares / AR(1).

    Observations are equally spaced by dt=1/N years. The fitted AR coefficient
    phi is converted with theta=-log(phi)/dt, level=intercept/(1-phi), and the
    innovation variance is mapped to the exact OU diffusion variance.
    """
    if interval not in ANNUALIZATION_PERIODS:
        raise ValueError("OU calibration supports 1d, 5d, 1wk, 1mo, and 3mo intervals only")
    _validate_window(frame, start, end)
    field, prices = historical_series(frame, price_field)
    if series == "log_return":
        values = np.log(prices).diff().dropna()
        units = "log return per observation interval"
    elif series == "log_price":
        if not allow_price_series:
            raise ValueError("log-price OU calibration requires explicit confirmation that this series is suitable and mean reverting")
        values = np.log(prices)
        units = "natural-log price level"
    elif series == "price":
        if not allow_price_series:
            raise ValueError("raw-price OU calibration requires explicit confirmation that this series is suitable and mean reverting")
        values = prices
        units = "price units"
    else:
        raise ValueError("series must be 'log_return', 'log_price', or 'price'")
    x = values.to_numpy(dtype=np.float64)
    if len(x) < 20:
        raise ValueError("OU conditional least squares requires at least 20 observations")
    if not np.isfinite(x).all() or np.ptp(x) <= np.finfo(float).eps * max(1.0, abs(float(np.mean(x)))):
        raise ValueError("OU input series must be finite and non-constant")
    x0, x1 = x[:-1], x[1:]
    centered = x0 - x0.mean()
    denom = float(centered @ centered)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("OU autoregression is numerically unidentified")
    phi = float(centered @ (x1 - x1.mean()) / denom)
    intercept = float(x1.mean() - phi * x0.mean())
    if not np.isfinite(phi) or not 0 < phi < 1:
        raise ValueError("OU mean reversion is not identified: fitted AR(1) coefficient must lie strictly between 0 and 1")
    dt = 1.0 / ANNUALIZATION_PERIODS[interval]
    theta = -float(np.log(phi)) / dt
    mean_level = intercept / (1.0 - phi)
    residuals = x1 - (intercept + phi * x0)
    innovation_variance = float(residuals @ residuals / (len(x) - 2))
    sigma_squared = innovation_variance * 2.0 * theta / (-np.expm1(-2.0 * theta * dt))
    sigma = float(np.sqrt(sigma_squared)) if sigma_squared > 0 else float("nan")
    if not np.isfinite([theta, mean_level, sigma, innovation_variance]).all() or theta <= 0 or sigma <= 0:
        raise ValueError("OU parameter conversion produced invalid or non-positive estimates")
    warnings = []
    if phi > .995:
        warnings.append("AR(1) coefficient is near one; continuous-time mean reversion is weakly identified.")
    if series in {"price", "log_price"}:
        warnings.append("Price-level OU suitability was explicitly asserted by the caller; equity price levels are not generally stationary.")
    if series == "log_return":
        warnings.append("Log returns were selected as the OU state series; their mean-reverting suitability is an empirical assumption.")
    return {
        "model": "ou", "ticker": ticker.upper(), "estimation_start": start, "estimation_end": end,
        "interval": interval, "price_field": field, "input_series": series, "input_units": units,
        "observation_count": len(x), "return_count": len(x) - 1, "dt_years": dt,
        "ar1_intercept": intercept, "ar1_coefficient": phi, "mean_level": float(mean_level),
        "mean_reversion_speed": float(theta), "volatility": sigma,
        "initial_value": float(x[-1]), "conditional_innovation_variance": innovation_variance,
        "calibrated_at": datetime.now(timezone.utc).isoformat(), "method": "ou_conditional_least_squares_ar1_v1",
        "optimizer_success": True, "warnings": warnings,
        "assumptions": ["Equally spaced observations with dt=1/annualization_periods years.",
                        "Gaussian OU transitions with constant parameters.",
                        "The observed series is an economically suitable mean-reverting state; raw prices are never assumed suitable automatically."],
    }


def calibrate_merton(ticker: str, frame: pd.DataFrame, *, price_field: str, interval: str,
                     start: str, end: str) -> dict:
    """Fit the exact discrete-time Merton compound-Poisson Gaussian mixture by MLE."""
    if interval not in ANNUALIZATION_PERIODS:
        raise ValueError("Merton calibration supports 1d, 5d, 1wk, 1mo, and 3mo intervals only")
    _validate_window(frame, start, end)
    prices, returns = _historical_returns(frame, price_field)
    x = returns.to_numpy(dtype=np.float64)
    if len(x) < 250:
        raise ValueError("Merton mixture calibration requires at least 250 log-return observations")
    dt = 1.0 / ANNUALIZATION_PERIODS[interval]
    per_year = ANNUALIZATION_PERIODS[interval]
    period_sd = float(np.std(x, ddof=1))
    if not np.isfinite(period_sd) or period_sd <= 0:
        raise ValueError("Merton calibration requires finite non-zero return variation")
    mean0 = float(np.mean(x) * per_year + 0.5 * period_sd**2 * per_year)
    sigma0 = float(np.clip(period_sd * np.sqrt(per_year) * .75, .005, 1.5))
    jump_scale = max(period_sd * 3.0, .01)
    max_k = min(100, int(poisson.ppf(.999999, 10.0 * dt)))
    counts = np.arange(max(1, max_k) + 1, dtype=np.float64)
    log_factorial = gammaln(counts + 1.0)

    # Parameter order follows the simulator: annual expected price drift mu,
    # annual diffusion sigma, annual Poisson intensity lambda, mean and SD of
    # normally distributed log jump sizes.
    bounds = [(-2.0, 2.0), (.003, 3.0), (1e-5, 10.0), (-2.0, 2.0), (.001, 2.0)]

    def objective(params: np.ndarray) -> float:
        mu, sigma, intensity, jump_mean, jump_sd = params
        compensator = np.expm1(jump_mean + .5 * jump_sd**2)
        base_mean = (mu - intensity * compensator - .5 * sigma**2) * dt
        variance = sigma**2 * dt + counts * jump_sd**2
        poisson_mean = max(intensity * dt, 1e-300)
        log_weight = -poisson_mean + counts * np.log(poisson_mean) - log_factorial
        residual = x[:, None] - base_mean - counts[None, :] * jump_mean
        component = log_weight[None, :] - .5 * (np.log(2 * np.pi * variance)[None, :] + residual**2 / variance[None, :])
        value = -float(np.sum(logsumexp(component, axis=1)))
        return value if np.isfinite(value) else 1e100

    # Deterministic starts reduce sensitivity to mixture-likelihood local optima.
    extreme = np.abs(x - np.median(x)) > max(3.0 * period_sd, .02)
    tail = x[extreme]
    jump_mean0 = float(np.clip(np.median(tail) if len(tail) else -jump_scale, -1.0, 1.0))
    jump_sd0 = float(np.clip(np.std(tail, ddof=1) if len(tail) > 1 else jump_scale, .01, .5))
    starts = [
        [np.clip(mean0, -1.9, 1.9), sigma0, .25, jump_mean0, jump_sd0],
        [np.clip(mean0, -1.9, 1.9), sigma0, 1.0, jump_mean0, max(jump_sd0, .08)],
        [np.clip(mean0, -1.9, 1.9), sigma0, 3.0, np.clip(-2 * jump_scale, -1.5, 1.5), .15],
        [np.clip(mean0, -1.9, 1.9), sigma0, 1.0, 0.0, .15],
    ]
    fits = [minimize(objective, np.asarray(initial), method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": 1200, "ftol": 1e-10, "maxls": 40}) for initial in starts]
    finite_fits = [fit for fit in fits if np.isfinite(fit.fun) and np.all(np.isfinite(fit.x))]
    if not finite_fits:
        raise ValueError("Merton maximum-likelihood optimizer failed to produce finite estimates")
    best = min(finite_fits, key=lambda fit: fit.fun)
    mu, sigma, intensity, jump_mean, jump_sd = map(float, best.x)
    expected_jumps = intensity * len(x) * dt
    compensator = np.expm1(jump_mean + .5 * jump_sd**2)
    base_mean = (mu - intensity * compensator - .5 * sigma**2) * dt
    variance = sigma**2 * dt + counts * jump_sd**2
    log_weight = -intensity * dt + counts * np.log(max(intensity * dt, 1e-300)) - log_factorial
    residual = x[:, None] - base_mean - counts[None, :] * jump_mean
    component = log_weight[None, :] - .5 * (np.log(2 * np.pi * variance)[None, :] + residual**2 / variance[None, :])
    posterior = np.exp(component - logsumexp(component, axis=1, keepdims=True))
    posterior_jump_count = float(np.sum(posterior @ counts))
    warnings = []
    if not best.success:
        warnings.append(f"Optimizer did not converge: {best.message}")
    if posterior_jump_count < 20:
        warnings.append("Fewer than 20 jumps are supported by the fitted posterior in this sample; jump parameters are weakly identified.")
    if intensity < .01 or jump_sd < .003:
        warnings.append("Jump component is near a boundary; a diffusion-only explanation may fit nearly as well.")
    if any(np.isclose(param, bound, rtol=0, atol=1e-5) for param, pair in zip(best.x, bounds) for bound in pair):
        warnings.append("At least one estimate is near its numerical bound; inspect parameter-identification uncertainty.")
    return {
        "model": "jump_diffusion", "ticker": ticker.upper(), "estimation_start": start, "estimation_end": end,
        "interval": interval, "price_field": prices.name, "input_series": "close_to_close_log_return",
        "observation_count": len(prices), "return_count": len(x), "dt_years": dt,
        "drift": mu, "volatility": sigma, "jump_intensity": intensity,
        "mean_log_jump": jump_mean, "jump_volatility": jump_sd,
        "initial_price": float(prices.iloc[-1]), "log_likelihood": float(-best.fun),
        "aic": float(2 * 5 + 2 * best.fun), "estimated_expected_jumps_in_sample": float(expected_jumps),
        "posterior_expected_jump_count_in_sample": posterior_jump_count,
        "optimizer_success": bool(best.success), "optimizer_status": int(best.status),
        "optimizer_message": str(best.message), "optimizer_iterations": int(best.nit),
        "calibrated_at": datetime.now(timezone.utc).isoformat(), "method": "merton_exact_poisson_normal_mixture_mle_v1",
        "warnings": warnings,
        "assumptions": ["Each interval return follows the simulator's exact compound-Poisson/Gaussian increment mixture.",
                        "Jump counts are Poisson with annual intensity; log jump sizes are iid normal.",
                        "Drift is expected proportional price growth and includes the simulator's jump compensator.",
                        "The finite mixture sums counts through a conservative Poisson-tail truncation based on maximum allowed intensity.",
                        "A three-standard-deviation tail rule is used only to initialize optimizer starts; returns are not classified as confirmed jumps."],
        "limitations": "Extreme returns are not labeled as confirmed jumps; mixture MLE can be weakly identified and sensitive to sample length.",
    }


def calibrate_garch(ticker: str, frame: pd.DataFrame, *, price_field: str, interval: str,
                    start: str, end: str) -> dict:
    """Gaussian conditional maximum likelihood for the existing stationary GARCH(1,1)."""
    if interval not in ANNUALIZATION_PERIODS:
        raise ValueError("GARCH calibration supports 1d, 5d, 1wk, 1mo, and 3mo intervals only")
    _validate_window(frame, start, end)
    prices, returns = _historical_returns(frame, price_field)
    x = returns.to_numpy(dtype=np.float64)
    if len(x) < 100:
        raise ValueError("GARCH(1,1) maximum likelihood requires at least 100 log-return observations")
    if not np.isfinite(x).all() or np.var(x) <= 0:
        raise ValueError("GARCH input returns must be finite with positive variance")
    sample_var = float(np.var(x, ddof=1))
    eps = 1e-8

    def sigmoid(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(value, -40, 40))))

    def decode(z: np.ndarray) -> tuple[float, float, float, float, float]:
        mean = float(z[0])
        omega = float(np.exp(z[1]))
        persistence = .999 * sigmoid(float(z[2]))
        share = sigmoid(float(z[3]))
        alpha, beta = persistence * share, persistence * (1.0 - share)
        initial_variance = omega / (1.0 - persistence)
        return mean, omega, alpha, beta, initial_variance

    def objective(z: np.ndarray) -> float:
        mean, omega, alpha, beta, variance0 = decode(z)
        if not np.isfinite(variance0) or variance0 <= 0:
            return 1e100
        variance = variance0
        residual = x[0] - mean
        total = 0.5 * (np.log(2 * np.pi) + np.log(variance) + residual**2 / variance)
        for value in x[1:]:
            variance = omega + alpha * residual**2 + beta * variance
            if not np.isfinite(variance) or variance <= 0:
                return 1e100
            residual = value - mean
            total += 0.5 * (np.log(2 * np.pi) + np.log(variance) + residual**2 / variance)
        return float(total) if np.isfinite(total) else 1e100

    # Persistence is parametrically kept below one; alpha,beta are positive.
    starts = []
    for persistence, alpha_share in ((.90, .10), (.97, .05), (.75, .20)):
        total_logit = np.log((persistence / .999) / (1 - persistence / .999))
        share_logit = np.log(alpha_share / (1 - alpha_share))
        starts.append(np.array([float(np.mean(x)), np.log(max(sample_var * (1 - persistence), 1e-12)), total_logit, share_logit]))
    fits = [minimize(objective, start_z, method="L-BFGS-B",
                     bounds=[(float(np.mean(x) - 10 * np.sqrt(sample_var)), float(np.mean(x) + 10 * np.sqrt(sample_var))),
                             (np.log(max(sample_var * 1e-8, 1e-16)), np.log(max(sample_var * 10, 1e-12))),
                             (-12, 8), (-12, 8)], options={"maxiter": 2000, "ftol": 1e-11, "maxls": 40}) for start_z in starts]
    finite_fits = [fit for fit in fits if np.isfinite(fit.fun) and np.all(np.isfinite(fit.x))]
    if not finite_fits:
        raise ValueError("GARCH maximum-likelihood optimizer failed to produce finite estimates")
    best = min(finite_fits, key=lambda fit: fit.fun)
    mean, omega, alpha, beta, variance = decode(best.x)
    conditional_volatility = []
    for value in x:
        conditional_volatility.append(float(np.sqrt(variance)))
        residual = value - mean
        variance = omega + alpha * residual**2 + beta * variance
    warnings = []
    if not best.success:
        warnings.append(f"Optimizer did not converge: {best.message}")
    if alpha + beta > .99:
        warnings.append("Estimated persistence is close to the stationarity boundary; long-run variance is highly sensitive.")
    dates = returns.index
    return {
        "model": "garch", "ticker": ticker.upper(), "estimation_start": start, "estimation_end": end,
        "interval": interval, "price_field": prices.name, "input_series": "close_to_close_log_return",
        "return_convention": "log", "observation_count": len(prices), "return_count": len(x),
        "mean": mean, "omega": omega, "alpha": alpha, "beta": beta,
        "stationary": alpha + beta < 1.0, "initial_variance": omega / (1.0 - alpha - beta),
        "conditional_volatility": [{"date": index.date().isoformat(), "value": vol} for index, vol in zip(dates, conditional_volatility)],
        "negative_log_likelihood": float(best.fun), "optimizer_success": bool(best.success),
        "optimizer_status": int(best.status), "optimizer_message": str(best.message),
        "optimizer_iterations": int(best.nit), "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "method": "garch_gaussian_conditional_mle_stationary_v1", "warnings": warnings,
        "assumptions": ["Historical log returns have conditionally Gaussian innovations.",
                        "omega > 0, alpha >= 0, beta >= 0, alpha+beta < 0.999 by parameterization.",
                        "All GARCH parameters and variance are per selected observation interval.",
                        "Initial conditional variance uses the fitted unconditional variance, matching the simulator default."],
        "limitations": "Gaussian conditional likelihood is sensitive to outliers and does not model jumps explicitly.",
    }


@dataclass(frozen=True)
class GBMCalibration:
    ticker: str
    start: str
    end: str
    interval: str
    price_field: str
    observation_count: int
    return_count: int
    annualization_periods: int
    drift: float
    volatility: float
    mean_period_log_return: float
    sample_std_period_log_return: float
    calibrated_at: str
    warnings: list[str]

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker, "estimation_start": self.start, "estimation_end": self.end,
            "interval": self.interval, "price_field": self.price_field,
            "observation_count": self.observation_count, "return_count": self.return_count,
            "annualization_periods": self.annualization_periods,
            "annualization_convention": f"sqrt({self.annualization_periods}) for volatility; {self.annualization_periods} periods/year",
            "drift": self.drift, "volatility": self.volatility,
            "mean_period_log_return": self.mean_period_log_return,
            "sample_std_period_log_return": self.sample_std_period_log_return,
            "calibrated_at": self.calibrated_at, "warnings": self.warnings,
            "method": "historical_gbm_moments_v1",
            "assumptions": [
                "Observed close-to-close log returns are treated as equally spaced observations at the requested interval.",
                "GBM has constant annualized drift and volatility with independent Gaussian diffusion increments.",
                "The annualization factors are conventional approximations, not exchange-calendar estimates.",
                "Drift is expected proportional price growth in dS/S = mu dt + sigma dW, not annualized mean log return.",
            ],
            "limitations": "Descriptive historical estimate; not a guarantee or forecast of future parameters.",
        }


def estimate_gbm(ticker: str, frame: pd.DataFrame, interval: str, requested_start: str, requested_end: str) -> GBMCalibration:
    """Estimate annualized GBM mu/sigma from validated period log returns.

    Sigma is the sample standard deviation (ddof=1) times sqrt(periods/year).
    Mu follows the GBM price-drift convention: mean log return times periods/year
    plus half annualized variance.
    """
    if interval not in ANNUALIZATION_PERIODS:
        raise ValueError("GBM calibration supports intervals 1d, 5d, 1wk, 1mo, and 3mo only; intraday annualization is undefined here")
    try:
        start_date, end_date = pd.Timestamp(requested_start).date(), pd.Timestamp(requested_end).date()
    except (TypeError, ValueError) as exc:
        raise ValueError("estimation dates must be valid ISO calendar dates") from exc
    if start_date >= end_date:
        raise ValueError("estimation start must be earlier than end")
    report = validate_market_data(frame)
    if not report.valid:
        raise MarketDataQualityError("Historical observations failed validation")
    if not frame.index.is_monotonic_increasing:
        raise MarketDataQualityError("Historical observations must be in strictly chronological order")
    observed_dates = frame.index.date
    if (observed_dates < start_date).any() or (observed_dates >= end_date).any():
        raise MarketDataQualityError("Historical observations fall outside the requested half-open estimation date range")
    price_field = "adj_close" if "adj_close" in frame.columns else "close"
    prices = pd.to_numeric(frame[price_field], errors="coerce")
    if prices.isna().any() or not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any():
        raise MarketDataQualityError(f"Historical {price_field} prices contain missing, non-finite, or non-positive values")
    if len(prices) < 3:
        raise ValueError("At least three valid price observations (two returns) are required for sample volatility")
    returns = compute_log_returns(frame).dropna().to_numpy(dtype=np.float64)
    if len(returns) < 2 or not np.isfinite(returns).all():
        raise ValueError("At least two finite historical log returns are required")
    periods = ANNUALIZATION_PERIODS[interval]
    mean_return = float(np.mean(returns))
    sample_std = float(np.std(returns, ddof=1))
    volatility = sample_std * np.sqrt(periods)
    drift = mean_return * periods + 0.5 * volatility**2
    if not np.isfinite([mean_return, sample_std, volatility, drift]).all():
        raise ValueError("Calibration produced non-finite parameters")
    if volatility <= 0:
        raise ValueError("Historical volatility estimate is zero; GBM calibration requires positive volatility")
    warnings = []
    if len(returns) < 30:
        warnings.append("Fewer than 30 returns: estimates are highly uncertain.")
    return GBMCalibration(ticker.upper(), requested_start, requested_end, interval, price_field, len(prices), len(returns), periods,
                          float(drift), float(volatility), mean_return, sample_std,
                          datetime.now(timezone.utc).isoformat(), warnings)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.05, 0.5, 0.95])
    return {"p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2])}


def _max_drawdown(price_series: pd.Series) -> float:
    _, value = drawdown(price_series.reset_index(drop=True), input_type="price")
    return value


def evaluate_gbm(calibration: GBMCalibration, evaluation_frame: pd.DataFrame, simulated_prices: np.ndarray) -> dict:
    """Compare held-out returns and equal-horizon simulated GBM paths descriptively."""
    report = validate_market_data(evaluation_frame)
    if not report.valid:
        raise MarketDataQualityError("Evaluation observations failed validation")
    if not evaluation_frame.index.is_monotonic_increasing:
        raise MarketDataQualityError("Evaluation observations must be in strictly chronological order")
    if evaluation_frame.index.min().date() <= pd.Timestamp(calibration.end).date():
        raise MarketDataQualityError("Evaluation observations must occur after the estimation period")
    price_field = "adj_close" if "adj_close" in evaluation_frame.columns else "close"
    prices = pd.to_numeric(evaluation_frame[price_field], errors="coerce")
    if prices.isna().any() or (prices <= 0).any() or not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise MarketDataQualityError("Evaluation prices contain missing or invalid values")
    observed = compute_log_returns(evaluation_frame).dropna().to_numpy(dtype=np.float64)
    if len(observed) < 2:
        raise ValueError("Evaluation period must contain at least two returns")
    paths = np.asarray(simulated_prices, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[1] != len(observed) + 1 or not np.isfinite(paths).all() or (paths <= 0).any():
        raise ValueError("Simulated price paths must be positive, finite, and match the evaluation horizon")
    simulated = np.diff(np.log(paths), axis=1).reshape(-1)
    periods = calibration.annualization_periods
    sim_period_std = float(np.std(simulated, ddof=1)) if len(simulated) > 1 else 0.0
    hist_period_std = float(np.std(observed, ddof=1))
    sim_total = paths[:, -1] / paths[:, 0] - 1
    simulated_terminal_risk = empirical_var_es(-sim_total, .95)
    hist_total = float(prices.iloc[-1] / prices.iloc[0] - 1)
    sim_dd = np.asarray([_max_drawdown(pd.Series(row)) for row in paths])
    hist_dd = _max_drawdown(prices)
    return {
        "estimation_period": {"start": calibration.start, "end": calibration.end, "interval": calibration.interval,
                              "observations": calibration.observation_count, "returns": calibration.return_count},
        "evaluation_period": {"start": str(evaluation_frame.index.min().date()), "end": str(evaluation_frame.index.max().date()),
                              "interval": calibration.interval, "observations": len(prices), "returns": len(observed)},
        "comparison_type": "out_of_sample_chronological",
        "metrics": {
            "historical_annualized_volatility": hist_period_std * np.sqrt(periods),
            "simulated_annualized_volatility": sim_period_std * np.sqrt(periods),
            "historical_period_log_return_quantiles": _quantiles(observed),
            "simulated_period_log_return_quantiles": _quantiles(simulated),
            "historical_total_return": hist_total,
            "historical_terminal_return_quantiles": None,
            "historical_terminal_return_sample_size": 1,
            "simulated_terminal_return_quantiles": _quantiles(sim_total),
            "simulated_terminal_var_95": simulated_terminal_risk["var"],
            "simulated_terminal_expected_shortfall_95": simulated_terminal_risk["expected_shortfall"],
            "simulated_terminal_risk_confidence": simulated_terminal_risk["confidence_level"],
            "simulated_terminal_risk_tail_scenario_count": simulated_terminal_risk["tail_observation_count"],
            "historical_maximum_drawdown": hist_dd,
            "simulated_maximum_drawdown_quantiles": _quantiles(sim_dd),
        },
        "methodology": {
            "return_definition": "Close-to-close log returns using adjusted close when available; otherwise close.",
            "simulation": "Equal number of intervals and same annualized horizon as the held-out observations; simulated increments use GBM log-return convention.",
            "volatility": "Sample standard deviation (ddof=1) times square root of the interval annualization factor.",
            "terminal_return": "Historical first-to-last simple return compared with simulated simple terminal returns.",
            "terminal_loss_risk": "Simulated loss L=-R; VaR is the linear empirical 95% loss quantile and ES is the mean of losses >= VaR, including ties. Historical terminal VaR/ES is unavailable because the holdout contains one terminal outcome.",
            "drawdown": "Maximum peak-to-trough percentage drawdown on positive price paths.",
            "interpretation": "Descriptive held-out comparison, not evidence of profitability or reliable predictive accuracy.",
        },
    }
