from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantum_market_lab.api.app import app
from quantum_market_lab.data.market_data import MarketDataQualityError
from quantum_market_lab.models.calibration import estimate_gbm


def market_frame(log_returns):
    prices = 100 * np.exp(np.r_[0.0, np.cumsum(log_returns)])
    index = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"open": prices, "high": prices * 1.01, "low": prices * .99,
                         "close": prices, "adj_close": prices, "volume": np.full(len(prices), 1000)}, index=index)


def test_gbm_estimator_uses_sample_std_and_drift_convention():
    values = np.array([-.02, .01, .025, -.005, .015], dtype=float)
    result = estimate_gbm("TEST", market_frame(values), "1d", "2020-01-01", "2021-01-01")
    daily_std = values.std(ddof=1)
    expected_sigma = daily_std * np.sqrt(252)
    assert result.sample_std_period_log_return == pytest.approx(daily_std)
    assert result.volatility == pytest.approx(expected_sigma)
    assert result.drift == pytest.approx(values.mean() * 252 + .5 * expected_sigma**2)
    assert result.annualization_periods == 252
    assert result.return_count == len(values)


@pytest.mark.parametrize(("interval", "factor"), [("1d", 252), ("5d", 52), ("1wk", 52), ("1mo", 12), ("3mo", 4)])
def test_annualization_factors(interval, factor):
    result = estimate_gbm("TEST", market_frame([.01, -.005, .02]), interval, "2020-01-01", "2021-01-01")
    assert result.annualization_periods == factor


def test_estimator_rejects_insufficient_zero_volatility_invalid_prices_and_interval():
    with pytest.raises(ValueError, match="three valid"):
        estimate_gbm("TEST", market_frame([.01]), "1d", "2020-01-01", "2021-01-01")
    with pytest.raises(ValueError, match="zero"):
        estimate_gbm("TEST", market_frame([0, 0, 0]), "1d", "2020-01-01", "2021-01-01")
    invalid = market_frame([.01, .02, -.01])
    invalid.loc[invalid.index[1], "close"] = 0
    with pytest.raises(MarketDataQualityError):
        estimate_gbm("TEST", invalid, "1d", "2020-01-01", "2021-01-01")
    with pytest.raises(ValueError, match="supports intervals"):
        estimate_gbm("TEST", market_frame([.01, -.01, .005]), "1h", "2020-01-01", "2021-01-01")


def test_estimator_rejects_non_chronological_and_out_of_range_prices():
    frame = market_frame([.01, -.01, .005])
    with pytest.raises(MarketDataQualityError, match="chronological"):
        estimate_gbm("TEST", frame.iloc[::-1], "1d", "2020-01-01", "2021-01-01")
    with pytest.raises(MarketDataQualityError, match="outside"):
        estimate_gbm("TEST", frame, "1d", "2020-01-02", "2021-01-01")


def test_estimator_fails_clearly_on_non_finite_estimates(monkeypatch):
    monkeypatch.setattr("quantum_market_lab.models.calibration.np.std", lambda *args, **kwargs: np.inf)
    with pytest.raises(ValueError, match="non-finite"):
        estimate_gbm("TEST", market_frame([.01, -.005, .02]), "1d", "2020-01-01", "2021-01-01")


def test_api_calibration_metadata_and_gbm_simulation_compatibility(monkeypatch):
    frame = market_frame(np.random.default_rng(7).normal(.0002, .01, 120))
    monkeypatch.setattr("quantum_market_lab.api.app._market_download", lambda *args: (SimpleNamespace(ticker="TEST"), frame))
    client = TestClient(app)
    response = client.post("/calibration/gbm", json={"ticker": "TEST", "start": "2020-01-01", "end": "2021-01-01", "interval": "1d"})
    assert response.status_code == 200, response.text
    calibration = response.json()
    assert calibration["ticker"] == "TEST"
    assert calibration["return_count"] == 120
    assert calibration["method"] == "historical_gbm_moments_v1"
    provenance = {key: calibration[key] for key in ["ticker", "estimation_start", "estimation_end", "interval", "drift", "volatility", "calibrated_at", "method", "observation_count", "return_count"]}
    sim = client.post("/simulate-gbm/paths", json={"drift": calibration["drift"], "volatility": calibration["volatility"], "calibration": provenance, "paths": 4, "steps": 5})
    assert sim.status_code == 200, sim.text
    assert sim.json()["parameters"]["calibration"]["ticker"] == "TEST"
    assert len(sim.json()["trajectories"]) == 4
    repeated = client.post("/simulate-gbm/paths", json={"drift": calibration["drift"], "volatility": calibration["volatility"], "calibration": provenance, "paths": 4, "steps": 5})
    assert repeated.json()["trajectories"] == sim.json()["trajectories"]
    mismatch = client.post("/simulate-gbm/paths", json={"drift": 0.1, "volatility": calibration["volatility"], "calibration": provenance})
    assert mismatch.status_code == 422


def test_calibration_endpoint_rejects_invalid_ranges_and_intraday_annualization(monkeypatch):
    monkeypatch.setattr("quantum_market_lab.api.app._market_download", lambda *args: pytest.fail("must reject before loading data"))
    client = TestClient(app)
    invalid_dates = client.post("/calibration/gbm", json={"ticker": "TEST", "start": "2021-01-01", "end": "2020-01-01"})
    assert invalid_dates.status_code == 422
    intraday = client.post("/calibration/gbm", json={"ticker": "TEST", "start": "2020-01-01", "end": "2021-01-01", "interval": "1h"})
    assert intraday.status_code == 422
    assert "supports intervals" in intraday.json()["detail"]


def test_evaluation_rejects_overlapping_chronological_windows():
    client = TestClient(app)
    request = {"ticker": "TEST", "estimation_start": "2020-01-01", "estimation_end": "2021-01-01",
               "evaluation_start": "2021-01-01", "evaluation_end": "2022-01-01"}
    response = client.post("/evaluation/gbm", json=request)
    assert response.status_code == 422
    assert "start after" in response.json()["detail"]


def test_evaluation_api_returns_separate_holdout_metrics(monkeypatch):
    estimation = market_frame(np.random.default_rng(31).normal(.0003, .009, 80))
    estimation.index = pd.date_range("2019-01-01", periods=len(estimation), freq="B")
    held_out = market_frame(np.random.default_rng(32).normal(0, .012, 25))
    held_out.index = pd.date_range("2022-01-03", periods=len(held_out), freq="B")
    frames = iter([estimation, held_out])
    monkeypatch.setattr("quantum_market_lab.api.app._market_download", lambda *args: (SimpleNamespace(ticker="TEST"), next(frames)))
    client = TestClient(app)
    response = client.post("/evaluation/gbm", json={"ticker": "TEST", "estimation_start": "2019-01-01",
        "estimation_end": "2021-01-01", "evaluation_start": "2022-01-01", "evaluation_end": "2023-01-01",
        "paths": 64, "seed": 12})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["evaluation"]["comparison_type"] == "out_of_sample_chronological"
    assert body["evaluation"]["estimation_period"]["end"] == "2021-01-01"
    assert body["evaluation"]["evaluation_period"]["start"] == "2022-01-03"
    assert body["evaluation"]["metrics"]["simulated_annualized_volatility"] > 0


def test_evaluation_api_rejects_actual_provider_window_overlap(monkeypatch):
    same = market_frame(np.random.default_rng(8).normal(0, .01, 30))
    monkeypatch.setattr("quantum_market_lab.api.app._market_download", lambda *args: (SimpleNamespace(ticker="TEST"), same))
    response = TestClient(app).post("/evaluation/gbm", json={"ticker": "TEST", "estimation_start": "2020-01-01",
        "estimation_end": "2020-03-01", "evaluation_start": "2020-03-02", "evaluation_end": "2020-04-01"})
    assert response.status_code == 422
    assert "observations overlap" in response.json()["detail"]


def test_out_of_sample_metrics_use_matching_return_units_and_horizon():
    calibration = estimate_gbm("TEST", market_frame(np.random.default_rng(1).normal(0, .01, 100)), "1d", "2020-01-01", "2020-06-01")
    from quantum_market_lab.models.calibration import evaluate_gbm
    evaluation = market_frame(np.random.default_rng(2).normal(0, .012, 20))
    # Establish a visibly separate evaluation calendar window.
    evaluation.index = pd.date_range("2021-01-01", periods=len(evaluation), freq="B")
    steps = len(evaluation) - 1
    from quantum_market_lab.gbm import simulate_gbm
    _, simulated = simulate_gbm(float(evaluation.close.iloc[0]), calibration.drift, calibration.volatility,
                                steps / 252, steps, 128, seed=17)
    metrics = evaluate_gbm(calibration, evaluation, simulated)
    assert metrics["comparison_type"] == "out_of_sample_chronological"
    assert metrics["evaluation_period"]["returns"] == steps
    assert set(metrics["metrics"]["historical_period_log_return_quantiles"]) == {"p05", "p50", "p95"}
    assert len(metrics["metrics"]["simulated_terminal_return_quantiles"]) == 3
    assert metrics["metrics"]["simulated_terminal_risk_confidence"] == .95
    assert np.isfinite(metrics["metrics"]["simulated_terminal_var_95"])
    assert np.isfinite(metrics["metrics"]["simulated_terminal_expected_shortfall_95"])
    assert metrics["metrics"]["historical_terminal_return_sample_size"] == 1
