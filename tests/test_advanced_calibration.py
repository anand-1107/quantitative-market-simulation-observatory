from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantum_market_lab.api.app import app
from quantum_market_lab.models.calibration import calibrate_garch, calibrate_merton, calibrate_ou
from quantum_market_lab.models.garch import simulate_garch
from quantum_market_lab.models.jump_diffusion import simulate_jump_diffusion
from quantum_market_lab.models.ou import simulate_ou


def market_frame(prices, start="1980-01-01"):
    values = np.asarray(prices, dtype=float)
    dates = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({"open": values, "high": values * 1.001, "low": values * .999,
                         "close": values, "adj_close": values, "volume": np.full(len(values), 1000)}, index=dates)


def test_ou_cls_recovers_synthetic_mean_reverting_log_price_and_applies_to_simulation(monkeypatch):
    years, steps = 40, 10080
    _, states = simulate_ou(.02, .04, 1.3, .16, years, steps, 1, seed=81)
    frame = market_frame(np.exp(states[0]))
    fitted = calibrate_ou("SYN", frame, series="log_price", price_field="auto", interval="1d",
                          start="1980-01-01", end="2030-01-01", allow_price_series=True)
    assert fitted["optimizer_success"]
    assert fitted["mean_reversion_speed"] == pytest.approx(1.3, rel=.3)
    assert fitted["mean_level"] == pytest.approx(.04, abs=.03)
    assert fitted["volatility"] == pytest.approx(.16, rel=.2)
    assert "explicitly asserted" in " ".join(fitted["warnings"])

    monkeypatch.setattr("quantum_market_lab.api.app._market_download", lambda *args: (SimpleNamespace(ticker="SYN"), frame))
    client = TestClient(app)
    response = client.post("/calibration/ou", json={"ticker": "SYN", "start": "1980-01-01", "end": "2030-01-01",
        "interval": "1d", "input_series": "log_price", "confirm_mean_reverting_price_series": True})
    assert response.status_code == 200, response.text
    result = response.json()
    provenance = {key: result[key] for key in ["model", "ticker", "estimation_start", "estimation_end", "interval", "input_series", "mean_level", "mean_reversion_speed", "volatility", "initial_value", "calibrated_at", "method", "observation_count"]}
    payload = {"initial_value": result["initial_value"], "mean_level": result["mean_level"],
        "mean_reversion_speed": result["mean_reversion_speed"], "volatility": result["volatility"],
        "calibration": provenance, "steps": 4, "paths": 2}
    simulated = client.post("/simulate-ou/paths", json=payload)
    assert simulated.status_code == 200, simulated.text
    assert simulated.json()["parameters"]["calibration"]["model"] == "ou"
    payload["mean_level"] = result["mean_level"] + 1
    assert client.post("/simulate-ou/paths", json=payload).status_code == 422


def test_ou_blocks_price_level_unless_user_explicitly_confirms_and_rejects_unidentified_series():
    frame = market_frame(np.exp(np.where(np.arange(500) % 2, .1, -.1)))
    with pytest.raises(ValueError, match="explicit confirmation"):
        calibrate_ou("SYN", frame, series="log_price", price_field="auto", interval="1d",
                      start="1980-01-01", end="1983-01-01")
    with pytest.raises(ValueError, match="not identified"):
        calibrate_ou("SYN", frame, series="log_price", price_field="auto", interval="1d",
                      start="1980-01-01", end="1983-01-01", allow_price_series=True)


def test_merton_exact_mixture_mle_matches_simulator_convention_and_integrates(monkeypatch):
    years, steps = 45, 11340
    _, price_paths = simulate_jump_diffusion(100, .06, .16, 1.2, -.13, .09, years, steps, 1, seed=22)
    frame = market_frame(price_paths[0])
    fitted = calibrate_merton("SYN", frame, price_field="auto", interval="1d", start="1980-01-01", end="2030-01-01")
    assert fitted["optimizer_success"]
    assert fitted["volatility"] == pytest.approx(.16, abs=.04)
    assert fitted["jump_intensity"] == pytest.approx(1.2, abs=.8)
    assert fitted["mean_log_jump"] < 0
    assert fitted["jump_volatility"] > 0
    assert fitted["estimated_expected_jumps_in_sample"] > 5
    assert "compound-Poisson" in " ".join(fitted["assumptions"])

    monkeypatch.setattr("quantum_market_lab.api.app._market_download", lambda *args: (SimpleNamespace(ticker="SYN"), frame))
    client = TestClient(app)
    result = client.post("/calibration/jump-diffusion", json={"ticker": "SYN", "start": "1980-01-01", "end": "2030-01-01", "interval": "1d"})
    assert result.status_code == 200, result.text
    calibration = result.json()
    fields = ["model", "ticker", "estimation_start", "estimation_end", "interval", "drift", "volatility", "jump_intensity", "mean_log_jump", "jump_volatility", "initial_price", "calibrated_at", "method", "observation_count", "return_count"]
    provenance = {key: calibration[key] for key in fields}
    payload = {key: calibration[key] for key in fields if key in {"drift", "volatility", "jump_intensity", "mean_log_jump", "jump_volatility", "initial_price"}}
    payload.update({"calibration": provenance, "steps": 4, "paths": 2})
    simulated = client.post("/simulate-jump-diffusion/paths", json=payload)
    assert simulated.status_code == 200, simulated.text
    assert simulated.json()["parameters"]["calibration"]["ticker"] == "SYN"


def test_merton_reports_sparse_jumps_and_optimizer_failure(monkeypatch):
    rng = np.random.default_rng(71)
    sparse = market_frame(100 * np.exp(np.r_[0, np.cumsum(rng.normal(0, .01, 260))]))
    with pytest.raises(ValueError, match="at least 250"):
        calibrate_merton("SYN", sparse.iloc[:200], price_field="auto", interval="1d", start="1980-01-01", end="1982-01-01")
    class FailedFit:
        success = False
        status = 2
        message = "synthetic optimizer failure"
        nit = 1
        fun = 12.0
        x = np.array([0.0, .2, .1, -.1, .1])
    monkeypatch.setattr("quantum_market_lab.models.calibration.minimize", lambda *args, **kwargs: FailedFit())
    result = calibrate_merton("SYN", sparse, price_field="auto", interval="1d", start="1980-01-01", end="1982-01-01")
    assert not result["optimizer_success"]
    assert any("did not converge" in warning for warning in result["warnings"])
    assert any("weakly identified" in warning for warning in result["warnings"])


def test_garch_conditional_mle_converges_and_uses_fitted_variance_in_simulation(monkeypatch):
    steps = 1400
    values, _ = simulate_garch(.000004, .08, .9, steps, 1, mean=.0001, seed=17)
    frame = market_frame(100 * np.exp(np.r_[0, np.cumsum(values[0])]))
    fitted = calibrate_garch("SYN", frame, price_field="auto", interval="1d", start="1980-01-01", end="1990-01-01")
    assert fitted["optimizer_success"], fitted["optimizer_message"]
    assert fitted["omega"] > 0 and fitted["alpha"] >= 0 and fitted["beta"] >= 0
    assert fitted["alpha"] + fitted["beta"] < .999
    assert len(fitted["conditional_volatility"]) == steps

    monkeypatch.setattr("quantum_market_lab.api.app._market_download", lambda *args: (SimpleNamespace(ticker="SYN"), frame))
    client = TestClient(app)
    result = client.post("/calibration/garch", json={"ticker": "SYN", "start": "1980-01-01", "end": "1990-01-01", "interval": "1d"})
    assert result.status_code == 200, result.text
    calibration = result.json()
    keys = ["model", "ticker", "estimation_start", "estimation_end", "interval", "mean", "omega", "alpha", "beta", "initial_variance", "return_convention", "calibrated_at", "method", "observation_count", "return_count"]
    provenance = {key: calibration[key] for key in keys}
    payload = {key: calibration[key] for key in ["mean", "omega", "alpha", "beta", "initial_variance", "return_convention"]}
    payload.update({"calibration": provenance, "steps": 5, "paths": 2, "seed": 42})
    simulated = client.post("/simulate-garch/paths", json=payload)
    assert simulated.status_code == 200, simulated.text
    assert simulated.json()["parameters"]["calibration"]["model"] == "garch"
    payload["beta"] = calibration["beta"] + .01
    assert client.post("/simulate-garch/paths", json=payload).status_code == 422


def test_garch_insufficient_samples_and_optimizer_failure_are_reported(monkeypatch):
    tiny = market_frame(np.exp(np.arange(30) * .001))
    with pytest.raises(ValueError, match="at least 100"):
        calibrate_garch("SYN", tiny, price_field="auto", interval="1d", start="1980-01-01", end="1981-01-01")
    values, _ = simulate_garch(.00001, .05, .9, 200, 1, seed=5)
    frame = market_frame(100 * np.exp(np.r_[0, np.cumsum(values[0])]))
    class FailedFit:
        success = False
        status = 2
        message = "synthetic optimizer failure"
        nit = 1
        fun = 123.0
        x = np.array([0.0, np.log(1e-6), 2.0, -2.0])
    monkeypatch.setattr("quantum_market_lab.models.calibration.minimize", lambda *args, **kwargs: FailedFit())
    result = calibrate_garch("SYN", frame, price_field="auto", interval="1d", start="1980-01-01", end="1981-01-01")
    assert not result["optimizer_success"]
    assert any("did not converge" in warning for warning in result["warnings"])
