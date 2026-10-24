from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quantum_market_lab.api.app import app
from quantum_market_lab.data.tournament import run_model_tournament
from quantum_market_lab.models.ou import simulate_ou


def _frame(prices, dates):
    values = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": values, "high": values * 1.002, "low": values * .998,
                         "close": values, "adj_close": values, "volume": np.full(len(values), 1000)},
                        index=dates)


@pytest.fixture
def windows():
    rng = np.random.default_rng(412)
    train_dates = pd.bdate_range("2020-01-01", periods=320)
    test_dates = pd.bdate_range(train_dates[-1] + pd.offsets.BDay(2), periods=80)
    train_r = rng.normal(.0003, .012, len(train_dates) - 1)
    test_r = rng.normal(-.0001, .018, len(test_dates) - 1)
    train = _frame(100 * np.exp(np.r_[0, np.cumsum(train_r)]), train_dates)
    test = _frame(train.close.iloc[-1] * np.exp(np.r_[0, np.cumsum(test_r)]), test_dates)
    return train, test


def _run(train, test, models=("gbm",), seed=17):
    dates = _request_dates(train, test)
    return run_model_tournament(ticker="SYN", train=train, test=test,
        **dates, interval="1d", models=list(models), paths=128, seed=seed)


def _request_dates(train, test):
    return {"train_start": train.index.min().date().isoformat(),
        "train_end": (train.index.max() + pd.Timedelta(days=1)).date().isoformat(),
        "test_start": test.index.min().date().isoformat(),
        "test_end": (test.index.max() + pd.Timedelta(days=1)).date().isoformat()}


def test_gbm_tournament_is_reproducible_and_fit_uses_training_only(windows):
    train, test = windows
    first = _run(train, test)
    second = _run(train, test)
    a, b = first["results"][0], second["results"][0]
    assert a["status"] == "completed"
    assert a["parameters"]["observation_count"] == len(train)
    assert a["metrics"] == b["metrics"]
    changed_test = test.copy()
    changed_test["close"] *= np.linspace(1, 1.4, len(test))
    changed_test["adj_close"] = changed_test["close"]
    changed = _run(train, changed_test)["results"][0]
    assert changed["parameters"]["drift"] == a["parameters"]["drift"]
    assert changed["metrics"]["test_period_total_simple_return"] != a["metrics"]["test_period_total_simple_return"]
    assert a["metrics"]["return_convention"].startswith("close-to-close natural-log returns")
    assert a["metrics"]["sample_size"] == len(test) - 1
    surface = a["forecast_distribution_surface"]
    risk = a["forecast_risk_surface"]
    assert len(surface["origin_indices"]) <= 64
    assert len(surface["log_return_grid"]) == 49
    assert len(surface["density_by_origin"]) == len(surface["evaluation_dates"])
    assert np.trapezoid(surface["density_by_origin"][0], surface["log_return_grid"]) == pytest.approx(1, abs=.015)
    assert len(risk["var_loss_fraction_by_origin"]) == len(surface["origin_indices"])
    assert risk["loss_definition"].startswith("L=1-exp(r)")


def test_tournament_rejects_overlapping_or_invalid_windows(windows):
    train, test = windows
    overlapping = test.copy()
    overlapping.index = pd.bdate_range(train.index[-2], periods=len(test))
    dates = _request_dates(train, test)
    with pytest.raises(ValueError, match="overlap or are out of order"):
        run_model_tournament(ticker="SYN", train=train, test=overlapping, **dates,
            interval="1d", models=["gbm"], paths=128, seed=17)
    invalid = test.copy()
    invalid.iloc[2, invalid.columns.get_loc("adj_close")] = np.nan
    with pytest.raises(ValueError, match="finite and strictly positive"):
        _run(train, invalid)


def test_four_existing_engines_run_on_a_synthetic_ou_log_return_process():
    _, states = simulate_ou(.001, .0002, 7, .01, 2, 504, 1, seed=25)
    prices = 100 * np.exp(np.r_[0, np.cumsum(states[0, 1:])])
    dates = pd.bdate_range("2020-01-01", periods=len(prices))
    cut = 321
    train = _frame(prices[:cut], dates[:cut])
    test = _frame(prices[cut + 1:cut + 81], dates[cut + 1:cut + 81])
    dates_for_request = _request_dates(train, test)
    result = run_model_tournament(ticker="SYN", train=train, test=test, **dates_for_request,
        interval="1d", models=["gbm", "ou", "jump_diffusion", "garch"], paths=128, seed=17)
    assert [item["model"] for item in result["results"]] == ["gbm", "ou", "jump_diffusion", "garch"]
    assert all(item["status"] == "completed" for item in result["results"])
    assert all(item["forecast_distribution_surface"] is not None for item in result["results"])
    assert all(item["forecast_risk_surface"] is not None for item in result["results"])
    reference_surface = result["results"][0]["forecast_distribution_surface"]
    assert all(item["forecast_distribution_surface"]["evaluation_dates"] == reference_surface["evaluation_dates"]
               for item in result["results"])
    assert all(item["forecast_distribution_surface"]["log_return_grid"] == reference_surface["log_return_grid"]
               for item in result["results"])
    ou = next(item for item in result["results"] if item["model"] == "ou")
    assert ou["parameters"]["input_series"] == "log_return"
    assert any("Log returns were selected" in warning for warning in ou["warnings"])


def test_api_downloads_train_and_test_separately_and_saves_grouped_runs(monkeypatch, tmp_path, windows):
    train, test = windows
    calls = []

    def download(ticker, start, end, interval):
        calls.append((ticker, start, end, interval))
        return SimpleNamespace(ticker=ticker.upper()), train if start == "2020-01-01" else test

    monkeypatch.setattr("quantum_market_lab.api.app._market_download", download)
    monkeypatch.setenv("QML_EXPERIMENT_DB", str(tmp_path / "tournaments.sqlite3"))
    client = TestClient(app)
    payload = {"ticker": "SYN", "models": ["gbm"], **_request_dates(train, test),
        "interval": "1d", "paths": 128, "seed": 17, "save": True, "name": "test"}
    response = client.post("/model-tournaments", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert [call[1] for call in calls] == [payload["train_start"], payload["test_start"]]
    assert result["results"][0]["experiment_id"]
    saved = client.get("/model-tournaments")
    assert saved.status_code == 200
    assert saved.json()["tournaments"][0]["tournament_id"] == result["tournament_id"]
    assert saved.json()["tournaments"][0]["models"][0]["metrics"]["sample_size"] == len(test) - 1
    assert saved.json()["tournaments"][0]["models"][0]["forecast_distribution_surface"]["unit"] == "natural_log_return_per_observation_interval"


def test_api_rejects_nonchronological_tournament(windows, monkeypatch):
    train, test = windows
    monkeypatch.setattr("quantum_market_lab.api.app._market_download",
                        lambda *args: (SimpleNamespace(ticker="SYN"), train))
    dates = _request_dates(train, test)
    dates["test_start"] = dates["train_end"]
    response = TestClient(app).post("/model-tournaments", json={"ticker": "SYN", "models": ["gbm"],
        **dates, "save": False})
    assert response.status_code == 422
    assert "strictly after" in response.json()["detail"]


def test_saved_tournament_preserves_unavailable_ou_status_and_failure_reason(monkeypatch, tmp_path, windows):
    train, test = windows
    # Alternating returns imply a negative AR(1) coefficient, which cannot be
    # the sampled transition of a continuous-time OU process.
    alternating = np.tile([.01, -.01], (len(train) - 1) // 2 + 1)[:len(train) - 1]
    prices = 100 * np.exp(np.r_[0, np.cumsum(alternating)])
    train = _frame(prices, train.index)
    monkeypatch.setattr("quantum_market_lab.api.app._market_download",
        lambda ticker, start, end, interval: (SimpleNamespace(ticker=ticker.upper()), train if start == train.index.min().date().isoformat() else test))
    monkeypatch.setenv("QML_EXPERIMENT_DB", str(tmp_path / "tournament-with-failure.sqlite3"))
    client = TestClient(app)
    payload = {"ticker": "SYN", "models": ["gbm", "ou"], **_request_dates(train, test),
        "interval": "1d", "paths": 128, "seed": 17, "save": True, "name": "failure audit"}
    response = client.post("/model-tournaments", json=payload)
    assert response.status_code == 200, response.text
    immediate = response.json()
    ou = next(item for item in immediate["results"] if item["model"] == "ou")
    assert ou["status"] == "unavailable"
    assert "AR(1) coefficient" in ou["error"]
    assert ou["experiment_id"]

    saved = client.get("/model-tournaments")
    assert saved.status_code == 200, saved.text
    restored = saved.json()["tournaments"][0]
    saved_ou = next(item for item in restored["models"] if item["model"] == "ou")
    assert saved_ou["status"] == "unavailable"
    assert saved_ou["error"] == ou["error"]
