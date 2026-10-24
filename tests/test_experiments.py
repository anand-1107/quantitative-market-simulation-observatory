import quantum_market_lab.api.app as app_module
from fastapi.testclient import TestClient

from quantum_market_lab.data.experiments import ExperimentStore


def test_store_persists_lifecycle_across_reinitialization(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    store = ExperimentStore(path)
    experiment_id = store.create(name="persist", description=None, model="gbm",
        parameters={"seed": 7}, metadata={"schema_version": 1})
    store.set_running(experiment_id)
    store.complete(experiment_id, {"terminal_return": 0.1})
    reopened = ExperimentStore(path)
    result = reopened.get(experiment_id)
    assert result["status"] == "completed"
    assert result["parameters"] == {"seed": 7}
    assert result["metrics"] == {"terminal_return": 0.1}
    assert result["created_at"].endswith("Z")


def test_create_run_list_retrieve_reproduce_compare_and_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("QML_EXPERIMENT_DB", str(tmp_path / "api.sqlite3"))
    client = TestClient(app_module.app)
    body = {"name": "seeded GBM", "description": "repeatability", "model": "gbm",
            "ticker": "AAPL", "parameters": {"initial_price": 100, "drift": .04,
            "volatility": .2, "years": 1, "steps": 8, "paths": 12, "seed": 27}}
    first_response = client.post("/experiments", json=body)
    second_response = client.post("/experiments", json={**body, "name": "seeded GBM copy"})
    assert first_response.status_code == second_response.status_code == 201
    first, second = first_response.json()["experiment"], second_response.json()["experiment"]
    assert len(first_response.json()["simulation"]["trajectories"]) == 12
    assert first_response.json()["simulation"]["summary"]["terminal_mean"] == first["metrics"]["summary"]["terminal_mean"]
    assert first["status"] == "completed"
    assert first["parameters"]["seed"] == 27
    assert first["metadata"]["ticker"] == "AAPL"
    assert first["metadata"]["parameter_source"] == "manual_or_default"
    assert first["metrics"] == second["metrics"]
    assert "path_metrics" not in first["metrics"]["terminal_risk"]

    assert client.get("/experiments").json()["count"] == 2
    assert client.get(f"/experiments/{first['id']}").json()["id"] == first["id"]
    assert client.get(f"/experiments/{first['id']}/summary").json()["metrics"] == first["metrics"]
    compared = client.get("/experiments/compare", params=[("experiment_ids", first["id"]),
                                                           ("experiment_ids", second["id"])]).json()
    assert compared["comparable"] is True
    assert "var" in compared["common_metrics"]
    assert "maximum_drawdown_p50" in compared["common_metrics"]
    assert client.delete(f"/experiments/{first['id']}").status_code == 204
    assert client.get(f"/experiments/{first['id']}").status_code == 404
    assert client.delete(f"/experiments/{first['id']}").status_code == 404


def test_all_models_validate_and_run_experiment_configurations(tmp_path, monkeypatch):
    monkeypatch.setenv("QML_EXPERIMENT_DB", str(tmp_path / "models.sqlite3"))
    client = TestClient(app_module.app)
    examples = [
        ("ou", {"initial_value": 0.1, "mean_level": 0, "mean_reversion_speed": 1.2,
                 "volatility": .3, "years": .5, "steps": 5, "paths": 3, "seed": 3}),
        ("jump_diffusion", {"initial_price": 100, "drift": .03, "volatility": .15,
                 "jump_intensity": .4, "mean_log_jump": -.12, "jump_volatility": .2,
                 "years": .5, "steps": 5, "paths": 3, "seed": 3}),
        ("garch", {"omega": .00001, "alpha": .04, "beta": .9, "mean": 0,
                 "steps": 5, "paths": 3, "seed": 3, "return_convention": "log"}),
    ]
    for model, parameters in examples:
        response = client.post("/experiments", json={"name": model, "model": model, "parameters": parameters})
        assert response.status_code == 201, response.text
        record = response.json()["experiment"]
        assert record["status"] == "completed"
        assert record["metrics"]["quantity_kind"] == ("state" if model == "ou" else "return" if model == "garch" else "price")
        assert record["metadata"]["reproducible_with_same_seed"] is True
    garch = next(record for record in client.get("/experiments").json()["experiments"] if record["model"] == "garch")
    assert garch["parameters"]["initial_variance"] > 0
    assert "initial_variance" in garch["metadata"]["derived_parameters"]


def test_invalid_experiment_configuration_and_unknown_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("QML_EXPERIMENT_DB", str(tmp_path / "invalid.sqlite3"))
    client = TestClient(app_module.app)
    invalid = client.post("/experiments", json={"name": "bad", "model": "gbm",
        "parameters": {"volatility": -1, "paths": 2, "steps": 2}})
    assert invalid.status_code == 422
    assert client.post("/experiments", json={"name": "   ", "model": "gbm", "parameters": {}}).status_code == 422
    unknown = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/experiments/{unknown}").status_code == 404
    assert client.get("/experiments/not-a-uuid").status_code == 422
    assert client.get("/experiments/compare", params=[("experiment_ids", unknown),
                                                       ("experiment_ids", unknown)]).status_code == 422


def test_calibration_provenance_is_saved_and_cannot_cross_models(tmp_path, monkeypatch):
    monkeypatch.setenv("QML_EXPERIMENT_DB", str(tmp_path / "provenance.sqlite3"))
    client = TestClient(app_module.app)
    calibration = {"model": "gbm", "ticker": "MSFT", "estimation_start": "2020-01-01",
        "estimation_end": "2023-01-01", "interval": "1d", "drift": .03,
        "volatility": .18, "calibrated_at": "2024-01-01T00:00:00Z", "method": "historical_gbm_moments_v1",
        "observation_count": 500, "return_count": 499}
    parameters = {"initial_price": 100, "drift": .03, "volatility": .18,
        "years": 1, "steps": 4, "paths": 3, "seed": 18, "calibration": calibration}
    response = client.post("/experiments", json={"name": "calibrated", "model": "gbm", "parameters": parameters})
    assert response.status_code == 201, response.text
    record = response.json()["experiment"]
    assert record["metadata"]["parameter_source"] == "historical_calibration"
    assert record["metadata"]["calibration"]["ticker"] == "MSFT"
    assert record["parameters"]["calibration"]["method"] == calibration["method"]
    wrong_model = client.post("/experiments", json={"name": "cross model", "model": "ou",
        "parameters": {"steps": 4, "paths": 3, "calibration": calibration}})
    assert wrong_model.status_code == 422


def test_concurrent_experiment_creations_remain_independent(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("QML_EXPERIMENT_DB", str(tmp_path / "concurrent.sqlite3"))
    client = TestClient(app_module.app)
    def create(index: int):
        return client.post("/experiments", json={"name": f"parallel-{index}", "model": "gbm",
            "parameters": {"steps": 5, "paths": 4, "seed": index}})
    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(create, range(6)))
    assert all(response.status_code == 201 for response in responses)
    records = client.get("/experiments").json()["experiments"]
    assert len(records) == 6
    assert len({record["id"] for record in records}) == 6
    assert all(record["status"] == "completed" for record in records)


def test_failed_simulation_is_persisted_as_failed_not_completed(tmp_path, monkeypatch):
    monkeypatch.setenv("QML_EXPERIMENT_DB", str(tmp_path / "failed.sqlite3"))
    def fail(_request):
        raise RuntimeError("synthetic engine failure")
    monkeypatch.setattr(app_module, "simulate_gbm_paths", fail)
    client = TestClient(app_module.app)
    response = client.post("/experiments", json={"name": "failure", "model": "gbm",
        "parameters": {"steps": 2, "paths": 2, "seed": 1}})
    assert response.status_code == 500
    records = client.get("/experiments").json()["experiments"]
    assert len(records) == 1 and records[0]["status"] == "failed"
    assert "synthetic engine failure" in records[0]["error"]
    assert records[0]["metrics"] is None
