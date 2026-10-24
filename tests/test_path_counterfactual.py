import numpy as np
import pytest
from fastapi.testclient import TestClient

import quantum_market_lab.api.app as app_module


CASES = [
    ("/simulate-gbm/paths", {"initial_price": 100, "drift": .04, "volatility": .2,
        "years": 1, "steps": 8, "paths": 3, "seed": 2}, {"drift": .12}),
    ("/simulate-ou/paths", {"initial_value": .2, "mean_level": 0, "mean_reversion_speed": 1.4,
        "volatility": .3, "years": 1, "steps": 8, "paths": 3, "seed": 2}, {"mean_level": .5}),
    ("/simulate-jump-diffusion/paths", {"initial_price": 100, "drift": .04, "volatility": .2,
        "jump_intensity": 2, "mean_log_jump": -.1, "jump_volatility": .2,
        "years": 1, "steps": 8, "paths": 3, "seed": 2}, {"jump_intensity": 4}),
    ("/simulate-garch/paths", {"omega": .00001, "alpha": .05, "beta": .9,
        "steps": 8, "paths": 3, "seed": 2, "return_convention": "log"}, {"alpha": .1}),
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("QML_EXPERIMENT_DB", str(tmp_path / "counterfactual.sqlite3"))
    return TestClient(app_module.app)


@pytest.mark.parametrize("endpoint,payload,change", CASES)
def test_counterfactual_reuses_path_inputs_and_preserves_reference(client, endpoint, payload, change):
    simulation = client.post(endpoint, json={**payload, "retain_forensics": True})
    assert simulation.status_code == 200, simulation.text
    run = simulation.json()
    reference = client.get(f"/path-forensics/{run['forensics_run_id']}/paths/2").json()
    identity = client.post("/path-forensics/counterfactual", json={"run_id": run["forensics_run_id"],
        "path_id": 2, "candidate_parameters": {"steps": payload["steps"]}})
    assert identity.status_code == 200, identity.text
    np.testing.assert_allclose(identity.json()["candidate_path"], reference["original_path"], rtol=1e-12, atol=1e-12)
    assert identity.json()["exact_common_innovations"] is True
    response = client.post("/path-forensics/counterfactual", json={
        "run_id": run["forensics_run_id"], "path_id": 2,
        "candidate_parameters": {**change, "steps": payload["steps"]},
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["run_id"] == run["forensics_run_id"] and result["path_id"] == 2
    np.testing.assert_allclose(result["reference_path"], run["trajectories"][1], rtol=0, atol=0)
    assert result["candidate_path"] != result["reference_path"]
    assert len(result["differences"]) == len(result["candidate_path"])
    assert result["raw_path_rmse"] > 0
    assert result["exact_common_innovations"] is (run["model"] != "jump_diffusion")
    if run["model"] == "jump_diffusion":
        assert "thinning" in result["coupling_method"] or "superposition" in result["coupling_method"]
        assert result["warnings"]
    else:
        np.testing.assert_allclose(result["innovations"][next(iter(reference["innovations"]))],
                                   list(reference["innovations"].values())[0], rtol=0, atol=0)
    # A later forensic fetch proves the stored baseline was not overwritten.
    after = client.get(f"/path-forensics/{run['forensics_run_id']}/paths/2").json()
    assert after["original_path"] == reference["original_path"]


def test_counterfactual_run_list_selects_every_retained_path(client):
    first = client.post("/simulate-gbm/paths", json={"steps": 4, "paths": 3, "seed": 1,
        "retain_forensics": True}).json()
    listed = client.get("/path-forensics/runs")
    assert listed.status_code == 200
    run = next(value for value in listed.json()["runs"] if value["run_id"] == first["forensics_run_id"])
    assert run["path_count"] == 3 and run["step_count"] == 5
    for path_id in range(1, run["path_count"] + 1):
        result = client.post("/path-forensics/counterfactual", json={"run_id": run["run_id"],
            "path_id": path_id, "candidate_parameters": {}})
        assert result.status_code == 200, result.text
        assert result.json()["reference_path"] == first["trajectories"][path_id - 1]


def test_counterfactual_reset_reproduces_the_same_stored_input_comparison(client):
    payload = {"initial_price": 100, "drift": .03, "volatility": .22, "jump_intensity": 1.1,
        "mean_log_jump": -.12, "jump_volatility": .18, "years": 1, "steps": 12, "paths": 2,
        "seed": 37, "retain_forensics": True}
    run = client.post("/simulate-jump-diffusion/paths", json=payload).json()
    request = {"run_id": run["forensics_run_id"], "path_id": 2,
        "candidate_parameters": {"jump_intensity": 1.8, "volatility": .3, "steps": 12}}
    first = client.post("/path-forensics/counterfactual", json=request).json()
    replay = client.post("/path-forensics/counterfactual", json=request).json()
    assert first["candidate_path"] == replay["candidate_path"]
    assert first["innovations"] == replay["innovations"]
    np.testing.assert_allclose(first["differences"],
        np.asarray(first["candidate_path"]) - np.asarray(first["reference_path"]), rtol=0, atol=0)


def test_merton_intensity_horizon_change_with_same_step_rate_reuses_counts_exactly(client):
    run = client.post("/simulate-jump-diffusion/paths", json={"initial_price": 100, "drift": .03,
        "volatility": .2, "jump_intensity": 2, "mean_log_jump": -.1, "jump_volatility": .15,
        "years": 1, "steps": 8, "paths": 1, "seed": 3, "retain_forensics": True}).json()
    source = client.get(f"/path-forensics/{run['forensics_run_id']}/paths/1").json()
    result = client.post("/path-forensics/counterfactual", json={"run_id": run["forensics_run_id"],
        "path_id": 1, "candidate_parameters": {"jump_intensity": 1, "years": 2, "steps": 8}}).json()
    assert result["exact_common_innovations"] is True
    assert "λcandidate·Δtcandidate equals" in result["coupling_method"]
    assert result["innovations"]["poisson_jump_count"] == source["innovations"]["poisson_jump_count"]


def test_counterfactual_validation_rejects_invalid_provenance_and_incompatible_grid(client):
    run = client.post("/simulate-gbm/paths", json={"steps": 4, "paths": 2,
        "retain_forensics": True}).json()
    run_id = run["forensics_run_id"]
    assert client.post("/path-forensics/counterfactual", json={"run_id": run_id,
        "path_id": 3, "candidate_parameters": {}}).status_code == 404
    assert client.post("/path-forensics/counterfactual", json={"run_id": run_id,
        "path_id": 1, "candidate_parameters": {"steps": 5}}).status_code == 422
    assert client.post("/path-forensics/counterfactual", json={"run_id": run_id,
        "path_id": 1, "candidate_parameters": {"paths": 20}}).status_code == 422
    assert client.post("/path-forensics/counterfactual", json={"run_id": run_id,
        "path_id": 1, "candidate_parameters": {"volatility": -1}}).status_code == 422
    assert client.post("/path-forensics/counterfactual", json={"run_id": "not-a-uuid",
        "path_id": 1, "candidate_parameters": {}}).status_code == 422
    incomplete = app_module._experiment_store().save_path_forensics({"model": "gbm", "parameters": {},
        "trajectories": [[100, 101]], "times": [0, 1], "innovations": {}})
    assert client.post("/path-forensics/counterfactual", json={"run_id": incomplete,
        "path_id": 1, "candidate_parameters": {}}).status_code == 422
