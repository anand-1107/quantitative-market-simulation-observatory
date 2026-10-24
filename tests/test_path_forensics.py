import numpy as np
import pytest
from fastapi.testclient import TestClient

import quantum_market_lab.api.app as app_module
from quantum_market_lab.models.scenario_matching import match_objective, normalized_path_rmse


CASES = [
    ("/simulate-gbm/paths", {"initial_price": 100, "drift": .04, "volatility": .2,
        "years": 1, "steps": 6, "paths": 3, "seed": 11}, "standard_normal"),
    ("/simulate-ou/paths", {"initial_value": .2, "mean_level": 0, "mean_reversion_speed": 1.4,
        "volatility": .3, "years": 1, "steps": 6, "paths": 3, "seed": 11}, "conditional_mean"),
    ("/simulate-jump-diffusion/paths", {"initial_price": 100, "drift": .04, "volatility": .2,
        "jump_intensity": 2, "mean_log_jump": -.1, "jump_volatility": .2,
        "years": 1, "steps": 6, "paths": 3, "seed": 11}, "poisson_jump_count"),
    ("/simulate-garch/paths", {"omega": .00001, "alpha": .05, "beta": .9,
        "steps": 6, "paths": 3, "seed": 11, "return_convention": "log"}, "innovation"),
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("QML_EXPERIMENT_DB", str(tmp_path / "forensics.sqlite3"))
    return TestClient(app_module.app)


@pytest.mark.parametrize("endpoint,payload,innovation_key", CASES)
def test_all_models_reconstruct_selected_paths_from_stored_innovations(client, endpoint, payload, innovation_key):
    response = client.post(endpoint, json={**payload, "retain_forensics": True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["forensics_run_id"]
    assert body["trajectories"] == client.post(endpoint, json=payload).json()["trajectories"]
    inspected = client.get(f"/path-forensics/{body['forensics_run_id']}/paths/2")
    assert inspected.status_code == 200, inspected.text
    result = inspected.json()
    assert result["model"] == body["model"]
    assert result["path_id"] == 2
    assert result["reconstruction"]["verified"] is True
    assert result["reconstruction"]["maximum_absolute_error"] <= result["reconstruction"]["verification_tolerance"]
    np.testing.assert_allclose(result["reconstruction"]["path"], body["trajectories"][1], rtol=1e-12, atol=1e-12)
    assert innovation_key in result["innovations"]
    assert len(result["transitions"]) == len(result["innovations"][innovation_key])
    if body["model"] == "garch":
        assert len(result["conditional_variance"]) == len(body["trajectories"][1])
        assert "next_conditional_variance" in result["transitions"][0]
        first = result["transitions"][0]
        assert first["variance_intercept_term"] == pytest.approx(payload["omega"])
        assert first["arch_variance_term"] == pytest.approx(payload["alpha"] * first["innovation"] ** 2)
        assert first["garch_variance_term"] == pytest.approx(payload["beta"] * first["conditional_variance"])
        assert first["next_conditional_variance"] == pytest.approx(sum(first[key] for key in (
            "variance_intercept_term", "arch_variance_term", "garch_variance_term")))
        assert result["transitions"][-1]["next_conditional_variance"] is None
        assert result["transitions"][-1]["variance_intercept_term"] is None
    if body["model"] == "jump_diffusion":
        assert "jump_occurred" in result["transitions"][0]
        assert "compensator_rate" in result["transitions"][0]


def test_forensics_is_opt_in_and_invalid_ids_are_reported(client):
    response = client.post("/simulate-gbm/paths", json={"paths": 2, "steps": 3})
    assert response.status_code == 200
    assert response.json()["forensics_run_id"] is None
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/path-forensics/{missing}/paths/1").status_code == 404
    assert client.get("/path-forensics/not-a-uuid/paths/1").status_code == 422
    incomplete = app_module._experiment_store().save_path_forensics({"model": "gbm"})
    assert client.get(f"/path-forensics/{incomplete}/paths/1").status_code == 422
    valid = client.post("/simulate-gbm/paths", json={"paths": 2, "steps": 3,
        "seed": 4, "retain_forensics": True}).json()
    assert client.get(f"/path-forensics/{valid['forensics_run_id']}/paths/0").status_code == 422


def test_calibration_provenance_is_retained_with_reconstructable_path(client):
    calibration = {"model": "gbm", "ticker": "MSFT", "estimation_start": "2020-01-01",
        "estimation_end": "2023-01-01", "interval": "1d", "drift": .03,
        "volatility": .18, "calibrated_at": "2024-01-01T00:00:00Z",
        "method": "historical_gbm_moments_v1", "observation_count": 500, "return_count": 499}
    response = client.post("/simulate-gbm/paths", json={"initial_price": 100, "drift": .03,
        "volatility": .18, "years": 1, "steps": 4, "paths": 2, "seed": 3,
        "calibration": calibration, "retain_forensics": True})
    assert response.status_code == 200, response.text
    path = client.get(f"/path-forensics/{response.json()['forensics_run_id']}/paths/1").json()
    assert path["parameters"]["calibration"]["ticker"] == "MSFT"
    assert path["reconstruction"]["verified"] is True


def test_unseeded_run_reconstructs_from_retained_innovations_not_seed(client):
    response = client.post("/simulate-ou/paths", json={"steps": 4, "paths": 2,
        "seed": None, "retain_forensics": True})
    assert response.status_code == 200, response.text
    body = response.json()
    path = client.get(f"/path-forensics/{body['forensics_run_id']}/paths/1").json()
    assert path["parameters"]["seed"] is None
    assert path["reconstruction"]["verified"] is True


def test_bounded_scenario_search_ranks_candidates_using_same_stored_innovations(client):
    base = {"initial_price": 100, "drift": .04, "volatility": .2, "years": 1,
            "steps": 8, "paths": 2, "seed": 9, "retain_forensics": True}
    run = client.post("/simulate-gbm/paths", json=base).json()
    response = client.post("/path-forensics/scenario-matches", json={
        "run_id": run["forensics_run_id"], "path_id": 2,
        "candidate_parameters": [{}, {"volatility": .1}, {"drift": -.1}],
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert [candidate["rank"] for candidate in result["candidates"]] == [1, 2, 3]
    assert result["candidates"][0]["objective"] == 0
    assert all(candidate["exactly_replayable"] for candidate in result["candidates"])
    reference = client.get(f"/path-forensics/{run['forensics_run_id']}/paths/2").json()["original_path"]
    assert result["reference_path_id"] == 2
    np.testing.assert_allclose(result["candidates"][0]["candidate_path"], reference, rtol=1e-12, atol=1e-12)
    assert len(result["candidates"][1]["candidate_path"]) == len(reference)
    assert all(candidate["replay_provenance"]["same_stored_innovations"] for candidate in result["candidates"])
    assert "not an out-of-sample forecast" in result["warnings"][0]
    too_many = {"run_id": run["forensics_run_id"], "path_id": 1,
                "candidate_parameters": [{}] * 33}
    assert client.post("/path-forensics/scenario-matches", json=too_many).status_code == 422
    invalid_override = {"run_id": run["forensics_run_id"], "path_id": 1,
                        "candidate_parameters": [{"steps": 9}]}
    assert client.post("/path-forensics/scenario-matches", json=invalid_override).status_code == 422
    assert client.post("/path-forensics/scenario-matches", json={"run_id": run["forensics_run_id"],
        "path_id": 3, "candidate_parameters": [{}]}).status_code == 404


@pytest.mark.parametrize("endpoint,payload,_innovation_key", CASES)
def test_reference_scenario_matching_supports_all_models(client, endpoint, payload, _innovation_key):
    run = client.post(endpoint, json={**payload, "retain_forensics": True}).json()
    response = client.post("/path-forensics/scenario-matches", json={
        "run_id": run["forensics_run_id"], "path_id": 1, "candidate_parameters": [{}],
    })
    assert response.status_code == 200, response.text
    candidate = response.json()["candidates"][0]
    assert candidate["normalized_path_rmse"] == 0
    assert candidate["exactly_replayable"] is True
    if run["model"] == "ou":
        assert candidate["total_simple_return"] is None and candidate["maximum_drawdown"] is None
    else:
        assert candidate["total_simple_return"] is not None and candidate["maximum_drawdown"] is not None


def test_scenario_distance_handles_scaling_and_optional_scores():
    assert normalized_path_rmse([100, 101, 102], [100, 101, 102]) == 0
    assert normalized_path_rmse([1, 2, 3], [1, 2, 3, 4]) > 0
    objective, return_difference, drawdown_difference = match_objective(.5, None, None, None, None, .25, .25)
    assert objective == .5 and return_difference is None and drawdown_difference is None
