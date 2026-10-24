from fastapi.testclient import TestClient

from quantum_market_lab.api.app import app
from quantum_market_lab.gbm import simulate_gbm


client = TestClient(app)


def test_model_registry_and_api_summaries():
    assert len(client.get("/models").json()["models"]) == 4
    ou = client.post("/simulate-ou", json={"steps": 20, "paths": 100, "seed": 7}).json()
    assert ou["model"] == "ornstein_uhlenbeck" and "terminal_quantiles" in ou
    jump = client.post("/simulate-jump-diffusion", json={"steps": 20, "paths": 100, "jump_intensity": 0}).json()
    assert jump["model"] == "merton_jump_diffusion" and jump["theoretical_terminal_mean"] > 0
    garch = client.post("/simulate-garch", json={"steps": 20, "paths": 100, "initial_price": 100}).json()
    assert garch["model"] == "garch_1_1" and garch["price_terminal"]["terminal_mean"] > 0


def test_api_parameter_validation_and_resource_limits():
    assert client.post("/simulate-ou", json={"mean_reversion_speed": -1}).status_code == 422
    assert client.post("/simulate-jump-diffusion", json={"initial_price": 0}).status_code == 422
    assert client.post("/simulate-garch", json={"omega": 0}).status_code == 422
    # Each dimension meets its individual cap, but their product exceeds total compute cap.
    response = client.post("/simulate-ou", json={"steps": 10000, "paths": 10000})
    assert response.status_code == 422 and "resource limit" in response.json()["detail"]


def test_summary_simulation_routes_report_consistent_risk_or_reason():
    ou = client.post("/simulate-ou", json={"steps": 4, "paths": 4}).json()
    assert ou["terminal_risk"]["available"] is False
    assert "state-level" in ou["terminal_risk"]["unavailable_reason"]
    for endpoint, payload in [
        ("/simulate-jump-diffusion", {"steps": 4, "paths": 4}),
        ("/simulate-garch", {"steps": 4, "paths": 4}),
    ]:
        risk = client.post(endpoint, json=payload).json()["terminal_risk"]
        assert risk["available"] is True
        assert risk["measure"] == "terminal_horizon"
        assert risk["confidence_level"] == .95
        assert risk["scenario_count"] == 4
        assert risk["horizon_steps"] == 4


def test_gbm_trajectory_schema_dimensions_summary_and_reproducibility():
    payload = {"initial_price": 75, "drift": .04, "volatility": .15,
               "years": .5, "steps": 4, "paths": 3, "seed": 19}
    response = client.post("/simulate-gbm/paths", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "gbm" and body["quantity_kind"] == "price"
    assert len(body["times"]) == 5 and len(body["trajectories"]) == 3
    assert all(len(path) == 5 for path in body["trajectories"])
    assert body["trajectories"][0][0] == 75
    _, expected = simulate_gbm(**payload)
    expected[:, 0] = payload["initial_price"]
    assert body["trajectories"] == expected.tolist()
    assert body["summary"]["terminal_mean"] == sum(path[-1] for path in body["trajectories"]) / 3
    assert body["terminal_risk"]["available"] is True
    assert body["terminal_risk"]["measure"] == "terminal_horizon"
    assert body["terminal_risk"]["horizon_steps"] == payload["steps"]
    assert body["terminal_risk"]["horizon_years"] == payload["years"]
    assert body["terminal_risk"]["scenario_count"] == payload["paths"]
    assert len(body["terminal_risk"]["path_metrics"]) == payload["paths"]
    again = client.post("/simulate-gbm/paths", json=payload).json()
    assert body["trajectories"] == again["trajectories"]
    # Existing summary-only fields remain present with explicit terminal risk.
    legacy = client.post("/simulate-gbm", json={"paths": 2, "steps": 2}).json()
    assert "terminal_price" in legacy and legacy["terminal_risk"]["available"]
    assert legacy["terminal_risk"]["horizon_unit"] == "years"


def test_all_models_return_native_trajectory_quantities():
    cases = [
        ("/simulate-ou/paths", {"paths": 2, "steps": 3}, "state", "OU state level"),
        ("/simulate-jump-diffusion/paths", {"paths": 2, "steps": 3}, "price", "Simulated price"),
        ("/simulate-garch/paths", {"paths": 2, "steps": 3}, "return", "Per-period log return"),
    ]
    for endpoint, payload, kind, label in cases:
        response = client.post(endpoint, json=payload)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["quantity_kind"] == kind and result["quantity_label"] == label
        assert len(result["trajectories"]) == 2
        assert all(len(path) == 3 + (kind != "return") for path in result["trajectories"])
        assert len(result["summary"]["bands"]) == len(result["times"])
        assert result["terminal_risk"]["available"] is (kind != "state")
        if kind == "return":
            assert result["terminal_risk"]["horizon_unit"] == "observation_periods"
    garch = client.post("/simulate-garch/paths", json={"paths": 2, "steps": 3}).json()
    assert len(garch["conditional_variances"]) == 2
    assert all(len(path) == 3 for path in garch["conditional_variances"])


def test_formula_lab_trajectory_endpoints_are_reproducible_for_a_fixed_seed():
    cases = [
        ("/simulate-gbm/paths", {"initial_price": 100, "drift": .04, "volatility": .2,
            "years": 1, "steps": 5, "paths": 3, "seed": 83}),
        ("/simulate-ou/paths", {"initial_value": .2, "mean_level": 0, "mean_reversion_speed": 1.2,
            "volatility": .3, "years": 1, "steps": 5, "paths": 3, "seed": 83}),
        ("/simulate-jump-diffusion/paths", {"initial_price": 100, "drift": .04, "volatility": .2,
            "jump_intensity": .8, "mean_log_jump": -.1, "jump_volatility": .2,
            "years": 1, "steps": 5, "paths": 3, "seed": 83}),
        ("/simulate-garch/paths", {"omega": .00001, "alpha": .05, "beta": .9,
            "mean": .001, "initial_variance": None, "steps": 5, "paths": 3,
            "seed": 83, "return_convention": "log"}),
    ]
    for endpoint, payload in cases:
        first = client.post(endpoint, json=payload)
        second = client.post(endpoint, json=payload)
        assert first.status_code == second.status_code == 200
        assert first.json()["trajectories"] == second.json()["trajectories"]
        assert first.json()["summary"] == second.json()["summary"]


def test_seeded_adverse_drift_scenario_reuses_simulator_and_moves_loss_risk():
    base = {"initial_price": 100, "drift": .08, "volatility": .2, "years": 1,
            "steps": 32, "paths": 64, "seed": 71}
    adverse = {**base, "drift": -.12}
    base_response = client.post("/simulate-gbm/paths", json=base).json()
    adverse_response = client.post("/simulate-gbm/paths", json=adverse).json()
    repeated = client.post("/simulate-gbm/paths", json=adverse).json()
    assert adverse_response["trajectories"] == repeated["trajectories"]
    assert adverse_response["terminal_risk"]["var"] > base_response["terminal_risk"]["var"]
    assert adverse_response["terminal_risk"]["expected_shortfall"] > base_response["terminal_risk"]["expected_shortfall"]


def test_trajectory_endpoints_reject_visualization_limits():
    assert client.post("/simulate-gbm/paths", json={"paths": 129}).status_code == 422
    assert client.post("/simulate-gbm/paths", json={"steps": 513}).status_code == 422
