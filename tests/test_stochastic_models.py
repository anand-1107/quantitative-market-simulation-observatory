import numpy as np
import pytest

from quantum_market_lab.models.diagnostics import path_diagnostics, return_diagnostics
from quantum_market_lab.models.garch import garch_returns_to_prices, simulate_garch
from quantum_market_lab.models.jump_diffusion import simulate_jump_diffusion
from quantum_market_lab.models.ou import simulate_ou
from quantum_market_lab.models.registry import get_model, list_models


def test_ou_shape_reproducibility_and_exact_mean_reversion():
    args = (3., 1., 2., .5, 1., 20, 40)
    times, paths = simulate_ou(*args, seed=11)
    _, again = simulate_ou(*args, seed=11)
    assert times.shape == (21,) and paths.shape == (40, 21)
    np.testing.assert_array_equal(paths, again)
    assert paths[:, -1].mean() == pytest.approx(1 + 2 * np.exp(-2), abs=.08)
    stationary_var = .5**2 / (2 * 2)
    _, stationary = simulate_ou(1., 1., 2., .5, 8., 1, 20000, seed=31)
    assert stationary[:, -1].var() == pytest.approx(stationary_var, rel=.06)


@pytest.mark.parametrize("kwargs", [
    {"initial_value": 0, "mean_level": 0, "mean_reversion_speed": -1, "volatility": 1, "years": 1, "steps": 2, "paths": 1},
    {"initial_value": 0, "mean_level": 0, "mean_reversion_speed": 1, "volatility": -1, "years": 1, "steps": 2, "paths": 1},
])
def test_ou_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError): simulate_ou(**kwargs)


def test_jump_diffusion_shape_positive_reproducible_and_zero_jump_is_gbm():
    args = (100., .05, .2, 0., -.1, .3, 1., 50, 100)
    times, prices = simulate_jump_diffusion(*args, seed=2)
    _, repeated = simulate_jump_diffusion(*args, seed=2)
    assert prices.shape == (100, 51) and np.all(prices > 0)
    np.testing.assert_array_equal(prices, repeated)
    assert prices[:, 0].tolist() == [100.] * 100
    # With no jumps, drift compensation vanishes; terminal mean approaches GBM theory.
    assert prices[:, -1].mean() == pytest.approx(100 * np.exp(.05), rel=.08)


def test_jump_diffusion_validation_and_compensated_mean():
    with pytest.raises(ValueError): simulate_jump_diffusion(0, .1, .2, 1, 0, .1, 1, 2, 2)
    _, paths = simulate_jump_diffusion(100, .03, .15, 2., -.1, .2, 1., 50, 30000, seed=42)
    assert paths[:, -1].mean() == pytest.approx(100 * np.exp(.03), rel=.035)


def test_garch_shapes_stationarity_clustering_and_prices():
    returns, variances = simulate_garch(1e-6, .08, .9, 800, 200, seed=9)
    assert returns.shape == variances.shape == (200, 800)
    assert np.all(variances > 0)
    diag = return_diagnostics(returns, variances)
    assert diag["squared_return_lag1_autocorrelation"] > .02
    assert diag["mean_conditional_variance"] == pytest.approx(1e-6 / (1 - .08 - .9), rel=.2)
    prices = garch_returns_to_prices(returns, 50)
    assert prices.shape == (200, 801) and np.all(prices > 0)


def test_garch_valid_nonstationary_parameterization_and_errors():
    _, var = simulate_garch(.1, .7, .5, 10, 2, initial_variance=.2, seed=1)
    assert np.all(var > 0)
    with pytest.raises(ValueError): simulate_garch(0, .1, .2, 10)
    with pytest.raises(ValueError): simulate_garch(.1, -.1, .2, 10)
    with pytest.raises(ValueError): simulate_garch(.1, .1, .2, 10, return_convention="unknown")
    with pytest.raises(ValueError): garch_returns_to_prices(np.array([[ -1.1]]), return_convention="simple")


def test_model_registry_and_path_diagnostics():
    assert get_model("OU").name == "ou"
    assert {m["name"] for m in list_models()} == {"gbm", "ou", "jump_diffusion", "garch"}
    stats = path_diagnostics(np.array([[1., 2.], [1., 4.]]))
    assert stats["terminal_mean"] == 3 and stats["terminal_quantiles"]["0.5"] == 3
    with pytest.raises(KeyError): get_model("other")
