import numpy as np
import pandas as pd
import pytest

from quantum_market_lab.data.features import drawdown
from quantum_market_lab.models.diagnostics import empirical_var_es, path_return_drawdown, terminal_return_risk


def test_empirical_var_es_known_answer_including_threshold_ties():
    # At 75%, linear VaR lies halfway between losses .1 and .2.
    result = empirical_var_es(np.array([-.1, 0., .1, .2]), .75)
    assert result["var"] == pytest.approx(.125)
    assert result["expected_shortfall"] == pytest.approx(.2)
    assert result["tail_observation_count"] == 1
    assert result["quantile_method"].startswith("linear")
    tied = empirical_var_es(np.array([-.1, 0., .1, .2, .2]), .75)
    assert tied["var"] == pytest.approx(.2)
    assert tied["expected_shortfall"] == pytest.approx(.2)
    assert tied["tail_observation_count"] == 2


def test_empirical_var_es_constant_small_extreme_and_positive_gain_samples():
    constant = empirical_var_es(np.zeros(1), .95)
    assert constant["var"] == constant["expected_shortfall"] == 0
    losses = empirical_var_es(np.array([-.1, 0., .1, .2]), .75)
    assert losses["var"] == pytest.approx(.125)
    assert losses["expected_shortfall"] == pytest.approx(.2)
    extreme = empirical_var_es(np.array([-1e8, 1e8]), .5)
    assert extreme["var"] == pytest.approx(0)
    assert extreme["expected_shortfall"] == pytest.approx(1e8)
    # A negative VaR/ES is valid when even the worst empirical outcomes gain.
    gains = empirical_var_es(np.array([-.3, -.2, -.1]), .95)
    assert gains["var"] < 0 and gains["expected_shortfall"] < 0


@pytest.mark.parametrize("confidence", [0, 1, -0.1, 1.1, np.nan, np.inf, "invalid"])
def test_empirical_var_es_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        empirical_var_es(np.array([0., 1.]), confidence)


@pytest.mark.parametrize("losses", [np.array([]), np.array([[1., 2.]]), np.array([np.nan]), np.array([np.inf])])
def test_empirical_var_es_rejects_empty_nonvector_or_nonfinite_data(losses):
    with pytest.raises(ValueError):
        empirical_var_es(losses)


def test_drawdown_price_curve_rising_falling_recovery_and_initial_peak():
    _, rising = drawdown(pd.Series([100., 110., 125.]))
    _, falling = drawdown(pd.Series([100., 90., 80.]))
    curve, recovery = drawdown(pd.Series([100., 120., 90., 120.]))
    assert rising == pytest.approx(0)
    assert falling == pytest.approx(80 / 100 - 1)
    assert list(curve) == pytest.approx([0, 0, -.25, 0])
    assert recovery == pytest.approx(-.25)


def test_drawdown_simple_returns_includes_initial_wealth_and_rejects_invalid_data():
    curve, maximum = drawdown(pd.Series([-.1, .2]), input_type="returns")
    assert list(curve) == pytest.approx([-.1, 0])
    assert maximum == pytest.approx(-.1)
    _, recovery = drawdown(pd.Series([.1, -.2, .1]), input_type="returns")
    assert recovery == pytest.approx(-.2)
    for values in ([0.], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            drawdown(pd.Series(values))
    for returns in ([-1.], [np.nan], [np.inf]):
        with pytest.raises(ValueError):
            drawdown(pd.Series(returns), input_type="returns")


def test_terminal_risk_uses_cumulative_simple_returns_and_returns_drawdowns():
    paths = np.array([[100., 110.], [100., 80.], [100., 90.]])
    risk = terminal_return_risk(paths, "price", confidence_level=2 / 3)
    assert risk["available"] and risk["measure"] == "terminal_horizon"
    assert risk["var"] == pytest.approx(.1 + (.2 - .1) / 3)
    assert risk["expected_shortfall"] == pytest.approx(.2)
    assert risk["scenario_count"] == 3 and risk["tail_scenario_count"] == 1
    assert risk["path_metrics"][1]["maximum_drawdown"] == pytest.approx(-.2)
    assert risk["maximum_drawdown_quantiles"]["0.5"] == pytest.approx(-.1)
    log_path = np.log(np.array([[1.1, .8 / 1.1]]))
    log_risk = terminal_return_risk(log_path, "return", "log")
    assert log_risk["path_metrics"][0]["total_simple_return"] == pytest.approx(-.2)
    simple_risk = terminal_return_risk(np.array([[.1, -.2]]), "return", "simple")
    assert simple_risk["path_metrics"][0]["total_simple_return"] == pytest.approx(.1 * .8 + .8 - 1)


def test_terminal_risk_unavailable_for_states_invalid_wealth_and_invalid_arrays():
    assert not terminal_return_risk(np.array([[1., 2.]]), "state")["available"]
    simple = terminal_return_risk(np.array([[-1.]]), "return", "simple")
    assert not simple["available"] and simple["path_metrics"] is None
    for values in (np.array([[np.nan]]), np.array([[np.inf]]), np.empty((0, 2))):
        with pytest.raises(ValueError):
            terminal_return_risk(values, "price")
    with pytest.raises(ValueError):
        terminal_return_risk(np.array([[1.]]), "price", confidence_level=1)


def test_terminal_risk_reproducible_for_fixed_simulation_output():
    from quantum_market_lab.gbm import simulate_gbm

    _, first = simulate_gbm(100, .04, .2, .5, 20, 40, seed=123)
    _, repeated = simulate_gbm(100, .04, .2, .5, 20, 40, seed=123)
    first_risk = terminal_return_risk(first, "price")
    repeated_risk = terminal_return_risk(repeated, "price")
    assert first_risk == repeated_risk
