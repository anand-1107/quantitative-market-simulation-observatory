import numpy as np
import pytest

from quantum_market_lab.gbm import simulate_gbm


def test_shapes_and_initial_prices():
    times, prices = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 20, seed=7)
    assert times.shape == (11,)
    assert prices.shape == (20, 11)
    assert np.all(prices[:, 0] == pytest.approx(100.0))


def test_reproducibility_with_same_seed():
    _, first = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 20, seed=123)
    _, second = simulate_gbm(100, 0.05, 0.2, 1.0, 10, 20, seed=123)
    np.testing.assert_array_equal(first, second)


def test_prices_remain_positive():
    _, prices = simulate_gbm(100, 0.05, 0.8, 2.0, 100, 100, seed=9)
    assert np.all(prices > 0)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        simulate_gbm(0, 0.05, 0.2, 1.0, 10, 20)
    with pytest.raises(ValueError):
        simulate_gbm(100, 0.05, -0.2, 1.0, 10, 20)
    with pytest.raises(ValueError):
        simulate_gbm(100, 0.05, 0.2, 1.0, 0, 20)


def test_terminal_mean_is_reasonably_close_to_theory():
    s0, mu, sigma, years, paths = 100.0, 0.07, 0.2, 1.0, 50_000
    _, prices = simulate_gbm(s0, mu, sigma, years, 1, paths, seed=2026)
    theoretical_mean = s0 * np.exp(mu * years)
    simulated_mean = prices[:, -1].mean()
    standard_error = prices[:, -1].std(ddof=1) / np.sqrt(paths)

    # A 5-SE tolerance is generous enough to avoid a flaky random test.
    assert abs(simulated_mean - theoretical_mean) < 5 * standard_error
