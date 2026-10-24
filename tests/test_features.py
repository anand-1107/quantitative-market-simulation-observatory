import numpy as np
import pandas as pd
import pytest

from quantum_market_lab.data.features import (annualize_arithmetic_return, annualize_mean_log_return,
    annualize_volatility, compute_log_returns, drawdown, realized_volatility, summarize_returns)


def test_log_returns_known_values_and_fallback():
    df = pd.DataFrame({"close": [100., 110., 100.]})
    result = compute_log_returns(df)
    np.testing.assert_allclose(result.iloc[1:].values, [np.log(1.1), np.log(100 / 110)])
    assert result.iloc[0] != 0
    df.loc[1, "close"] = -1
    assert pd.isna(compute_log_returns(df).iloc[1])


def test_return_summary_and_short_series():
    result = summarize_returns(pd.Series([1., 2., 3., np.nan]))
    assert result["observation_count"] == 3 and result["missing_fraction"] == .25
    assert result["mean"] == 2 and result["excess_kurtosis"] is None
    assert summarize_returns(pd.Series([np.nan]))["mean"] is None


def test_rolling_volatility_and_annualization():
    result = realized_volatility(pd.Series([.01, -.01, .02]), window=2)
    assert pd.isna(result.iloc[0]) and result.iloc[1] == pytest.approx(np.std([.01, -.01], ddof=1))
    annual = realized_volatility(pd.Series([.01, -.01, .02]), window=2, annualized=True)
    assert annual.iloc[-1] == pytest.approx(result.iloc[-1] * np.sqrt(252))
    assert annualize_volatility(.2) == pytest.approx(.2 * np.sqrt(252))
    assert annualize_mean_log_return(.001) == pytest.approx(.252)
    assert annualize_arithmetic_return(.001, 2) == pytest.approx(1.001**2 - 1)


def test_drawdown_from_prices_and_returns():
    dd, maximum = drawdown(pd.Series([100., 120., 90., 110.]))
    assert list(dd) == pytest.approx([0., 0., -.25, -1/12]) and maximum == pytest.approx(-.25)
    _, ret_max = drawdown(pd.Series([.1, -.2, .1]), input_type="returns")
    assert ret_max < 0

