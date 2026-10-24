import numpy as np
import pandas as pd
import pytest

from quantum_market_lab.data.market_data import MarketDataLoader, MarketDataProviderError, MarketDataQualityError, validate_market_data


def prices(index=None):
    idx = index if index is not None else pd.date_range("2024-01-01", periods=3, tz="UTC")
    return pd.DataFrame({"Open": [10., 11., 12.], "High": [11., 12., 13.], "Low": [9., 10., 11.],
                         "Close": [10.5, 11.5, 12.5], "Adj Close": [10., 11., 12.], "Volume": [0, 100, 200]}, index=idx)


def test_normalize_download_and_timezone_and_duplicate_policy(monkeypatch):
    df = prices(pd.DatetimeIndex(["2024-01-02 04:00", "2024-01-01 04:00", "2024-01-01 18:00"], tz="America/New_York"))
    monkeypatch.setattr("quantum_market_lab.data.market_data.yf.download", lambda *a, **k: df)
    result = MarketDataLoader("aapl").download()
    assert result.index.name == "date" and result.index.is_monotonic_increasing
    assert result.index.tz is None and len(result) == 2
    assert result.iloc[0].volume == 100 and "adj_close" in result
    assert list(df.columns)[0] == "Open"  # input not mutated


def test_multiindex_columns(monkeypatch):
    df = prices()
    df.columns = pd.MultiIndex.from_product([df.columns, ["AAPL"]])
    monkeypatch.setattr("quantum_market_lab.data.market_data.yf.download", lambda *a, **k: df)
    assert MarketDataLoader("AAPL").download().shape == (3, 6)


def test_intraday_download_preserves_observation_times(monkeypatch):
    df = prices(pd.date_range("2024-01-02 14:30", periods=3, freq="15min", tz="UTC"))
    monkeypatch.setattr("quantum_market_lab.data.market_data.yf.download", lambda *a, **k: df)
    result = MarketDataLoader("AAPL", interval="15m").download()
    assert len(result) == 3
    assert result.index.is_monotonic_increasing
    assert list(result.index.hour) == [14, 14, 15]
    assert list(result.index.minute) == [30, 45, 0]


def test_provider_empty_failure_and_missing_columns(monkeypatch):
    monkeypatch.setattr("quantum_market_lab.data.market_data.yf.download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(MarketDataQualityError): MarketDataLoader("SPY").download()
    monkeypatch.setattr("quantum_market_lab.data.market_data.yf.download", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(MarketDataProviderError): MarketDataLoader("SPY").download()
    monkeypatch.setattr("quantum_market_lab.data.market_data.yf.download", lambda *a, **k: pd.DataFrame({"Close": [1]}, index=["2024-01-01"]))
    with pytest.raises(MarketDataQualityError, match="missing required"):
        MarketDataLoader("SPY").download()


def test_loader_validates_inputs():
    with pytest.raises(ValueError): MarketDataLoader("../etc/passwd")
    with pytest.raises(ValueError): MarketDataLoader("AAPL", "2024-02-01", "2024-01-01")
    with pytest.raises(ValueError): MarketDataLoader("AAPL", "yesterday")


def test_validation_finds_quality_problems_without_mutation():
    df = prices(pd.date_range("2024-01-01", periods=3))
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.iloc[1, df.columns.get_loc("close")] = -1
    df.iloc[2, df.columns.get_loc("high")] = 1
    before = df.copy(deep=True)
    report = validate_market_data(df)
    assert not report.valid and report.invalid_prices == 1 and report.invalid_ohlc_rows >= 1
    pd.testing.assert_frame_equal(df, before)

