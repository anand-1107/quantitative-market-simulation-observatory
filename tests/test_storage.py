import pandas as pd
import pytest

from quantum_market_lab.data.storage import MarketDataStore, MarketDataStorageError


def frame():
    return pd.DataFrame({"open": [10., 11.], "high": [11., 12.], "low": [9., 10.], "close": [10.5, 11.5], "volume": [0, 10]}, index=pd.date_range("2024-01-01", periods=2, name="date"))


def test_parquet_round_trip_and_metadata(tmp_path):
    store = MarketDataStore(tmp_path)
    metadata = store.save("AAPL", frame(), {"interval": "1d"})
    pd.testing.assert_frame_equal(store.load("AAPL"), frame(), check_freq=False)
    assert metadata["row_count"] == 2 and store.metadata("AAPL")["interval"] == "1d"
    assert len(store.list_datasets()) == 1
    with pytest.raises(FileExistsError): store.save("AAPL", frame())


def test_missing_corrupt_and_unsafe_path(tmp_path):
    store = MarketDataStore(tmp_path)
    with pytest.raises(FileNotFoundError): store.load("AAPL")
    for ticker in ("../escape", "/tmp/file", "AAPL/foo"):
        with pytest.raises(ValueError): store.load(ticker)
    folder = tmp_path / "MSFT"
    folder.mkdir()
    (folder / "daily.parquet").write_text("not parquet")
    with pytest.raises(MarketDataStorageError): store.load("MSFT")

