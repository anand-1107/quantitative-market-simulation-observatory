"""Safe deterministic Parquet storage for normalized market datasets."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .market_data import MarketDataError, validate_market_data

_SAFE_ID = re.compile(r"^[A-Za-z0-9^=._-]{1,32}$")


class MarketDataStorageError(MarketDataError):
    """Storage operation failed or data could not be read."""


class MarketDataStore:
    def __init__(self, root: str | Path = "data/market"):
        self.root = Path(root).expanduser().resolve()

    def _directory(self, ticker: str) -> Path:
        if not isinstance(ticker, str) or not _SAFE_ID.fullmatch(ticker):
            raise ValueError("invalid ticker identifier")
        path = (self.root / ticker.upper()).resolve()
        if path.parent != self.root:
            raise ValueError("ticker escapes configured storage directory")
        return path

    def save(self, ticker: str, df: pd.DataFrame, metadata: dict | None = None, overwrite: bool = False) -> dict:
        report = validate_market_data(df)
        if not report.valid:
            raise MarketDataStorageError(f"Refusing to store invalid market data: {report}")
        directory = self._directory(ticker)
        parquet = directory / "daily.parquet"
        if parquet.exists() and not overwrite:
            raise FileExistsError(f"Dataset already exists for {ticker}; pass overwrite=True to replace it")
        directory.mkdir(parents=True, exist_ok=True)
        payload = dict(metadata or {})
        payload.update({"ticker": ticker.upper(), "stored_at": datetime.now(timezone.utc).isoformat(),
                        "start": str(df.index.min().date()), "end": str(df.index.max().date()), "row_count": len(df)})
        try:
            df.to_parquet(parquet, engine="pyarrow", index=True)
            (directory / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as exc:
            raise MarketDataStorageError(f"Could not save dataset for {ticker}: {exc}") from exc
        return payload

    def load(self, ticker: str) -> pd.DataFrame:
        path = self._directory(ticker) / "daily.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"No stored market data found for {ticker}")
        try:
            result = pd.read_parquet(path, engine="pyarrow")
            result.index = pd.DatetimeIndex(result.index, name="date")
            return result
        except Exception as exc:
            raise MarketDataStorageError(f"Could not read Parquet dataset for {ticker}: {exc}") from exc

    def metadata(self, ticker: str) -> dict:
        path = self._directory(ticker) / "metadata.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(f"No metadata found for {ticker}") from None
        except (OSError, json.JSONDecodeError) as exc:
            raise MarketDataStorageError(f"Could not read metadata for {ticker}: {exc}") from exc

    def list_datasets(self) -> list[dict]:
        if not self.root.exists(): return []
        entries = []
        for directory in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if not _SAFE_ID.fullmatch(directory.name) or not (directory / "daily.parquet").is_file(): continue
            try: entries.append(self.metadata(directory.name))
            except MarketDataError: continue
        return entries
