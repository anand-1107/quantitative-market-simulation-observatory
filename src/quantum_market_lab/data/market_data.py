"""Yahoo Finance market data ingestion. yfinance availability and data quality are not guaranteed."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
_TICKER_RE = re.compile(r"^[A-Za-z0-9^=._-]{1,32}$")


class MarketDataError(Exception):
    """Base exception for market-data input, provider, and quality errors."""


class MarketDataProviderError(MarketDataError):
    """The upstream market-data provider failed."""


class MarketDataQualityError(MarketDataError):
    """Downloaded data could not be normalized safely."""


@dataclass(frozen=True)
class MarketDataValidationReport:
    valid: bool
    row_count: int
    missing_columns: list[str]
    duplicate_dates: int
    missing_values: dict[str, int]
    invalid_prices: int
    invalid_ohlc_rows: int
    date_range: dict[str, str | None]
    warnings: list[str]


def validate_market_data(df: pd.DataFrame) -> MarketDataValidationReport:
    """Inspect data without modifying it; duplicate count is excess duplicate rows."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    invalid_dates = not isinstance(df.index, pd.DatetimeIndex) or df.index.isna().any()
    duplicate_dates = int(df.index.duplicated(keep="first").sum()) if isinstance(df.index, pd.DatetimeIndex) else 0
    missing_values = {}
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        missing_values[col] = int((df[col].isna() | ~np.isfinite(values)).sum())
    invalid_prices = 0
    invalid_ohlc = 0
    if not missing_cols and len(df):
        prices = df.loc[:, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        invalid_prices = int((~np.isfinite(prices) | (prices <= 0)).any(axis=1).sum())
        invalid_ohlc = int(((prices["high"] < prices["low"]) | (prices["high"] < prices[["open", "close"]].max(axis=1)) | (prices["low"] > prices[["open", "close"]].min(axis=1))).fillna(False).sum())
    warnings = []
    if len(df) == 0:
        warnings.append("DataFrame is empty")
    if invalid_dates:
        warnings.append("Index must contain valid dates")
    if duplicate_dates:
        warnings.append("Duplicate dates are present")
    if invalid_ohlc:
        warnings.append("OHLC relationships are invalid")
    if any(missing_values.values()):
        warnings.append("Required fields contain missing or non-finite values")
    if invalid_prices:
        warnings.append("OHLC prices contain zero, negative, or non-finite values")
    valid = bool(len(df) and not missing_cols and not invalid_dates and not duplicate_dates and not any(missing_values.values()) and invalid_prices == 0 and invalid_ohlc == 0)
    date_range = {"start": None, "end": None}
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) and not df.index.isna().all():
        date_range = {"start": str(df.index.min()), "end": str(df.index.max())}
    return MarketDataValidationReport(valid, len(df), missing_cols, duplicate_dates, missing_values, invalid_prices, invalid_ohlc, date_range, warnings)


def _normalize(frame: pd.DataFrame, *, preserve_intraday: bool = False) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise MarketDataQualityError("Provider returned no market data")
    result = frame.copy(deep=True)
    if isinstance(result.columns, pd.MultiIndex):
        # yfinance can return either (field, ticker) or (ticker, field).
        levels = [{str(x).strip().lower() for x in result.columns.get_level_values(i)} for i in range(result.columns.nlevels)]
        field_level = next((i for i, vals in enumerate(levels) if set(REQUIRED_COLUMNS).issubset(vals)), None)
        if field_level is None:
            raise MarketDataQualityError("Provider response has unsupported MultiIndex columns")
        result.columns = [str(col[field_level]).strip().lower() for col in result.columns]
    else:
        result.columns = [str(col).strip().lower().replace(" ", "_") for col in result.columns]
    if "adj_close" not in result and "adjclose" in result:
        result = result.rename(columns={"adjclose": "adj_close"})
    if len(set(result.columns)) != len(result.columns):
        # Common single-ticker MultiIndex response can collapse duplicate field labels.
        result = result.loc[:, ~pd.Index(result.columns).duplicated(keep="first")]
    missing = [c for c in REQUIRED_COLUMNS if c not in result.columns]
    if missing:
        raise MarketDataQualityError(f"Provider response is missing required columns: {', '.join(missing)}")
    try:
        index = pd.DatetimeIndex(pd.to_datetime(result.index, errors="raise"))
    except (ValueError, TypeError) as exc:
        raise MarketDataQualityError("Provider response contains invalid dates") from exc
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    if not preserve_intraday:
        index = index.normalize()
    result.index = index
    result.index.name = "date"
    for col in result.columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    # Preserve first provider observation for duplicate calendar dates (documented policy).
    result = result.loc[~result.index.duplicated(keep="first")].sort_index()
    report = validate_market_data(result)
    if not report.valid:
        raise MarketDataQualityError(f"Invalid provider data: {report}")
    return result


class MarketDataLoader:
    """Download and normalize Yahoo Finance history. Requires network at call time."""

    def __init__(self, ticker: str, start: str | date | None = None, end: str | date | None = None, interval: str = "1d"):
        if not isinstance(ticker, str) or not _TICKER_RE.fullmatch(ticker.strip()):
            raise ValueError("ticker must be a valid symbol (letters, digits, ^, =, ., _, -)")
        self.ticker = ticker.strip().upper()
        self.start = self._date(start, "start") if start is not None else None
        self.end = self._date(end, "end") if end is not None else None
        if self.start and self.end and self.start >= self.end:
            raise ValueError("start must be earlier than end")
        if not isinstance(interval, str) or interval not in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}:
            raise ValueError("unsupported yfinance interval")
        self.interval = interval

    @staticmethod
    def _date(value: str | date, name: str) -> date:
        try:
            parsed = date.fromisoformat(value) if isinstance(value, str) else value
        except ValueError as exc:
            raise ValueError(f"{name} must be an ISO date (YYYY-MM-DD)") from exc
        if isinstance(parsed, datetime) or not isinstance(parsed, date):
            raise ValueError(f"{name} must be an ISO date")
        return parsed

    def download(self) -> pd.DataFrame:
        try:
            frame = yf.download(self.ticker, start=self.start.isoformat() if self.start else None,
                                end=self.end.isoformat() if self.end else None,
                                interval=self.interval, auto_adjust=False, progress=False, threads=False)
        except Exception as exc:
            raise MarketDataProviderError(f"Yahoo Finance download failed for {self.ticker}: {exc}") from exc
        try:
            return _normalize(frame, preserve_intraday=self.interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"})
        except MarketDataQualityError:
            raise
        except Exception as exc:
            raise MarketDataQualityError(f"Could not normalize response for {self.ticker}: {exc}") from exc

    def metadata(self, df: pd.DataFrame) -> dict[str, Any]:
        return {"ticker": self.ticker, "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "start": str(df.index.min().date()), "end": str(df.index.max().date()),
                "interval": self.interval, "row_count": len(df)}
