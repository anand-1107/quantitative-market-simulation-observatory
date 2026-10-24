"""Historical-news provider boundary and persistent SQLite response cache.

Finnhub is an optional runtime provider. Credentials are read only from FINNHUB_API_KEY.
"""
from __future__ import annotations

import hashlib
import json
import logging
from math import exp
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsQuery:
    ticker: str
    start: str
    end: str


class HistoricalNewsProvider(Protocol):
    name: str
    def historical_news(self, query: NewsQuery) -> list[dict[str, Any]]: ...


class NewsProviderError(RuntimeError):
    def __init__(self, message: str, *, http_status: int | None = None):
        super().__init__(message)
        self.http_status = http_status


class FinnhubNewsProvider:
    name = "Finnhub company news"
    endpoint = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key: str | None = None, *, timeout: float = 12):
        self.api_key = api_key if api_key is not None else os.getenv("FINNHUB_API_KEY", "")
        self.timeout = timeout
        self.last_diagnostics: dict[str, int | float | None] = {}

    def historical_news(self, query: NewsQuery) -> list[dict[str, Any]]:
        if not self.api_key:
            raise NewsProviderError("Historical news is not configured. Set FINNHUB_API_KEY on the backend, then restart the API. No news search was performed.")
        # The application uses half-open [start,end) periods; Finnhub's `to` date is inclusive.
        provider_end = (date.fromisoformat(query.end) - timedelta(days=1)).isoformat()
        params = urlencode({"symbol": query.ticker, "from": query.start, "to": provider_end, "token": self.api_key})
        request = Request(f"{self.endpoint}?{params}", headers={"Accept": "application/json", "User-Agent": "QuantitativeMarketObservatory/0.1"})
        started = time.perf_counter()
        status: int | None = None
        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", None) or (response.getcode() if hasattr(response, "getcode") else 200)
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # HTTPError includes the request URL, which contains the API token. Never echo it.
            if exc.code == 429:
                message = "Finnhub rate limit reached. Wait before searching again or review the account's request quota."
            elif exc.code in (401, 403):
                message = "Finnhub rejected the request (invalid credentials, permission, or account coverage). Verify the backend key and Finnhub plan."
            else:
                message = f"Finnhub historical news request failed with HTTP {exc.code}. Check provider availability and account coverage."
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            self.last_diagnostics = {"http_status": exc.code, "duration_ms": duration_ms, "raw_article_count": None, "accepted_article_count": 0, "rejected_item_count": 0}
            logger.warning("Finnhub company-news failed: status=%s duration_ms=%s symbol=%s from=%s to=%s error=%s",
                           exc.code, duration_ms, query.ticker.upper(), query.start, provider_end, message)
            raise NewsProviderError(message, http_status=exc.code) from None
        except TimeoutError:
            message = "Finnhub historical news request timed out. Retry later or check backend connectivity."
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            self.last_diagnostics = {"http_status": None, "duration_ms": duration_ms, "raw_article_count": None, "accepted_article_count": 0, "rejected_item_count": 0}
            logger.warning("Finnhub company-news failed: status=unavailable duration_ms=%s symbol=%s from=%s to=%s error=%s",
                           duration_ms, query.ticker.upper(), query.start, provider_end, message)
            raise NewsProviderError(message) from None
        except URLError as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                message = "Finnhub historical news request timed out. Retry later or check backend connectivity."
            else:
                message = "Could not connect to Finnhub for historical news. Check backend connectivity and provider availability."
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            self.last_diagnostics = {"http_status": None, "duration_ms": duration_ms, "raw_article_count": None, "accepted_article_count": 0, "rejected_item_count": 0}
            logger.warning("Finnhub company-news failed: status=unavailable duration_ms=%s symbol=%s from=%s to=%s error=%s",
                           duration_ms, query.ticker.upper(), query.start, provider_end, message)
            raise NewsProviderError(message) from None
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            message = "Finnhub returned an unreadable historical-news response. Retry later or check provider status."
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            self.last_diagnostics = {"http_status": status, "duration_ms": duration_ms, "raw_article_count": None, "accepted_article_count": 0, "rejected_item_count": 0}
            logger.warning("Finnhub company-news failed: status=%s duration_ms=%s symbol=%s from=%s to=%s error=%s",
                           status, duration_ms, query.ticker.upper(), query.start, provider_end, message)
            raise NewsProviderError(message) from None
        except Exception as exc:
            # Never log exception text or traceback here: URL-bearing exceptions can contain the token.
            message = "Finnhub historical news request failed unexpectedly. Check backend logs for the sanitized provider diagnostic."
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            self.last_diagnostics = {"http_status": status, "duration_ms": duration_ms, "raw_article_count": None, "accepted_article_count": 0, "rejected_item_count": 0}
            logger.error("Finnhub company-news unexpected failure: status=%s duration_ms=%s symbol=%s from=%s to=%s exception_type=%s",
                         status, duration_ms, query.ticker.upper(), query.start, provider_end, type(exc).__name__)
            raise NewsProviderError(message) from None
        if not isinstance(payload, list):
            # Finnhub may return a JSON error object with HTTP 200; do not expose arbitrary payloads.
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            self.last_diagnostics = {"http_status": status, "duration_ms": duration_ms, "raw_article_count": None, "accepted_article_count": 0, "rejected_item_count": 0}
            logger.warning("Finnhub company-news malformed response: status=%s duration_ms=%s symbol=%s from=%s to=%s payload_type=%s",
                           status, duration_ms, query.ticker.upper(), query.start, provider_end, type(payload).__name__)
            raise NewsProviderError("Finnhub returned an error or unexpected historical-news response. Check credentials, account access, and date coverage.")
        articles = []
        rejected = 0
        for item in payload:
            if not isinstance(item, dict) or not item.get("url") or not item.get("headline"):
                rejected += 1
                continue
            try:
                published = datetime.fromtimestamp(float(item["datetime"]), tz=timezone.utc)
            except (KeyError, TypeError, ValueError, OverflowError):
                published = None
            articles.append({
                "headline": str(item["headline"]), "publisher": str(item.get("source") or "Finnhub"),
                "original_url": str(item["url"]), "publication_timestamp": published.isoformat() if published else None,
                "publication_timezone": "UTC" if published else None,
                "retrieved_timestamp": datetime.now(timezone.utc).isoformat(),
                "relevant_instrument": query.ticker.upper() if query.ticker.upper() in {part.strip().upper() for part in str(item.get("related") or "").split(",")} else None,
                "provider_query_ticker": query.ticker.upper(), "excerpt": str(item.get("summary") or "") or None,
                "category": str(item.get("category") or "general"), "source_provider": self.name,
            })
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        self.last_diagnostics = {"http_status": status, "duration_ms": duration_ms, "raw_article_count": len(payload), "accepted_article_count": len(articles), "rejected_item_count": rejected}
        logger.info("Finnhub company-news completed: status=%s duration_ms=%s symbol=%s from=%s to=%s raw_articles=%s accepted_articles=%s rejected_items=%s",
                    status, duration_ms, query.ticker.upper(), query.start, provider_end, len(payload), len(articles), rejected)
        return articles


class NewsResponseCache:
    """Persistent exact-query cache. Cache empty responses briefly; errors are never cached."""
    def __init__(self, path: str | Path = "data/news_cache.sqlite3", *, ttl_seconds: int = 3600):
        self.path = Path(path).expanduser().resolve()
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def key(query: NewsQuery) -> str:
        normalized = f"{query.ticker.upper()}|{query.start}|{query.end}"
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: NewsQuery) -> list[dict[str, Any]] | None:
        if not self.path.exists():
            return None
        try:
            with sqlite3.connect(self.path) as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS news_cache (cache_key TEXT PRIMARY KEY, stored_at REAL NOT NULL, payload TEXT NOT NULL)")
                row = connection.execute("SELECT stored_at, payload FROM news_cache WHERE cache_key=?", (self.key(query),)).fetchone()
        except sqlite3.Error:
            return None
        if row is None or time.time() - float(row[0]) > self.ttl_seconds:
            return None
        try:
            value = json.loads(row[1])
            return value if isinstance(value, list) else None
        except json.JSONDecodeError:
            return None

    def put(self, query: NewsQuery, articles: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(self.path) as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS news_cache (cache_key TEXT PRIMARY KEY, stored_at REAL NOT NULL, payload TEXT NOT NULL)")
                connection.execute("INSERT OR REPLACE INTO news_cache(cache_key, stored_at, payload) VALUES(?,?,?)",
                                   (self.key(query), time.time(), json.dumps(articles, separators=(",", ":"))))
                connection.commit()
        except sqlite3.Error:
            return


def fetch_historical_news(query: NewsQuery, provider: HistoricalNewsProvider | None = None,
                          cache: NewsResponseCache | None = None) -> tuple[list[dict[str, Any]], str, bool]:
    provider = provider or FinnhubNewsProvider()
    cache = cache or NewsResponseCache()
    cached = cache.get(query)
    if cached is not None:
        return cached, provider.name, True
    articles = provider.historical_news(query)
    # Deduplicate by canonical source URL, preserving original records/timestamps.
    by_url = {article["original_url"].strip(): article for article in articles if article.get("original_url")}
    result = list(by_url.values())
    cache.put(query, result)
    return result, provider.name, False


def rank_news_for_events(articles: list[dict[str, Any]], events: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    """Rank factual provider articles; scores expose temporal, relevance, category, and corroboration signals."""
    import re
    from datetime import datetime, timezone

    token = re.compile(rf"(?<![A-Za-z0-9]){re.escape(ticker.upper())}(?![A-Za-z0-9])")
    category_terms = {
        "large_absolute_return": {"earnings", "guidance", "results", "revenue", "profit", "loss", "downgrade", "upgrade", "acquisition", "merger", "lawsuit", "regulator"},
        "unusual_volume": {"volume", "trading", "shares", "investor", "announcement", "filing"},
        "opening_gap": {"earnings", "announcement", "gap", "pre-market", "after-hours", "guidance", "deal"},
        "volatility_spike": {"volatility", "risk", "crash", "surge", "investigation", "halt", "uncertainty"},
        "sustained_movement": {"trend", "outlook", "forecast", "demand", "sales", "policy", "rates", "sector"},
    }
    def parsed(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
        except (TypeError, ValueError):
            return None

    output = []
    for event in events:
        event_time = parsed(str(event.get("timestamp", "")))
        if event_time is None:
            continue
        event_day = event_time.date()
        candidates = []
        for article in articles:
            published = parsed(article.get("publication_timestamp"))
            if published is None:
                continue
            headline = str(article.get("headline", ""))
            excerpt = str(article.get("excerpt") or "")
            text = f"{headline} {excerpt}".lower()
            hours = abs((published - event_time).total_seconds()) / 3600
            calendar_days = abs((published.date() - event_day).days)
            resolution = str(event.get("timestamp_resolution", "session_date"))
            temporal = float(exp(-(calendar_days / 2.0) if resolution == "session_date" else -hours / 48.0))
            related = str(article.get("relevant_instrument") or "").upper() == ticker.upper()
            mentioned = bool(token.search(f"{headline} {excerpt}"))
            relevance = 1.0 if related else .8 if mentioned else .4  # Finnhub company-news endpoint itself is ticker scoped.
            terms = category_terms.get(str(event.get("category")), set())
            category = .25 + .75 * min(1.0, sum(term in text for term in terms) / 2.0) if terms else .25
            publisher = str(article.get("publisher") or "unknown").strip().lower()
            corroborating = {str(other.get("publisher") or "unknown").strip().lower() for other in articles
                             if other is not article and parsed(other.get("publication_timestamp")) is not None
                             and abs((parsed(other.get("publication_timestamp")) - published).total_seconds()) <= 86400
                             and (str(other.get("relevant_instrument") or "").upper() == ticker.upper() or token.search(f"{other.get('headline','')} {other.get('excerpt') or ''}"))}
            corroboration = min(1.0, len(corroborating) / 2.0)
            score = .45 * temporal + .30 * relevance + .15 * category + .10 * corroboration
            delta = (published - event_time).total_seconds()
            if resolution == "session_date" and published.date() == event_day:
                phase = "same_session_date; pre-market/intraday/after-market phase unavailable without exchange-local bar time and calendar"
            elif published.date() != event_day:
                phase = "published_before_observation" if delta < 0 else "published_after_observation"
            elif abs(delta) <= 1:
                phase = "near_observed_intraday_bar; session phase unavailable without exchange calendar"
            else:
                phase = "published_before_observed_bar; exchange session phase unavailable" if delta < 0 else "published_after_observed_bar; exchange session phase unavailable"
            candidates.append({"article": article, "score": float(score), "signals": {
                "temporal_proximity_score": temporal, "instrument_relevance_score": relevance,
                "category_relevance_score": category, "corroborating_publisher_count": len(corroborating),
                "corroboration_score": corroboration,
                "published_minus_event_hours": delta / 3600 if resolution == "intraday_observation_utc" else "unavailable at session-date resolution",
                "absolute_calendar_days_from_event": calendar_days,
                "publication_phase": phase,
                "explanation": "Rank combines time proximity, ticker-scoped/company relevance, category terms, and distinct nearby publishers; it is a relevance heuristic, not evidence of causation."}})
        candidates.sort(key=lambda item: (-item["score"], str(item["article"].get("publication_timestamp", ""))))
        output.append({"event_id": event.get("event_id"), "category": event.get("category"), "ranked_articles": [
            {"rank": rank + 1, **candidate} for rank, candidate in enumerate(candidates[:20])
        ]})
    return output
