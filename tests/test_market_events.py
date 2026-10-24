import json
from io import BytesIO
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pytest

from quantum_market_lab.data.events import EventThresholds, analyze_historical_events
from quantum_market_lab.data.news import (FinnhubNewsProvider, NewsProviderError, NewsQuery,
    NewsResponseCache, fetch_historical_news, rank_news_for_events)


def synthetic_ohlcv(n=320):
    dates = pd.bdate_range("2023-01-02", periods=n, name="date")
    returns = np.full(n, 0.0002)
    spike = min(160, max(2, n // 2))
    returns[spike] = 0.12
    if n >= 220:
        returns[210:216] = -0.022
    close = 100 * np.exp(np.cumsum(returns))
    opn = close.copy()
    opn[spike] = close[spike - 1] * 1.05
    volume = np.full(n, 1_000_000.)
    volume[spike] = 4_000_000
    return pd.DataFrame({"open": opn, "high": np.maximum(opn, close) * 1.005,
                         "low": np.minimum(opn, close) * .995, "close": close,
                         "adj_close": close, "volume": volume}, index=dates)


def no_model_calibrations(monkeypatch):
    monkeypatch.setattr("quantum_market_lab.data.events._model_comparison", lambda *args, **kwargs: {
        "observed_log_return": .12, "gbm": {"status": "unavailable"},
        "merton": {"status": "unavailable"}, "garch": {"status": "unavailable"}})


def test_detects_return_volume_gap_and_sustained_events_with_numeric_evidence(monkeypatch):
    no_model_calibrations(monkeypatch)
    result = analyze_historical_events(synthetic_ohlcv(), ticker="AAPL", start="2023-01-01", end="2025-01-01",
        thresholds=EventThresholds(absolute_log_return=.05, volume_multiple=2, opening_gap=.03,
                                  volatility_multiple=5, sustained_log_return=.08, sustained_periods=5))
    kinds = {event["category"] for event in result["candidate_events"]}
    assert {"large_absolute_return", "unusual_volume", "opening_gap", "volatility_spike", "sustained_movement"} <= kinds
    spike = [event for event in result["candidate_events"] if event["date"] == synthetic_ohlcv().index[160].date().isoformat()]
    assert all(event["method"] and event["evidence"] for event in spike)
    assert any(event["category"] == "unusual_volume" and event["evidence"]["volume_multiple"] == 4 for event in spike)
    assert result["data_convention"] and result["causality_note"]


def test_missing_prices_and_irregular_intervals_are_reported_not_fabricated(monkeypatch):
    no_model_calibrations(monkeypatch)
    frame = synthetic_ohlcv(40).drop(index=pd.bdate_range("2023-01-02", periods=40)[8:12])
    frame.iloc[4, frame.columns.get_loc("close")] = np.nan
    result = analyze_historical_events(frame, ticker="AAPL", start="2023-01-01", end="2024-01-01")
    assert any("excluded" in warning for warning in result["warnings"])
    assert result["observation_count"] == 35
    assert result["source_observation_count"] == 36 and result["excluded_observation_count"] == 1
    with pytest.raises(ValueError, match="At least three"):
        analyze_historical_events(frame.iloc[:2], ticker="AAPL", start="2023-01-01", end="2024-01-01")


def test_impact_windows_and_benchmark_alignment_use_actual_prices(monkeypatch):
    no_model_calibrations(monkeypatch)
    frame = synthetic_ohlcv(30)
    benchmark = frame.copy()
    benchmark["close"] *= np.linspace(1, 1.2, len(frame))
    result = analyze_historical_events(frame, ticker="AAPL", start="2023-01-01", end="2024-01-01",
        thresholds=EventThresholds(absolute_log_return=.05, volume_multiple=100, opening_gap=.5,
                                  volatility_multiple=20, sustained_log_return=.5), impact_window=4, benchmark=benchmark)
    event = next(item for item in result["candidate_events"] if item["category"] == "large_absolute_return")
    assert event["impact"]["available_pre_observations"] == 4
    assert event["impact"]["available_post_observations"] == 4
    assert event["impact"]["benchmark_relative_post_return"] is not None
    assert event["impact"]["causal_claim"] is False


def test_finnhub_parsing_keeps_times_and_logs_sanitized_request_diagnostics(monkeypatch, tmp_path, caplog):
    caplog.set_level("INFO", logger="quantum_market_lab.data.news")
    requests = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps([{"headline":"AAPL reports results", "source":"Example", "url":"https://example.test/a", "datetime":1700000000, "related":"AAPL", "summary":"Excerpt", "category":"company"}]).encode()
    def fake_urlopen(request, **kwargs):
        requests.append(request)
        return Response()
    monkeypatch.setattr("quantum_market_lab.data.news.urlopen", fake_urlopen)
    provider = FinnhubNewsProvider("test-secret")
    query = NewsQuery("AAPL", "2023-01-01", "2024-01-01")
    cache = NewsResponseCache(tmp_path / "cache.sqlite3")
    articles, _, cached = fetch_historical_news(query, provider, cache)
    assert not cached and articles[0]["publication_timezone"] == "UTC"
    assert datetime.fromisoformat(articles[0]["publication_timestamp"]).tzinfo is not None
    assert datetime.fromisoformat(articles[0]["retrieved_timestamp"]).tzinfo is not None
    query_params = parse_qs(urlsplit(requests[0].full_url).query)
    assert query_params["symbol"] == ["AAPL"] and query_params["from"] == ["2023-01-01"]
    assert query_params["to"] == ["2023-12-31"]  # App is half-open; provider `to` is inclusive.
    assert query_params["token"] == ["test-secret"]  # Finnhub documents query-token authentication.
    assert provider.last_diagnostics["http_status"] == 200
    assert provider.last_diagnostics["raw_article_count"] == provider.last_diagnostics["accepted_article_count"] == 1
    assert "duration_ms=" in caplog.text and "accepted_articles=1" in caplog.text
    assert "test-secret" not in caplog.text and "token=" not in caplog.text
    repeated, _, cached = fetch_historical_news(query, provider, cache)
    assert cached and repeated == articles


def test_missing_news_credentials_and_provider_failures_are_explicit(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(NewsProviderError, match="No news search was performed"):
        FinnhubNewsProvider().historical_news(NewsQuery("AAPL", "2024-01-01", "2024-02-01"))
    class Broken:
        name = "broken"
        def historical_news(self, query): raise NewsProviderError("upstream unavailable")
    with pytest.raises(NewsProviderError, match="upstream unavailable"):
        fetch_historical_news(NewsQuery("AAPL", "2024-01-01", "2024-02-01"), Broken(), NewsResponseCache("/tmp/no-cache-event-test.sqlite3"))


@pytest.mark.parametrize(("status", "message"), [
    (401, "invalid credentials"), (403, "account coverage"), (429, "rate limit"),
])
def test_finnhub_http_errors_are_specific_and_never_echo_credentials(monkeypatch, status, message):
    def fail(request, **kwargs):
        raise HTTPError(request.full_url, status, "provider error", {}, BytesIO(b"secret response"))
    monkeypatch.setattr("quantum_market_lab.data.news.urlopen", fail)
    with pytest.raises(NewsProviderError, match=message) as error:
        FinnhubNewsProvider("test-secret").historical_news(NewsQuery("AAPL", "2024-01-01", "2024-02-01"))
    assert error.value.http_status == status
    assert "test-secret" not in str(error.value)


def test_finnhub_timeout_and_provider_error_payload_are_reported(monkeypatch):
    monkeypatch.setattr("quantum_market_lab.data.news.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(NewsProviderError, match="timed out"):
        FinnhubNewsProvider("test-secret").historical_news(NewsQuery("AAPL", "2024-01-01", "2024-02-01"))
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return b'{"error":"invalid token"}'
    monkeypatch.setattr("quantum_market_lab.data.news.urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(NewsProviderError, match="unexpected historical-news response"):
        FinnhubNewsProvider("test-secret").historical_news(NewsQuery("AAPL", "2024-01-01", "2024-02-01"))


def test_finnhub_empty_and_fully_filtered_payloads_are_distinguishable(monkeypatch, caplog):
    caplog.set_level("INFO", logger="quantum_market_lab.data.news")
    class Response:
        status = 200
        def __init__(self, payload): self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps(self.payload).encode()
    query = NewsQuery("AAPL", "2024-01-01", "2024-02-01")
    provider = FinnhubNewsProvider("test-secret")
    monkeypatch.setattr("quantum_market_lab.data.news.urlopen", lambda *args, **kwargs: Response([]))
    assert provider.historical_news(query) == []
    assert provider.last_diagnostics["raw_article_count"] == 0
    assert provider.last_diagnostics["rejected_item_count"] == 0
    monkeypatch.setattr("quantum_market_lab.data.news.urlopen", lambda *args, **kwargs: Response([{"datetime": 1}, {"headline": "missing url"}]))
    assert provider.historical_news(query) == []
    assert provider.last_diagnostics["raw_article_count"] == 2
    assert provider.last_diagnostics["rejected_item_count"] == 2
    assert "raw_articles=0" in caplog.text and "rejected_items=2" in caplog.text
    assert "test-secret" not in caplog.text


def test_unexpected_provider_exception_is_wrapped_without_logging_exception_text(monkeypatch, caplog):
    def fail(*args, **kwargs): raise RuntimeError("request url contains test-secret")
    monkeypatch.setattr("quantum_market_lab.data.news.urlopen", fail)
    with pytest.raises(NewsProviderError, match="failed unexpectedly"):
        FinnhubNewsProvider("test-secret").historical_news(NewsQuery("AAPL", "2024-01-01", "2024-02-01"))
    assert "RuntimeError" in caplog.text
    assert "test-secret" not in caplog.text and "request url contains" not in caplog.text


def test_news_ranking_exposes_time_relevance_category_and_corroboration_signals():
    event = {"event_id":"e1", "timestamp":"2024-04-10T00:00:00", "category":"large_absolute_return", "timestamp_resolution":"session_date"}
    articles = [
        {"headline":"AAPL earnings results beat estimates", "excerpt":"Apple reports revenue", "publisher":"One", "publication_timestamp":"2024-04-10T12:00:00+00:00", "relevant_instrument":"AAPL"},
        {"headline":"Apple shares move", "excerpt":"AAPL activity", "publisher":"Two", "publication_timestamp":"2024-04-10T13:00:00+00:00", "relevant_instrument":"AAPL"},
        {"headline":"AAPL earnings", "excerpt":"", "publisher":"Three", "publication_timestamp":"2024-04-20T12:00:00+00:00", "relevant_instrument":"AAPL"},
    ]
    ranked = rank_news_for_events(articles, [event], "AAPL")[0]["ranked_articles"]
    assert ranked[0]["article"]["publisher"] == "One"
    assert ranked[0]["signals"]["corroborating_publisher_count"] >= 1
    assert ranked[0]["signals"]["publication_phase"].startswith("same_session_date")
    assert "causation" in ranked[0]["signals"]["explanation"]
    intraday = rank_news_for_events(articles, [{**event, "timestamp_resolution":"intraday_observation_utc"}], "AAPL")[0]["ranked_articles"]
    assert "exchange session phase unavailable" in intraday[0]["signals"]["publication_phase"]


def test_event_analysis_cache_keys_full_request_and_expires(tmp_path):
    from quantum_market_lab.data.events import EventAnalysisCache
    cache = EventAnalysisCache(tmp_path / "events.sqlite3", ttl_seconds=3600)
    request = {"ticker":"AAPL", "start":"2024-01-01", "threshold":.04}
    result = {"events":[{"id":"actual-event"}]}
    assert cache.get(request) is None
    cache.put(request, result)
    assert cache.get(request) == result
    assert cache.get({**request, "threshold":.05}) is None


def test_model_comparison_uses_project_calibrators_and_returns_rarity_diagnostics(monkeypatch):
    from types import SimpleNamespace
    from quantum_market_lab.data.events import _model_comparison
    frame = synthetic_ohlcv(20)
    monkeypatch.setattr("quantum_market_lab.data.events.estimate_gbm", lambda *args, **kwargs: SimpleNamespace(as_dict=lambda: {
        "method":"existing_gbm_moments", "mean_period_log_return":.001, "sample_std_period_log_return":.01}))
    monkeypatch.setattr("quantum_market_lab.data.events.calibrate_merton", lambda *args, **kwargs: {
        "method":"existing_merton_mle", "drift":.08, "volatility":.2, "jump_intensity":.2, "mean_log_jump":-.1,
        "jump_volatility":.2, "warnings":[]})
    monkeypatch.setattr("quantum_market_lab.data.events.calibrate_garch", lambda *args, **kwargs: {
        "method":"existing_garch_mle", "conditional_volatility":[{"value":.01} for _ in range(19)],
        "mean":0., "omega":1e-6, "alpha":.05, "beta":.9, "warnings":[], "limitations":"in sample"})
    result = _model_comparison(frame, 10, "1d", "AAPL", "2023-01-01", "2024-01-01", {})
    assert result["gbm"]["method"] == "existing_gbm_moments"
    assert result["merton"]["method"] == "existing_merton_mle"
    assert result["garch"]["method"] == "existing_garch_mle"
    assert result["garch"]["gaussian_two_sided_tail_probability"] >= 0


def test_empty_news_results_are_cached_without_inventing_articles(tmp_path):
    class EmptyProvider:
        name = "mock empty provider"
        calls = 0
        def historical_news(self, query):
            self.calls += 1
            return []
    provider = EmptyProvider()
    query = NewsQuery("AAPL", "2024-01-01", "2024-02-01")
    cache = NewsResponseCache(tmp_path / "empty.sqlite3")
    articles, _, cached = fetch_historical_news(query, provider, cache)
    assert articles == [] and cached is False
    articles, _, cached = fetch_historical_news(query, provider, cache)
    assert articles == [] and cached is True and provider.calls == 1


def test_intraday_opening_gap_detection_is_disabled_without_a_session_calendar(monkeypatch):
    no_model_calibrations(monkeypatch)
    frame = synthetic_ohlcv(20)
    frame.index = pd.date_range("2024-01-02 14:30", periods=len(frame), freq="15min", tz="UTC", name="date")
    frame.iloc[10, frame.columns.get_loc("open")] = frame.iloc[9]["close"] * 1.1
    result = analyze_historical_events(frame, ticker="AAPL", start="2024-01-02", end="2024-01-03", interval="15m")
    assert not any(item["category"] == "opening_gap" for item in result["candidate_events"])
    assert any("Opening-gap detection is disabled" in warning for warning in result["warnings"])
