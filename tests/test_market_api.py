import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from uuid import uuid4
from fastapi.testclient import TestClient

from quantum_market_lab.api import app as api_module


def test_market_endpoints_offline(monkeypatch, tmp_path):
    frame = pd.DataFrame({"open": [10., 11., 12.], "high": [11., 12., 13.], "low": [9., 10., 9.],
                          "close": [10., 11., 10.5], "adj_close": [10., 11., 10.5], "volume": [0, 10, 20]},
                         index=pd.date_range("2024-01-01", periods=3, name="date"))
    class Loader:
        def __init__(self, ticker, start=None, end=None, interval="1d"):
            self.ticker = ticker.upper(); self.interval = interval
        def download(self):
            if self.interval == "15m":
                intraday = frame.copy()
                intraday.index = pd.date_range("2024-01-01 14:30", periods=len(intraday), freq="15min")
                return intraday
            return frame
        def metadata(self, data): return {"ticker": self.ticker}
    monkeypatch.setattr(api_module, "MarketDataLoader", Loader)
    monkeypatch.setattr(api_module, "MarketDataStore", lambda: __import__("quantum_market_lab.data.storage", fromlist=["MarketDataStore"]).MarketDataStore(tmp_path))
    client = TestClient(api_module.app)
    assert client.get("/market-data/validate/AAPL").json()["valid"]
    summary = client.get("/market-data/summary/AAPL").json()
    assert summary["row_count"] == 3 and summary["maximum_drawdown"] < 0
    assert summary["historical_period_risk"]["measure"] == "historical_one_period"
    assert summary["historical_period_risk"]["confidence_level"] == .95
    assert summary["historical_period_risk"]["sample_size"] == 2
    assert summary["historical_period_risk"]["unit"] == "simple_return_fraction"
    period_log_returns = np.diff(np.log(np.array([10., 11., 10.5])))
    assert summary["annualized_volatility"] == pytest.approx(np.std(period_log_returns, ddof=1) * np.sqrt(252))
    assert summary["drawdown_price_field"] == "adj_close"
    response = client.post("/market-data/download", json={"ticker": "AAPL"})
    assert response.status_code == 200
    assert client.post("/market-data/download", json={"ticker": "../x"}).status_code == 422
    chart = client.get("/market-data/prices/AAPL?max_points=2").json()
    assert chart["observation_count"] == 2 and chart["downsampled"]
    assert chart["observations"][0]["price"] == 10.0
    candles = client.get("/market-data/ohlcv/AAPL?max_points=2").json()
    assert candles["ticker"] == "AAPL" and candles["downsampled"]
    assert candles["observation_count"] == 2 and candles["source_observation_count"] == 3
    assert candles["observations"][0] == {
        "date": "2024-01-02", "open": 10.0, "high": 12.0,
        "low": 9.0, "close": 11.0, "volume": 10.0,
    }
    assert candles["observations"][1]["high"] == 13.0
    assert candles["observations"][1]["low"] == 9.0
    assert candles["observations"][1]["close"] == 10.5
    assert candles["observations"][1]["volume"] == 20.0
    intraday = client.get("/market-data/ohlcv/AAPL?interval=15m").json()
    assert intraday["timestamp_convention"] == "UTC"
    assert intraday["observations"][1]["date"] == "2024-01-01T14:45:00"


def test_vite_origins_are_explicitly_allowed():
    client = TestClient(api_module.app)
    response = client.options("/health", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_market_event_api_uses_shared_market_loader_and_reports_missing_news_provider(monkeypatch):
    loader_calls=[]
    index = pd.date_range("2024-01-01", periods=4, name="date")
    frame = pd.DataFrame({"open":[10.,10.,11.,11.],"high":[11.,11.,12.,12.],"low":[9.,9.,10.,10.],"close":[10.,10.5,11.,11.5],"volume":[100.,110.,120.,130.]},index=index)
    class Loader:
        def __init__(self,ticker,start=None,end=None,interval="1d"):
            self.ticker=ticker.upper(); self.interval=interval
            loader_calls.append((self.ticker,start,end,interval))
        def download(self): return frame
    monkeypatch.setattr(api_module,"MarketDataLoader",Loader)
    monkeypatch.setenv("MARKET_EVENT_ANALYSIS_CACHE",str(Path("/tmp") / f"events-{uuid4()}.sqlite3"))
    monkeypatch.setattr(api_module,"analyze_historical_events",lambda frame,**kwargs:{"ticker":kwargs["ticker"],"bars":[],"candidate_events":[],"warnings":[],"observation_count":4,"source_observation_count":4,"detection_configuration":kwargs["thresholds"].__dict__})
    monkeypatch.delenv("FINNHUB_API_KEY",raising=False)
    client=TestClient(api_module.app)
    analysis=client.post("/market-events/analyze",json={"ticker":"AAPL","start":"2024-01-01","end":"2024-02-01","interval":"1d","benchmark":"SPY"})
    assert analysis.status_code==200 and analysis.json()["ticker"]=="AAPL"
    assert analysis.json()["detection_configuration"]["absolute_log_return"]==.04
    assert loader_calls[:2]==[("AAPL","2024-01-01","2024-02-01","1d"),("SPY","2024-01-01","2024-02-01","1d")]
    assert analysis.json()["candidate_events"]==[]  # Successful analysis with no detected candidates.
    provider=client.get("/market-events/news/provider").json()
    assert provider["configured"] is False and "FINNHUB_API_KEY" in provider["setup_message"]
    news=client.post("/market-events/news",json={"ticker":"AAPL","start":"2024-01-01","end":"2024-02-01","events":[]}).json()
    assert news["searched"] is False and news["articles"]==[] and "No news search" in news["message"]
    assert client.post("/market-events/analyze",json={"ticker":"AAPL","start":"2024-02-01","end":"2024-01-01"}).status_code==422


def test_market_event_analysis_market_data_failure_is_logged_and_returned(monkeypatch, caplog, tmp_path):
    class FailingLoader:
        def __init__(self, *args, **kwargs): raise api_module.MarketDataProviderError("synthetic test fixture: provider unavailable")
    monkeypatch.setattr(api_module,"MarketDataLoader",FailingLoader)
    monkeypatch.setenv("MARKET_EVENT_ANALYSIS_CACHE",str(tmp_path / "failed-analysis.sqlite3"))
    response=TestClient(api_module.app).post("/market-events/analyze",json={
        "ticker":"AAPL","start":"2025-10-20","end":"2026-10-24","interval":"1d","benchmark":"SPY"})
    assert response.status_code==502
    assert "Yahoo Finance could not retrieve historical data for AAPL" in response.json()["detail"]
    assert "synthetic test fixture" not in response.json()["detail"]
    assert "Historical market-data provider failed" in caplog.text
    assert "AAPL" in caplog.text and "2025-10-20" in caplog.text


def test_market_event_news_provider_failure_is_an_http_error(monkeypatch):
    class FailingProvider:
        name = "Finnhub company news"
        api_key = "configured"
        def historical_news(self, query): raise api_module.NewsProviderError("provider unavailable")
    monkeypatch.setattr(api_module,"FinnhubNewsProvider",lambda:FailingProvider())
    monkeypatch.setenv("MARKET_EVENT_NEWS_CACHE",str(Path("/tmp") / f"news-{uuid4()}.sqlite3"))
    response=TestClient(api_module.app).post("/market-events/news",json={"ticker":"AAPL","start":"2024-01-01","end":"2024-02-01","events":[]})
    assert response.status_code==502 and response.json()["detail"]=="provider unavailable"


def test_market_event_news_returns_provider_articles_and_explicit_rate_limit(monkeypatch, tmp_path):
    requested = {}
    class Provider:
        name = "Finnhub company news"
        api_key = "configured"
    def fetch(query, provider, cache):
        requested.update(ticker=query.ticker, start=query.start, end=query.end)
        return ([{"headline":"Test fixture: AAPL report", "publisher":"Test Fixture Publisher",
                  "original_url":"https://fixture.invalid/aapl", "publication_timestamp":"2024-01-02T12:00:00+00:00",
                  "publication_timezone":"UTC", "retrieved_timestamp":"2024-01-03T00:00:00+00:00",
                  "relevant_instrument":"AAPL", "provider_query_ticker":"AAPL", "excerpt":None,
                  "category":"company", "source_provider":"fixture"}], provider.name, False)
    monkeypatch.setattr(api_module,"FinnhubNewsProvider",lambda:Provider())
    monkeypatch.setattr(api_module,"fetch_historical_news",fetch)
    monkeypatch.setenv("MARKET_EVENT_NEWS_CACHE",str(tmp_path / "news.sqlite3"))
    payload={"ticker":"AAPL","start":"2024-01-01","end":"2024-02-01","events":[
        {"event_id":"evt1","timestamp":"2024-01-02","category":"large_absolute_return"}]}
    response=TestClient(api_module.app).post("/market-events/news",json=payload)
    assert response.status_code==200
    data=response.json()
    assert requested=={"ticker":"AAPL","start":"2024-01-01","end":"2024-02-01"}
    assert data["searched"] is True and data["articles"][0]["headline"].startswith("Test fixture:")
    assert data["event_matches"][0]["ranked_articles"][0]["article"]["original_url"]=="https://fixture.invalid/aapl"

    class RateLimitedProvider(Provider):
        def historical_news(self, query): raise api_module.NewsProviderError("Finnhub rate limit reached",http_status=429)
    monkeypatch.setattr(api_module,"FinnhubNewsProvider",lambda:RateLimitedProvider())
    monkeypatch.setattr(api_module,"fetch_historical_news",lambda query,provider,cache:provider.historical_news(query))
    limited=TestClient(api_module.app).post("/market-events/news",json={**payload,"end":"2024-03-01"})
    assert limited.status_code==429 and "rate limit" in limited.json()["detail"]


def test_market_event_news_reports_provider_empty_list_diagnostics(monkeypatch, tmp_path):
    class EmptyProvider:
        name="Finnhub company news"
        api_key="configured"
        last_diagnostics={"http_status":200,"raw_article_count":0,"accepted_article_count":0,"rejected_item_count":0}
    monkeypatch.setattr(api_module,"FinnhubNewsProvider",lambda:EmptyProvider())
    monkeypatch.setattr(api_module,"fetch_historical_news",lambda query,provider,cache:([],provider.name,False))
    monkeypatch.setenv("MARKET_EVENT_NEWS_CACHE",str(tmp_path / "empty.sqlite3"))
    response=TestClient(api_module.app).post("/market-events/news",json={"ticker":"TEST","start":"2024-01-01","end":"2024-02-01","events":[]})
    assert response.status_code==200
    data=response.json()
    assert data["searched"] is True and data["articles"]==[]
    assert "empty company-news list (HTTP 200)" in data["message"]
    assert "TEST" in data["message"]
