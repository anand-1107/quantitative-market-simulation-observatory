import { useEffect, useMemo, useRef, useState } from 'react'
import { Activity, Newspaper, Search, ShieldAlert } from 'lucide-react'
import { api } from './api'
import { Button, ErrorNotice, Field, PageHeading, Panel, PanelTitle, SelectField } from './components'
import { CandlestickChart } from './CandlestickChart'
import type { MarketEventAnalysis, MarketEventNews, NewsProviderStatus } from './types'
import { buildNewsCards, isLatestRequest } from './market-news-view'
import { DETECTION_NUMERIC_DEFAULTS, marketEventAnalysisViewState, marketEventErrorMessage } from './market-events-view'

type Props = { ticker: string; onTickerChange: (ticker: string) => void }
type Form = { ticker: string; start: string; end: string; interval: string; benchmark: string; returnThreshold: string; volumeMultiple: string; gapThreshold: string; volatilityMultiple: string; sustainedThreshold: string; sustainedPeriods: string; impactWindow: string }
const today = () => new Date().toISOString().slice(0, 10)
const yearAgo = () => { const value = new Date(); value.setFullYear(value.getFullYear() - 1); return value.toISOString().slice(0, 10) }
const percent = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : 'Unavailable'
const number = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 5 }) : 'Unavailable'
const object = (value: unknown): Record<string, unknown> => value && typeof value === 'object' ? value as Record<string, unknown> : {}
const eventTitle = (value: string) => value.replaceAll('_', ' ').replace(/\b\w/g, (char) => char.toUpperCase())

export function MarketEventIntelligence({ ticker, onTickerChange }: Props) {
  const [form, setForm] = useState<Form>({ ticker, start: yearAgo(), end: today(), interval: '1d', benchmark: 'SPY', returnThreshold: '0.04', volumeMultiple: '2.5', gapThreshold: '0.03', volatilityMultiple: '2', sustainedThreshold: '0.08', sustainedPeriods: '5', impactWindow: '10' })
  const [analysis, setAnalysis] = useState<MarketEventAnalysis | null>(null)
  const [selectedId, setSelectedId] = useState('')
  const [news, setNews] = useState<MarketEventNews | null>(null)
  const [provider, setProvider] = useState<NewsProviderStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [newsBusy, setNewsBusy] = useState(false)
  const [error, setError] = useState('')
  const [newsError, setNewsError] = useState('')
  const newsRequest = useRef<AbortController | null>(null)
  const newsRequestId = useRef(0)
  const analysisRequest = useRef<AbortController | null>(null)
  const analysisRequestId = useRef(0)
  useEffect(() => { api.marketEventNewsProvider().then(setProvider).catch(() => setProvider(null)) }, [])
  useEffect(() => {
    setForm((old) => ({ ...old, ticker }))
    setAnalysis(null); setSelectedId(''); setNews(null); setNewsError('')
    setBusy(false); setError('')
    newsRequest.current?.abort(); newsRequest.current = null; newsRequestId.current += 1
    analysisRequest.current?.abort(); analysisRequest.current = null; analysisRequestId.current += 1
  }, [ticker])
  useEffect(() => () => {
    newsRequest.current?.abort()
    analysisRequest.current?.abort()
  }, [])
  const selected = useMemo(() => analysis?.candidate_events.find((item) => item.event_id === selectedId) ?? analysis?.candidate_events[0] ?? null, [analysis, selectedId])
  const set = (key: keyof Form, value: string) => {
    const next = key === 'ticker' ? value.toUpperCase() : value
    setForm((old) => ({ ...old, [key]: next }))
    if (key === 'ticker' || key === 'start' || key === 'end') {
      setAnalysis(null); setSelectedId(''); setNews(null); setNewsError('')
      setBusy(false); setError('')
      newsRequest.current?.abort(); newsRequest.current = null; newsRequestId.current += 1
      analysisRequest.current?.abort(); analysisRequest.current = null; analysisRequestId.current += 1
    }
    if (key === 'ticker') onTickerChange(next)
  }
  const analyze = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!form.ticker.trim() || form.start >= form.end) { setError('Enter an instrument and a valid start/end date range.'); return }
    analysisRequest.current?.abort()
    const controller = new AbortController(); analysisRequest.current = controller
    const requestId = ++analysisRequestId.current
    newsRequest.current?.abort(); newsRequest.current = null; newsRequestId.current += 1
    setBusy(true); setError(''); setAnalysis(null); setNews(null); setNewsError(''); setSelectedId('')
    try {
      const result = await api.analyzeMarketEvents({ ticker: form.ticker.trim(), start: form.start, end: form.end, interval: form.interval,
        ...(form.benchmark.trim() ? { benchmark: form.benchmark.trim() } : {}), absolute_log_return_threshold: Number(form.returnThreshold),
        volume_multiple_threshold: Number(form.volumeMultiple), opening_gap_threshold: Number(form.gapThreshold),
        volatility_multiple_threshold: Number(form.volatilityMultiple), sustained_log_return_threshold: Number(form.sustainedThreshold),
        sustained_periods: Number(form.sustainedPeriods), impact_window_observations: Number(form.impactWindow) }, controller.signal)
      if (!isLatestRequest(requestId, analysisRequestId.current)) return
      setAnalysis(result); setSelectedId(result.candidate_events[0]?.event_id ?? '')
    } catch (reason) { if (isLatestRequest(requestId, analysisRequestId.current) && !(reason instanceof DOMException && reason.name === 'AbortError')) setError(marketEventErrorMessage(reason, 'Historical event analysis failed.')) }
    finally { if (isLatestRequest(requestId, analysisRequestId.current)) setBusy(false) }
  }
  const retrieveNews = async () => {
    if (!analysis) return
    newsRequest.current?.abort()
    const controller = new AbortController(); newsRequest.current = controller
    const requestId = ++newsRequestId.current
    setNews(null); setNewsBusy(true); setNewsError('')
    try {
      const result = await api.marketEventNews({ ticker: analysis.ticker, start: analysis.estimation_period.start, end: analysis.estimation_period.end_exclusive,
        events: analysis.candidate_events.slice(0, 200).map(({ event_id, timestamp, category, timestamp_resolution }) => ({ event_id, timestamp, category, timestamp_resolution })) }, controller.signal)
      if (isLatestRequest(requestId, newsRequestId.current)) setNews(result)
    } catch (reason) { if (isLatestRequest(requestId, newsRequestId.current) && !(reason instanceof DOMException && reason.name === 'AbortError')) setNewsError(marketEventErrorMessage(reason, 'Historical news request failed.')) }
    finally { if (isLatestRequest(requestId, newsRequestId.current)) setNewsBusy(false) }
  }
  const newsCards = news ? buildNewsCards(news, selected?.event_id, analysis?.candidate_events ?? []) : []
  const analysisViewState = marketEventAnalysisViewState(busy, error, analysis !== null)
  const comparison = object(selected?.model_comparison)
  const impact = object(selected?.impact)
  return <div className="event-intelligence-page">
    <PageHeading eyebrow="ANALYSIS / HISTORICAL EVENTS" title="Market Event Intelligence" description="Investigate unusual observed price and volume movements, inspect timestamped news evidence, and compare the observations with calibrated statistical baselines. Event windows are descriptive, not causal." aside={<span className="source-tag"><Activity size={12} /> HISTORICAL EVIDENCE</span>} />
    <Panel className="event-query-panel"><PanelTitle title="Instrument and date range" subtitle="Historical OHLCV is loaded through the existing Yahoo Finance backend pipeline." />
      <form className="event-query-grid" onSubmit={(event) => void analyze(event)}>
        <Field name="ticker" label="Instrument" value={form.ticker} maxLength={32} required onChange={(event) => set('ticker', event.target.value)} />
        <Field name="start" label="Start date" type="date" value={form.start} required onChange={(event) => set('start', event.target.value)} />
        <Field name="end" label="End date (exclusive)" type="date" value={form.end} required onChange={(event) => set('end', event.target.value)} />
        <SelectField name="interval" label="Observation interval" value={form.interval} onChange={(event) => set('interval', event.target.value)}><option value="15m">15 minutes</option><option value="30m">30 minutes</option><option value="60m">60 minutes</option><option value="1h">1 hour</option><option value="1d">Daily</option><option value="1wk">Weekly</option><option value="1mo">Monthly</option></SelectField>
        <Field name="benchmark" label="Benchmark (optional)" value={form.benchmark} maxLength={32} onChange={(event) => set('benchmark', event.target.value)} placeholder="SPY" />
        <Button type="submit" busy={busy}><Search size={15} /> DETECT EVENTS</Button>
        <details className="event-thresholds"><summary>Detection and impact settings</summary><div>
          <Field name="absolute_log_return_threshold" label={DETECTION_NUMERIC_DEFAULTS[0].label} type="number" min={DETECTION_NUMERIC_DEFAULTS[0].min} max={DETECTION_NUMERIC_DEFAULTS[0].max} step={DETECTION_NUMERIC_DEFAULTS[0].step} value={form.returnThreshold} onChange={(event) => set('returnThreshold', event.target.value)} />
          <Field name="volume_multiple_threshold" label={DETECTION_NUMERIC_DEFAULTS[1].label} type="number" min={DETECTION_NUMERIC_DEFAULTS[1].min} max={DETECTION_NUMERIC_DEFAULTS[1].max} step={DETECTION_NUMERIC_DEFAULTS[1].step} value={form.volumeMultiple} onChange={(event) => set('volumeMultiple', event.target.value)} />
          <Field name="opening_gap_threshold" label={DETECTION_NUMERIC_DEFAULTS[2].label} type="number" min={DETECTION_NUMERIC_DEFAULTS[2].min} max={DETECTION_NUMERIC_DEFAULTS[2].max} step={DETECTION_NUMERIC_DEFAULTS[2].step} value={form.gapThreshold} onChange={(event) => set('gapThreshold', event.target.value)} />
          <Field name="volatility_multiple_threshold" label={DETECTION_NUMERIC_DEFAULTS[3].label} type="number" min={DETECTION_NUMERIC_DEFAULTS[3].min} max={DETECTION_NUMERIC_DEFAULTS[3].max} step={DETECTION_NUMERIC_DEFAULTS[3].step} value={form.volatilityMultiple} onChange={(event) => set('volatilityMultiple', event.target.value)} />
          <Field name="sustained_log_return_threshold" label={DETECTION_NUMERIC_DEFAULTS[4].label} type="number" min={DETECTION_NUMERIC_DEFAULTS[4].min} max={DETECTION_NUMERIC_DEFAULTS[4].max} step={DETECTION_NUMERIC_DEFAULTS[4].step} value={form.sustainedThreshold} onChange={(event) => set('sustainedThreshold', event.target.value)} />
          <Field name="sustained_periods" label="Sustained observations" type="number" min="2" max="60" step="1" value={form.sustainedPeriods} onChange={(event) => set('sustainedPeriods', event.target.value)} />
          <Field name="impact_window_observations" label="Impact window on each side" type="number" min="2" max="60" step="1" value={form.impactWindow} onChange={(event) => set('impactWindow', event.target.value)} />
        </div></details>
        {['15m','30m','60m','1h'].includes(form.interval) && <p className="event-note">For intraday bars, session opening-gap detection is disabled because this app has no exchange-session calendar; close-to-close bar moves, volume, volatility and sustained movement remain available.</p>}
      </form>
    </Panel>
    {analysisViewState === 'loading' && <Panel><div className="event-analysis-status" role="status"><strong>Loading historical observations…</strong><span>Requesting {form.ticker} OHLCV for {form.start} through {form.end} (end date exclusive), interval {form.interval}{form.benchmark.trim() ? `, with ${form.benchmark.trim().toUpperCase()} benchmark data` : ''}.</span></div></Panel>}
    {analysisViewState === 'error' && <Panel><div className="event-analysis-status event-analysis-failure" role="alert"><strong>Historical event analysis failed</strong><span>{error}</span></div></Panel>}
    {analysisViewState === 'idle' && <Panel className="event-empty"><ShieldAlert size={21} /><strong>Load historical observations to begin</strong><span>The detector uses actual provider OHLCV. No synthetic events or news articles are generated.</span></Panel>}
    {analysis && <>
      <Panel><div className="event-analysis-status" role="status"><strong>Historical analysis complete</strong><span>{analysis.observation_count} valid observations analyzed; {analysis.candidate_events.length} candidate event{analysis.candidate_events.length === 1 ? '' : 's'} detected.</span></div></Panel>
      <Panel><PanelTitle title="Historical price and volume" subtitle={`${analysis.ticker} · ${analysis.interval} · ${analysis.observation_count} valid of ${analysis.source_observation_count} source observations${analysis.excluded_observation_count ? ` · ${analysis.excluded_observation_count} excluded` : ''}${analysis.downsampled ? ' · display bars downsampled; calculations use all valid observations' : ''}`} />
        <CandlestickChart key={`${analysis.ticker}-${analysis.interval}-${analysis.estimation_period.start}`} bars={analysis.bars} ticker={analysis.ticker} interval={analysis.interval} downsampled={analysis.downsampled} timestampConvention={analysis.interval === '1d' || analysis.interval === '1wk' || analysis.interval === '1mo' ? 'provider session date' : 'UTC'} />
        <p className="event-note">{analysis.chart_note} {analysis.data_convention}</p>
      </Panel>
      <Panel><PanelTitle title="Candidate market movements" subtitle={`${analysis.candidate_events.length} detected from observed OHLCV using the configured thresholds; each method and numeric evidence is shown.`} />
        {analysis.warnings.map((warning) => <p className="event-warning" key={warning}>{warning}</p>)}
        {analysis.candidate_events.length ? <div className="event-table-wrap"><table className="event-table"><thead><tr><th>Observed time</th><th>Type</th><th>Evidence / trigger</th><th>Close</th><th></th></tr></thead><tbody>{analysis.candidate_events.map((candidate) => <tr key={candidate.event_id} className={candidate.event_id === selected?.event_id ? 'event-selected' : ''} onClick={() => setSelectedId(candidate.event_id)}><td>{candidate.timestamp}</td><td>{eventTitle(candidate.category)}</td><td><span>{candidate.method}</span><code>{Object.entries(candidate.evidence).map(([key, value]) => `${key.replaceAll('_', ' ')}: ${typeof value === 'number' ? number(value) : value}`).join(' · ')}</code></td><td>{number(candidate.close)}</td><td><button type="button" onClick={() => setSelectedId(candidate.event_id)}>Inspect</button></td></tr>)}</tbody></table></div> : <div className="event-empty"><strong>No candidates passed these thresholds</strong><span>Lower a threshold or widen the observation period. This is an absence of detected candidates, not proof that no important event occurred.</span></div>}
      </Panel>
      <div className="event-analysis-grid">
        <Panel><PanelTitle title="Historical event timeline" subtitle="Select a detected date to inspect its impact and model comparison." />
          <div className="event-timeline">{analysis.candidate_events.map((candidate) => <button key={candidate.event_id} type="button" className={candidate.event_id === selected?.event_id ? 'timeline-selected' : ''} onClick={() => setSelectedId(candidate.event_id)}><time>{candidate.date}</time><b>{eventTitle(candidate.category)}</b><small>{candidate.event_log_return == null ? 'movement not defined' : `log return ${percent(candidate.event_log_return)}`}</small></button>)}{!analysis.candidate_events.length && <span>No detected events in this range.</span>}</div>
        </Panel>
        <Panel><PanelTitle title="News evidence" subtitle={provider?.configured ? `${provider.provider} · article publication time is distinct from retrieval time` : 'Provider status and setup instructions'} action={<Newspaper size={16} />} />
          {provider && <p className="event-note">{provider.configured ? provider.historical_coverage : provider.setup_message}</p>}
          {newsError && <ErrorNotice>{newsError}</ErrorNotice>}
          <Button type="button" variant="secondary" disabled={!analysis || newsBusy} busy={newsBusy} onClick={() => void retrieveNews()}><Newspaper size={14} /> SEARCH NEWS</Button>
          {newsBusy && <div className="event-empty event-news-state" role="status"><strong>Searching historical news…</strong><span>Querying {analysis?.ticker} from {analysis?.estimation_period.start} to {analysis?.estimation_period.end_exclusive} (end exclusive).</span></div>}
          {!newsBusy && !news && !newsError && <div className="event-empty event-news-state"><strong>{provider?.configured === false ? 'Provider not configured' : analysis ? 'Search not performed' : 'News search unavailable'}</strong><span>{provider?.configured === false ? 'Live retrieval requires FINNHUB_API_KEY in the backend environment. The key is never sent to the browser.' : analysis ? `Search historical articles for ${analysis.ticker} over the displayed estimation period.` : 'Detect events first to choose the ticker and date range.'}</span></div>}
          {newsError && <div className="event-empty event-news-state" role="alert"><strong>Provider request failed</strong><span>{newsError}</span></div>}
          {news && <p className="event-note">{news.cached ? 'Showing cached provider results; no new external request was made.' : news.searched ? `Historical search completed using ${news.provider}.` : 'No historical news search was made.'}{news.retrieved_at ? ` Response retrieved at ${news.retrieved_at}.` : ''}</p>}
          {news?.message && <div className="event-empty event-news-state"><strong>{news.searched && news.articles.length === 0 ? 'Search completed with no articles' : news.searched ? 'Provider article diagnostic' : news.configured ? 'Search not performed' : 'Provider not configured'}</strong><span>{news.message}</span></div>}
          {news && !news.message && !news.cached && news.searched && news.articles.length === 0 && <div className="event-empty event-news-state"><strong>Search completed with no results</strong><span>{news.provider} returned no articles for {analysis.ticker} from {analysis.estimation_period.start} to {analysis.estimation_period.end_exclusive}. Coverage varies by ticker, date range, and Finnhub account plan.</span></div>}
          {news && !news.message && news.articles.length > 0 && <div className="event-news-list">{newsCards.map(({ article, rank, score, signals, eventLabel, eventTimestamp }) => <article key={article.original_url}>
            {rank !== null && score !== null ? <div className="event-news-rank">RANK {rank} · MATCH {score.toFixed(3)}</div> : <div className="event-news-rank event-news-unranked">RETRIEVED · NOT RANKED</div>}
            {article.original_url ? <a href={article.original_url} target="_blank" rel="noopener noreferrer">{article.headline || 'Untitled article'}</a> : <strong>{article.headline || 'Untitled article'}</strong>}
            <strong>{article.publisher || 'Publisher unavailable'}</strong>
            <div><span>Published: {article.publication_timestamp ? <time dateTime={article.publication_timestamp}>{article.publication_timestamp}{article.publication_timezone ? ` (${article.publication_timezone})` : ''}</time> : 'timestamp unavailable'}</span><span>Retrieved: {article.retrieved_timestamp ?? news.retrieved_at ?? 'unavailable'}</span></div>
            {article.relevant_instrument && <small>Finnhub related-ticker match: {article.relevant_instrument}. This article may discuss other companies too.</small>}
            <div className="event-news-match">Matched market event: {eventLabel ? `${eventTitle(eventLabel)} · ${eventTimestamp ?? 'time unavailable'}` : 'No timestamped event match available'}</div>
            {article.excerpt && <p>{article.excerpt}</p>}
            {signals ? <small>Relevance: proximity {number(signals.temporal_proximity_score)} · instrument {number(signals.instrument_relevance_score)} · category {number(signals.category_relevance_score)} · corroborating publishers {signals.corroborating_publisher_count} · {signals.publication_phase}. {String(signals.explanation ?? '')}</small> : <small>Article returned by the provider; no event-ranking record was returned. No relevance score is inferred.</small>}
          </article>)}</div>}
          {news?.matching_note && <p className="event-note">{news.matching_note}</p>}
        </Panel>
      </div>
      {selected && <>
        <Panel><PanelTitle title="Event impact analysis" subtitle={`${selected.timestamp} · ${impact.available_pre_observations ?? 0} pre-event and ${impact.available_post_observations ?? 0} post-event observations available · target window ${analysis.impact_window_observations_each_side} on each side · descriptive, not causal`} action={<span className="source-tag">{eventTitle(selected.category)}</span>} />
          <div className="event-impact-grid">{[
            ['Pre-event return', percent(impact.pre_return)], ['Post-event return', percent(impact.post_return)],
            ['Pre-event realized volatility', percent(impact.pre_realized_volatility_log_return)], ['Post-event realized volatility', percent(impact.post_realized_volatility_log_return)],
            ['Volume change', percent(impact.volume_change_fraction)], ['Pre-window max drawdown', percent(impact.pre_maximum_drawdown)], ['Post-window max drawdown', percent(impact.post_maximum_drawdown)],
            ['Benchmark relative pre return', percent(impact.benchmark_relative_pre_return)], ['Benchmark relative post return', percent(impact.benchmark_relative_post_return)],
          ].map(([label, value]) => <div key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
          <p className="event-note">{String(impact.return_convention ?? '')}. {String(impact.drawdown_convention ?? '')}{analysis.benchmark_ticker ? ` Benchmark: ${analysis.benchmark_ticker}; relative return is instrument simple return minus aligned benchmark simple return.` : ''}</p>
        </Panel>
        <Panel><PanelTitle title="Calibrated model comparison" subtitle="Full selected period is used for fitting, so an event may be part of its own baseline; in-sample rarity is descriptive and not a pre-event forecast." />
          <div className="event-model-grid">{(['gbm','merton','garch'] as const).map((key) => { const model = object(comparison[key]); return <article key={key}><h3>{key === 'gbm' ? 'GBM' : key === 'merton' ? 'Merton jump-diffusion' : 'GARCH(1,1)'}</h3><span className={`event-model-status ${model.status === 'available' ? 'available' : ''}`}>{String(model.status ?? 'unavailable').toUpperCase()}</span>{model.status === 'available' ? <><p>{String(model.method ?? '')}</p>{key === 'gbm' && <><div>Observed log return: {percent(comparison.observed_log_return)}</div><div>Standardized return: {number(model.standardized_return)}</div><div>Empirical two-sided tail fraction: {percent(model.empirical_two_sided_tail_fraction)}</div></>}{key === 'merton' && <><div>Fitted two-sided mixture tail probability: {percent(model.fitted_two_sided_tail_probability)}</div><pre>{JSON.stringify(model.parameters, null, 2)}</pre></>}{key === 'garch' && <><div>Conditional volatility: {percent(model.conditional_volatility)}</div><div>Standardized residual: {number(model.standardized_residual)}</div><div>Gaussian two-sided tail probability: {percent(model.gaussian_two_sided_tail_probability)}</div></>}{Array.isArray(model.warnings) && model.warnings.map((item) => <small key={String(item)}>{String(item)}</small>)}</> : <p>{String(model.reason ?? 'The selected interval or available sample does not support this calibration.')}</p>}<small>{String(model.limitation ?? comparison.warning ?? '')}</small></article>})}</div>
        </Panel>
      </>}
    </>}
  </div>
}
