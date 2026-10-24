import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity, BarChart3, BookOpen, Boxes, ChartNoAxesCombined, ChevronRight,
  CircleHelp, FlaskConical, Gauge, Layers3, LineChart, Menu, Microscope,
  Network, Radio, RefreshCw, ShieldAlert, Waves, Trash2, GitCompareArrows, Play, Orbit, Sigma,
} from 'lucide-react'
import { API_BASE, api, ApiError } from './api'
import { Button, EmptyState, ErrorNotice, Field, PageHeading, Panel, PanelTitle, SelectField, Stat, SuccessNotice } from './components'
import type { DownloadMetadata, ExperimentComparison, ExperimentRecord, MarketSummary, ModelDescription, OHLCVSeries, Page, ValidationReport } from './types'
import { ModelLaboratory } from './Cockpit'
import { CandlestickChart } from './CandlestickChart'
const SimulationUniverse = lazy(() => import('./SimulationUniverse'))
const FormulaLab = lazy(() => import('./FormulaLab'))
const MarketEventIntelligence = lazy(() => import('./MarketEventIntelligence').then((module) => ({ default: module.MarketEventIntelligence })))
const ModelTournament = lazy(() => import('./ModelTournament').then((module) => ({ default: module.ModelTournament })))

const NAV: { id: Page; label: string; icon: typeof Activity }[] = [
  { id: 'overview', label: 'Overview', icon: Gauge },
  { id: 'market', label: 'Market data', icon: LineChart },
  { id: 'models', label: 'Model laboratory', icon: FlaskConical },
  { id: 'formula', label: 'Quant Formula Lab', icon: Sigma },
  { id: 'universe', label: 'Simulation Universe', icon: Orbit },
  { id: 'risk', label: 'Risk analytics', icon: ShieldAlert },
  { id: 'experiments', label: 'Experiments', icon: Boxes },
  { id: 'events', label: 'Market Event Intelligence', icon: Activity },
  { id: 'tournament', label: 'Model Tournament', icon: BarChart3 },
]

function formatNumber(value: number | null | undefined, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—'
}
function formatPercent(value: number | null | undefined, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—'
}
function today(): string { return new Date().toISOString().slice(0, 10) }
function yearAgo(): string { const date = new Date(); date.setFullYear(date.getFullYear() - 1); return date.toISOString().slice(0, 10) }

export default function App() {
  const [page, setPage] = useState<Page>('models')
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [health, setHealth] = useState<'checking' | 'connected' | 'offline'>('checking')
  const [models, setModels] = useState<ModelDescription[]>([])
  const [modelsError, setModelsError] = useState('')
  const [marketSummary, setMarketSummary] = useState<MarketSummary | null>(null)
  const [selectedTicker, setSelectedTicker] = useState('AAPL')
  const [experimentToLoad, setExperimentToLoad] = useState<ExperimentRecord | null>(null)
  const selectTicker = useCallback((ticker: string) => { setSelectedTicker(ticker); setMarketSummary(null) }, [])

  useEffect(() => {
    let active = true
    let controller: AbortController | undefined
    const check = async () => {
      controller?.abort()
      controller = new AbortController()
      try { await api.health(controller.signal); if (active) setHealth('connected') }
      catch { if (active && !controller.signal.aborted) setHealth('offline') }
    }
    void check()
    const interval = window.setInterval(check, 15000)
    return () => { active = false; window.clearInterval(interval); controller?.abort() }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    api.models(controller.signal).then(({ models: found }) => {
      if (!Array.isArray(found) || !found.every((model) => typeof model.name === 'string')) throw new Error('The backend returned an unexpected model registry format.')
      setModels(found); setModelsError('')
    }).catch((error: unknown) => {
      if (error instanceof DOMException && error.name === 'AbortError') return
      setModelsError(error instanceof Error ? error.message : 'Could not load available models.')
    })
    return () => controller.abort()
  }, [])

  const navigate = useCallback((next: Page) => { setPage(next); setMobileNavOpen(false) }, [])
  const activeNav = NAV.find((item) => item.id === page)!
  return <div className="app-shell min-h-screen">
    <aside className={`sidebar ${mobileNavOpen ? 'sidebar-open' : ''}`}>
      <a className="brand" href="#overview" onClick={(event) => { event.preventDefault(); navigate('overview') }}>
        <span className="brand-mark"><Microscope size={19} /></span>
        <span><strong>QUANT<span> / </span>OBSERVATORY</strong><small>RESEARCH TERMINAL</small></span>
      </a>
      <div className="workspace-label">WORKSPACE</div>
      <nav aria-label="Primary navigation" className="nav-list">
        {NAV.map(({ id, label, icon: Icon }) => <button key={id} className={`nav-item ${page === id ? 'nav-active' : ''}`} onClick={() => navigate(id)}>
          <Icon size={17} strokeWidth={1.8} /><span>{label}</span>{page === id && <ChevronRight size={14} className="nav-chevron" />}
        </button>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="connection-card">
          <div className="connection-row"><span className={`status-dot status-${health}`} /><span>{health === 'connected' ? 'Backend connected' : health === 'offline' ? 'Backend unavailable' : 'Checking backend'}</span></div>
          <code>{API_BASE}</code>
        </div>
        <div className="sidebar-foot"><span>QUANTITATIVE RESEARCH</span><span>v0.1.0</span></div>
      </div>
    </aside>
    <main className="main-area">
      <header className="topbar">
        <button className="mobile-menu" aria-label="Open navigation" onClick={() => setMobileNavOpen((open) => !open)}><Menu size={19} /></button>
        <div className="breadcrumb"><span>Observatory</span><ChevronRight size={13} /><strong>{activeNav.label}</strong></div>
        <div className="topbar-right"><span className="topbar-caption">STOCHASTIC RESEARCH ENVIRONMENT</span><div className={`health-pill health-${health}`}><span className="status-dot" />{health === 'connected' ? 'API LIVE' : health === 'offline' ? 'API OFFLINE' : 'CONNECTING'}</div></div>
      </header>
      <div className="content-area" key={page}>
        {page === 'overview' && <Overview models={models} modelsError={modelsError} onNavigate={navigate} health={health} />}
        {page === 'market' && <MarketExplorer ticker={selectedTicker} onTickerChange={selectTicker} onSummary={setMarketSummary} />}
        {page === 'models' && <ModelLaboratory models={models} modelsError={modelsError} ticker={selectedTicker} onTickerChange={selectTicker} experimentToLoad={experimentToLoad} onExperimentLoaded={() => setExperimentToLoad(null)} />}
        {page === 'formula' && <Suspense fallback={<Panel><div className="inline-loading"><RefreshCw size={14} className="spin" /> Loading Quant Formula Lab…</div></Panel>}><FormulaLab /></Suspense>}
        {page === 'universe' && <Suspense fallback={<Panel><div className="inline-loading"><RefreshCw size={14} className="spin" /> Loading Simulation Universe…</div></Panel>}><SimulationUniverse ticker={selectedTicker} onTickerChange={selectTicker} /></Suspense>}
        {page === 'risk' && <RiskAnalytics summary={marketSummary} onNavigate={navigate} />}
        {page === 'experiments' && <Experiments onReopen={(record) => { setExperimentToLoad(record); navigate('models') }} />}
        {page === 'events' && <Suspense fallback={<Panel><div className="inline-loading"><RefreshCw size={14} className="spin" /> Loading Market Event Intelligence…</div></Panel>}><MarketEventIntelligence ticker={selectedTicker} onTickerChange={selectTicker} /></Suspense>}
        {page === 'tournament' && <Suspense fallback={<Panel><div className="inline-loading"><RefreshCw size={14} className="spin" /> Loading Model Tournament…</div></Panel>}><ModelTournament /></Suspense>}
      </div>
      <footer className="app-footer"><span>Quantitative Observatory</span><span>Historical analysis and stochastic simulation · not investment advice</span></footer>
    </main>
    {mobileNavOpen && <button aria-label="Close navigation" className="mobile-backdrop" onClick={() => setMobileNavOpen(false)} />}
  </div>
}

function Overview({ models, modelsError, onNavigate, health }: { models: ModelDescription[]; modelsError: string; onNavigate: (page: Page) => void; health: string }) {
  return <>
    <PageHeading eyebrow="RESEARCH OVERVIEW / 01" title="Quantitative Observatory" description="A research workspace for historical market data and stochastic process experiments." aside={<div className={`connection-large ${health === 'connected' ? 'connected' : ''}`}><span className="status-dot" />{health === 'connected' ? 'Backend available' : health === 'offline' ? 'Backend unavailable' : 'Checking backend'}</div>} />
    <div className="overview-intro">
      <div className="intro-copy"><span className="eyebrow cyan">RESEARCH WORKFLOW</span><h2>From observed data<br />to modeled uncertainty.</h2><p>Inspect historical series, validate source data, then explore stochastic assumptions with reproducible simulations.</p><Button onClick={() => onNavigate('market')}>Open market explorer <ChevronRight size={16} /></Button></div>
      <div className="workflow-visual" aria-label="Research workflow: observe, model, analyze">
        {[{ n: '01', title: 'Observe', desc: 'Historical market data', icon: Activity }, { n: '02', title: 'Model', desc: 'Stochastic processes', icon: Network }, { n: '03', title: 'Analyze', desc: 'Risk and distributions', icon: ChartNoAxesCombined }].map(({ n, title, desc, icon: Icon }, index) => <div className="workflow-step" key={n}><div className="workflow-node"><Icon size={18} /></div><div className="workflow-meta"><span>{n} / {title}</span><small>{desc}</small></div>{index < 2 && <div className="workflow-link" />}</div>)}
      </div>
    </div>
    <div className="section-label"><span>RESEARCH MODULES</span><span className="section-rule" /></div>
    <div className="module-grid">
      <ModuleCard index="01" icon={<LineChart size={19} />} title="Market data explorer" description="Validate and summarize historical OHLCV data with a bounded price chart." action="Explore market data" accent="green" onClick={() => onNavigate('market')} />
      <ModuleCard index="02" icon={<Waves size={19} />} title="Stochastic model lab" description="Run GBM, OU, jump-diffusion, and GARCH models from their actual API contracts." action="Open model lab" accent="violet" onClick={() => onNavigate('models')} />
      <ModuleCard index="03" icon={<ShieldAlert size={19} />} title="Risk analytics" description="Review realized volatility and drawdown from a market summary you request." action="View risk analytics" accent="red" onClick={() => onNavigate('risk')} />
      <ModuleCard index="04" icon={<Orbit size={19} />} title="Simulation Universe" description="Explore evolving model density surfaces with synchronized, seeded path playback." action="Enter simulation universe" accent="cyan" onClick={() => onNavigate('universe')} />
    </div>
    <div className="lower-grid">
      <Panel><PanelTitle title="Available model registry" subtitle="Loaded from GET /models" action={<span className="source-tag"><Radio size={12} /> LIVE SOURCE</span>} />
        {modelsError ? <ErrorNotice>{modelsError}</ErrorNotice> : models.length ? <div className="model-registry">{models.map((model) => <div className="registry-row" key={model.name}><span className="registry-icon"><Activity size={15} /></span><div><strong>{model.name.replaceAll('_', ' ')}</strong><small>{model.description}</small></div><code>{model.name === 'garch' ? 'RETURN PATH' : 'PRICE / STATE PATH'}</code></div>)}</div> : <div className="inline-loading"><RefreshCw size={14} className="spin" /> Loading model registry…</div>}
      </Panel>
      <Panel className="empty-panel"><PanelTitle title="Experiment history" subtitle="Persistent research records" /><EmptyState icon={<BookOpen size={18} />} title="Review saved runs">Experiment configurations, provenance, and compact results are available in the Experiments workspace.</EmptyState><Button variant="quiet" onClick={() => onNavigate('experiments')}>Open experiments <ChevronRight size={15} /></Button></Panel>
    </div>
  </>
}

function ModuleCard({ index, icon, title, description, action, accent, onClick }: { index: string; icon: React.ReactNode; title: string; description: string; action: string; accent: string; onClick: () => void }) {
  return <button className={`module-card module-${accent}`} onClick={onClick}><div className="module-top"><span className="module-icon">{icon}</span><span className="module-index">{index} / MODULE</span></div><h3>{title}</h3><p>{description}</p><span className="module-link">{action}<ChevronRight size={15} /></span></button>
}

type MarketForm = { ticker: string; start: string; end: string; interval: string }
const initialMarket: MarketForm = { ticker: 'AAPL', start: yearAgo(), end: today(), interval: '1d' }

function MarketExplorer({ ticker, onTickerChange, onSummary }: { ticker: string; onTickerChange: (ticker: string) => void; onSummary: (summary: MarketSummary | null) => void }) {
  const [form, setForm] = useState<MarketForm>({ ...initialMarket, ticker })
  const [validation, setValidation] = useState<ValidationReport | null>(null)
  const [summary, setSummary] = useState<MarketSummary | null>(null)
  const [series, setSeries] = useState<OHLCVSeries | null>(null)
  const [download, setDownload] = useState<DownloadMetadata | null>(null)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState('')
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  useEffect(() => { setForm((old) => old.ticker === ticker ? old : { ...old, ticker }); setSeries(null); setSummary(null); onSummary(null) }, [ticker, onSummary])
  const begin = (key: string) => { controller.current?.abort(); controller.current = new AbortController(); setBusy(key); setErrors((old) => ({ ...old, [key]: '' })); return controller.current.signal }
  const finish = () => setBusy('')
  const params = useMemo(() => ({ start: form.start, end: form.end, interval: form.interval }), [form])
  const validForm = (event?: { preventDefault: () => void }) => {
    event?.preventDefault()
    if (!form.ticker.trim() || (form.start && form.end && form.start >= form.end)) { setErrors((old) => ({ ...old, form: 'Enter a ticker and ensure the start date is earlier than the end date.' })); return false }
    setErrors((old) => ({ ...old, form: '' })); return true
  }
  const validate = async (event?: { preventDefault: () => void }) => {
    if (!validForm(event)) return
    const signal = begin('validate')
    try { setValidation(await api.validate(form.ticker.trim(), params, signal)) }
    catch (error) { if (!(error instanceof DOMException && error.name === 'AbortError')) setErrors((old) => ({ ...old, validate: errorMessage(error) })) }
    finally { finish() }
  }
  const loadSummary = async () => {
    if (!validForm()) return
    const signal = begin('summary')
    setSummary(null); onSummary(null)
    try { const result = await api.summary(form.ticker.trim(), params, signal); setSummary(result); onSummary(result); setErrors((old) => ({ ...old, summary: '' })) }
    catch (error) { if (!(error instanceof DOMException && error.name === 'AbortError')) { setSummary(null); onSummary(null); setErrors((old) => ({ ...old, summary: errorMessage(error) })) } }
    finally { finish() }
  }
  const loadPrices = async () => {
    if (!form.ticker.trim() || (form.start && form.end && form.start >= form.end)) { setErrors((old) => ({ ...old, chart: 'Check the ticker and date range before loading the chart.' })); return }
    const signal = begin('chart')
    setSeries(null)
    try { setSeries(await api.ohlcv(form.ticker.trim(), { ...params, max_points: 1500 }, signal)); setErrors((old) => ({ ...old, chart: '' })) }
    catch (error) { if (!(error instanceof DOMException && error.name === 'AbortError')) setErrors((old) => ({ ...old, chart: errorMessage(error) })) }
    finally { finish() }
  }
  const persist = async () => {
    if (!validForm()) return
    const signal = begin('download')
    try { const result = await api.download({ ticker: form.ticker.trim(), ...(form.start ? { start: form.start } : {}), ...(form.end ? { end: form.end } : {}), interval: form.interval }, signal); setDownload(result); setErrors((old) => ({ ...old, download: '' })) }
    catch (error) { if (!(error instanceof DOMException && error.name === 'AbortError')) setErrors((old) => ({ ...old, download: errorMessage(error) })) }
    finally { finish() }
  }
  const update = (key: keyof MarketForm, value: string) => setForm((old) => ({ ...old, [key]: value }))
  return <>
    <PageHeading eyebrow="DATA / HISTORICAL SERIES" title="Market data explorer" description="Inspect, validate, and persist historical observations from the configured market-data provider." aside={<span className="source-tag"><Radio size={12} /> YFINANCE VIA API</span>} />
    <Panel className="query-panel"><PanelTitle title="Series query" subtitle="Requests use the backend's Yahoo Finance loader. Provider availability and data quality are not guaranteed." />
      <div className="market-watchlist" aria-label="Instrument watchlist"><span>WATCHLIST / SELECTED INSTRUMENTS</span>{['AAPL', 'MSFT', 'NVDA', 'SPY', '^GSPC'].map((symbol) => <button key={symbol} className={form.ticker === symbol ? 'watch-active' : ''} onClick={() => { onTickerChange(symbol); update('ticker', symbol) }}>{symbol}</button>)}<small>Symbols only; all values load from the backend on request.</small></div>
      <form className="query-form" onSubmit={(event) => void validate(event)}>
        <Field label="Ticker symbol" value={form.ticker} onChange={(e) => { const value = e.target.value.toUpperCase(); update('ticker', value); onTickerChange(value) }} maxLength={32} required placeholder="AAPL" />
        <Field label="Start date" type="date" value={form.start} onChange={(e) => update('start', e.target.value)} />
        <Field label="End date" type="date" value={form.end} onChange={(e) => update('end', e.target.value)} />
        <SelectField label="Interval" value={form.interval} onChange={(e) => update('interval', e.target.value)}><option value="15m">15 minutes</option><option value="1h">Hourly</option><option value="1d">Daily</option><option value="1wk">Weekly</option><option value="1mo">Monthly</option><option value="3mo">Quarterly</option></SelectField>
        <Button type="submit" busy={busy === 'validate'} disabled={Boolean(busy && busy !== 'validate')}><ShieldAlert size={15} /> Validate data</Button>
      </form>
      {errors.form && <ErrorNotice>{errors.form}</ErrorNotice>}
      <div className="query-actions"><Button type="button" variant="secondary" disabled={Boolean(busy)} busy={busy === 'summary'} onClick={() => void loadSummary()}><ChartNoAxesCombined size={15} /> Load summary</Button><Button type="button" variant="secondary" disabled={Boolean(busy)} busy={busy === 'chart'} onClick={loadPrices}><LineChart size={15} /> Load OHLCV chart</Button><Button type="button" variant="quiet" disabled={Boolean(busy)} busy={busy === 'download'} onClick={() => void persist()}><Layers3 size={15} /> Download &amp; persist</Button></div>
      {errors.validate && <ErrorNotice>{errors.validate}</ErrorNotice>}{errors.summary && <ErrorNotice>{errors.summary}</ErrorNotice>}{errors.chart && <ErrorNotice>{errors.chart}</ErrorNotice>}{errors.download && <ErrorNotice>{errors.download}{errors.download.includes('409') ? ' An existing dataset is not overwritten automatically.' : ''}</ErrorNotice>}
      {download && <SuccessNotice>Saved {download.ticker}: {download.row_count.toLocaleString()} observations, {download.start} to {download.end}. Existing datasets require an explicit overwrite option.</SuccessNotice>}
    </Panel>
    <div className="market-results-grid">
      <Panel><PanelTitle title="Data quality" subtitle={validation ? `${validation.row_count.toLocaleString()} rows inspected` : 'Validation report'} action={validation && <span className={`validation-badge ${validation.valid ? 'is-valid' : 'is-invalid'}`}>{validation.valid ? 'VALID' : 'ISSUES FOUND'}</span>} />
        {validation ? <div className="quality-content"><div className="quality-range"><span>Date range</span><strong>{validation.date_range.start?.slice(0, 10) ?? '—'} <span>→</span> {validation.date_range.end?.slice(0, 10) ?? '—'}</strong></div><div className="quality-grid"><QualityItem label="Duplicate dates" value={validation.duplicate_dates} bad={validation.duplicate_dates > 0} /><QualityItem label="Invalid prices" value={validation.invalid_prices} bad={validation.invalid_prices > 0} /><QualityItem label="Invalid OHLC rows" value={validation.invalid_ohlc_rows} bad={validation.invalid_ohlc_rows > 0} /><QualityItem label="Missing columns" value={validation.missing_columns.length} bad={validation.missing_columns.length > 0} /></div>{validation.warnings.length > 0 && <ul className="warning-list">{validation.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>}</div> : <EmptyState icon={<CircleHelp size={18} />} title="No validation report">Choose a symbol and run data validation to inspect source quality.</EmptyState>}
      </Panel>
      <Panel><PanelTitle title="Historical summary" subtitle={summary ? `${summary.ticker} · ${summary.row_count.toLocaleString()} observations` : 'Descriptive statistics'} />
        {summary ? <div className="stats-grid"><Stat label="Latest close" value={`$${formatNumber(summary.latest_close)}`} detail={(summary.latest_timestamp ?? summary.last_date).replace('T', ' ').slice(0, 16)} /><Stat label="Mean log return / period" value={formatPercent(summary.mean_daily_log_return, 3)} detail={`${summary.interval ?? form.interval} · log return`} tone={summary.mean_daily_log_return != null && summary.mean_daily_log_return < 0 ? 'negative' : 'positive'} /><Stat label="Volatility / period" value={formatPercent(summary.daily_volatility)} detail="Sample standard deviation" /><Stat label="Annualized volatility" value={formatPercent(summary.annualized_volatility)} detail={`√${summary.annualization_periods ?? 252} period scaling`} tone="cyan" /><Stat label="Maximum drawdown" value={formatPercent(summary.maximum_drawdown)} detail="Peak-to-trough decline" tone="negative" /><Stat label="Date range" value={`${summary.first_date} — ${summary.last_date}`} detail={`${summary.row_count.toLocaleString()} rows`} /></div> : <EmptyState icon={<BarChart3 size={18} />} title="No summary loaded">Request a summary to view the metrics returned by the backend.</EmptyState>}
      </Panel>
    </div>
    <Panel className="chart-panel"><PanelTitle title="Historical OHLCV / price discovery" subtitle={series ? `${series.ticker} · ${series.observation_count.toLocaleString()} bars${series.downsampled ? ` aggregated from ${series.source_observation_count.toLocaleString()} source observations` : ''}` : 'Provider bars · maximum 1,500 visible candles'} action={series && <span className="chart-legend"><i /> HISTORICAL / API</span>} />
      {series?.observations.length ? <CandlestickChart key={`${series.ticker}-${series.interval}`} bars={series.observations} ticker={series.ticker} interval={series.interval} downsampled={series.downsampled} timestampConvention={series.timestamp_convention} /> : series ? <EmptyState icon={<LineChart size={18} />} title="No historical bars returned">The provider returned no usable OHLC observations for this ticker, interval, and estimation window.</EmptyState> : busy === 'chart' ? <div className="chart-loading"><RefreshCw size={17} className="spin" /> Requesting historical OHLCV…</div> : errors.chart ? <EmptyState icon={<LineChart size={18} />} title="Historical chart unavailable">The provider request failed. Check the symbol, date range, timeframe, or provider availability, then retry.</EmptyState> : <EmptyState icon={<LineChart size={18} />} title="Historical OHLCV not loaded">Load the chart to request actual open, high, low, close, and volume observations from the backend. No prices are generated in the browser.</EmptyState>}
    </Panel>
  </>
}

function QualityItem({ label, value, bad }: { label: string; value: number; bad: boolean }) { return <div className={`quality-item ${bad ? 'quality-bad' : ''}`}><span>{label}</span><strong>{value}</strong></div> }
function RiskAnalytics({ summary, onNavigate }: { summary: MarketSummary | null; onNavigate: (page: Page) => void }) {
  return <>
    <PageHeading eyebrow="ANALYSIS / HISTORICAL RISK" title="Risk analytics" description="Descriptive risk measures from the requested historical market summary. Simulated VaR and Expected Shortfall are calculated separately from model paths in the cockpit." aside={<span className="source-tag"><Activity size={12} /> REAL BACKEND OUTPUT</span>} />
    {summary ? <><div className="risk-source-line"><span className="status-dot status-connected" /> Showing backend summary for <strong>{summary.ticker}</strong> · {summary.first_date} to {summary.last_date} · {summary.row_count.toLocaleString()} observations</div><div className="risk-grid"><Panel><PanelTitle title="Historical return profile" subtitle="Log-return summary · provider data" /><div className="stats-grid risk-stats"><Stat label="Mean log return / period" value={formatPercent(summary.mean_daily_log_return, 3)} detail={`${summary.interval ?? '1d'} · log return`} tone={summary.mean_daily_log_return != null && summary.mean_daily_log_return < 0 ? 'negative' : 'positive'} /><Stat label="Volatility / period" value={formatPercent(summary.daily_volatility)} detail="Sample standard deviation" /><Stat label="Annualized volatility" value={formatPercent(summary.annualized_volatility)} detail={`Period σ × √${summary.annualization_periods ?? 252}`} tone="cyan" /></div></Panel><Panel><PanelTitle title="Drawdown & reference" subtitle={`${summary.drawdown_price_field ?? 'adj_close'} historical price curve`} /><div className="drawdown-block"><span>Historical maximum drawdown</span><strong>{formatPercent(summary.maximum_drawdown)}</strong><p>Peak-to-trough decline from the running historical close peak.</p></div><div className="risk-reference"><span>Latest close</span><strong>${formatNumber(summary.latest_close)}</strong><small>As of {(summary.latest_timestamp ?? summary.last_date).replace('T', ' ').slice(0, 16)}</small></div></Panel></div><Panel className="planned-panel"><PanelTitle title="Historical one-period loss risk" subtitle={summary.historical_period_risk ? `${(summary.historical_period_risk.confidence_level * 100).toFixed(0)}% confidence · ${summary.historical_period_risk.interval} observations` : 'Unavailable for this sample'} />{summary.historical_period_risk ? <><div className="risk-estimate-grid"><RiskMetric label={`Historical VaR ${(summary.historical_period_risk.confidence_level * 100).toFixed(0)}%`} value={formatPercent(summary.historical_period_risk.var)} detail="Quantile of one-period simple losses" /><RiskMetric label={`Historical ES ${(summary.historical_period_risk.confidence_level * 100).toFixed(0)}%`} value={formatPercent(summary.historical_period_risk.expected_shortfall)} detail="Mean empirical worst-tail loss" /><div className="risk-estimate-foot">Loss L = −R, where R is the close-to-close simple return for one {summary.historical_period_risk.interval} interval. Linear empirical VaR; ES includes all losses tied at or above VaR ({summary.historical_period_risk.tail_observation_count} of {summary.historical_period_risk.sample_size} observations). This is not terminal-horizon or portfolio risk.</div>{summary.historical_period_risk.warnings.map((warning) => <div className="risk-estimate-foot" key={warning}>{warning}</div>)}</div></> : <div className="diagnostic-unavailable"><ShieldAlert size={17} /><span>{summary.historical_period_risk_unavailable_reason ?? 'Historical one-period loss risk is unavailable.'}</span></div>}</Panel></> : <Panel className="risk-empty"><EmptyState icon={<ShieldAlert size={21} />} title="No historical summary in this session">Load a market summary first. Risk analytics will display only metrics returned by the connected API.</EmptyState><Button onClick={() => onNavigate('market')}>Open market data explorer <ChevronRight size={16} /></Button></Panel>}
  </>
}

function RiskMetric({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="risk-estimate"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>
}

function Experiments({ onReopen }: { onReopen: (record: ExperimentRecord) => void }) {
  const [items, setItems] = useState<ExperimentRecord[]>([])
  const [selected, setSelected] = useState<ExperimentRecord | null>(null)
  const [checked, setChecked] = useState<string[]>([])
  const [comparison, setComparison] = useState<ExperimentComparison | null>(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const refresh = useCallback(async () => {
    setLoading(true); setError('')
    try { const result = await api.experiments(); setItems(result.experiments) }
    catch (reason) { setError(errorMessage(reason)) }
    finally { setLoading(false) }
  }, [])
  useEffect(() => { void refresh() }, [refresh])
  const toggle = (id: string) => setChecked((old) => old.includes(id) ? old.filter((value) => value !== id) : [...old, id])
  const compare = async () => {
    setBusy(true); setError(''); setComparison(null)
    try { setComparison(await api.compareExperiments(checked)) }
    catch (reason) { setError(errorMessage(reason)) }
    finally { setBusy(false) }
  }
  const remove = async (record: ExperimentRecord) => {
    if (!window.confirm(`Delete experiment “${record.name}”? This cannot be undone.`)) return
    setBusy(true); setError('')
    try { await api.deleteExperiment(record.id); if (selected?.id === record.id) setSelected(null); setChecked((old) => old.filter((id) => id !== record.id)); await refresh() }
    catch (reason) { setError(errorMessage(reason)) }
    finally { setBusy(false) }
  }
  const modelTitle: Record<string, string> = { gbm: 'GBM', ou: 'Ornstein–Uhlenbeck', jump_diffusion: 'Merton jump-diffusion', garch: 'GARCH(1,1)' }
  const metrics = selected?.metrics ?? null
  const risk = metrics?.terminal_risk as Record<string, unknown> | undefined
  const riskAvailable = risk?.available === true
  const summary = metrics?.summary as Record<string, unknown> | undefined
  const horizon = selected?.metadata.horizon as { value?: number; unit?: string; steps?: number } | undefined
  const renderMetric = (label: string, value: unknown, unit = '') => <div className="experiment-metric" key={label}><span>{label}</span><strong>{typeof value === 'number' && Number.isFinite(value) ? `${(value * (unit === '%' ? 100 : 1)).toFixed(4)}${unit}` : 'Unavailable'}</strong></div>
  return <>
    <PageHeading eyebrow="RESEARCH / REPRODUCIBILITY" title="Experiments" description="Browse durable run records, reopen configurations in the model cockpit, and compare compatible simulation results." aside={<span className="source-tag"><Layers3 size={12} /> SQLITE · PERSISTENT</span>} />
    {error && <ErrorNotice>{error}</ErrorNotice>}
    <div className="experiment-workspace">
      <Panel className="experiment-list-panel">
        <PanelTitle title="Saved runs" subtitle={`${items.length} experiment${items.length === 1 ? '' : 's'}`} action={<Button type="button" variant="quiet" onClick={() => void refresh()}><RefreshCw size={13} /> Refresh</Button>} />
        <div className="experiment-compare-actions"><span>Select 2–10 completed runs to compare.</span><Button type="button" variant="secondary" disabled={checked.length < 2 || checked.length > 10 || busy} busy={busy && checked.length > 1} onClick={() => void compare()}><GitCompareArrows size={14} /> Compare selected</Button></div>
        {loading ? <div className="inline-loading">Loading saved experiment records…</div> : items.length === 0 ? <EmptyState icon={<Boxes size={18} />} title="No saved experiments">Configure and run a model in the Model Laboratory, then use “Save &amp; Run Experiment”.</EmptyState> : <div className="experiment-list">{items.map((record) => <article className={`experiment-row ${selected?.id === record.id ? 'experiment-row-active' : ''}`} key={record.id}>
          <label className="experiment-check"><input type="checkbox" checked={checked.includes(record.id)} onChange={() => toggle(record.id)} aria-label={`Select ${record.name} for comparison`} /></label>
          <button type="button" className="experiment-row-main" onClick={() => { setSelected(record); setComparison(null) }}><span className="experiment-row-title">{record.name}</span><small>{modelTitle[record.model]} · {record.created_at.replace('T', ' ').slice(0, 19)} · {record.metadata.ticker ? `${String(record.metadata.ticker)} · ` : ''}{String(record.metadata.path_count ?? '—')} paths · seed {String(record.metadata.random_seed ?? 'none')}</small></button>
          <span className={`experiment-status experiment-status-${record.status}`}>{record.status}</span>
          <Button type="button" variant="quiet" className="experiment-icon-button" title="Reopen configuration" onClick={() => onReopen(record)}><Play size={14} /></Button>
          <Button type="button" variant="quiet" className="experiment-icon-button danger-button" title="Delete experiment" disabled={busy} onClick={() => void remove(record)}><Trash2 size={14} /></Button>
        </article>)}</div>}
      </Panel>
      <div className="experiment-detail-column">
        {comparison && <Panel><PanelTitle title="Run comparison" subtitle={comparison.comparable ? 'Metrics share quantity units and horizon.' : 'Only compatible metrics are compared.'} />{comparison.comparison_reason && <p className="experiment-note">{comparison.comparison_reason}</p>}
          <div className="comparison-table-wrap"><table className="comparison-table"><thead><tr><th>Run metadata</th>{comparison.experiments.map((record) => <th key={record.id}>{record.name}</th>)}</tr></thead><tbody>
            <tr><td>Model</td>{comparison.experiments.map((record) => <td key={record.id}>{modelTitle[record.model]}</td>)}</tr>
            <tr><td>Seed</td>{comparison.experiments.map((record) => <td key={record.id}>{String(record.metadata.random_seed ?? 'unseeded')}</td>)}</tr>
            <tr><td>Parameter source</td>{comparison.experiments.map((record) => <td key={record.id}>{String(record.metadata.parameter_source ?? 'unknown')}</td>)}</tr>
            <tr><td>Calibration</td>{comparison.experiments.map((record) => { const provenance = record.metadata.calibration as Record<string, unknown> | null; return <td key={record.id}>{provenance ? `${String(provenance.ticker)} · ${String(provenance.estimation_start)}–${String(provenance.estimation_end)}` : 'None'}</td> })}</tr>
            <tr><td>Parameters</td>{comparison.experiments.map((record) => <td key={record.id} title={JSON.stringify(record.parameters)}>{Object.entries(record.parameters).filter(([key]) => key !== 'calibration').map(([key, value]) => `${key}=${String(value)}`).join(' · ')}</td>)}</tr>
            {Object.entries(comparison.common_metrics).map(([metric, values]) => <tr key={metric}><td>{metric.replaceAll('_', ' ')}</td>{comparison.experiments.map((record) => <td key={record.id}>{typeof values[record.id] === 'number' ? Number(values[record.id]).toFixed(5) : 'Unavailable'}</td>)}</tr>)}
          </tbody></table></div>
          {Object.keys(comparison.common_metrics).length === 0 && <EmptyState title="No common numeric metrics">The selected records do not have completed comparable results.</EmptyState>}
        </Panel>}
        {selected && <Panel className="experiment-detail"><PanelTitle title={selected.name} subtitle={`${modelTitle[selected.model]} · ${selected.status}`} action={<Button type="button" variant="quiet" onClick={() => onReopen(selected)}><Play size={13} /> Reopen in cockpit</Button>} />
          {selected.description && <p className="experiment-note">{selected.description}</p>}
          <div className="experiment-meta-grid"><span>Created UTC<strong>{selected.created_at}</strong></span><span>Horizon<strong>{horizon?.value ?? '—'} {horizon?.unit ?? ''} · {horizon?.steps ?? '—'} steps</strong></span><span>Paths / seed<strong>{String(selected.metadata.path_count ?? '—')} / {String(selected.metadata.random_seed ?? 'unseeded')}</strong></span><span>Parameter source<strong>{String(selected.metadata.parameter_source ?? 'unknown')}</strong></span><span>Instrument<strong>{String(selected.metadata.ticker ?? '—')}</strong></span><span>Reproducibility<strong>{selected.metadata.reproducible_with_same_seed ? 'Seeded' : 'Unseeded'}</strong></span></div>
          {selected.status === 'failed' && <ErrorNotice>{selected.error ?? 'Simulation failed.'}</ErrorNotice>}
          {selected.status === 'completed' && <><div className="experiment-metrics-grid">{selected.model === 'ou' ? <>{renderMetric('Terminal state mean', summary?.terminal_mean)}{renderMetric('Terminal state variance', summary?.terminal_variance)}</> : <>{renderMetric('Mean terminal return', risk?.mean_total_simple_return, '%')}{renderMetric('Terminal return dispersion', risk?.terminal_simple_return_std, '%')}{renderMetric('VaR 95%', riskAvailable ? risk?.var : null, '%')}{renderMetric('Expected Shortfall 95%', riskAvailable ? risk?.expected_shortfall : null, '%')}{renderMetric('Maximum drawdown P50', (risk?.maximum_drawdown_quantiles as Record<string, number> | undefined)?.['0.5'], '%')}</>}</div>
            {!riskAvailable && selected.model !== 'ou' && <p className="experiment-note">{String(risk?.unavailable_reason ?? 'Some risk metrics were unavailable for this run.')}</p>}
          </>}
          <details className="experiment-json"><summary>Effective parameters and calibration provenance</summary><pre>{JSON.stringify({ parameters: selected.parameters, calibration: selected.metadata.calibration, manually_specified_parameters: selected.metadata.manually_specified_parameters, defaulted_parameters: selected.metadata.defaulted_parameters, derived_parameters: selected.metadata.derived_parameters, observation_frequency: selected.metadata.observation_frequency, annualization_convention: selected.metadata.annualization_convention }, null, 2)}</pre></details>
          <details className="experiment-json"><summary>Saved summary metadata</summary><pre>{JSON.stringify(selected.metrics, null, 2)}</pre></details>
        </Panel>}
      </div>
    </div>
  </>
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message
  if (error instanceof Error) return error.message
  return 'An unexpected error occurred while contacting the backend.'
}
