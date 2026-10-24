import { useEffect, useMemo, useState } from 'react'
import { Activity, BarChart3, Clock3, Database, Play, RefreshCw } from 'lucide-react'
import { api } from './api'
import { ModelTournamentResearch } from './ModelTournamentResearch'
import { Button, ErrorNotice, Field, PageHeading, Panel, PanelTitle } from './components'
import type { ModelTournamentResult, TournamentModel, TournamentModelResult } from './types'
import { metricNumber, tournamentResultFromSaved, tournamentViewState, validTournamentPeriods } from './model-tournament-view'
import './model-tournament.css'

const MODELS: { key: TournamentModel; label: string; short: string; note: string }[] = [
  { key: 'gbm', label: 'Geometric Brownian Motion', short: 'GBM', note: 'Constant Gaussian log-return increments' },
  { key: 'ou', label: 'Ornstein–Uhlenbeck', short: 'OU', note: 'Mean-reverting log-return state' },
  { key: 'jump_diffusion', label: 'Merton Jump-Diffusion', short: 'Merton', note: 'Gaussian diffusion plus compound-Poisson log jumps' },
  { key: 'garch', label: 'GARCH(1,1)', short: 'GARCH', note: 'Conditional variance recursion on log returns' },
]
const today = () => new Date().toISOString().slice(0, 10)
const year = (n: number) => { const d = new Date(); d.setFullYear(d.getFullYear() - n); return d.toISOString().slice(0, 10) }
const text = (v: unknown) => typeof v === 'number' && Number.isFinite(v) ? v.toLocaleString(undefined, { maximumFractionDigits: 5 }) : 'Unavailable'
const percent = (v: number | null) => v == null ? 'Unavailable' : `${(100 * v).toFixed(2)}%`
const title = (key: string) => MODELS.find((m) => m.key === key)?.short ?? key

export function ModelTournament() {
  const [ticker, setTicker] = useState('AAPL')
  const [trainStart, setTrainStart] = useState(year(7))
  const [trainEnd, setTrainEnd] = useState(year(3))
  const [testStart, setTestStart] = useState(year(2))
  const [testEnd, setTestEnd] = useState(today())
  const [interval, setInterval] = useState('1d')
  const [paths, setPaths] = useState('1000')
  const [seed, setSeed] = useState('42')
  const [selectedModels, setSelectedModels] = useState<TournamentModel[]>(MODELS.map(({ key }) => key))
  const [save, setSave] = useState(true)
  const [name, setName] = useState('Chronological tournament')
  const [result, setResult] = useState<ModelTournamentResult | null>(null)
  const [saved, setSaved] = useState<Record<string, unknown>[]>([])
  const [savedError, setSavedError] = useState('')
  const [busy, setBusy] = useState(false)
  const [savedBusy, setSavedBusy] = useState(false)
  const [error, setError] = useState('')

  const loadSaved = async () => {
    setSavedBusy(true)
    try { setSaved((await api.modelTournaments()).tournaments); setSavedError('') }
    catch (reason) { setSaved([]); setSavedError(reason instanceof Error ? reason.message : 'Could not load saved tournaments.') }
    finally { setSavedBusy(false) }
  }
  useEffect(() => { void loadSaved() }, [])

  const toggleModel = (model: TournamentModel) => setSelectedModels((current) => current.includes(model)
    ? current.filter((item) => item !== model) : [...current, model])
  const run = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!selectedModels.length) { setError('Select at least one model.'); return }
    if (!validTournamentPeriods(trainStart, trainEnd, testStart, testEnd)) {
      setError('Use valid dates with training start < training end < test start < test end. End dates are exclusive.')
      return
    }
    setBusy(true); setError(''); setResult(null)
    try {
      const response = await api.runModelTournament({ ticker: ticker.trim().toUpperCase(), models: selectedModels,
        train_start: trainStart, train_end: trainEnd, test_start: testStart, test_end: testEnd,
        interval, paths: Number(paths), seed: Number(seed), save, ...(save && name.trim() ? { name: name.trim() } : {}) })
      setResult(response)
      if (response.tournament_id) void loadSaved()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Model tournament failed.')
    } finally { setBusy(false) }
  }
  const restore = (item: Record<string, unknown>) => {
    setResult(tournamentResultFromSaved(item)); setError('')
  }
  const state = tournamentViewState(busy, error, result !== null)
  const completed = result?.results.filter((item) => item.status === 'completed') ?? []
  const chartDomain = useMemo(() => {
    const values = completed.flatMap(({ metrics }) => {
      const q = metrics?.simulated_terminal_simple_return_p05_p50_p95
      return Array.isArray(q) ? q.filter((v): v is number => typeof v === 'number' && Number.isFinite(v)) : []
    })
    const observed = completed.map(({ metrics }) => metricNumber(metrics, 'test_period_total_simple_return')).filter((v): v is number => v !== null)
    const all = [...values, ...observed]
    const min = all.length ? Math.min(...all) : -.25, max = all.length ? Math.max(...all) : .25
    const pad = Math.max((max - min) * .12, .01)
    return [min - pad, max + pad] as const
  }, [completed])
  const xAt = (value: number) => 48 + (value - chartDomain[0]) / (chartDomain[1] - chartDomain[0]) * 884
  const fitParams: Record<TournamentModel, string[]> = {
    gbm: ['drift', 'volatility', 'mean_period_log_return', 'sample_std_period_log_return'],
    ou: ['mean_level', 'mean_reversion_speed', 'volatility', 'initial_value'],
    jump_diffusion: ['drift', 'volatility', 'jump_intensity', 'mean_log_jump', 'jump_volatility'],
    garch: ['mean', 'omega', 'alpha', 'beta', 'initial_variance'],
  }

  return <div className="tournament-page">
    <PageHeading eyebrow="EVALUATION / MODEL TOURNAMENT" title="Model Tournament" description="Compare train-only calibrated one-step return distributions on a later chronological holdout. Scores describe this sample and do not select a universally best model." aside={<span className="source-tag"><BarChart3 size={13} /> OUT-OF-SAMPLE RESEARCH</span>} />
    <Panel className="tournament-config"><PanelTitle title="Tournament configuration" subtitle="The backend downloads each half-open period independently and fits every selected model using training observations only." />
      <form onSubmit={(event) => void run(event)} className="tournament-form">
        <Field label="Instrument" name="ticker" value={ticker} maxLength={32} required onChange={(event) => setTicker(event.target.value.toUpperCase())} />
        <Field label="Training start" type="date" name="train_start" value={trainStart} required onChange={(event) => setTrainStart(event.target.value)} />
        <Field label="Training end (exclusive)" type="date" name="train_end" value={trainEnd} required onChange={(event) => setTrainEnd(event.target.value)} />
        <Field label="Test start" type="date" name="test_start" value={testStart} required onChange={(event) => setTestStart(event.target.value)} />
        <Field label="Test end (exclusive)" type="date" name="test_end" value={testEnd} required onChange={(event) => setTestEnd(event.target.value)} />
        <label className="field"><span>Observation interval</span><select value={interval} onChange={(event) => setInterval(event.target.value)}>{[['1d','Daily'],['5d','5 day'],['1wk','Weekly'],['1mo','Monthly'],['3mo','Quarterly']].map(([key,label]) => <option value={key} key={key}>{label}</option>)}</select></label>
        <Field label="Simulation paths" type="number" min="100" max="5000" step="100" value={paths} onChange={(event) => setPaths(event.target.value)} />
        <Field label="Random seed" type="number" min="0" max="4294967295" step="1" value={seed} onChange={(event) => setSeed(event.target.value)} />
        <div className="tournament-model-select"><span>Models</span>{MODELS.map((model) => <label key={model.key}><input type="checkbox" checked={selectedModels.includes(model.key)} onChange={() => toggleModel(model.key)} /><b>{model.short}</b><small>{model.note}</small></label>)}</div>
        <div className="tournament-save"><label><input type="checkbox" checked={save} onChange={(event) => setSave(event.target.checked)} /> Save completed model results</label>{save && <Field label="Tournament name" value={name} maxLength={100} onChange={(event) => setName(event.target.value)} />}</div>
        <Button type="submit" busy={busy}><Play size={14} /> RUN MODEL TOURNAMENT</Button>
      </form>
      <div className="tournament-period-visual" aria-label="Chronological train and test periods"><span>TRAIN · {trainStart} → {trainEnd} exclusive</span><i/><span>TEST · {testStart} → {testEnd} exclusive</span></div>
      <p className="tournament-note">OU models log returns as its mean-reverting state; outputs are compounded into prices only for horizon risk summaries. This does not make OU a stock-price process. All models use within-test close-to-close log returns, preferring adjusted close when available.</p>
      {error && <ErrorNotice>{error}</ErrorNotice>}
    </Panel>

    {state === 'loading' && <Panel><div className="tournament-state" role="status"><RefreshCw size={17} className="spin"/><div><strong>Calibrating and evaluating selected models…</strong><span>Fetching training and test windows separately; each test forecast is scored before its observed return updates the next forecast.</span></div></div></Panel>}
    {state === 'idle' && <Panel className="tournament-empty"><Activity size={20}/><strong>Configure a chronological holdout to begin</strong><span>Results will show valid scores only for models whose training sample supports calibration.</span></Panel>}
    {result && <>
      <Panel><PanelTitle title={`${result.ticker} · Evaluation windows`} subtitle={result.comparison_method} />
        <p className="tournament-version">Model implementation version: {result.model_version ?? 'not recorded in this saved run'}</p>
        <div className="tournament-window-grid"><div><span>Training · end exclusive</span><strong>{String(result.training_period.start)} — {String(result.training_period.end_exclusive)}</strong><small>{String(result.training_period.observation_count)} observations · actual {String(result.training_period.first_observation)} to {String(result.training_period.last_observation)}</small></div><div><span>Test · end exclusive</span><strong>{String(result.test_period.start)} — {String(result.test_period.end_exclusive)}</strong><small>{String(result.test_period.observation_count)} observations · {String(result.test_period.return_count)} scored returns</small></div><div><span>Seed / paths</span><strong>{result.reproducibility.seed} / {result.reproducibility.paths}</strong><small>Deterministic model simulation replay within the same runtime</small></div></div>
      </Panel>
      <Panel><PanelTitle title="Model comparison" subtitle="Lower predictive log score, PIT KS statistic, and QLIKE are preferable; VaR breach rate should be near 5% over a sufficiently long test sample. No single metric defines a winner." />
        <div className="tournament-table-wrap"><table className="tournament-table"><thead><tr><th>Model</th><th>Status</th><th>Mean negative log score ↓</th><th>PIT KS ↓</th><th>Volatility QLIKE ↓</th><th>95% VaR breaches</th><th>Observed test return</th><th>Observed max drawdown</th></tr></thead><tbody>{result.results.map((item) => <tr key={item.model}><th>{MODELS.find((m) => m.key === item.model)?.label ?? item.model}</th><td><span className={`tournament-status ${item.status}`}>{item.status}</span>{item.error && <small>{item.error}</small>}</td><td>{text(metricNumber(item.metrics, 'mean_negative_log_score_nats_per_observation'))}</td><td>{text(metricNumber(item.metrics, 'pit_ks_statistic'))}</td><td>{text(metricNumber(item.metrics, 'volatility_qlike_mean'))}</td><td>{item.metrics ? `${percent(metricNumber(item.metrics, 'one_period_95pct_var_breach_rate'))} · ${text(item.metrics.historical_risk_tail_observations)} tail obs` : 'Unavailable'}</td><td>{percent(metricNumber(item.metrics, 'test_period_total_simple_return'))}</td><td>{percent(metricNumber(item.metrics, 'test_period_maximum_drawdown'))}</td></tr>)}</tbody></table></div>
        <p className="tournament-note">Predictive log score is the mean negative log density in nats for the same held-out log-return target. PIT KS is an in-sample diagnostic on held-out probability integral transforms, not a significance-adjusted model-selection test. Historical test VaR/ES are ex-post sample diagnostics; predictive VaR/ES are forecast before each observation.</p>
      </Panel>
      <Panel><PanelTitle title="Simulated terminal return distributions" subtitle="Model-based horizon outcomes from the existing simulation engines using train-only parameters, the test return horizon, the same path count, and the recorded seed. These are distributions, not point forecasts." />
        <svg className="tournament-terminal-chart" viewBox="0 0 980 250" role="img" aria-label="Simulated terminal return quantiles compared with the observed test return">
          {[0, .25, .5, .75, 1].map((fraction) => { const value = chartDomain[0] + (chartDomain[1] - chartDomain[0]) * fraction; return <g key={fraction}><line x1={48 + 884 * fraction} x2={48 + 884 * fraction} y1="28" y2="210"/><text x={48 + 884 * fraction} y="233">{percent(value)}</text></g> })}
          {completed.map((item, index) => {
            const q = item.metrics?.simulated_terminal_simple_return_p05_p50_p95
            const actual = metricNumber(item.metrics, 'test_period_total_simple_return')
            if (!Array.isArray(q) || q.length < 3 || !q.every((v) => typeof v === 'number') || actual == null) return null
            const y = 52 + index * 42
            return <g key={item.model}><text x="4" y={y + 4}>{title(item.model)}</text><line x1={xAt(q[0] as number)} x2={xAt(q[2] as number)} y1={y} y2={y} className="tournament-range"/><circle cx={xAt(q[1] as number)} cy={y} r="5" className="tournament-median"/><path d={`M${xAt(actual)},${y - 8} v16`} className="tournament-observed"/><text x="945" y={y + 4}>{percent(actual)}</text></g>
          })}
        </svg><div className="tournament-chart-legend"><span><i className="range-key"/> Simulated P05–P95</span><span><i className="median-key"/> Simulated median</span><span><i className="actual-key"/> Observed test return</span></div>
      </Panel>
      <Panel className="tournament-observatory-panel"><ModelTournamentResearch result={result} /></Panel>
      <Panel><PanelTitle title="Model-specific diagnostics" subtitle="Training calibration outputs, assumptions, warnings, and horizon risk summaries. Training parameters are never refit on the test period." />
        <div className="tournament-diagnostics">{result.results.map((item: TournamentModelResult) => <article key={item.model}><h3>{MODELS.find((m) => m.key === item.model)?.label ?? item.model}</h3>{item.status !== 'completed' ? <p>{item.error ?? 'This model could not be evaluated for the selected data and calibration requirements.'}</p> : <>
          <dl>{fitParams[item.model].map((key) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{text(item.parameters?.[key])}</dd></div>)}{item.model === 'garch' && <div><dt>Stationary</dt><dd>{String(item.parameters?.stationary)}</dd></div>}</dl>
          <p>Sim terminal loss VaR / ES 95%: {percent(metricNumber(item.metrics, 'simulated_terminal_95pct_var_loss_fraction'))} / {percent(metricNumber(item.metrics, 'simulated_terminal_95pct_expected_shortfall_loss_fraction'))}</p>
          <p>Sim max drawdown P05 / P50 / P95: {Array.isArray(item.metrics?.simulated_maximum_drawdown_p05_p50_p95) ? (item.metrics?.simulated_maximum_drawdown_p05_p50_p95 as number[]).map(percent).join(' / ') : 'Unavailable'}</p>
          {item.warnings.map((warning) => <small className="tournament-warning" key={warning}>{warning}</small>)}
          {item.assumptions.length > 0 && <details><summary>Calibration method and assumptions</summary><ul>{item.assumptions.map((assumption) => <li key={assumption}>{assumption}</li>)}</ul><p>{item.limitation}</p></details>}
          {item.experiment_id && <small className="tournament-saved-id">Saved experiment {item.experiment_id}</small>}
        </>}</article>)}</div>
      </Panel>
    </>}

    <Panel><PanelTitle title="Saved tournament runs" subtitle="Saved model rows are stored by the existing experiment store and grouped by tournament ID." action={<Button type="button" variant="quiet" busy={savedBusy} onClick={() => void loadSaved()}><RefreshCw size={13}/> Refresh</Button>} />
      {savedError && <ErrorNotice>{savedError}</ErrorNotice>}
      {savedBusy ? <div className="inline-loading">Loading saved tournament results…</div> : saved.length === 0 ? <div className="tournament-empty compact"><Database size={18}/><strong>{savedError ? 'Saved tournaments unavailable' : 'No saved tournaments found'}</strong><span>{savedError ? 'Refresh after the experiment store is available.' : 'Enable “Save completed model results” before running a tournament.'}</span></div> : <div className="saved-tournaments">{saved.map((item) => <button type="button" key={String(item.tournament_id)} onClick={() => restore(item)}><span><strong>{String(item.name)}</strong><small>{String(item.ticker)} · {String(item.interval)} · {String((item.training_period as Record<string, unknown> | undefined)?.start)}–{String((item.test_period as Record<string, unknown> | undefined)?.end_exclusive)}</small></span><span>{Array.isArray(item.models) ? item.models.map((model) => title(String((model as Record<string, unknown>).model))).join(' / ') : ''}</span><Clock3 size={14}/></button>)}</div>}
    </Panel>
  </div>
}
