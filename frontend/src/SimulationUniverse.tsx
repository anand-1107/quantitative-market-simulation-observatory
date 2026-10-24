import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Activity, Pause, Play, RotateCcw, SkipBack, Waves } from 'lucide-react'
import { api, ApiError } from './api'
import { Button, ErrorNotice, Field, PageHeading, Panel, PanelTitle, SelectField, SuccessNotice } from './components'
import { DistributionSurface, SynchronizedPerformanceChart } from './UniverseVisuals'
import { estimateDensitySurface, inspectPath, pathJourneySnapshot, playbackAdvance, playbackPause, playbackReset, playbackStart, scenarioComparisonPaths } from './simulation-universe-math'
import type { GARCHCalibration, GBMCalibration, MertonCalibration, OUCalibration, PathForensics, ScenarioMatchingResult, SimulationTrajectories } from './types'

type ModelKey = SimulationTrajectories['model']
type Calibration = (GBMCalibration & { model: 'gbm' }) | OUCalibration | MertonCalibration | GARCHCalibration
type FieldSpec = { key: string; label: string; min?: number; max?: number; step?: string; optional?: boolean }

const TITLES: Record<ModelKey, string> = { gbm: 'Geometric Brownian Motion', ou: 'Ornstein–Uhlenbeck', jump_diffusion: 'Merton jump-diffusion', garch: 'GARCH(1,1)' }
const ROUTES: Record<ModelKey, string> = { gbm: '/simulate-gbm/paths', ou: '/simulate-ou/paths', jump_diffusion: '/simulate-jump-diffusion/paths', garch: '/simulate-garch/paths' }
const BASE: Record<ModelKey, Record<string, string>> = {
  gbm: { initial_price: '100', drift: '.08', volatility: '.2', years: '1', steps: '160', paths: '64', seed: '42' },
  ou: { initial_value: '0', mean_level: '0', mean_reversion_speed: '1', volatility: '1', years: '1', steps: '160', paths: '64', seed: '42' },
  jump_diffusion: { initial_price: '100', drift: '.08', volatility: '.2', jump_intensity: '.5', mean_log_jump: '-.1', jump_volatility: '.2', years: '1', steps: '160', paths: '64', seed: '42' },
  garch: { omega: '.000001', alpha: '.05', beta: '.9', mean: '0', initial_variance: '', steps: '160', paths: '64', seed: '42', return_convention: 'log' },
}
const FIELDS: Record<ModelKey, FieldSpec[]> = {
  gbm: [
    { key: 'initial_price', label: 'Initial price S₀', min: 0.000001, max: 1e9 },
    { key: 'drift', label: 'Drift μ / year', min: -2, max: 2 },
    { key: 'volatility', label: 'Volatility σ / √year', min: 0, max: 3 },
    { key: 'years', label: 'Horizon / years', min: .000001, max: 10 },
  ],
  ou: [
    { key: 'initial_value', label: 'Initial state X₀', min: -1e9, max: 1e9 },
    { key: 'mean_level', label: 'Long-run mean μ', min: -1e9, max: 1e9 },
    { key: 'mean_reversion_speed', label: 'Reversion θ / year', min: 0, max: 100000 },
    { key: 'volatility', label: 'Level volatility σ / √year', min: 0, max: 1e6 },
    { key: 'years', label: 'Horizon / years', min: .000001, max: 10 },
  ],
  jump_diffusion: [
    { key: 'initial_price', label: 'Initial price S₀', min: .000001, max: 1e9 },
    { key: 'drift', label: 'Compensated drift / year', min: -2, max: 2 },
    { key: 'volatility', label: 'Diffusion σ / √year', min: 0, max: 3 },
    { key: 'jump_intensity', label: 'Jump intensity λ / year', min: 0, max: 10 },
    { key: 'mean_log_jump', label: 'Mean log jump m', min: -2, max: 2 },
    { key: 'jump_volatility', label: 'Jump-size σJ', min: 0, max: 2 },
    { key: 'years', label: 'Horizon / years', min: .000001, max: 10 },
  ],
  garch: [
    { key: 'omega', label: 'Variance intercept ω', min: .000000001, max: 1 },
    { key: 'alpha', label: 'Shock response α', min: 0, max: 10 },
    { key: 'beta', label: 'Variance persistence β', min: 0, max: 10 },
    { key: 'mean', label: 'Mean return / period', min: -10, max: 10 },
    { key: 'initial_variance', label: 'Initial variance · optional', min: .000000001, max: 100, optional: true },
  ],
}

function messageFor(error: unknown): string {
  if (error instanceof ApiError || error instanceof Error) return error.message
  return 'An unexpected error occurred while contacting the simulation API.'
}
function numberText(value: number | null | undefined, digits = 5): string {
  return value == null || !Number.isFinite(value) ? 'Unavailable' : Number(value).toPrecision(digits)
}
function payloadFrom(parameters: Record<string, string>): Record<string, unknown> {
  const payload: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(parameters)) {
    if (key === 'return_convention') payload[key] = value
    else if (value === '' && (key === 'initial_variance' || key === 'seed')) payload[key] = null
    else payload[key] = Number(value)
  }
  return payload
}

export default function SimulationUniverse({ ticker: selectedTicker, onTickerChange }: { ticker: string; onTickerChange: (ticker: string) => void }) {
  const [model, setModel] = useState<ModelKey>('gbm')
  const [parameters, setParameters] = useState(BASE.gbm)
  const [ticker, setTicker] = useState(selectedTicker)
  const [calibrationStart, setCalibrationStart] = useState('2021-01-01')
  const [calibrationEnd, setCalibrationEnd] = useState('2024-01-01')
  const [calibrationInterval, setCalibrationInterval] = useState('1d')
  const [ouInputSeries, setOuInputSeries] = useState<'log_return' | 'log_price' | 'price'>('log_return')
  const [confirmPriceOU, setConfirmPriceOU] = useState(false)
  const [calibration, setCalibration] = useState<Calibration | null>(null)
  const [appliedCalibration, setAppliedCalibration] = useState<Calibration | null>(null)
  const [calibrationBusy, setCalibrationBusy] = useState(false)
  const [calibrationError, setCalibrationError] = useState('')
  const [calibrationNotice, setCalibrationNotice] = useState('')
  const [data, setData] = useState<SimulationTrajectories | null>(null)
  const [forensics, setForensics] = useState<PathForensics | null>(null)
  const [forensicsBusy, setForensicsBusy] = useState(false)
  const [forensicsError, setForensicsError] = useState('')
  const [simulationError, setSimulationError] = useState('')
  const [busy, setBusy] = useState(false)
  const [selectedPath, setSelectedPath] = useState(0)
  const [playback, setPlayback] = useState({ index: 0, playing: false })
  const [speed, setSpeed] = useState(1)
  const [draftChanged, setDraftChanged] = useState(false)
  const playbackRef = useRef(playback)
  const animationRef = useRef<number | null>(null)
  const simulationRequestRef = useRef(0)
  const calibrationRequestRef = useRef(0)
  const forensicsRequestRef = useRef(0)
  useEffect(() => { playbackRef.current = playback }, [playback])
  useEffect(() => () => { simulationRequestRef.current += 1; calibrationRequestRef.current += 1 }, [])

  useEffect(() => {
    if (!playback.playing || !data) return
    const finalIndex = data.times.length - 1
    let lastTime = 0
    let accumulated = 0
    const frame = (timestamp: number) => {
      if (!lastTime) lastTime = timestamp
      accumulated += timestamp - lastTime
      lastTime = timestamp
      const frameDuration = 150 / speed
      if (accumulated >= frameDuration) {
        accumulated %= frameDuration
        const next = playbackAdvance(playbackRef.current, finalIndex)
        playbackRef.current = next
        setPlayback(next)
        if (!next.playing) return
      }
      animationRef.current = requestAnimationFrame(frame)
    }
    animationRef.current = requestAnimationFrame(frame)
    return () => { if (animationRef.current !== null) cancelAnimationFrame(animationRef.current); animationRef.current = null }
  }, [playback.playing, speed, data])

  useEffect(() => {
    const runId = data?.forensics_run_id
    if (!runId) { setForensics(null); setForensicsError(''); return }
    const requestId = ++forensicsRequestRef.current
    const controller = new AbortController()
    setForensicsBusy(true); setForensics(null); setForensicsError('')
    api.pathForensics(runId, selectedPath + 1, controller.signal)
      .then((result) => { if (requestId === forensicsRequestRef.current) setForensics(result) })
      .catch((error: unknown) => { if (requestId === forensicsRequestRef.current && !(error instanceof DOMException && error.name === 'AbortError')) setForensicsError(messageFor(error)) })
      .finally(() => { if (requestId === forensicsRequestRef.current) setForensicsBusy(false) })
    return () => { controller.abort(); if (requestId === forensicsRequestRef.current) forensicsRequestRef.current += 1 }
  }, [data?.forensics_run_id, selectedPath])

  const densityResult = useMemo(() => {
    if (!data) return { surface: null, error: '' }
    try { return { surface: estimateDensitySurface(data.trajectories), error: '' } }
    catch (error) { return { surface: null, error: messageFor(error) } }
  }, [data])
  const activeCalibration = appliedCalibration?.model === model ? appliedCalibration : null
  const selectedMetrics = data ? inspectPath(data, selectedPath, playback.index) : null

  const changeParameter = (key: string, value: string) => {
    setParameters((old) => ({ ...old, [key]: value }))
    setAppliedCalibration(null); setDraftChanged(true); setCalibrationNotice('')
  }
  const switchModel = (next: ModelKey) => {
    simulationRequestRef.current += 1; calibrationRequestRef.current += 1
    setBusy(false); setCalibrationBusy(false)
    setModel(next); setParameters({ ...BASE[next] }); setCalibration(null); setAppliedCalibration(null)
    setData(null); setSelectedPath(0); setPlayback(playbackReset()); playbackRef.current = playbackReset()
    setSimulationError(''); setCalibrationError(''); setCalibrationNotice(''); setDraftChanged(false)
  }
  const runSimulation = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const payload = payloadFrom(parameters)
    const steps = Number(parameters.steps), paths = Number(parameters.paths)
    if (!Number.isInteger(steps) || steps < 1 || steps > 512 || !Number.isInteger(paths) || paths < 1 || paths > 128) {
      setSimulationError('Use 1–512 time steps and 1–128 paths.'); return
    }
    const requestConfiguration = { ...payload }
    if (appliedCalibration && appliedCalibration.model === model) requestConfiguration.calibration = provenance(appliedCalibration)
    requestConfiguration.retain_forensics = true
    const requestId = ++simulationRequestRef.current
    setBusy(true); setSimulationError(''); setData(null); setPlayback(playbackReset()); playbackRef.current = playbackReset()
    try {
      const result = await api.simulateTrajectories(ROUTES[model], requestConfiguration)
      if (requestId !== simulationRequestRef.current) return
      setData(result); setSelectedPath(0)
      setPlayback({ index: 0, playing: false }); playbackRef.current = { index: 0, playing: false }
      setDraftChanged(false)
    } catch (error) { if (requestId === simulationRequestRef.current) setSimulationError(messageFor(error)) }
    finally { if (requestId === simulationRequestRef.current) setBusy(false) }
  }

  const calibrate = async () => {
    const requestId = ++calibrationRequestRef.current
    setCalibrationBusy(true); setCalibrationError(''); setCalibrationNotice(''); setCalibration(null); setAppliedCalibration(null)
    try {
      const base = { ticker, start: calibrationStart, end: calibrationEnd, interval: calibrationInterval, price_field: 'auto' }
      let fitted: Calibration
      if (model === 'gbm') fitted = { ...await api.calibrateGBM(base), model: 'gbm' }
      else if (model === 'ou') fitted = await api.calibrateOU({ ...base, input_series: ouInputSeries, confirm_mean_reverting_price_series: confirmPriceOU })
      else if (model === 'jump_diffusion') fitted = await api.calibrateMerton(base)
      else fitted = await api.calibrateGARCH(base)
      if (requestId !== calibrationRequestRef.current) return
      setCalibration(fitted)
    } catch (error) { if (requestId === calibrationRequestRef.current) setCalibrationError(messageFor(error)) }
    finally { if (requestId === calibrationRequestRef.current) setCalibrationBusy(false) }
  }

  const applyEstimate = () => {
    if (!calibration || calibration.model !== model) return
    const next = { ...parameters }
    if (calibration.model === 'gbm') { next.drift = String(calibration.drift); next.volatility = String(calibration.volatility) }
    else if (calibration.model === 'ou') {
      next.initial_value = String(calibration.initial_value); next.mean_level = String(calibration.mean_level)
      next.mean_reversion_speed = String(calibration.mean_reversion_speed); next.volatility = String(calibration.volatility)
    } else if (calibration.model === 'jump_diffusion') {
      next.initial_price = String(calibration.initial_price); next.drift = String(calibration.drift)
      next.volatility = String(calibration.volatility); next.jump_intensity = String(calibration.jump_intensity)
      next.mean_log_jump = String(calibration.mean_log_jump); next.jump_volatility = String(calibration.jump_volatility)
    } else {
      next.mean = String(calibration.mean); next.omega = String(calibration.omega); next.alpha = String(calibration.alpha)
      next.beta = String(calibration.beta); next.initial_variance = String(calibration.initial_variance); next.return_convention = 'log'
    }
    setParameters(next); setAppliedCalibration(calibration); setDraftChanged(true)
    setCalibrationNotice('Calibration explicitly applied to the draft. Run a new simulation to use these estimates.')
  }

  const updatePlayback = (fn: (state: { index: number; playing: boolean }) => { index: number; playing: boolean }) => {
    setPlayback((old) => { const next = fn(old); playbackRef.current = next; return next })
  }
  const start = () => data && updatePlayback((state) => playbackStart(state, data.times.length - 1))
  const pause = () => updatePlayback(playbackPause)
  const reset = () => updatePlayback(playbackReset)
  const replay = () => updatePlayback(() => ({ index: 0, playing: true }))

  return <div className="simulation-universe-page">
    <PageHeading eyebrow="RESEARCH / STOCHASTIC DISTRIBUTION" title="Simulation Universe" description="Explore the evolving probability density of actual backend simulation paths, synchronized with a selected-path chart and step-by-step playback." aside={<span className="universe-source-badge"><i /> PYTHON-GENERATED PATHS</span>} />
    <div className="universe-layout">
      <Panel className="universe-control-panel">
        <PanelTitle title="Model configuration" subtitle="Draft inputs are separate from the immutable generated run." />
        <form className="universe-form" onSubmit={(event) => void runSimulation(event)}>
          <SelectField label="Stochastic model" value={model} onChange={(event) => switchModel(event.target.value as ModelKey)}>
            {(Object.keys(TITLES) as ModelKey[]).map((key) => <option value={key} key={key}>{TITLES[key]}</option>)}
          </SelectField>
          <div className="universe-input-grid">{FIELDS[model].map((field) => <Field key={field.key} label={field.label} type="number" min={field.min} max={field.max} step={field.step ?? 'any'} required={!field.optional} value={parameters[field.key] ?? ''} onChange={(event) => changeParameter(field.key, event.target.value)} />)}</div>
          {model === 'garch' && <SelectField label="Return convention" value={parameters.return_convention} onChange={(event) => changeParameter('return_convention', event.target.value)}><option value="log">Log returns</option><option value="simple">Simple returns</option></SelectField>}
          <div className="universe-input-grid universe-run-inputs">
            <Field label={model === 'garch' ? 'Observation periods' : 'Simulation time steps'} type="number" min={1} max={512} step="1" required value={parameters.steps} onChange={(event) => changeParameter('steps', event.target.value)} />
            <Field label="Independent paths" type="number" min={1} max={128} step="1" required value={parameters.paths} onChange={(event) => changeParameter('paths', event.target.value)} />
            <Field label="Random seed · blank = unseeded" type="number" min={0} max={4294967295} step="1" value={parameters.seed} onChange={(event) => changeParameter('seed', event.target.value)} />
          </div>
          <div className="universe-calibration-box">
            <div className="universe-calibration-title"><strong>Historical calibration</strong><span>{activeCalibration ? 'CALIBRATED DRAFT' : 'MANUAL DRAFT'}</span></div>
            <Field label="Ticker / instrument" value={ticker} maxLength={32} onChange={(event) => { calibrationRequestRef.current += 1; setCalibrationBusy(false); setTicker(event.target.value.toUpperCase()); onTickerChange(event.target.value.toUpperCase()); setCalibration(null); setAppliedCalibration(null); setCalibrationNotice('') }} />
            <div className="universe-input-grid"><Field label="Estimation start" type="date" value={calibrationStart} onChange={(event) => setCalibrationStart(event.target.value)} /><Field label="Estimation end" type="date" value={calibrationEnd} onChange={(event) => setCalibrationEnd(event.target.value)} /></div>
            <SelectField label="Observation interval" value={calibrationInterval} onChange={(event) => setCalibrationInterval(event.target.value)}><option value="1d">Daily</option><option value="5d">5 day</option><option value="1wk">Weekly</option><option value="1mo">Monthly</option><option value="3mo">Quarterly</option></SelectField>
            {model === 'ou' && <><SelectField label="OU input series" value={ouInputSeries} onChange={(event) => setOuInputSeries(event.target.value as typeof ouInputSeries)}><option value="log_return">Close-to-close log returns</option><option value="log_price">Log price</option><option value="price">Raw price</option></SelectField>{ouInputSeries !== 'log_return' && <label className="universe-confirm"><input type="checkbox" checked={confirmPriceOU} onChange={(event) => setConfirmPriceOU(event.target.checked)} /> I confirm this selected series is a suitable mean-reverting state.</label>}</>}
            <Button type="button" variant="quiet" busy={calibrationBusy} onClick={() => void calibrate()}>ESTIMATE {model.toUpperCase()} FROM HISTORY</Button>
            {calibrationError && <ErrorNotice>{calibrationError}</ErrorNotice>}
            {calibrationNotice && <SuccessNotice>{calibrationNotice}</SuccessNotice>}
            {calibration && calibration.model === model && <div className="universe-calibration-result"><small>{calibration.ticker} · {calibration.estimation_start} → {calibration.estimation_end} · {calibration.observation_count} observations · {calibration.interval}</small>
              <strong>{calibrationValues(calibration)}</strong>
              {calibration.warnings.map((warning) => <small className="universe-warning" key={warning}>{warning}</small>)}
              <details className="universe-calibration-details"><summary>Method and assumptions</summary><small>{calibration.method}</small>{calibration.assumptions.map((assumption) => <small key={assumption}>• {assumption}</small>)}{'limitations' in calibration && <small>Limitations: {calibration.limitations}</small>}</details>
              <Button type="button" variant="secondary" disabled={'optimizer_success' in calibration && !calibration.optimizer_success} onClick={applyEstimate}>APPLY ESTIMATES TO DRAFT</Button>
            </div>}
          </div>
          <p className="universe-model-note">{model === 'ou' ? 'OU is plotted as a state level. Its state values have no implied asset return or portfolio P&amp;L.' : model === 'garch' ? 'GARCH density uses per-period simulated returns; the chart does not represent portfolio P&amp;L.' : 'Price paths are simulated model outputs, not strategy equity or forecasts.'}</p>
          <Button type="submit" className="universe-generate-button" busy={busy}><Waves size={15} /> GENERATE SIMULATION UNIVERSE</Button>
          {simulationError && <ErrorNotice>{simulationError}</ErrorNotice>}
        </form>
      </Panel>

      <section className="universe-visual-column">
        <Panel className="density-panel">
          <PanelTitle title="Probability-density surface" subtitle={densityResult.surface ? `${densityResult.surface.method} Density uses ${data?.trajectories.length ?? 0} actual paths at each time; all individual trajectories reveal together.` : 'X = time · Y = model output value · Z = estimated probability density'} action={data && <span className="universe-run-meta">SEED {data.parameters.seed ?? 'NONE'} · {data.trajectories.length} PATHS</span>} />
          {busy ? <div className="universe-empty"><span className="universe-loading-orb"><Activity size={20} /></span><strong>Generating backend trajectories</strong><small>Animation will be enabled after a valid simulation response arrives.</small></div>
            : data && densityResult.surface ? <div className="density-canvas"><DistributionSurface data={data} density={densityResult.surface} selectedPath={selectedPath} currentIndex={playback.index} /><div className="density-color-key"><span>LOWER DENSITY</span><i /><span>HIGHER DENSITY</span></div></div>
              : data ? <div className="universe-empty universe-error-state"><Waves size={23} /><strong>Density surface unavailable</strong><small>{densityResult.error}</small><small>The returned paths and performance chart remain available below.</small></div>
                : <div className="universe-empty"><Waves size={23} /><strong>Universe is waiting for a simulation</strong><small>Select a model and generate paths to estimate its evolving distribution.</small></div>}
        </Panel>

        <Panel className="universe-playback-panel">
          <div className="universe-playback-row">
            <div className="universe-playback-buttons">
              <Button type="button" variant="secondary" disabled={!data || playback.playing} onClick={start}><Play size={14} /> PLAY</Button>
              <Button type="button" variant="quiet" disabled={!playback.playing} onClick={pause}><Pause size={14} /> PAUSE</Button>
              <Button type="button" variant="quiet" disabled={!data} onClick={reset}><RotateCcw size={14} /> RESET</Button>
              <Button type="button" variant="quiet" disabled={!data} onClick={replay}><SkipBack size={14} /> REPLAY</Button>
            </div>
            <SelectField label="Speed" value={String(speed)} onChange={(event) => setSpeed(Number(event.target.value))}><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option><option value="4">4×</option></SelectField>
          </div>
          <div className="universe-progress">
            <span>STEP {data ? playback.index + 1 : 0} / {data?.times.length ?? 0}</span>
            <input aria-label="Simulation time scrubber" type="range" min={0} max={Math.max(0, (data?.times.length ?? 1) - 1)} step={1} value={data ? playback.index : 0} disabled={!data} onChange={(event) => { const index = Number(event.target.value); const next = { index, playing: false }; playbackRef.current = next; setPlayback(next) }} />
            <span>{data ? `${data.time_label}: ${numberText(data.times[playback.index], 4)}` : 'NO RUN'}</span>
          </div>
          {draftChanged && data && <div className="universe-draft-note">Draft parameters changed. The displayed paths still belong to the previous run; generate a new simulation to apply edits.</div>}
        </Panel>

        {data && <>
          <Panel className="universe-performance-panel"><PanelTitle title={data.quantity_kind === 'price' ? 'Simulated price paths' : data.quantity_kind === 'return' ? 'Simulated return paths' : 'Simulated state paths'} subtitle={`All ${data.trajectories.length} trajectories reveal in sync · path ${selectedPath + 1} highlighted · ensemble statistics from the same backend result · ${data.quantity_label}`} action={<span className="universe-run-meta">{TITLES[data.model]}</span>} />
            <SynchronizedPerformanceChart data={data} density={densityResult.surface} selectedPath={selectedPath} currentIndex={playback.index} />
          </Panel>
          <Panel className="universe-inspector"><PanelTitle title="Path inspector" subtitle="Original path metrics from this simulation run" action={<span className="universe-run-meta">{activeCalibration ? `CALIBRATED · ${activeCalibration.ticker}` : 'MODEL PARAMETERS FIXED AT RUN'}</span>} />
            <div className="universe-inspector-layout">
              <SelectField label="Selected trajectory" value={String(selectedPath)} onChange={(event) => setSelectedPath(Number(event.target.value))}>{data.trajectories.map((_, index) => <option value={index} key={index}>Path {index + 1}{index === selectedPath ? ' · selected' : ''}</option>)}</SelectField>
              <div className="universe-inspector-stats"><UniverseStat label="Current model value" value={numberText(selectedMetrics?.currentValue)} detail={`${data.time_label} ${numberText(selectedMetrics?.time, 4)} · step ${playback.index + 1}`} /><UniverseStat label="Final model value" value={numberText(selectedMetrics?.finalValue)} detail={data.quantity_label} /><UniverseStat label="Cumulative simple return" value={selectedMetrics?.totalReturn == null ? 'Unavailable' : `${(selectedMetrics.totalReturn * 100).toFixed(3)}%`} detail="Defined for positive price/return wealth paths" /><UniverseStat label="Maximum drawdown" value={selectedMetrics?.maximumDrawdown == null ? 'Unavailable' : `${(selectedMetrics.maximumDrawdown * 100).toFixed(3)}%`} detail="Path maximum from its running wealth peak" /></div>
            </div>
            <details className="universe-parameters"><summary>Effective run configuration and provenance</summary><pre>{JSON.stringify({ model: TITLES[data.model], parameters: data.parameters, random_seed: data.parameters.seed ?? null, calibration: data.parameters.calibration ?? null, displayed_configuration_matches_run: !draftChanged }, null, 2)}</pre></details>
          </Panel>
          <PathForensicsPanel data={forensics} busy={forensicsBusy} error={forensicsError} runId={data.forensics_run_id ?? null} pathId={selectedPath + 1} pathCount={data.trajectories.length} onPathChange={(pathId) => setSelectedPath(pathId - 1)} />
        </>}
      </section>
    </div>
    {data && <div className="universe-method-foot">Density method: {densityResult.surface?.method ?? densityResult.error} KDEs are estimated cross-sectionally from the actual simulated values at every observation time on a common value axis. The lower chart is a simulated model path, not a trading strategy or portfolio P&amp;L.</div>}
  </div>
}

function provenance(calibration: Calibration): Record<string, unknown> {
  const base = { model: calibration.model, ticker: calibration.ticker, estimation_start: calibration.estimation_start,
    estimation_end: calibration.estimation_end, interval: calibration.interval, calibrated_at: calibration.calibrated_at,
    method: calibration.method, observation_count: calibration.observation_count, return_count: calibration.return_count }
  if (calibration.model === 'gbm') return { ...base, drift: calibration.drift, volatility: calibration.volatility }
  if (calibration.model === 'ou') return { ...base, input_series: calibration.input_series, mean_level: calibration.mean_level,
    mean_reversion_speed: calibration.mean_reversion_speed, volatility: calibration.volatility, initial_value: calibration.initial_value }
  if (calibration.model === 'jump_diffusion') return { ...base, drift: calibration.drift, volatility: calibration.volatility,
    jump_intensity: calibration.jump_intensity, mean_log_jump: calibration.mean_log_jump,
    jump_volatility: calibration.jump_volatility, initial_price: calibration.initial_price }
  return { ...base, mean: calibration.mean, omega: calibration.omega, alpha: calibration.alpha, beta: calibration.beta,
    initial_variance: calibration.initial_variance, return_convention: 'log' }
}

function calibrationValues(calibration: Calibration): string {
  if (calibration.model === 'gbm') return `μ ${calibration.drift.toPrecision(4)} / year · σ ${calibration.volatility.toPrecision(4)} / √year`
  if (calibration.model === 'ou') return `θ ${calibration.mean_reversion_speed.toPrecision(4)} / year · μ ${calibration.mean_level.toPrecision(4)} · σ ${calibration.volatility.toPrecision(4)}`
  if (calibration.model === 'jump_diffusion') return `μ ${calibration.drift.toPrecision(4)} · σ ${calibration.volatility.toPrecision(4)} · λ ${calibration.jump_intensity.toPrecision(4)} · m ${calibration.mean_log_jump.toPrecision(4)} · σJ ${calibration.jump_volatility.toPrecision(4)}`
  return `ω ${calibration.omega.toExponential(3)} · α ${calibration.alpha.toPrecision(4)} · β ${calibration.beta.toPrecision(4)} · mean ${calibration.mean.toExponential(3)}`
}

function UniverseStat({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="universe-stat"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>
}

function PathForensicsPanel({ data, busy, error, runId, pathId, pathCount, onPathChange }: { data: PathForensics | null; busy: boolean; error: string; runId: string | null; pathId: number; pathCount: number; onPathChange: (pathId: number) => void }) {
  const matchingRequestRef = useRef(0)
  const [matchingBusy, setMatchingBusy] = useState(false)
  const [matchingError, setMatchingError] = useState('')
  const [matches, setMatches] = useState<ScenarioMatchingResult | null>(null)
  const [comparisonIndex, setComparisonIndex] = useState(0)
  const [journeyIndex, setJourneyIndex] = useState(0)
  useEffect(() => {
    matchingRequestRef.current += 1
    setMatches(null); setMatchingError(''); setMatchingBusy(false)
    return () => { matchingRequestRef.current += 1 }
  }, [data?.run_id, data?.path_id])
  useEffect(() => { setJourneyIndex(0) }, [data?.run_id, data?.path_id])
  const runMatching = async () => {
    if (!data || !runId) return
    const requestId = ++matchingRequestRef.current
    setMatchingBusy(true); setMatchingError(''); setMatches(null)
    try {
      const result = await api.matchScenarios({ run_id: runId, path_id: data.path_id, candidate_parameters: nearbyCandidates(data) })
      if (requestId === matchingRequestRef.current) { setComparisonIndex(0); setMatches(result) }
    } catch (reason) { if (requestId === matchingRequestRef.current) setMatchingError(messageFor(reason)) }
    finally { if (requestId === matchingRequestRef.current) setMatchingBusy(false) }
  }
  const pathSelector = <SelectField label="Reference trajectory" value={String(pathId)} onChange={(event) => onPathChange(Number(event.target.value))}>{Array.from({ length: pathCount }, (_, index) => <option value={index + 1} key={index}>Path {index + 1}{index + 1 === pathId ? ' · selected' : ''}</option>)}</SelectField>
  if (busy) return <Panel className="universe-forensics"><PanelTitle title="Path forensics" subtitle={`Loading exact innovations for path ${pathId}…`} />{pathCount > 0 && pathSelector}</Panel>
  if (error) return <Panel className="universe-forensics"><PanelTitle title="Path forensics" subtitle="Forensic reconstruction is unavailable" />{pathCount > 0 && pathSelector}<ErrorNotice>{error}</ErrorNotice></Panel>
  if (!data) return <Panel className="universe-forensics"><PanelTitle title="Path forensics" subtitle="Exact path-level innovations were not retained for this run." />{pathCount > 0 && pathSelector}</Panel>
  const fields = Object.keys(data.transitions[0] ?? {}).filter((key) => !['time', 'value', 'return'].includes(key))
  const initialOffset = data.model === 'garch' ? 0 : 1
  return <Panel className="universe-forensics">
    <PanelTitle title={`Path forensics · path ${data.path_id}`} subtitle={`Run ${data.run_id} · ${data.model_version} · generated ${data.created_at}`} action={<span className={`universe-forensic-status ${data.reconstruction.verified ? 'verified' : 'unverified'}`}>{data.reconstruction.verified ? 'RECONSTRUCTION VERIFIED' : 'RECONSTRUCTION MISMATCH'}</span>} />
    <div className="universe-forensic-summary">
      <div><span>Maximum absolute error</span><strong>{numberText(data.reconstruction.maximum_absolute_error, 6)}</strong><small>Tolerance {numberText(data.reconstruction.verification_tolerance, 4)}</small></div>
      <div><span>Initial conditions</span><strong>{JSON.stringify(data.initial_conditions)}</strong><small>Seed {String(data.parameters.seed ?? 'unseeded')} · {data.original_path.length} observations</small></div>
      <div><span>Terminal metrics</span><strong>{numberText(Number(data.terminal_metrics.terminal_value), 6)}</strong><small>{JSON.stringify(data.terminal_metrics.path_risk ?? 'No return-risk metric for this state series')}</small></div>
    </div>
    <details className="universe-parameters"><summary>Effective parameters, calibration provenance, and stored innovations</summary><pre>{JSON.stringify({ parameters: data.parameters, initial_conditions: data.initial_conditions, innovations: data.innovations, conditional_variance: data.conditional_variance, reconstruction_method: data.reconstruction.method }, null, 2)}</pre></details>
    <div className="universe-forensic-table-wrap"><table className="universe-forensic-table"><thead><tr><th>Step</th><th>{data.times.length === data.original_path.length && data.model !== 'garch' ? 'Time' : 'Period'}</th>{fields.map((field) => <th key={field}>{field.replaceAll('_', ' ')}</th>)}<th>Original</th><th>Reconstructed</th><th>Error</th></tr></thead><tbody>{data.transitions.map((transition, index) => {
      const valueIndex = index + initialOffset
      const original = data.original_path[valueIndex]
      const rebuilt = data.reconstruction.path[valueIndex]
      const errorValue = Math.abs(original - rebuilt)
      return <tr key={index} className={valueIndex === journeyIndex ? 'path-journey-selected-row' : undefined}><td>{index + 1}</td><td>{numberText(Number(transition.time), 5)}</td>{fields.map((field) => <td key={field}>{typeof transition[field] === 'boolean' ? String(transition[field]) : transition[field] == null ? '—' : numberText(Number(transition[field]), 5)}</td>)}<td>{numberText(original, 7)}</td><td>{numberText(rebuilt, 7)}</td><td>{numberText(errorValue, 3)}</td></tr>
    })}</tbody></table></div>
    <PathJourneyExplorer data={data} selectedIndex={journeyIndex} onSelectIndex={setJourneyIndex} pathSelector={pathSelector} />
    {data.model === 'jump_diffusion' && <small className="universe-forensic-note">Jump sizes are stored as the simulator’s per-period aggregate log jump: it draws a Poisson count and one aggregate normal, then computes N·m + √N·σJ·Z. Individual per-jump normals are not generated by this engine.</small>}
    <div className="universe-scenario-box"><div><strong>REFERENCE-PATH SCENARIO MATCHING</strong><small>Four bounded parameter variations use this path’s stored innovations as common random numbers. This is an in-sample counterfactual comparison.</small></div><Button type="button" variant="secondary" busy={matchingBusy} onClick={() => void runMatching()}>MATCH SCENARIOS</Button></div>
    {matchingError && <ErrorNotice>{matchingError}</ErrorNotice>}
    {matches && <div className="universe-scenario-results"><small>{matches.method}</small><SelectField label="Candidate path comparison" value={String(comparisonIndex)} onChange={(event) => setComparisonIndex(Number(event.target.value))}>{matches.candidates.map((candidate, index) => <option value={index} key={candidate.rank}>Candidate #{candidate.rank} · score {numberText(candidate.objective, 4)}</option>)}</SelectField>{matches.candidates[comparisonIndex] && <ScenarioPathComparison reference={data.original_path} candidate={matches.candidates[comparisonIndex].candidate_path} candidateLabel={`Candidate #${matches.candidates[comparisonIndex].rank}`} />}{matches.candidates.map((candidate) => <div className="universe-scenario-candidate" key={candidate.rank}><strong>#{candidate.rank} · score {numberText(candidate.objective, 5)}</strong><span>Path RMSE {numberText(candidate.normalized_path_rmse, 5)} · terminal {numberText(candidate.terminal_value, 5)} · return {candidate.total_simple_return == null ? 'N/A' : `${(candidate.total_simple_return * 100).toFixed(2)}%`} · drawdown {candidate.maximum_drawdown == null ? 'N/A' : `${(candidate.maximum_drawdown * 100).toFixed(2)}%`} · replayable {candidate.exactly_replayable ? 'YES' : 'NO'}</span><code>{JSON.stringify(candidate.parameters)}</code></div>)}{matches.warnings.map((warning) => <small className="universe-forensic-note" key={warning}>{warning}</small>)}</div>}
  </Panel>
}

function ScenarioPathComparison({ reference, candidate, candidateLabel }: { reference: number[]; candidate: number[]; candidateLabel: string }) {
  const paths = scenarioComparisonPaths(reference, candidate)
  if (!paths) return <small className="universe-forensic-note">Path comparison is unavailable for these candidate observations.</small>
  const count = paths.reference.length
  const width = 900, height = 180, pad = { left: 48, right: 16, top: 14, bottom: 24 }
  const values = [...paths.reference, ...paths.candidate]
  const low = Math.min(...values), high = Math.max(...values)
  const span = Math.max(high - low, Math.max(Math.abs(low), Math.abs(high), 1) * 1e-6)
  const x = (index: number) => pad.left + index / (count - 1) * (width - pad.left - pad.right)
  const y = (value: number) => pad.top + (high + span * .04 - value) / (span * 1.08) * (height - pad.top - pad.bottom)
  const points = (series: number[]) => series.slice(0, count).map((value, index) => `${x(index)},${y(value)}`).join(' ')
  return <div className="scenario-comparison" aria-label="Selected scenario compared with unchanged reference path">
    <strong>Same-innovation replay · reference path {reference.length} observations</strong>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${candidateLabel} compared with unchanged selected reference path`}>
      {[0, .5, 1].map((fraction) => { const yy = pad.top + fraction * (height - pad.top - pad.bottom); return <line key={fraction} x1={pad.left} x2={width - pad.right} y1={yy} y2={yy} className="journey-gridline" /> })}
      <polyline points={points(paths.reference)} className="scenario-reference-line" />
      <polyline points={points(paths.candidate)} className="scenario-candidate-line" />
    </svg>
    <div className="journey-legend"><span className="scenario-reference-key">REFERENCE · UNCHANGED</span><span className="scenario-candidate-key">{candidateLabel.toUpperCase()}</span></div>
  </div>
}

function PathJourneyExplorer({ data, selectedIndex, onSelectIndex, pathSelector }: { data: PathForensics; selectedIndex: number; onSelectIndex: (index: number) => void; pathSelector: ReactNode }) {
  const snapshot = pathJourneySnapshot(data, selectedIndex)
  const path = data.original_path
  if (!snapshot || !path.length) return <section className="path-journey"><div className="path-journey-heading"><strong>Path Journey Explorer</strong><small>No stored path observations are available.</small></div></section>
  const width = 900, height = 230, pad = { left: 54, right: 18, top: 17, bottom: 32 }
  const rawMin = Math.min(...path), rawMax = Math.max(...path)
  const span = Math.max(rawMax - rawMin, Math.max(Math.abs(rawMin), Math.abs(rawMax), 1) * 1e-6)
  const min = rawMin - span * .05, max = rawMax + span * .05, displaySpan = max - min
  const x = (index: number) => pad.left + (path.length < 2 ? (width - pad.left - pad.right) / 2 : index / (path.length - 1) * (width - pad.left - pad.right))
  const y = (value: number) => pad.top + (max - value) / displaySpan * (height - pad.top - pad.bottom)
  const points = path.map((value, index) => ({ value, x: x(index), y: y(value), index }))
  const transition = snapshot.transition
  const transitionFields: Array<[string, string, string]> = data.model === 'gbm' ? [
    ['Previous price', 'previous_value', 'price'], ['Standard-normal innovation', 'standard_normal', 'standard normal'],
    ['Drift log component', 'drift_log_component', 'log return'], ['Diffusion component', 'diffusion_component', 'log return'],
    ['Total log-return increment', 'log_return_increment', 'log return'], ['Original next price', 'value', 'price'],
  ] : data.model === 'ou' ? [
    ['Previous state', 'current_value', 'state units'], ['Conditional mean', 'conditional_mean', 'state units'],
    ['Mean-reversion contribution', 'mean_reversion_component', 'state units'], ['Standard-normal innovation', 'standard_normal', 'standard normal'],
    ['Diffusion contribution', 'diffusion_component', 'state units'],
  ] : data.model === 'jump_diffusion' ? [
    ['Diffusion innovation', 'diffusion_standard_normal', 'standard normal'], ['Poisson jump count', 'poisson_jump_count', 'jumps / period'],
    ['Aggregate jump draw', 'jump_aggregate_standard_normal', 'standard normal'], ['Drift log term', 'drift_log_term', 'log return'],
    ['Diffusion contribution', 'diffusion_component', 'log return'], ['Aggregate jump log sum', 'jump_log_sum', 'log return'],
    ['Total log-return increment', 'log_price_increment', 'log return'], ['Compensator rate', 'compensator_rate', 'per year'],
  ] : [
    ['Previous conditional variance hₜ₋₁', 'previous_conditional_variance', 'return²'],
    ['Conditional variance hₜ', 'conditional_variance', 'return²'], ['Standard-normal innovation', 'standard_normal', 'standard normal'],
    ['Return innovation εₜ', 'innovation', 'return'],
    ['Next conditional variance', 'next_conditional_variance', 'return²'], ['Variance intercept ω', 'variance_intercept_term', 'return²'],
    ['ARCH term αε²', 'arch_variance_term', 'return²'], ['GARCH term βh', 'garch_variance_term', 'return²'],
  ]
  const groups = [...new Set(snapshot.components.map((component) => component.group))]
  const groupTitles: Record<string, string> = { log: 'Log-return contributions', state: 'State-level contributions', return: 'Return contributions', variance: 'Conditional-variance recursion' }
  const maxByGroup = Object.fromEntries(groups.map((group) => [group, Math.max(...snapshot.components.filter((item) => item.group === group).map((item) => Math.abs(item.value)), 1e-12)]))
  const pricePath = data.model === 'gbm' || data.model === 'jump_diffusion'
  const valueLabel = data.model === 'gbm' || data.model === 'jump_diffusion' ? 'Simulated price' : data.model === 'garch' ? 'Simulated return' : 'Simulated state'
  const reconstructionIndex = data.model === 'garch' ? selectedIndex : selectedIndex
  const reconstructed = data.reconstruction.path[reconstructionIndex]
  const original = path[selectedIndex]
  const selectedError = Number.isFinite(reconstructed) ? Math.abs(original - reconstructed) : null
  return <section className="path-journey" aria-label="Path Journey Explorer">
    <div className="path-journey-heading"><div><strong>Path Journey Explorer</strong><small>{valueLabel} · stored path {data.path_id} · step cursor is independent of universe playback</small></div>
      <span>{data.model.toUpperCase()} · {data.times.length === path.length && data.model === 'garch' ? 'observation periods' : 'simulation time'}</span></div>
    <div className="path-journey-path-select">{pathSelector}</div>
    <div className="path-journey-chart" role="group" aria-label={`${valueLabel} chart; click a point to inspect its transition`}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Complete ${valueLabel.toLowerCase()} trajectory with positive and negative movements`}>
        {[0, .5, 1].map((fraction) => { const yy = pad.top + fraction * (height - pad.top - pad.bottom); const value = max - fraction * displaySpan; return <g key={fraction}><line x1={pad.left} x2={width - pad.right} y1={yy} y2={yy} className="journey-gridline" /><text x={pad.left - 7} y={yy + 3} textAnchor="end" className="journey-axis-text">{numberText(value, 4)}</text></g> })}
        {points.slice(1).map((point, index) => { const prior = points[index]; const rising = point.value >= prior.value; return <line key={point.index} x1={prior.x} y1={prior.y} x2={point.x} y2={point.y} className={rising ? 'journey-segment-up' : 'journey-segment-down'} /> })}
        {points.map((point) => <g key={point.index} className="journey-point" role="button" tabIndex={0} aria-label={`Step ${point.index + 1}, ${valueLabel.toLowerCase()} ${numberText(point.value, 6)}`} onClick={() => onSelectIndex(point.index)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelectIndex(point.index) } }}><circle cx={point.x} cy={point.y} r={point.index === selectedIndex ? 7 : 5} className={point.index === selectedIndex ? 'journey-selected-hit' : 'journey-hit'} /><circle cx={point.x} cy={point.y} r={point.index === selectedIndex ? 3.5 : 1.8} className={point.index === selectedIndex ? 'journey-selected-point' : 'journey-point-core'} /></g>)}
        <text x={pad.left} y={height - 7} className="journey-axis-text">STEP 1</text><text x={width - pad.right} y={height - 7} textAnchor="end" className="journey-axis-text">STEP {path.length}</text>
      </svg>
      <div className="journey-legend"><span className="journey-up-key">UP MOVE</span><span className="journey-down-key">DOWN MOVE</span><span className="journey-selected-key">SELECTED STEP</span></div>
    </div>
    <div className="path-journey-controls"><Button type="button" variant="quiet" disabled={selectedIndex <= 0} onClick={() => onSelectIndex(Math.max(0, selectedIndex - 1))}>← PREVIOUS</Button>
      <label><span>STEP {selectedIndex + 1} / {path.length}</span><input aria-label="Path journey step" type="range" min={0} max={path.length - 1} step={1} value={selectedIndex} onChange={(event) => onSelectIndex(Number(event.target.value))} /></label>
      <Button type="button" variant="quiet" disabled={selectedIndex >= path.length - 1} onClick={() => onSelectIndex(Math.min(path.length - 1, selectedIndex + 1))}>NEXT →</Button>
    </div>
    <div className="path-journey-stats">
      <UniverseStat label={valueLabel} value={numberText(snapshot.currentValue, 7)} detail={`${data.model === 'garch' ? 'period' : 'time'} ${snapshot.time == null ? '—' : numberText(snapshot.time, 5)} · observation ${selectedIndex + 1}`} />
      <UniverseStat label="Previous observation" value={snapshot.previousValue == null ? 'Initial point' : numberText(snapshot.previousValue, 7)} detail={snapshot.absoluteChange == null ? 'No prior value in this path' : `Absolute change ${numberText(snapshot.absoluteChange, 7)}`} />
      <UniverseStat label={pricePath ? 'Step simple return' : 'Step change'} value={snapshot.percentageChange == null ? (snapshot.absoluteChange == null ? '—' : numberText(snapshot.absoluteChange, 7)) : `${(snapshot.percentageChange * 100).toFixed(4)}%`} detail={pricePath ? 'Computed from adjacent positive prices' : data.model === 'garch' ? 'Return observations are not expressed as percentage changes' : 'Change in model state units'} />
      <UniverseStat label="Cumulative simple return" value={snapshot.cumulativeReturn == null ? 'Unavailable' : `${(snapshot.cumulativeReturn * 100).toFixed(4)}%`} detail={data.model === 'garch' ? `${data.parameters.return_convention === 'simple' ? 'Compounded simple' : 'Exponentiated cumulative log'} returns to this period` : pricePath ? 'Selected price relative to initial price' : 'Not defined for this state series'} />
      <UniverseStat label="Path maximum drawdown" value={snapshot.maximumDrawdown == null ? 'Unavailable' : `${(snapshot.maximumDrawdown * 100).toFixed(4)}%`} detail="Running-peak drawdown through selected observation" />
      <UniverseStat label="Reconstruction error" value={selectedError == null ? '—' : numberText(selectedError, 5)} detail={data.reconstruction.verified ? `Stored reconstruction verified · tolerance ${numberText(data.reconstruction.verification_tolerance, 3)}` : 'Stored reconstruction did not verify'} />
    </div>
    <div className="path-journey-transition"><div className="path-journey-subhead"><strong>Stored transition explanation</strong><span>{transition ? `transition ${data.model === 'garch' ? selectedIndex + 1 : selectedIndex}` : 'initial condition · no incoming transition'}</span></div>
      {transition ? <div className="path-journey-detail-grid">{transitionFields.map(([label, key, unit]) => { const value = key === 'previous_value' ? snapshot.previousValue : key === 'previous_conditional_variance' ? data.transitions[selectedIndex - 1]?.next_conditional_variance : transition[key]; return value == null || typeof value === 'boolean' || typeof value !== 'number' ? null : <div key={key}><span>{label}</span><strong>{numberText(value, 7)}</strong><small>{unit}</small></div> })}
        <div><span>Original next value</span><strong>{numberText(original, 7)}</strong><small>{data.model === 'gbm' || data.model === 'jump_diffusion' ? 'price' : data.model === 'garch' ? 'return' : 'state units'}</small></div>
        <div><span>Reconstructed value</span><strong>{numberText(reconstructed, 7)}</strong><small>{data.model === 'gbm' || data.model === 'jump_diffusion' ? 'price' : data.model === 'garch' ? 'return' : 'state units'}</small></div>
        <div><span>Absolute reconstruction error</span><strong>{selectedError == null ? '—' : numberText(selectedError, 5)}</strong><small>{selectedError != null && selectedError <= data.reconstruction.verification_tolerance ? 'within verification tolerance' : 'compare against stored tolerance'}</small></div>
      </div> : <small className="universe-forensic-note">Select a later point to inspect its incoming model transition. The initial point has no previous transition.</small>}
      {groups.length > 0 && <div className="path-journey-contributions">{groups.map((group) => <div className="journey-contribution-group" key={group}><strong>{groupTitles[group]}</strong><small>{group === 'variance' ? 'Variance units are squared return units; these bars are separate from return contributions.' : group === 'log' ? 'These terms combine additively in log space.' : group === 'state' ? 'These terms combine additively in model state units.' : 'These terms combine additively in return units.'}</small>{snapshot.components.filter((component) => component.group === group).map((component) => { const ratio = Math.min(1, Math.abs(component.value) / maxByGroup[group]); return <div className="journey-contribution" key={component.label}><span>{component.label}</span><div className="journey-bar-track"><i className={component.value >= 0 ? 'positive' : 'negative'} style={{ left: component.value >= 0 ? '50%' : `${50 - ratio * 48}%`, width: `${Math.max(ratio * 48, component.value === 0 ? 0 : 1)}%` }} /></div><b className={component.value >= 0 ? 'positive-text' : 'negative-text'}>{numberText(component.value, 7)}</b><small>{component.unit}</small></div>})}</div>)}</div>}
      <small className="universe-forensic-note">{snapshot.modelNote}{data.model === 'garch' && transition?.next_conditional_variance == null ? ' No next variance is stored for the final simulated period.' : ''}</small>
    </div>
  </section>
}

function nearbyCandidates(data: PathForensics): Record<string, number>[] {
  const parameters = data.parameters
  const value = (key: string, fallback: number) => typeof parameters[key] === 'number' ? parameters[key] as number : fallback
  const clamp = (number: number, min: number, max: number) => Math.max(min, Math.min(max, number))
  if (data.model === 'gbm') {
    const drift = value('drift', .08), volatility = value('volatility', .2)
    return [{ drift: clamp(drift - .05, -2, 2) }, { drift: clamp(drift + .05, -2, 2) },
      { volatility: clamp(volatility * .8, 0, 3) }, { volatility: clamp(volatility * 1.2, 0, 3) }]
  }
  if (data.model === 'ou') {
    const theta = value('mean_reversion_speed', 1), volatility = value('volatility', 1)
    return [{ mean_reversion_speed: clamp(theta * .75, 0, 100000) }, { mean_reversion_speed: clamp(theta * 1.25, 0, 100000) },
      { volatility: clamp(volatility * .8, 0, 1e6) }, { volatility: clamp(volatility * 1.2, 0, 1e6) }]
  }
  if (data.model === 'jump_diffusion') {
    const intensity = value('jump_intensity', .5), jumpMean = value('mean_log_jump', -.1)
    return [{ jump_intensity: clamp(intensity * .5, 0, 10) }, { jump_intensity: clamp(intensity * 1.5, 0, 10) },
      { mean_log_jump: clamp(jumpMean - .05, -2, 2) }, { mean_log_jump: clamp(jumpMean + .05, -2, 2) }]
  }
  const alpha = value('alpha', .05), beta = value('beta', .9), omega = value('omega', .000001)
  return [{ alpha: clamp(alpha * .8, 0, 10) }, { alpha: clamp(alpha * 1.2, 0, 10) },
    { beta: clamp(beta - .03, 0, 10) }, { omega: clamp(omega * 1.2, Number.MIN_VALUE, 1) }]
}
