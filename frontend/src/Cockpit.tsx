import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Activity, Box, Camera, ChartNoAxesCombined, CircleHelp, FlaskConical, RotateCcw, ShieldAlert, Waves } from 'lucide-react'
import { api, ApiError } from './api'
import { Button, EmptyState, ErrorNotice, Field, PageHeading, Panel, PanelTitle, SelectField, SuccessNotice } from './components'
import type { ExperimentRecord, GARCHCalibration, GBMCalibration, GBMEvaluation, MertonCalibration, ModelDescription, OUCalibration, SimulationTrajectories } from './types'

const TrajectoryScene = lazy(() => import('./TrajectoryScene'))

const ENDPOINTS: Record<string, string> = {
  gbm: '/simulate-gbm/paths', ou: '/simulate-ou/paths',
  jump_diffusion: '/simulate-jump-diffusion/paths', garch: '/simulate-garch/paths',
}
const DEFAULTS: Record<string, Record<string, string>> = {
  gbm: { initial_price: '100', drift: '0.08', volatility: '0.2', years: '1', steps: '160', paths: '64', seed: '42' },
  ou: { initial_value: '0', mean_level: '0', mean_reversion_speed: '1', volatility: '1', years: '1', steps: '160', paths: '64', seed: '42' },
  jump_diffusion: { initial_price: '100', drift: '0.08', volatility: '0.2', jump_intensity: '0.5', mean_log_jump: '-0.1', jump_volatility: '0.2', years: '1', steps: '160', paths: '64', seed: '42' },
  garch: { omega: '0.000001', alpha: '0.05', beta: '0.9', steps: '160', paths: '64', mean: '0', initial_variance: '', seed: '42', return_convention: 'log' },
}
type Parameter = { key: string; label: string; min?: number; max?: number; step?: string; optional?: boolean }
const FIELDS: Record<string, Parameter[]> = {
  gbm: [
    { key: 'initial_price', label: 'Initial price S₀', min: 0.000001, max: 1e9 },
    { key: 'drift', label: 'Drift μ / year', min: -2, max: 2 },
    { key: 'volatility', label: 'Diffusion σ / √year', min: 0, max: 3 },
    { key: 'years', label: 'Horizon / years', min: 0.000001, max: 10 },
    { key: 'steps', label: 'Time steps', min: 1, max: 512, step: '1' },
    { key: 'paths', label: 'Simulated paths', min: 1, max: 128, step: '1' },
    { key: 'seed', label: 'Random seed', min: 0, max: 4294967295, step: '1' },
  ],
  ou: [
    { key: 'initial_value', label: 'Initial level X₀', min: -1e9, max: 1e9 },
    { key: 'mean_level', label: 'Long-run mean μ', min: -1e9, max: 1e9 },
    { key: 'mean_reversion_speed', label: 'Reversion θ / year', min: 0, max: 100000 },
    { key: 'volatility', label: 'Level σ / √year', min: 0, max: 1e6 },
    { key: 'years', label: 'Horizon / years', min: 0.000001, max: 10 },
    { key: 'steps', label: 'Time steps', min: 1, max: 512, step: '1' },
    { key: 'paths', label: 'Simulated paths', min: 1, max: 128, step: '1' },
    { key: 'seed', label: 'Random seed', min: 0, max: 4294967295, step: '1' },
  ],
  jump_diffusion: [
    { key: 'initial_price', label: 'Initial price S₀', min: 0.000001, max: 1e9 },
    { key: 'drift', label: 'Compensated drift / year', min: -2, max: 2 },
    { key: 'volatility', label: 'Diffusion σ / √year', min: 0, max: 3 },
    { key: 'jump_intensity', label: 'Jump intensity λ / year', min: 0, max: 10 },
    { key: 'mean_log_jump', label: 'Mean log jump m', min: -2, max: 2 },
    { key: 'jump_volatility', label: 'Jump-size σJ', min: 0, max: 2 },
    { key: 'years', label: 'Horizon / years', min: 0.000001, max: 10 },
    { key: 'steps', label: 'Time steps', min: 1, max: 512, step: '1' },
    { key: 'paths', label: 'Simulated paths', min: 1, max: 128, step: '1' },
    { key: 'seed', label: 'Random seed', min: 0, max: 4294967295, step: '1' },
  ],
  garch: [
    { key: 'omega', label: 'ω — variance intercept', min: 0.000000001, max: 1 },
    { key: 'alpha', label: 'α — shock response', min: 0, max: 10 },
    { key: 'beta', label: 'β — variance persistence', min: 0, max: 10 },
    { key: 'mean', label: 'Mean return / period', min: -10, max: 10 },
    { key: 'initial_variance', label: 'Initial variance · optional', min: 0.000000001, max: 100, optional: true },
    { key: 'steps', label: 'Observation periods', min: 1, max: 512, step: '1' },
    { key: 'paths', label: 'Simulated paths', min: 1, max: 128, step: '1' },
    { key: 'seed', label: 'Random seed', min: 0, max: 4294967295, step: '1' },
  ],
}

const MODEL_TITLES: Record<string, string> = { gbm: 'Geometric Brownian Motion', ou: 'Ornstein–Uhlenbeck', jump_diffusion: 'Merton Jump-Diffusion', garch: 'GARCH(1,1)' }
const DEFAULT_VISIBLE = [16, 32, 48, 64, 96, 128]

export function ModelLaboratory({ models, modelsError, ticker: selectedTicker, onTickerChange, experimentToLoad, onExperimentLoaded }: { models: ModelDescription[]; modelsError: string; ticker: string; onTickerChange: (ticker: string) => void; experimentToLoad?: ExperimentRecord | null; onExperimentLoaded?: () => void }) {
  const [model, setModel] = useState('gbm')
  const [parameters, setParameters] = useState<Record<string, string>>(DEFAULTS.gbm)
  const [result, setResult] = useState<SimulationTrajectories | null>(null)
  const [selectedPath, setSelectedPath] = useState<number | null>(null)
  const [currentStep, setCurrentStep] = useState(0)
  const [visiblePathCount, setVisiblePathCount] = useState(48)
  const [flat, setFlat] = useState(false)
  const [resetToken, setResetToken] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [ticker, setTicker] = useState(selectedTicker)
  const [calibrationInterval, setCalibrationInterval] = useState('1d')
  const [calibrationStart, setCalibrationStart] = useState('2021-01-01')
  const [calibrationEnd, setCalibrationEnd] = useState('2024-01-01')
  const [evaluationStart, setEvaluationStart] = useState('2024-01-02')
  const [evaluationEnd, setEvaluationEnd] = useState('2025-01-01')
  const [calibration, setCalibration] = useState<GBMCalibration | null>(null)
  const [calibrationApplied, setCalibrationApplied] = useState(false)
  const [calibrationBusy, setCalibrationBusy] = useState(false)
  const [calibrationError, setCalibrationError] = useState('')
  const [evaluationBusy, setEvaluationBusy] = useState(false)
  const [evaluationError, setEvaluationError] = useState('')
  const [evaluation, setEvaluation] = useState<GBMEvaluation | null>(null)
  const [advancedCalibration, setAdvancedCalibration] = useState<OUCalibration | MertonCalibration | GARCHCalibration | null>(null)
  const [advancedCalibrationApplied, setAdvancedCalibrationApplied] = useState(false)
  const [advancedCalibrationBusy, setAdvancedCalibrationBusy] = useState(false)
  const [advancedCalibrationError, setAdvancedCalibrationError] = useState('')
  const [ouInputSeries, setOuInputSeries] = useState('log_return')
  const [confirmPriceMeanReversion, setConfirmPriceMeanReversion] = useState(false)
  const [calibrationPriceField, setCalibrationPriceField] = useState('auto')
  const [experimentName, setExperimentName] = useState('')
  const [experimentBusy, setExperimentBusy] = useState(false)
  const [experimentMessage, setExperimentMessage] = useState('')
  const [experimentError, setExperimentError] = useState('')
  const [restoredCalibration, setRestoredCalibration] = useState<Record<string, unknown> | null>(null)
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])
  useEffect(() => { setTicker(selectedTicker); setCalibration(null); setAdvancedCalibration(null); setCalibrationApplied(false); setAdvancedCalibrationApplied(false); setEvaluation(null) }, [selectedTicker])
  useEffect(() => {
    if (!experimentToLoad) return
    const record = experimentToLoad
    setModel(record.model)
    const values = { ...(DEFAULTS[record.model] ?? {}) }
    for (const [key, value] of Object.entries(record.parameters)) {
      if (key === 'calibration') continue
      values[key] = value == null ? '' : String(value)
    }
    setParameters(values)
    const provenance = record.parameters.calibration
    setRestoredCalibration(provenance && typeof provenance === 'object' ? provenance as Record<string, unknown> : null)
    setTicker(String(record.metadata.ticker ?? selectedTicker))
    setResult(null); setSelectedPath(null); setCurrentStep(0); setError('')
    setExperimentName(`${record.name} (re-run)`)
    setExperimentMessage('Saved configuration loaded. Run it to regenerate trajectories.')
    onExperimentLoaded?.()
  }, [experimentToLoad])

  const selectedModel = models.find((entry) => entry.name === model)
  const update = (key: string, value: string) => {
    setParameters((old) => ({ ...old, [key]: value }))
    setRestoredCalibration(null)
    if (model === 'gbm' && (key === 'drift' || key === 'volatility')) setCalibrationApplied(false)
    if (model !== 'gbm') setAdvancedCalibrationApplied(false)
  }
  const changeModel = (next: string) => {
    controller.current?.abort()
    setModel(next); setParameters(DEFAULTS[next] ?? {}); setResult(null); setCalibrationApplied(false); setAdvancedCalibrationApplied(false); setRestoredCalibration(null)
    setSelectedPath(null); setCurrentStep(0); setError('')
  }
  const resetDefaults = () => { setParameters(DEFAULTS[model]); setError('') }
  const buildPayload = () => {
    const payload: Record<string, unknown> = {}
    for (const [key, value] of Object.entries(parameters)) {
      if (key === 'return_convention') payload[key] = value
      else if (value === '' && key === 'initial_variance') payload[key] = null
      else payload[key] = Number(value)
    }
    if (restoredCalibration) payload.calibration = restoredCalibration
    else if (model === 'gbm' && calibrationApplied && calibration) {
      payload.calibration = { model: 'gbm', ticker: calibration.ticker,
        estimation_start: calibration.estimation_start, estimation_end: calibration.estimation_end,
        interval: calibration.interval, drift: calibration.drift, volatility: calibration.volatility,
        calibrated_at: calibration.calibrated_at, method: calibration.method,
        observation_count: calibration.observation_count, return_count: calibration.return_count }
    } else if (model !== 'gbm' && advancedCalibrationApplied && advancedCalibration?.model === model) {
      const provenance = { model: advancedCalibration.model, ticker: advancedCalibration.ticker,
        estimation_start: advancedCalibration.estimation_start, estimation_end: advancedCalibration.estimation_end,
        interval: advancedCalibration.interval, calibrated_at: advancedCalibration.calibrated_at,
        method: advancedCalibration.method, observation_count: advancedCalibration.observation_count,
        return_count: advancedCalibration.return_count }
      payload.calibration = advancedCalibration.model === 'ou' ? { ...provenance,
        input_series: advancedCalibration.input_series, mean_level: advancedCalibration.mean_level,
        mean_reversion_speed: advancedCalibration.mean_reversion_speed, volatility: advancedCalibration.volatility,
        initial_value: advancedCalibration.initial_value,
      } : advancedCalibration.model === 'jump_diffusion' ? { ...provenance,
        drift: advancedCalibration.drift, volatility: advancedCalibration.volatility,
        jump_intensity: advancedCalibration.jump_intensity, mean_log_jump: advancedCalibration.mean_log_jump,
        jump_volatility: advancedCalibration.jump_volatility, initial_price: advancedCalibration.initial_price,
      } : { ...provenance, mean: advancedCalibration.mean, omega: advancedCalibration.omega,
        alpha: advancedCalibration.alpha, beta: advancedCalibration.beta,
        initial_variance: advancedCalibration.initial_variance, return_convention: 'log' }
    }
    return payload
  }

  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const count = Number(parameters.paths), steps = Number(parameters.steps)
    if (count > 128 || steps > 512 || count < 1 || steps < 1) {
      setError('Trajectory requests are limited to 128 paths and 512 time steps.'); return
    }
    const endpoint = ENDPOINTS[model]
    if (!endpoint) { setError('No trajectory endpoint exists for this model.'); return }
    const payload = buildPayload()
    controller.current?.abort()
    controller.current = new AbortController()
    setBusy(true); setError(''); setResult(null); setSelectedPath(null); setCurrentStep(0)
    try {
      const response = await api.simulateTrajectories(endpoint, payload, controller.current.signal)
      setResult(response)
      setVisiblePathCount(Math.min(48, response.trajectories.length))
    } catch (reason) {
      if (!(reason instanceof DOMException && reason.name === 'AbortError')) setError(errorMessage(reason))
    } finally { setBusy(false) }
  }

  const saveExperiment = async () => {
    if (!experimentName.trim()) { setExperimentError('Enter an experiment name.'); return }
    setExperimentBusy(true); setExperimentError(''); setExperimentMessage('')
    try {
      const experiment = await api.createExperiment({ name: experimentName.trim(), model: model as SimulationTrajectories['model'],
        ticker, parameters: buildPayload() })
      setResult(experiment.simulation)
      setVisiblePathCount(Math.min(48, experiment.simulation.trajectories.length))
      setExperimentMessage(`Saved and ran ${experiment.experiment.name} · ${experiment.experiment.id.slice(0, 8)} · seed ${experiment.experiment.metadata.random_seed ?? 'unseeded'}.`)
    } catch (reason) { setExperimentError(errorMessage(reason)) }
    finally { setExperimentBusy(false) }
  }

  const requestCalibration = async () => {
    setCalibrationBusy(true); setCalibrationError(''); setCalibration(null); setCalibrationApplied(false); setEvaluation(null)
    try {
      const response = await api.calibrateGBM({ ticker, start: calibrationStart, end: calibrationEnd, interval: calibrationInterval })
      setCalibration(response)
    } catch (reason) { setCalibrationError(errorMessage(reason)) }
    finally { setCalibrationBusy(false) }
  }

  const applyCalibration = () => {
    if (!calibration) return
    setParameters((old) => ({ ...old, drift: String(calibration.drift), volatility: String(calibration.volatility) }))
    setCalibrationApplied(true); setModel('gbm'); setResult(null); setSelectedPath(null)
  }

  const requestEvaluation = async () => {
    if (!calibration) return
    setEvaluationBusy(true); setEvaluationError(''); setEvaluation(null)
    try {
      const response = await api.evaluateGBM({ ticker, estimation_start: calibrationStart, estimation_end: calibrationEnd,
        evaluation_start: evaluationStart, evaluation_end: evaluationEnd, interval: calibration.interval, paths: 512,
        seed: Number(parameters.seed || 42) })
      setEvaluation(response)
    } catch (reason) { setEvaluationError(errorMessage(reason)) }
    finally { setEvaluationBusy(false) }
  }

  const requestAdvancedCalibration = async () => {
    setAdvancedCalibrationBusy(true); setAdvancedCalibrationError(''); setAdvancedCalibration(null); setAdvancedCalibrationApplied(false)
    try {
      const base = { ticker, start: calibrationStart, end: calibrationEnd, interval: calibrationInterval, price_field: 'auto' }
      const result = model === 'ou'
        ? await api.calibrateOU({ ...base, input_series: ouInputSeries, confirm_mean_reverting_price_series: confirmPriceMeanReversion })
        : model === 'jump_diffusion'
          ? await api.calibrateMerton(base)
          : await api.calibrateGARCH(base)
      setAdvancedCalibration(result)
    } catch (reason) { setAdvancedCalibrationError(errorMessage(reason)) }
    finally { setAdvancedCalibrationBusy(false) }
  }

  const applyAdvancedCalibration = () => {
    if (!advancedCalibration || advancedCalibration.model !== model || !advancedCalibration.optimizer_success) return
    if (advancedCalibration.model === 'ou') {
      setParameters((old) => ({ ...old, initial_value: String(advancedCalibration.initial_value), mean_level: String(advancedCalibration.mean_level),
        mean_reversion_speed: String(advancedCalibration.mean_reversion_speed), volatility: String(advancedCalibration.volatility) }))
    } else if (advancedCalibration.model === 'jump_diffusion') {
      setParameters((old) => ({ ...old, initial_price: String(advancedCalibration.initial_price), drift: String(advancedCalibration.drift),
        volatility: String(advancedCalibration.volatility), jump_intensity: String(advancedCalibration.jump_intensity),
        mean_log_jump: String(advancedCalibration.mean_log_jump), jump_volatility: String(advancedCalibration.jump_volatility) }))
    } else {
      setParameters((old) => ({ ...old, mean: String(advancedCalibration.mean), omega: String(advancedCalibration.omega),
        alpha: String(advancedCalibration.alpha), beta: String(advancedCalibration.beta),
        initial_variance: String(advancedCalibration.initial_variance), return_convention: 'log' }))
    }
    setAdvancedCalibrationApplied(true); setResult(null); setSelectedPath(null)
  }

  const availableModels = models.length ? models : Object.keys(ENDPOINTS).map((name) => ({ name, description: '', parameters: [], output_convention: '' }))
  const visibleOptions = result ? DEFAULT_VISIBLE.filter((count) => count <= result.trajectories.length) : [16, 32, 48, 64]
  return <>
    <PageHeading eyebrow="STOCHASTIC RESEARCH / TRAJECTORY SPACE" title="Simulation cockpit" description="Inspect Python-generated paths in 3D, select scenarios, and compare empirical path distributions." aside={<span className="source-tag"><Activity size={12} /> SEEDED BACKEND RUNS</span>} />
    {modelsError && <ErrorNotice>Model registry could not be loaded: {modelsError}. The known model endpoints remain selectable.</ErrorNotice>}
    <div className="cockpit-layout">
      <Panel className="cockpit-controls">
        <PanelTitle title="Scenario controls" subtitle="Model-specific inputs · explicit run" action={<button className="reset-link" type="button" onClick={resetDefaults}><RotateCcw size={12} /> DEFAULTS</button>} />
        <form onSubmit={(event) => void submit(event)}>
          <SelectField label="Stochastic process" value={model} onChange={(event) => changeModel(event.target.value)}>
            {availableModels.map((entry) => <option key={entry.name} value={entry.name}>{MODEL_TITLES[entry.name] ?? entry.name}</option>)}
          </SelectField>
          {selectedModel?.description && <p className="cockpit-description">{selectedModel.description}</p>}
          <div className="cockpit-fields">{(FIELDS[model] ?? []).map((field) => <Field key={field.key} label={field.label} type="number" value={parameters[field.key] ?? ''}
            onChange={(event) => update(field.key, event.target.value)} min={field.min} max={field.max} step={field.step ?? 'any'} required={!field.optional}
            placeholder={field.optional ? 'Model default' : undefined} />)}</div>
          {model === 'garch' && <SelectField label="Return convention" value={parameters.return_convention} onChange={(event) => update('return_convention', event.target.value)}><option value="log">Log return</option><option value="simple">Simple return</option></SelectField>}
          {model === 'ou' && <p className="model-callout">OU is a state or spread process. The vertical axis is a state level, not a stock price.</p>}
          {model === 'jump_diffusion' && <p className="model-callout">Drift is compensated as expected asset growth. σ shown here is diffusion volatility; jumps add return variability.</p>}
          {model === 'garch' && <p className="model-callout">Positive variance requires ω &gt; 0, α, β ≥ 0. Stationarity additionally requires α + β &lt; 1.</p>}
          {model !== 'gbm' && <p className="model-callout">GBM historical calibration does not apply to this model. Its displayed parameters are manual unless this model’s calibration is explicitly applied.</p>}
          {model !== 'gbm' && <div className="calibration-box">
            <div className="calibration-heading"><strong>{MODEL_TITLES[model]} calibration</strong><span>{advancedCalibrationApplied && advancedCalibration?.model === model ? 'CALIBRATED PARAMETERS' : 'MANUAL PARAMETERS'}</span></div>
            <div className="calibration-inputs">
              <Field label="Ticker" value={ticker} onChange={(event) => { const value = event.target.value.toUpperCase(); setTicker(value); onTickerChange(value); setAdvancedCalibration(null); setAdvancedCalibrationApplied(false) }} />
              <SelectField label="Observation interval" value={calibrationInterval} onChange={(event) => { setCalibrationInterval(event.target.value); setAdvancedCalibration(null); setAdvancedCalibrationApplied(false) }}><option value="1d">Daily</option><option value="5d">5 day</option><option value="1wk">Weekly</option><option value="1mo">Monthly</option><option value="3mo">Quarterly</option></SelectField>
              <Field label="Estimation start" type="date" value={calibrationStart} onChange={(event) => { setCalibrationStart(event.target.value); setAdvancedCalibration(null); setAdvancedCalibrationApplied(false) }} />
              <Field label="Estimation end" type="date" value={calibrationEnd} onChange={(event) => { setCalibrationEnd(event.target.value); setAdvancedCalibration(null); setAdvancedCalibrationApplied(false) }} />
              <SelectField label="Historical price field" value={calibrationPriceField} onChange={(event) => { setCalibrationPriceField(event.target.value); setAdvancedCalibration(null); setAdvancedCalibrationApplied(false) }}><option value="auto">Adjusted close, fallback close</option><option value="adj_close">Adjusted close</option><option value="close">Close</option></SelectField>
            </div>
            {model === 'ou' && <>
              <SelectField label="OU state series" value={ouInputSeries} onChange={(event) => { setOuInputSeries(event.target.value); setAdvancedCalibration(null); setAdvancedCalibrationApplied(false) }}><option value="log_return">Close-to-close log returns (per interval)</option><option value="log_price">Log adjusted/close price</option><option value="price">Raw adjusted/close price</option></SelectField>
              {ouInputSeries !== 'log_return' && <label className="model-callout calibration-confirm"><input type="checkbox" checked={confirmPriceMeanReversion} onChange={(event) => { setConfirmPriceMeanReversion(event.target.checked); setAdvancedCalibration(null); setAdvancedCalibrationApplied(false) }} /> I confirm this price level is an appropriate mean-reverting state.</label>}
              <p className="model-callout">Raw equity prices are not assumed mean reverting. Price-level calibration is blocked until you explicitly confirm suitability.</p>
            </>}
            {model === 'jump_diffusion' && <p className="model-callout">Uses historical close-to-close log returns and the simulator’s annual Poisson intensity / normal log-jump convention. Extreme returns are candidate mixture observations, not confirmed jumps.</p>}
            {model === 'garch' && <p className="model-callout">Gaussian conditional MLE on historical log returns. Parameters are per observation interval; fit enforces α + β &lt; 0.999.</p>}
            <Button type="button" variant="quiet" className="calibration-button" busy={advancedCalibrationBusy} onClick={() => void requestAdvancedCalibration()}>CALIBRATE {model.toUpperCase()}</Button>
            {advancedCalibrationError && <ErrorNotice>{advancedCalibrationError}</ErrorNotice>}
            {advancedCalibration?.model === model && <div className="calibration-result">
              <div className="calibration-stats"><span>{advancedCalibration.ticker}</span><span>{advancedCalibration.estimation_start} → {advancedCalibration.estimation_end}</span><span>{advancedCalibration.observation_count.toLocaleString()} observations · {advancedCalibration.return_count.toLocaleString()} returns</span><span>{advancedCalibration.interval}</span></div>
              {advancedCalibration.model === 'ou' && <><div className="calibration-stats"><span>θ {advancedCalibration.mean_reversion_speed.toPrecision(4)} / year</span><span>Long-run level {advancedCalibration.mean_level.toPrecision(5)} ({advancedCalibration.input_units})</span><span>σ {advancedCalibration.volatility.toPrecision(4)} / √year</span></div><small>State: {advancedCalibration.input_series} · initial value {advancedCalibration.initial_value.toPrecision(5)} · AR(1) coefficient {advancedCalibration.ar1_coefficient?.toPrecision?.(4) ?? 'estimated'}</small></>}
              {advancedCalibration.model === 'jump_diffusion' && <><div className="calibration-stats"><span>μ {advancedCalibration.drift.toPrecision(4)} / year</span><span>σ {advancedCalibration.volatility.toPrecision(4)} / √year</span><span>λ {advancedCalibration.jump_intensity.toPrecision(4)} / year</span><span>Jump m {advancedCalibration.mean_log_jump.toPrecision(4)}</span><span>Jump σJ {advancedCalibration.jump_volatility.toPrecision(4)}</span></div><small>Expected jumps {advancedCalibration.estimated_expected_jumps_in_sample.toPrecision(4)} · posterior count {advancedCalibration.posterior_expected_jump_count_in_sample.toPrecision(4)} · optimizer: {advancedCalibration.optimizer_success ? 'converged' : advancedCalibration.optimizer_message}</small></>}
              {advancedCalibration.model === 'garch' && <><div className="calibration-stats"><span>ω {advancedCalibration.omega.toExponential(3)}</span><span>α {advancedCalibration.alpha.toPrecision(4)}</span><span>β {advancedCalibration.beta.toPrecision(4)}</span><span>mean {advancedCalibration.mean.toExponential(3)} / interval</span></div><small>Optimizer: {advancedCalibration.optimizer_success ? 'converged' : advancedCalibration.optimizer_message} · final fitted conditional σ {advancedCalibration.conditional_volatility.at(-1)?.value.toPrecision(4) ?? 'unavailable'} / √interval</small></>}
              {advancedCalibration.warnings.map((warning) => <small className="calibration-warning" key={warning}>{warning}</small>)}
              {advancedCalibration.assumptions.map((assumption) => <small key={assumption}>{assumption}</small>)}
              <Button type="button" variant="quiet" className="calibration-button" disabled={!advancedCalibration.optimizer_success} onClick={applyAdvancedCalibration}>APPLY TO {model.toUpperCase()} CONTROLS</Button>
            </div>}
          </div>}
          {model === 'gbm' && <div className="calibration-box">
            <div className="calibration-heading"><strong>Historical GBM calibration</strong><span>{calibrationApplied ? 'CALIBRATED PARAMETERS' : 'MANUAL PARAMETERS'}</span></div>
            <p className="model-callout">This estimator fits GBM drift and volatility. Calibration controls for OU, Merton jump-diffusion, and GARCH are model-specific; applied provenance is retained in saved experiments.</p>
            <div className="calibration-inputs">
              <Field label="Ticker" value={ticker} onChange={(event) => { const value = event.target.value.toUpperCase(); setTicker(value); onTickerChange(value); setCalibration(null); setAdvancedCalibration(null); setCalibrationApplied(false); setAdvancedCalibrationApplied(false); setEvaluation(null) }} />
              <SelectField label="Observation interval" value={calibrationInterval} onChange={(event) => { setCalibrationInterval(event.target.value); setCalibration(null); setAdvancedCalibration(null); setCalibrationApplied(false); setAdvancedCalibrationApplied(false); setEvaluation(null) }}><option value="1d">Daily</option><option value="5d">5 day</option><option value="1wk">Weekly</option><option value="1mo">Monthly</option><option value="3mo">Quarterly</option></SelectField>
              <Field label="Estimation start" type="date" value={calibrationStart} onChange={(event) => { setCalibrationStart(event.target.value); setCalibration(null); setAdvancedCalibration(null); setCalibrationApplied(false); setAdvancedCalibrationApplied(false); setEvaluation(null) }} />
              <Field label="Estimation end" type="date" value={calibrationEnd} onChange={(event) => { setCalibrationEnd(event.target.value); setCalibration(null); setAdvancedCalibration(null); setCalibrationApplied(false); setAdvancedCalibrationApplied(false); setEvaluation(null) }} />
            </div>
            <Button type="button" variant="quiet" className="calibration-button" busy={calibrationBusy} onClick={() => void requestCalibration()}>ESTIMATE FROM HISTORY</Button>
            {calibrationError && <ErrorNotice>{calibrationError}</ErrorNotice>}
            {calibration && <div className="calibration-result">
              <div className="calibration-stats"><span><b>μ</b> {calibration.drift.toFixed(4)} / year</span><span><b>σ</b> {calibration.volatility.toFixed(4)} / √year</span><span>{calibration.return_count} returns · {calibration.interval}</span></div>
              <small>{calibration.ticker} · {calibration.estimation_start} to {calibration.estimation_end} · {calibration.price_field} · σ uses sample SD (ddof=1), √{calibration.annualization_periods}; μ = mean log return × {calibration.annualization_periods} + ½σ².</small>
              {calibration.assumptions.map((assumption) => <small key={assumption}>{assumption}</small>)}
              {calibration.warnings.map((warning) => <small className="calibration-warning" key={warning}>{warning}</small>)}
              <Button type="button" variant="quiet" className="calibration-button" onClick={applyCalibration}>APPLY μ AND σ</Button>
            </div>}
            <div className="evaluation-box">
              <strong>Chronological holdout comparison</strong>
              <div className="calibration-inputs"><Field label="Evaluation start" type="date" value={evaluationStart} onChange={(event) => { setEvaluationStart(event.target.value); setEvaluation(null) }} /><Field label="Evaluation end" type="date" value={evaluationEnd} onChange={(event) => { setEvaluationEnd(event.target.value); setEvaluation(null) }} /></div>
              <Button type="button" variant="quiet" className="calibration-button" busy={evaluationBusy} disabled={!calibration} onClick={() => void requestEvaluation()}>EVALUATE LATER WINDOW</Button>
              {evaluationError && <ErrorNotice>{evaluationError}</ErrorNotice>}
              {evaluation && <div className="evaluation-result"><small>Estimation: {evaluation.evaluation.estimation_period.start} → {evaluation.evaluation.estimation_period.end} · Holdout: {evaluation.evaluation.evaluation_period.start} → {evaluation.evaluation.evaluation_period.end}</small><div className="calibration-stats"><span>Historical vol {formatPercent(Number(evaluation.evaluation.metrics.historical_annualized_volatility))}</span><span>Simulated vol {formatPercent(Number(evaluation.evaluation.metrics.simulated_annualized_volatility))}</span><span>Observed total return {formatPercent(Number(evaluation.evaluation.metrics.historical_total_return))}</span><span>Historical max DD {formatPercent(Number(evaluation.evaluation.metrics.historical_maximum_drawdown))}</span></div><small>Held-out period log-return quantiles · historical: {evaluationQuantiles(evaluation, 'historical_period_log_return_quantiles')} · simulated: {evaluationQuantiles(evaluation, 'simulated_period_log_return_quantiles')}</small><small>Terminal simple return: one historical holdout outcome {formatPercent(Number(evaluation.evaluation.metrics.historical_total_return))}; historical terminal VaR/ES unavailable (one outcome).</small><small>Simulated terminal VaR/ES 95%: {formatPercent(Number(evaluation.evaluation.metrics.simulated_terminal_var_95))} / {formatPercent(Number(evaluation.evaluation.metrics.simulated_terminal_expected_shortfall_95))} loss over the full held-out horizon; ES averages losses at or above VaR, ties included.</small><small>Simulated terminal-return P05/P50/P95: {evaluationQuantiles(evaluation, 'simulated_terminal_return_quantiles')} · maximum drawdown P05/P50/P95: {evaluationQuantiles(evaluation, 'simulated_maximum_drawdown_quantiles')}</small><small>Descriptive out-of-sample comparison; not a forecast accuracy or profitability claim.</small></div>}
            </div>
          </div>}
          {error && <ErrorNotice>{error}</ErrorNotice>}
          <Button type="submit" className="cockpit-run" busy={busy}><FlaskConical size={15} /> {busy ? 'GENERATING PATHS' : 'RUN SIMULATION'}</Button>
          <div className="experiment-save-box"><Field label="Experiment name" value={experimentName} maxLength={120} placeholder={`${MODEL_TITLES[model]} run`} onChange={(event) => setExperimentName(event.target.value)} />
            <Button type="button" variant="secondary" busy={experimentBusy} onClick={() => void saveExperiment()}>SAVE &amp; RUN EXPERIMENT</Button>
            {experimentError && <ErrorNotice>{experimentError}</ErrorNotice>}{experimentMessage && <SuccessNotice>{experimentMessage}</SuccessNotice>}
            <small>Stores the effective parameters and compact results. Trajectories are regenerated from the saved seed when reopened.</small>
          </div>
          <div className="control-limit"><span>SERVER PATH DATA</span><span>≤128 × 512</span></div>
        </form>
      </Panel>

      <Panel className="universe-panel">
        <PanelTitle title="Simulation universe" subtitle={result ? `${MODEL_TITLES[result.model]} · ${result.trajectories.length} trajectories · ${result.times.length} coordinates` : '3D stochastic trajectory environment'} action={result && <span className="response-tag"><span /> PYTHON RESPONSE</span>} />
        <div className="universe-toolbar">
          <div className="universe-legend"><span><i className="legend-amber" /> SELECTED</span><span><i className="legend-muted" /> OTHER PATHS</span></div>
          <div className="universe-actions">
            {result && <SelectField aria-label="Visible path count" label="Visible" value={String(visiblePathCount)} onChange={(event) => setVisiblePathCount(Number(event.target.value))}>{visibleOptions.map((count) => <option key={count} value={count}>{count} paths</option>)}</SelectField>}
            {result && <Button type="button" variant="quiet" className="compact-button" onClick={() => setFlat((old) => !old)}><Box size={13} /> {flat ? '2D' : '3D'}</Button>}
            {result && <Button type="button" variant="quiet" className="compact-button" onClick={() => setResetToken((old) => old + 1)}><Camera size={13} /> Reset view</Button>}
          </div>
        </div>
        {busy ? <div className="universe-empty"><div className="loading-orbit"><Activity size={20} /></div><strong>Generating trajectories</strong><span>Sampling the selected Python stochastic process…</span></div>
          : result ? <Suspense fallback={<div className="universe-empty"><div className="loading-orbit"><Activity size={20} /></div><strong>Preparing 3D renderer</strong><span>Loading the trajectory visualization module…</span></div>}><TrajectoryScene data={result} selectedIndex={selectedPath} visiblePathCount={visiblePathCount} flat={flat} resetToken={resetToken} onSelect={setSelectedPath} /></Suspense>
            : <div className="universe-empty"><Waves size={22} /><strong>Simulation space is empty</strong><span>Configure a model and run it to load its actual backend trajectories here.</span></div>}
        <div className="coordinate-note"><span>X <b>TIME</b></span><span>Y <b>{result?.quantity_label.toUpperCase() ?? 'MODEL QUANTITY'}</b></span><span>Z <b>{flat ? 'FLAT VIEW' : 'PATH INDEX'}</b></span></div>
      </Panel>

      <PathInspector data={result} selectedPath={selectedPath} onSelect={setSelectedPath} step={currentStep} onStep={setCurrentStep} />
    </div>
    {result && <AnalyticsStrip data={result} />}
    <Panel className="assumption-panel cockpit-assumptions"><PanelTitle title="Model convention" subtitle="Parameter metadata returned with this run" />
      <div className="convention-copy"><span className="eyebrow amber-label">{MODEL_TITLES[model]}</span><p>{selectedModel?.description ?? 'Model description unavailable from registry.'}</p><small>Quantity plotted: {result?.quantity_label ?? selectedModel?.output_convention ?? 'Run a simulation to inspect output metadata.'}</small></div>
      {selectedModel && <div className="parameter-tags">{selectedModel.parameters.map((parameter) => <code key={parameter}>{parameter}</code>)}</div>}
    </Panel>
  </>
}

function PathInspector({ data, selectedPath, onSelect, step, onStep }: {
  data: SimulationTrajectories | null; selectedPath: number | null; onSelect: (value: number | null) => void;
  step: number; onStep: (value: number) => void;
}) {
  const path = data && selectedPath !== null ? data.trajectories[selectedPath] : undefined
  const pathRisk = data && selectedPath !== null ? data.terminal_risk.path_metrics?.[selectedPath] : undefined
  const index = Math.min(step, (data?.times.length ?? 1) - 1)
  return <Panel className="inspector-panel">
    <PanelTitle title="Path inspector" subtitle={selectedPath === null ? 'Run metadata' : `Scenario ${String(selectedPath).padStart(3, '0')}`} action={selectedPath !== null && <span className="selection-mark"><i /> SELECTED</span>} />
    {!data ? <EmptyState icon={<CircleHelp size={17} />} title="No active run">Run a simulation to inspect paths and model-specific diagnostics.</EmptyState> : <>
      <div className="inspector-meta"><span>MODEL</span><strong>{MODEL_TITLES[data.model]}</strong><span>QUANTITY</span><strong>{data.quantity_label}</strong><span>PARAMETER SOURCE</span><strong>{data.parameters.calibration ? `Historical · ${data.parameters.calibration.ticker}` : 'Manual'}</strong>{data.parameters.calibration && <><span>CALIBRATION PERIOD</span><strong>{data.parameters.calibration.estimation_start} → {data.parameters.calibration.estimation_end}</strong></>}<span>PATHS × STEPS</span><strong>{data.trajectories.length} × {data.times.length}</strong><span>SEED</span><strong>{data.parameters.seed ?? 'unfixed'}</strong></div>
      <SelectField label="Select trajectory" value={selectedPath === null ? '' : String(selectedPath)} onChange={(event) => onSelect(event.target.value === '' ? null : Number(event.target.value))}>
        <option value="">Overall run</option>{data.trajectories.map((_, pathIndex) => <option key={pathIndex} value={pathIndex}>Path {String(pathIndex).padStart(3, '0')}</option>)}
      </SelectField>
      {selectedPath === null ? <div className="inspector-overall"><ChartNoAxesCombined size={15} /><span>Select a path in the scene or from the list to inspect its values.</span></div> : path ? <>
        <div className="inspect-value"><span>VALUE AT TIME STEP</span><strong>{formatValue(path[index], data.quantity_kind)}</strong><small>{data.time_label} {formatAxis(data.times[index])}</small></div>
        <label className="step-control"><span>TIME STEP <b>{index + 1} / {path.length}</b></span><input type="range" min={0} max={path.length - 1} value={index} onChange={(event) => onStep(Number(event.target.value))} /></label>
        <div className="inspector-stats"><InspectStat label={data.quantity_kind === 'price' ? 'Initial price' : 'First value'} value={formatValue(path[0], data.quantity_kind)} /><InspectStat label={data.quantity_kind === 'price' ? 'Terminal price' : 'Last value'} value={formatValue(path.at(-1)!, data.quantity_kind)} />
          <InspectStat label="Total simple return · horizon" value={pathRisk ? formatPercent(pathRisk.total_simple_return) : 'N/A'} />
          <InspectStat label="Maximum drawdown" value={pathRisk ? formatPercent(pathRisk.maximum_drawdown) : 'N/A'} negative />
          {data.model === 'garch' && <InspectStat label="Conditional σ / period" value={formatValue(Math.sqrt(data.conditional_variances?.[selectedPath]?.[index] ?? NaN), 'state')} />}
          {data.model === 'gbm' && <InspectStat label="Diffusion σ / √year" value={formatPercent(Number(data.parameters.volatility))} />}
          {data.model === 'jump_diffusion' && <InspectStat label="Diffusion σ / √year" value={formatPercent(Number(data.parameters.volatility))} />}
          {data.model === 'ou' && <InspectStat label="Level σ / √year" value={formatValue(Number(data.parameters.volatility), 'state')} />}
        </div>
        {!pathRisk && <p className="inspector-warning">{data.terminal_risk.unavailable_reason ?? 'Return and drawdown measures are unavailable for this path quantity.'}</p>}
      </> : null}
    </>}
  </Panel>
}

function InspectStat({ label, value, negative = false }: { label: string; value: string; negative?: boolean }) {
  return <div className={`inspect-stat ${negative && value !== 'N/A' ? 'inspect-negative' : ''}`}><span>{label}</span><strong>{value}</strong></div>
}

type ScenarioKey = 'adverse_drift' | 'higher_volatility' | 'jump_stress' | 'lower_ou_mean' | 'garch_mean_shock' | 'garch_shock_response'
const SCENARIO_OPTIONS: Record<string, { key: ScenarioKey; label: string }[]> = {
  gbm: [{ key: 'adverse_drift', label: 'Adverse drift: −10 percentage points / year' }, { key: 'higher_volatility', label: 'Higher volatility: ×1.5' }],
  ou: [{ key: 'lower_ou_mean', label: 'Lower long-run state level: −0.5σ' }],
  jump_diffusion: [{ key: 'adverse_drift', label: 'Adverse drift: −10 percentage points / year' }, { key: 'higher_volatility', label: 'Higher diffusion volatility: ×1.5' }, { key: 'jump_stress', label: 'Jump stress: higher intensity and severity' }],
  garch: [{ key: 'garch_mean_shock', label: 'Adverse mean-return shift: −1% / period' }, { key: 'garch_shock_response', label: 'Higher ARCH shock response: target +0.05 α' }],
}

function scenarioParameters(data: SimulationTrajectories, scenario: ScenarioKey) {
  const parameters: Record<string, unknown> = { ...data.parameters }
  delete parameters.calibration
  if (scenario === 'adverse_drift') parameters.drift = Math.max(-2, Number(parameters.drift) - .1)
  if (scenario === 'higher_volatility') parameters.volatility = Math.min(3, Math.max(.05, Number(parameters.volatility) * 1.5))
  if (scenario === 'jump_stress') {
    parameters.jump_intensity = Math.min(10, Math.max(1, Number(parameters.jump_intensity) * 2))
    parameters.mean_log_jump = Math.max(-2, Number(parameters.mean_log_jump) - .15)
    parameters.jump_volatility = Math.min(2, Math.max(.05, Number(parameters.jump_volatility) * 1.25))
  }
  if (scenario === 'lower_ou_mean') parameters.mean_level = Number(parameters.mean_level) - Math.max(Math.abs(Number(parameters.volatility)) * .5, 1e-6)
  if (scenario === 'garch_mean_shock') parameters.mean = Math.max(-10, Number(parameters.mean) - .01)
  if (scenario === 'garch_shock_response') {
    const alpha = Number(parameters.alpha), beta = Number(parameters.beta)
    const target = Math.min(10, alpha + .05)
    parameters.alpha = alpha + beta < 1 ? Math.min(target, Math.max(0, .999 - beta)) : target
  }
  return parameters
}

function AnalyticsStrip({ data }: { data: SimulationTrajectories }) {
  const [scenario, setScenario] = useState<ScenarioKey>(SCENARIO_OPTIONS[data.model][0].key)
  const [scenarioResult, setScenarioResult] = useState<SimulationTrajectories | null>(null)
  const [scenarioBusy, setScenarioBusy] = useState(false)
  const [scenarioError, setScenarioError] = useState('')
  const scenarioController = useRef<AbortController | null>(null)
  const baseIdentity = `${data.model}:${JSON.stringify(data.parameters)}:${data.times.length}:${data.trajectories.length}`
  useEffect(() => {
    scenarioController.current?.abort()
    setScenarioBusy(false); setScenarioResult(null); setScenarioError(''); setScenario(SCENARIO_OPTIONS[data.model][0].key)
    return () => scenarioController.current?.abort()
  }, [baseIdentity, data.model])
  const terminal = useMemo(() => data.trajectories.map((path) => path.at(-1)!), [data])
  const histogram = useMemo(() => makeHistogram(terminal), [terminal])
  const returnAnalysis = data.terminal_risk.available ? data.terminal_risk : null
  const riskHorizon = data.terminal_risk.horizon_years != null
    ? `${data.terminal_risk.horizon_years} years · ${data.terminal_risk.horizon_steps} steps`
    : `${data.terminal_risk.horizon_steps} observation periods`
  const garchVol = useMemo(() => data.conditional_variances?.[0]?.map((_, step) => ({
    time: data.times[step], volatility: Math.sqrt(data.conditional_variances!.reduce((sum, row) => sum + row[step], 0) / data.conditional_variances!.length),
  })), [data])
  const scenarioDescription = SCENARIO_OPTIONS[data.model].find((item) => item.key === scenario)?.label ?? 'Model parameter perturbation'
  const runScenario = async () => {
    scenarioController.current?.abort()
    const controller = new AbortController()
    scenarioController.current = controller
    setScenarioBusy(true); setScenarioResult(null); setScenarioError('')
    try { setScenarioResult(await api.simulateTrajectories(ENDPOINTS[data.model], scenarioParameters(data, scenario), controller.signal)) }
    catch (error) { if (!(error instanceof DOMException && error.name === 'AbortError')) setScenarioError(errorMessage(error)) }
    finally { if (!controller.signal.aborted) setScenarioBusy(false) }
  }
  return <section className="analytics-strip">
    <div className="analytics-heading"><span>SIMULATION ANALYTICS</span><span>EMPIRICAL · {data.trajectories.length} SCENARIOS</span></div>
    <div className="analytics-grid">
      <Panel className="analytic-panel"><PanelTitle title="Terminal distribution" subtitle={`${data.quantity_label} · last simulation coordinate`} />
        <div className="analytic-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={histogram} margin={{ top: 6, right: 2, bottom: 0, left: -19 }}><CartesianGrid stroke="#282d2c" strokeDasharray="3 5" vertical={false} /><XAxis dataKey="label" tickLine={false} axisLine={false} tick={{ fill: '#817968', fontSize: 8 }} minTickGap={12} /><YAxis tickLine={false} axisLine={false} tick={{ fill: '#817968', fontSize: 8 }} width={36} /><Tooltip formatter={(value) => [`${value} paths`, 'Count']} contentStyle={{ background: '#111514', border: '1px solid #454033', fontSize: 10 }} /><Bar dataKey="count" radius={[1, 1, 0, 0]}>{histogram.map((_, index) => <Cell key={index} fill={index === Math.floor(histogram.length / 2) ? '#d99a3b' : '#766443'} fillOpacity={index === Math.floor(histogram.length / 2) ? .9 : .65} />)}</Bar></BarChart></ResponsiveContainer></div>
        <div className="analytic-caption">Observed terminal {data.quantity_kind === 'price' ? 'prices' : data.quantity_kind === 'state' ? 'states' : 'period returns'}; bins use the returned path sample.</div>
      </Panel>
      <Panel className="analytic-panel"><PanelTitle title="Path quantile envelope" subtitle="P05 / median P50 / P95 · across simulated paths" />
        <div className="analytic-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={data.summary.bands} margin={{ top: 6, right: 5, bottom: 0, left: -15 }}><CartesianGrid stroke="#282d2c" strokeDasharray="3 5" vertical={false} /><XAxis dataKey="time" tickLine={false} axisLine={false} tick={{ fill: '#817968', fontSize: 8 }} minTickGap={35} /><YAxis domain={['auto', 'auto']} tickLine={false} axisLine={false} tick={{ fill: '#817968', fontSize: 8 }} width={42} /><Tooltip labelFormatter={(time) => `${data.time_label}: ${formatAxis(Number(time))}`} contentStyle={{ background: '#111514', border: '1px solid #454033', fontSize: 10 }} /><Line type="monotone" dataKey="p05" name="P05" stroke="#617e7b" strokeWidth={1} dot={false} isAnimationActive={false} /><Line type="monotone" dataKey="p50" name="P50 median" stroke="#f3b64d" strokeWidth={1.8} dot={false} isAnimationActive={false} /><Line type="monotone" dataKey="p95" name="P95" stroke="#798a63" strokeWidth={1} dot={false} isAnimationActive={false} /></LineChart></ResponsiveContainer></div>
        <div className="analytic-caption">Quantiles describe this simulated {data.quantity_kind === 'price' ? 'price' : data.quantity_kind}; not an investment-return percentile.</div>
      </Panel>
      <Panel className="analytic-panel analytic-diagnostics"><PanelTitle title={data.quantity_kind === 'state' ? 'Model diagnostics' : 'Terminal-horizon loss risk'} subtitle={returnAnalysis ? `${(returnAnalysis.confidence_level! * 100).toFixed(0)}% confidence · ${returnAnalysis.scenario_count} simulated paths` : 'Risk measures unavailable for this path quantity'} />
        {!returnAnalysis ? <div className="diagnostic-unavailable">{data.quantity_kind === 'state' ? <Waves size={17} /> : <ShieldAlert size={17} />}<span>{data.terminal_risk.unavailable_reason ?? 'Return risk cannot be computed for these trajectories.'}</span></div> : <div className="risk-estimate-grid"><RiskEstimate label={`VaR ${(returnAnalysis.confidence_level! * 100).toFixed(0)}%`} value={formatPercent(returnAnalysis.var!)} detail="Confidence quantile of terminal loss" /><RiskEstimate label={`Expected Shortfall ${(returnAnalysis.confidence_level! * 100).toFixed(0)}%`} value={formatPercent(returnAnalysis.expected_shortfall!)} detail="Mean loss in empirical worst tail" /><RiskEstimate label="Mean terminal return" value={formatPercent(returnAnalysis.mean_total_simple_return!)} detail={`${returnAnalysis.scenario_count} complete scenarios`} /><div className="risk-estimate-foot">Horizon: {riskHorizon}. Return unit: cumulative simple return; loss L = −R. VaR uses a linearly interpolated empirical quantile; ES averages all losses ≥ VaR, including ties ({returnAnalysis.tail_scenario_count} tail scenarios). These are simulated outputs, not forecasts.</div>{returnAnalysis.warnings.map((warning) => <div className="risk-estimate-foot" key={warning}>{warning}</div>)}</div>}
        {garchVol ? <div className="garch-vol-chart"><div className="chart-subtitle"><strong>Mean conditional volatility</strong><span>√ mean hₜ · per period</span></div><ResponsiveContainer width="100%" height={96}><LineChart data={garchVol}><XAxis dataKey="time" hide /><YAxis hide domain={['auto', 'auto']} /><Tooltip formatter={(value) => formatValue(Number(value), 'state')} contentStyle={{ background: '#111514', border: '1px solid #454033', fontSize: 9 }} /><Line dataKey="volatility" stroke="#70bbc1" strokeWidth={1.4} dot={false} isAnimationActive={false} /></LineChart></ResponsiveContainer></div> : null}
      </Panel>
    </div>
    <Panel className="scenario-analysis-panel"><PanelTitle title="Hypothetical scenario analysis" subtitle="Reruns the selected model through its existing backend simulator" action={<span className="source-tag">{data.parameters.calibration ? 'CALIBRATED BASE' : 'MANUAL BASE'}</span>} />
      <div className="scenario-controls"><SelectField label="Scenario assumption" value={scenario} onChange={(event) => { scenarioController.current?.abort(); setScenarioBusy(false); setScenario(event.target.value as ScenarioKey); setScenarioResult(null); setScenarioError('') }}>{SCENARIO_OPTIONS[data.model].map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}</SelectField><Button type="button" variant="secondary" busy={scenarioBusy} onClick={() => void runScenario()}>RUN HYPOTHETICAL SCENARIO</Button><span>Same path count, horizon, and seed as baseline.</span></div>
      {scenarioError && <ErrorNotice>{scenarioError}</ErrorNotice>}
      {scenarioBusy && <div className="scenario-result-note">Running {data.model} scenario through the existing simulation endpoint…</div>}
      {scenarioResult && <><div className="scenario-result-note"><strong>Scenario:</strong> {scenarioDescription}. Model-based hypothetical change, not a forecast or guaranteed outcome.</div>
        {data.quantity_kind === 'state' ? <div className="scenario-compare-grid"><RiskEstimate label="Baseline terminal state mean" value={formatValue(data.summary.terminal_mean, 'state')} detail={`${data.trajectories.length} state paths`} /><RiskEstimate label="Scenario terminal state mean" value={formatValue(scenarioResult.summary.terminal_mean, 'state')} detail={`${scenarioResult.trajectories.length} state paths`} /><div className="risk-estimate-foot">OU paths are state levels, so VaR, Expected Shortfall, and asset-return drawdown are not defined.</div></div> : returnAnalysis && scenarioResult.terminal_risk.available ? <div className="scenario-compare-grid"><RiskEstimate label="Baseline terminal VaR 95%" value={formatPercent(returnAnalysis.var!)} detail={`${returnAnalysis.scenario_count} paths · ${riskHorizon}`} /><RiskEstimate label="Scenario terminal VaR 95%" value={formatPercent(scenarioResult.terminal_risk.var!)} detail={`${scenarioResult.terminal_risk.scenario_count} paths · same horizon`} /><RiskEstimate label="Baseline terminal ES 95%" value={formatPercent(returnAnalysis.expected_shortfall!)} detail="Empirical worst tail" /><RiskEstimate label="Scenario terminal ES 95%" value={formatPercent(scenarioResult.terminal_risk.expected_shortfall!)} detail="Empirical worst tail" /><RiskEstimate label="Baseline maximum drawdown P50" value={formatPercent(returnAnalysis.maximum_drawdown_quantiles?.['0.5'] ?? NaN)} detail="Across simulated paths" /><RiskEstimate label="Scenario maximum drawdown P50" value={formatPercent(scenarioResult.terminal_risk.maximum_drawdown_quantiles?.['0.5'] ?? NaN)} detail="Across simulated paths" /></div> : <div className="scenario-result-note">Risk comparison unavailable: {scenarioResult.terminal_risk.unavailable_reason ?? data.terminal_risk.unavailable_reason ?? 'the simulated paths did not define valid positive wealth.'}</div>}
      </>}
    </Panel>
  </section>
}

function RiskEstimate({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="risk-estimate"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>
}

function evaluationQuantiles(result: GBMEvaluation, key: string): string {
  const value = result.evaluation.metrics[key]
  if (!value || typeof value !== 'object') return '—'
  return `P05 ${formatPercent(value.p05)} / P50 ${formatPercent(value.p50)} / P95 ${formatPercent(value.p95)}`
}

function makeHistogram(values: number[]) {
  const min = Math.min(...values), max = Math.max(...values)
  const binCount = Math.min(16, Math.max(5, Math.ceil(Math.sqrt(values.length))))
  const width = (max - min) / binCount || 1
  const bins = Array.from({ length: binCount }, (_, index) => ({
    label: formatNumber(min + (index + .5) * width, 1), count: 0,
  }))
  for (const value of values) bins[Math.min(binCount - 1, Math.floor((value - min) / width))].count++
  return bins
}

function formatValue(value: number, kind: SimulationTrajectories['quantity_kind']) {
  if (!Number.isFinite(value)) return '—'
  if (kind === 'price') return `$${formatNumber(value, 3)}`
  if (kind === 'return') return `${(value * 100).toFixed(3)}%`
  return formatNumber(value, 5)
}

function formatPercent(value: number) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '—'
}

function formatNumber(value: number, digits = 3) {
  return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—'
}

function formatAxis(value: number) {
  return Number.isFinite(value) ? value.toPrecision(4) : '—'
}

function errorMessage(error: unknown) {
  if (error instanceof ApiError) return error.message
  return error instanceof Error ? error.message : 'The trajectory request failed unexpectedly.'
}
