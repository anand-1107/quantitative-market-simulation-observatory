import { useEffect, useMemo, useRef, useState } from 'react'
import { Area, Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Activity, FlaskConical, RotateCcw, Save, Sigma } from 'lucide-react'
import { api } from './api'
import { Button, ErrorNotice, Field, PageHeading, Panel, PanelTitle, SelectField } from './components'
import type { CounterfactualPathResult, PathForensics, RetainedForensicsRun, SimulationTrajectories } from './types'
import {
  buildFormulaRequest, deriveGarchPricePaths, FORMULA_FIELDS, FORMULA_MODEL_TITLES,
  FORMULA_TEXT, createFormulaDraft, formulaTrajectoryLabel, parameterDifferences, quantileLinear, representativePathIndices,
  buildCounterfactualParameters, rmse,
  changedSensitivityKeys, clampObservationIndex, formulaObservationSnapshot, observationIndexFromChartEvent,
} from './formula-lab-math'
import type { FormulaDraft, FormulaModel } from './formula-lab-math'

type LabRun = { model: FormulaModel; payload: Record<string, unknown>; data: SimulationTrajectories; garchView: 'returns' | 'price'; displayInitialPrice: number }
type DisplaySeries = { paths: number[][]; times: number[]; values: Array<{ mean: number; p05: number; p50: number; p95: number }> }
const colors = ['#51d5de', '#648af3', '#aa81ec', '#42c990', '#e9ad53', '#48b2d1', '#ba7dc4', '#77b4ff', '#6ab88b', '#d4817d', '#77ccd0', '#9286d6', '#a5c168', '#d5a061', '#57a6a7', '#c27ea0']

const freshDefaults = createFormulaDraft

function finiteMean(values: number[]): number | null { return values.length ? values.reduce((sum, value) => sum + value / values.length, 0) : null }
function sampleDeviation(values: number[]): number | null {
  if (values.length < 2) return null
  const mean = finiteMean(values)!
  return Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1))
}

function displayedSeries(run: LabRun): DisplaySeries {
  const { data } = run
  const invalidPrice = data.model === 'garch' && run.garchView === 'price' && (!Number.isFinite(run.displayInitialPrice) || run.displayInitialPrice <= 0)
  const paths = invalidPrice ? [] : data.model === 'garch' && run.garchView === 'price'
    ? deriveGarchPricePaths(data, run.displayInitialPrice)
    : data.trajectories
  const bands = data.model !== 'garch' || run.garchView === 'returns'
    ? data.summary.bands.map(({ mean, p05, p50, p95 }) => ({ mean, p05, p50, p95 }))
    : paths[0]?.map((_, step) => {
    const crossSection = paths.map((path) => path[step])
    return { mean: finiteMean(crossSection)!, p05: quantileLinear(crossSection, .05),
      p50: quantileLinear(crossSection, .5), p95: quantileLinear(crossSection, .95) }
  }) ?? []
  const times = data.model === 'garch' && run.garchView === 'price'
    ? Array.from({ length: (data.times.length || 0) + 1 }, (_, index) => index)
    : data.times
  return { paths, times, values: bands }
}

function histogram(values: number[], bins = 20) {
  if (!values.length) return []
  const min = Math.min(...values), max = Math.max(...values)
  const span = Math.max(max - min, Math.max(Math.abs(min), Math.abs(max), 1) * 1e-8)
  const width = span / bins
  const result = Array.from({ length: bins }, (_, index) => ({ x: min + (index + .5) * width, count: 0 }))
  for (const value of values) result[Math.min(bins - 1, Math.floor((value - min) / width))].count += 1 / values.length
  return result
}

function terminalOutcomes(run: LabRun): { values: number[]; label: string } {
  if (run.data.model === 'garch') {
    if (run.garchView === 'price') {
      const series = displayedSeries(run)
      return { values: series.paths.map((path) => path.at(-1)!), label: 'Terminal derived price index' }
    }
    const returns = run.data.terminal_risk.path_metrics?.map((metric) => metric.total_simple_return) ?? []
    return { values: returns, label: 'Cumulative simple return over simulated horizon' }
  }
  return { values: run.data.trajectories.map((path) => path.at(-1)!),
    label: run.data.quantity_kind === 'price' ? 'Terminal simulated price' : 'Terminal simulated state' }
}

function sameGrid(first: LabRun, second: LabRun): boolean {
  const a = displayedSeries(first), b = displayedSeries(second)
  return first.model === second.model && first.data.quantity_kind === second.data.quantity_kind &&
    first.garchView === second.garchView && a.times.length === b.times.length &&
    a.times.every((time, index) => Math.abs(time - b.times[index]) < 1e-12)
}

function lineRows(run: LabRun, prefix: string) {
  const series = displayedSeries(run)
  const selected = representativePathIndices(series.paths.length)
  const rows = series.times.map((time, step) => {
    const stats = series.values[step]
    const row: Record<string, number | number[]> = { time, p05: stats?.p05 ?? NaN, p50: stats?.p50 ?? NaN,
      p95: stats?.p95 ?? NaN, bandRange: [stats?.p05 ?? NaN, stats?.p95 ?? NaN] }
    selected.forEach((pathIndex) => { row[`${prefix}path${pathIndex}`] = series.paths[pathIndex]?.[step] ?? NaN })
    return row
  })
  return { rows, selected }
}

function histogramComparison(first: number[], second: number[], bins = 20) {
  const values = [...first, ...second]
  if (!values.length) return []
  const min = Math.min(...values), max = Math.max(...values)
  const width = Math.max(max - min, Math.max(Math.abs(min), Math.abs(max), 1) * 1e-8) / bins
  const rows = Array.from({ length: bins }, (_, index) => ({ x: min + (index + .5) * width, baseline: 0, scenario: 0 }))
  for (const value of first) rows[Math.min(bins - 1, Math.floor((value - min) / width))].baseline += 1 / first.length
  for (const value of second) rows[Math.min(bins - 1, Math.floor((value - min) / width))].scenario += 1 / second.length
  return rows
}

export default function FormulaLab() {
  const [model, setModel] = useState<FormulaModel>('gbm')
  const [draft, setDraft] = useState<FormulaDraft>(freshDefaults('gbm'))
  const [lastChanged, setLastChanged] = useState('')
  const [run, setRun] = useState<LabRun | null>(null)
  const [runPayload, setRunPayload] = useState<Record<string, unknown> | null>(null)
  const [baseline, setBaseline] = useState<LabRun | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [garchView, setGarchView] = useState<'returns' | 'price'>('returns')
  const [displayInitialPrice, setDisplayInitialPrice] = useState('100')
  const [retainedRuns, setRetainedRuns] = useState<RetainedForensicsRun[]>([])
  const [referenceKey, setReferenceKey] = useState('')
  const [forensic, setForensic] = useState<PathForensics | null>(null)
  const [forensicBusy, setForensicBusy] = useState(false)
  const [forensicError, setForensicError] = useState('')
  const [counterfactual, setCounterfactual] = useState<CounterfactualPathResult | null>(null)
  const [counterfactualSubmitted, setCounterfactualSubmitted] = useState<Record<string, number | string | null> | null>(null)
  const [counterfactualBusy, setCounterfactualBusy] = useState(false)
  const [counterfactualError, setCounterfactualError] = useState('')
  const [stepIndex, setStepIndex] = useState(0)
  const [showEnsemble, setShowEnsemble] = useState(false)
  const requestRef = useRef<AbortController | null>(null)
  const forensicRef = useRef<AbortController | null>(null)
  const counterfactualRef = useRef<AbortController | null>(null)
  useEffect(() => () => requestRef.current?.abort(), [])

  useEffect(() => {
    const controller = new AbortController()
    api.retainedForensicsRuns(controller.signal).then(({ runs }) => setRetainedRuns(runs))
      .catch((reason) => { if (!(reason instanceof DOMException && reason.name === 'AbortError')) setForensicError(reason instanceof Error ? reason.message : 'Could not load retained runs.') })
    return () => controller.abort()
  }, [])

  const selectedReference = useMemo(() => {
    const [runId, pathText] = referenceKey.split('|')
    const runInfo = retainedRuns.find((item) => item.run_id === runId)
    const pathId = Number(pathText)
    return runInfo && Number.isInteger(pathId) && pathId >= 1 && pathId <= runInfo.path_count ? { runInfo, pathId } : null
  }, [referenceKey, retainedRuns])

  useEffect(() => {
    forensicRef.current?.abort(); counterfactualRef.current?.abort()
    setForensic(null); setCounterfactual(null); setCounterfactualSubmitted(null); setCounterfactualError(''); setStepIndex(0)
    if (!selectedReference) return
    const controller = new AbortController(); forensicRef.current = controller
    setForensicBusy(true); setForensicError('')
    api.pathForensics(selectedReference.runInfo.run_id, selectedReference.pathId, controller.signal)
      .then((data) => {
        setForensic(data)
        setModel(data.model)
        const next = createFormulaDraft(data.model)
        for (const field of FORMULA_FIELDS[data.model]) {
          const value = data.parameters[field.key]
          if (value === null || value === undefined) continue
          next[field.key] = String(value)
        }
        if (data.model === 'garch' && data.parameters.return_convention) next.return_convention = String(data.parameters.return_convention)
        setDraft(next); setLastChanged('')
      })
      .catch((reason) => { if (!(reason instanceof DOMException && reason.name === 'AbortError')) setForensicError(reason instanceof Error ? reason.message : 'Forensic data unavailable for this path.') })
      .finally(() => { if (forensicRef.current === controller) setForensicBusy(false) })
    return () => controller.abort()
  }, [selectedReference?.runInfo.run_id, selectedReference?.pathId])

  const request = useMemo(() => buildFormulaRequest(model, draft), [model, draft])
  const counterfactualDraft = useMemo(() => buildCounterfactualParameters(model, draft), [model, draft])
  const counterfactualStale = Boolean(counterfactual && counterfactualSubmitted && (counterfactualDraft.error ||
    parameterDifferences(counterfactualSubmitted, counterfactualDraft.parameters).length > 0))
  const dirty = run != null && (model !== run.model || JSON.stringify(request.payload) !== JSON.stringify(runPayload))
  const currentRun = useMemo(() => run ? { ...run, garchView, displayInitialPrice: Number(displayInitialPrice) } : null, [run, garchView, displayInitialPrice])
  const baselineRun = baseline
  const chart = useMemo(() => currentRun ? lineRows(currentRun, 'current') : null, [currentRun])
  const selectedModelTerminal = currentRun ? terminalOutcomes(currentRun) : null
  const currentHistogram = useMemo(() => selectedModelTerminal ? histogram(selectedModelTerminal.values) : [], [currentRun, garchView, displayInitialPrice])
  const comparisonCompatible = Boolean(baselineRun && currentRun && sameGrid(baselineRun, currentRun))
  const comparisonHistogram = useMemo(() => baselineRun && currentRun && comparisonCompatible
    ? histogramComparison(terminalOutcomes(baselineRun).values, terminalOutcomes(currentRun).values) : [],
  [baselineRun, currentRun, comparisonCompatible])
  const comparisonChart = useMemo(() => baselineRun && currentRun && comparisonCompatible
    ? { baseline: displayedSeries(baselineRun), scenario: displayedSeries(currentRun) } : null,
  [baselineRun, currentRun, comparisonCompatible])
  const differences = useMemo(() => baselineRun && currentRun
    ? parameterDifferences({ ...baselineRun.payload, ...(baselineRun.model === 'garch' && baselineRun.garchView === 'price' ? { display_initial_price: baselineRun.displayInitialPrice } : {}) },
      { ...currentRun.payload, ...(currentRun.model === 'garch' && currentRun.garchView === 'price' ? { display_initial_price: currentRun.displayInitialPrice } : {}) }) : [], [baselineRun, currentRun])

  const changeModel = (next: FormulaModel) => {
    requestRef.current?.abort()
    setModel(next); setDraft(freshDefaults(next)); setLastChanged('')
    setRun(null); setRunPayload(null); setError('')
  }
  const update = (key: string, value: string) => { setDraft((old) => ({ ...old, [key]: value })); setLastChanged(key); setError(''); setCounterfactualError('') }
  const submit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (request.error) { setError(request.error); return }
    if (model === 'garch' && garchView === 'price' && (!Number.isFinite(Number(displayInitialPrice)) || Number(displayInitialPrice) <= 0 || Number(displayInitialPrice) > 1e9)) {
      setError('Derived GARCH index starting price must be finite, positive, and no greater than 1,000,000,000.')
      return
    }
    requestRef.current?.abort()
    const controller = new AbortController(); requestRef.current = controller
    setBusy(true); setError('')
    setRun(null); setRunPayload(null)
    try {
      const data = await api.simulateTrajectories(request.endpoint, { ...request.payload, retain_forensics: true }, controller.signal)
      if (requestRef.current !== controller) return
      const startPrice = Number(displayInitialPrice)
      const next: LabRun = { model, payload: { ...request.payload }, data, garchView, displayInitialPrice: startPrice }
      if (model === 'garch' && garchView === 'price') deriveGarchPricePaths(data, startPrice)
      setRun(next); setRunPayload({ ...request.payload }); setLastChanged('')
      if (data.forensics_run_id) {
        setReferenceKey(`${data.forensics_run_id}|1`)
        setRetainedRuns((old) => [{ run_id: data.forensics_run_id!, model, created_at: new Date().toISOString(),
          path_count: data.trajectories.length, step_count: data.times.length, parameters: data.parameters }, ...old.filter((item) => item.run_id !== data.forensics_run_id)])
        try {
          const retained = await api.retainedForensicsRuns(controller.signal)
          if (requestRef.current === controller) setRetainedRuns(retained.runs)
        } catch (reason) {
          if (!(reason instanceof DOMException && reason.name === 'AbortError') && requestRef.current === controller) {
            setForensicError(reason instanceof Error ? `Simulation completed; refreshing retained runs failed: ${reason.message}` : 'Simulation completed; retained-run refresh failed.')
          }
        }
      }
    } catch (reason) {
      if (requestRef.current === controller && !(reason instanceof DOMException && reason.name === 'AbortError')) setError(reason instanceof Error ? reason.message : 'Simulation request failed.')
    } finally { if (requestRef.current === controller) setBusy(false) }
  }
  const saveBaseline = () => { if (run && !dirty) setBaseline({ ...run, garchView, displayInitialPrice: Number(displayInitialPrice) }) }
  const resetParameters = () => { setDraft(freshDefaults(model)); setLastChanged(''); setError(''); setCounterfactualError('') }
  const restoreReferenceParameters = () => {
    if (!forensic) return
    const next = createFormulaDraft(forensic.model)
    for (const field of FORMULA_FIELDS[forensic.model]) {
      const value = forensic.parameters[field.key]
      if (value !== undefined && value !== null) next[field.key] = String(value)
    }
    if (forensic.model === 'garch' && forensic.parameters.return_convention) next.return_convention = String(forensic.parameters.return_convention)
    setModel(forensic.model); setDraft(next); setLastChanged(''); setCounterfactualError('')
  }
  const runCounterfactual = async (parametersOverride?: Record<string, number | string | null>) => {
    if (!forensic || !selectedReference) return
    const built = buildCounterfactualParameters(model, draft)
    const candidateParameters = parametersOverride ?? built.parameters
    if (!parametersOverride && built.error) { setCounterfactualError(built.error); return }
    if (!parametersOverride) {
      const changed = changedSensitivityKeys(model, forensic.parameters, candidateParameters)
      if (changed.length === 0) { setCounterfactualError('No supported model parameter differs from the selected reference. Change one parameter before running a counterfactual.'); return }
      if (changed.length > 1) { setCounterfactualError(`Change one model parameter at a time. Currently changed: ${changed.join(', ')}.`); return }
    }
    const referenceSteps = model === 'garch' ? forensic.original_path.length : forensic.original_path.length - 1
    if (candidateParameters.steps !== undefined && Number(candidateParameters.steps) !== referenceSteps) {
      setCounterfactualError(`Keep the step count at ${referenceSteps} to compare against this stored path.`); return
    }
    counterfactualRef.current?.abort()
    const controller = new AbortController(); counterfactualRef.current = controller
    setCounterfactualBusy(true); setCounterfactualError('')
    try {
      const result = await api.counterfactualPath({ run_id: selectedReference.runInfo.run_id, path_id: selectedReference.pathId, candidate_parameters: candidateParameters }, controller.signal)
      setCounterfactual(result); setCounterfactualSubmitted({ ...candidateParameters }); setStepIndex(0)
    } catch (reason) {
      if (!(reason instanceof DOMException && reason.name === 'AbortError')) setCounterfactualError(reason instanceof Error ? reason.message : 'Counterfactual request failed.')
    } finally { if (counterfactualRef.current === controller) setCounterfactualBusy(false) }
  }
  const replayCounterfactual = () => counterfactual && counterfactualSubmitted && void runCounterfactual({ ...counterfactualSubmitted })

  const fields = FORMULA_FIELDS[model]
  const fieldByKey = Object.fromEntries(fields.map((field) => [field.key, field]))
  const quantityUnit = run?.data.quantity_kind === 'price' ? 'price units' : run?.data.quantity_kind === 'state' ? 'state units' : run?.data.quantity_label ?? 'model units'
  const primaryPaths = chart
  const currentRisk = currentRun?.data.terminal_risk
  const horizon = currentRun ? currentRun.data.terminal_risk.horizon_years != null
    ? `${currentRun.data.terminal_risk.horizon_years} years · ${currentRun.data.terminal_risk.horizon_steps} steps`
    : `${currentRun.data.terminal_risk.horizon_steps ?? currentRun.data.times.length} observation periods` : ''
  const outcomes = selectedModelTerminal?.values ?? []
  const terminalMean = finiteMean(outcomes), terminalSd = sampleDeviation(outcomes)
  const observation = forensic ? formulaObservationSnapshot(forensic.model, forensic.original_path, forensic.times,
    stepIndex, counterfactual?.candidate_path, counterfactual?.times) : null
  const observationCount = observation?.length ?? 0
  const selectedObservation = observation?.index ?? 0
  const transitionIndex = observation?.transitionIndex ?? null
  const referenceTransition = transitionIndex == null ? null : forensic?.transitions[transitionIndex] ?? null
  const candidateTransition = transitionIndex == null ? null : counterfactual?.transitions[transitionIndex] ?? null
  const referenceValue = observation?.currentValue
  const previousReferenceValue = observation?.previousValue ?? null
  const referenceAbsoluteChange = observation?.absoluteChange ?? null
  const referencePercentChange = observation?.percentageChange ?? null
  const referenceTime = observation?.referenceTime
  const candidateTime = observation?.candidateTime
  const timeGridsMatch = observation?.timeGridsMatch ?? false
  const plotTimes = counterfactual?.times ?? forensic?.times ?? []
  const plotLength = Math.min(plotTimes.length, forensic?.original_path.length ?? 0, counterfactual?.candidate_path.length ?? Infinity)
  const pathRows = Array.from({ length: plotLength }, (_, index) => ({
    time: timeGridsMatch || !counterfactual ? plotTimes[index] : index,
    reference: forensic!.original_path[index],
    candidate: counterfactual?.candidate_path[index],
    difference: counterfactual?.differences[index],
  }))
  const activePlotIndex = clampObservationIndex(selectedObservation, pathRows.length)
  const activePlotTime = pathRows[activePlotIndex]?.time
  const selectedCandidateValue = observation?.candidateValue
  const selectedDifference = observation?.difference
  const handlePathChartClick = (event: unknown) => {
    const index = observationIndexFromChartEvent(event, observationCount)
    if (index != null) setStepIndex(index)
  }
  const selectedInnovations = transitionIndex == null || !forensic ? {} : Object.fromEntries(
    Object.entries(forensic.innovations).filter(([, values]) => values[transitionIndex] !== undefined)
      .map(([key, values]) => [key, values[transitionIndex]]))
  const submittedDifferences = forensic && counterfactual
    ? changedSensitivityKeys(forensic.model, forensic.parameters, counterfactual.parameters as Record<string, number | string | null>) : []
  const candidateDraftDiffers = forensic ? changedSensitivityKeys(forensic.model, forensic.parameters, counterfactualDraft.parameters).length > 0 : false
  const displayedRmse = counterfactual && Number.isFinite(counterfactual.raw_path_rmse) ? counterfactual.raw_path_rmse
    : counterfactual ? rmse(counterfactual.reference_path, counterfactual.candidate_path) : NaN
  const metricValue = (metrics: Record<string, unknown> | undefined, key: string) => {
    const value = metrics?.[key]
    return typeof value === 'number' && Number.isFinite(value) ? value : null
  }

  return <div className="formula-lab-page">
    <PageHeading eyebrow="RESEARCH / MODEL MECHANICS" title="Quant Formula Lab" description="Explore how model assumptions map to equations, simulated trajectories, terminal outcomes, and risk. Every displayed simulation comes from the existing Python model API." aside={<span className="formula-source-tag"><Activity size={12} /> BACKEND SIMULATIONS</span>} />
    <div className="formula-lab-layout">
      <Panel className="formula-control-panel">
        <PanelTitle title={forensic ? 'Draft counterfactual parameters' : 'Model & parameters'} subtitle={forensic ? 'Draft inputs are separate from the stored reference configuration.' : 'Edit inputs, then run explicitly to request a new backend simulation.'} action={<button type="button" className="formula-reset" onClick={resetParameters}><RotateCcw size={12} /> DEFAULTS</button>} />
        <form onSubmit={(event) => void submit(event)}>
          <SelectField label="Stochastic model" value={model} onChange={(event) => changeModel(event.target.value as FormulaModel)} disabled={Boolean(forensic) || busy}>
            {(Object.keys(FORMULA_MODEL_TITLES) as FormulaModel[]).map((key) => <option key={key} value={key}>{FORMULA_MODEL_TITLES[key]}</option>)}
          </SelectField>
          <div className="formula-parameter-grid">{fields.map((field) => <label className={`formula-parameter ${lastChanged === field.key ? 'parameter-touched' : ''}`} key={field.key}>
            <span><b>{field.symbol}</b> {field.label}<small>{field.unit}</small></span>
            <input type="number" value={draft[field.key] ?? ''} min={field.min} max={field.max} step={field.step ?? 'any'} required={!field.optional} disabled={busy}
              onChange={(event) => update(field.key, event.target.value)} aria-label={`${field.label} (${field.unit})`} />
            <em>{field.description}</em>
          </label>)}</div>
          {model === 'garch' && <>
            <SelectField label="Return convention" value={draft.return_convention} disabled={busy} onChange={(event) => update('return_convention', event.target.value)}><option value="log">Log returns</option><option value="simple">Simple returns</option></SelectField>
            <SelectField label="Displayed trajectory" value={garchView} disabled={busy} onChange={(event) => setGarchView(event.target.value as 'returns' | 'price')}><option value="returns">Per-period returns (simulator output)</option><option value="price">Derived price index (from returns)</option></SelectField>
            {garchView === 'price' && <Field label="Derived index starting price S₀" type="number" min={0.000001} max={1e9} step="any" value={displayInitialPrice} disabled={busy} onChange={(event) => setDisplayInitialPrice(event.target.value)} />}
            <p className="formula-control-note">GARCH parameters generate per-period returns. The optional price index is a deterministic display transform of those returned values; its starting price is not sent as a GARCH parameter.</p>
          </>}
          {model === 'ou' && <p className="formula-control-note">OU is a state process, usually suited to spreads or transformed variables. Raw equity prices are not automatically treated as mean reverting.</p>}
          {request.error && <ErrorNotice>{request.error}</ErrorNotice>}{error && <ErrorNotice>{error}</ErrorNotice>}
          {busy && <div className="formula-change-note" role="status"><strong>Simulation request in progress</strong><span>The prior completed output was cleared. Results update only after this request succeeds.</span></div>}
          {lastChanged && <div className="formula-change-note"><strong>Modified: {fieldByKey[lastChanged]?.symbol ?? lastChanged}</strong><span>{fieldByKey[lastChanged]?.description ?? 'The display settings changed.'} No new random simulation has run yet.</span></div>}
          <Button type="submit" className="formula-run" busy={busy}><FlaskConical size={15} /> RUN SIMULATION</Button>
        </form>
      </Panel>

      <section className="formula-lab-results">
        <Panel className="formula-live-explorer">
          <PanelTitle title="Live Formula Explorer" subtitle={`${FORMULA_MODEL_TITLES[forensic?.model ?? model]} · actual backend equation and selected-observation values`} action={<span className="formula-model-pill">{forensic ? `OBS ${selectedObservation + 1}` : 'DRAFT'}</span>} />
          <pre className="formula-equation">{FORMULA_TEXT[forensic?.model ?? model]}</pre>
          <div className="formula-reference-configs">
            <div><strong>COUNTERFACTUAL PARAMETERS</strong><pre>{JSON.stringify(counterfactualDraft.error ? { error: counterfactualDraft.error } : counterfactualDraft.parameters, null, 2)}</pre><small>{forensic ? candidateDraftDiffers ? 'Editable draft; changes are applied only by Run Counterfactual.' : 'Draft currently matches the reference configuration.' : 'Select a retained path to load its effective reference parameters.'}</small></div>
            <div><strong>EFFECTIVE REFERENCE PARAMETERS</strong><pre>{JSON.stringify(forensic ? { run_id: forensic.run_id, path_id: forensic.path_id, model: forensic.model, parameters: forensic.parameters } : { unavailable: 'Select a retained path to load its effective configuration.' }, null, 2)}</pre><small>Reference provenance remains immutable while draft parameters change.</small></div>
          </div>
          <SelectField label="Stored run and path" value={referenceKey} disabled={busy} onChange={(event) => setReferenceKey(event.target.value)}>
            <option value="">Select a retained run/path…</option>
            {retainedRuns.map((item) => <optgroup key={item.run_id} label={`${FORMULA_MODEL_TITLES[item.model]} · ${item.created_at} · ${item.path_count} paths`}>
              {Array.from({ length: item.path_count }, (_, index) => index + 1).map((pathId) => <option key={pathId} value={`${item.run_id}|${pathId}`}>Path {pathId} · {item.run_id.slice(0, 8)}</option>)}
            </optgroup>)}
          </SelectField>
          {forensic && <Button type="button" variant="secondary" disabled={counterfactualBusy} onClick={() => setReferenceKey('')}>CLEAR REFERENCE · unlock model controls</Button>}
          {forensic && <div className="formula-counterfactual-actions">
            <Button type="button" className="formula-run" busy={counterfactualBusy} disabled={model !== forensic.model} onClick={() => void runCounterfactual()}><FlaskConical size={14} /> RUN COUNTERFACTUAL</Button>
            <Button type="button" variant="secondary" disabled={counterfactualBusy} onClick={restoreReferenceParameters}><RotateCcw size={13} /> RESET TO REFERENCE</Button>
            <Button type="button" variant="secondary" disabled={!counterfactual || counterfactualBusy} onClick={replayCounterfactual}><RotateCcw size={13} /> REPLAY SAME COMPARISON</Button>
          </div>}
          {forensicError && <ErrorNotice>{forensicError}</ErrorNotice>}
          {counterfactualError && <ErrorNotice>{counterfactualError}</ErrorNotice>}
          {forensicBusy && <p className="formula-control-note">Loading the selected path’s stored provenance and innovations…</p>}
          {!forensic && !forensicBusy && !forensicError && <p className="formula-control-note">Retained runs are listed only when path-level forensic information was saved. Runs without stored innovations cannot be used for a defensible common-innovation comparison.</p>}
          {forensic && <>
            <div className="formula-live-values">
              <FormulaMetric label="Selected observation" value={`${selectedObservation + 1} / ${observationCount}`} note={`${forensic.model === 'garch' ? 'period' : 'time'} ${referenceTime == null ? 'Unavailable' : formatValue(referenceTime)}${candidateTime == null || timeGridsMatch ? '' : ` · candidate time ${formatValue(candidateTime)}`}`} />
              <FormulaMetric label="Previous reference value" value={previousReferenceValue == null ? 'Initial observation' : formatValue(previousReferenceValue)} />
              <FormulaMetric label="Current reference value" value={referenceValue == null ? 'Unavailable' : formatValue(referenceValue)} />
              <FormulaMetric label="Current counterfactual value" value={selectedCandidateValue == null ? 'Not yet available' : formatValue(selectedCandidateValue)} />
              <FormulaMetric label="Reference step change" value={referenceAbsoluteChange == null ? 'Initial observation' : formatValue(referenceAbsoluteChange)} note={referencePercentChange == null ? 'Percentage change unavailable for this model output.' : formatPercent(referencePercentChange)} />
              <FormulaMetric label="Stepwise difference" value={selectedDifference == null ? 'Unavailable until counterfactual runs' : formatValue(selectedDifference)} note="Counterfactual − reference at this observation." />
            </div>
            <details className="formula-forensic-details"><summary>Stored stochastic inputs and path provenance</summary><pre>{JSON.stringify({ parameters: forensic.parameters, initial_conditions: forensic.initial_conditions, innovations: forensic.innovations }, null, 2)}</pre></details>
            {counterfactualStale && <div className="formula-stale-banner"><strong>Displayed comparison is stale</strong><span>Draft parameters changed after the last successful counterfactual. Run it again to evaluate this configuration.</span></div>}
          </>}
        </Panel>

        <Panel className="formula-main-comparison">
          <PanelTitle title="Reference Path vs. Counterfactual Path" subtitle={forensic ? `${FORMULA_MODEL_TITLES[forensic.model]} · shared time axis and value scale` : 'Select a retained reference path in Live Formula Explorer.'} action={<span className="formula-model-pill">{counterfactual ? counterfactualStale ? 'STALE RESULT' : 'COMPARISON' : 'REFERENCE'}</span>} />
          {forensic ? <div className="formula-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={pathRows} onClick={handlePathChartClick}><CartesianGrid stroke="#1d2b31" strokeDasharray="3 5" vertical={false} /><XAxis dataKey="time" tick={{ fill: '#708488', fontSize: 8 }} label={{ value: timeGridsMatch || !counterfactual ? forensic.model === 'garch' ? 'Observation period' : 'Time (years)' : 'Aligned observation index', position: 'insideBottom', offset: -3, fill: '#708488', fontSize: 8 }} /><YAxis domain={['auto', 'auto']} tick={{ fill: '#708488', fontSize: 8 }} width={56} /><Tooltip contentStyle={{ background: '#0c1417', border: '1px solid #284047', fontSize: 9 }} /><Legend /><ReferenceLine x={activePlotTime} stroke="#a4c9d7" strokeDasharray="3 3" /><Line dataKey="reference" name="Stored reference path" stroke="#7190ff" strokeWidth={2} dot={{ r: 2, fill: '#7190ff', onClick: handlePathChartClick }} activeDot={{ r: 5 }} isAnimationActive={false} /><Line dataKey="candidate" name="Counterfactual path" stroke="#f2bd59" strokeWidth={2} dot={{ r: 2, fill: '#f2bd59', onClick: handlePathChartClick }} activeDot={{ r: 5 }} isAnimationActive={false} connectNulls={false} />{pathRows[activePlotIndex]?.reference != null && <ReferenceDot x={activePlotTime} y={pathRows[activePlotIndex].reference} r={6} fill="#7190ff" stroke="#e4f5ff" strokeWidth={2} />}{pathRows[activePlotIndex]?.candidate != null && <ReferenceDot x={activePlotTime} y={pathRows[activePlotIndex].candidate} r={6} fill="#f2bd59" stroke="#fff0cc" strokeWidth={2} />}</LineChart></ResponsiveContainer></div> : <div className="formula-comparison-empty">Select a stored path to display the reference trajectory and compare it with a counterfactual.</div>}
          {forensic && <label className="formula-observation-control">Observation {selectedObservation + 1} / {observationCount}<input aria-label="Selected observation" type="range" min={0} max={Math.max(0, observationCount - 1)} value={selectedObservation} onChange={(event) => setStepIndex(clampObservationIndex(Number(event.target.value), observationCount))} /></label>}
          {forensic && <div className="formula-step-buttons"><button type="button" disabled={selectedObservation <= 0} onClick={() => setStepIndex(clampObservationIndex(selectedObservation - 1, observationCount))}>PREVIOUS</button><button type="button" disabled={selectedObservation >= observationCount - 1} onClick={() => setStepIndex(clampObservationIndex(selectedObservation + 1, observationCount))}>NEXT</button></div>}
        </Panel>

        <Panel className="formula-transition-panel">
          <PanelTitle title="Selected-step transition breakdown" subtitle={forensic ? `Observation ${selectedObservation + 1} · ${transitionIndex == null ? 'initial observation has no incoming transition' : `transition ${transitionIndex + 1}`}` : 'Select a retained path to inspect its actual transition.'} />
          {forensic ? <div className="formula-transition-pair">
            <div><strong>REFERENCE · STORED TRANSITION {transitionIndex == null ? '· INITIAL OBSERVATION' : `· ${transitionIndex + 1}`}</strong><small>Previous value: {previousReferenceValue == null ? 'none' : formatValue(previousReferenceValue)}</small><small>Stored innovation: {transitionIndex == null ? 'none' : JSON.stringify(selectedInnovations)}</small><pre>{referenceTransition ? JSON.stringify(referenceTransition, null, 2) : 'Initial observation has no incoming transition.'}</pre><small>Next value: {referenceTransition ? formatValue(Number(referenceTransition.value ?? referenceTransition.return)) : 'not applicable'} · observed: {referenceValue == null ? 'Unavailable' : formatValue(referenceValue)} · reconstruction error: {transitionIndex == null ? 'not applicable' : formatValue(Math.abs((referenceValue ?? 0) - forensic.reconstruction.path[selectedObservation]))}</small></div>
            <div><strong>{counterfactual ? counterfactualStale ? 'COUNTERFACTUAL · LAST SUCCESSFUL CONFIGURATION (STALE)' : 'COUNTERFACTUAL · LAST SUCCESSFUL CONFIGURATION' : 'COUNTERFACTUAL · NOT YET EVALUATED'}</strong><small>Previous reference value: {previousReferenceValue == null ? 'none' : formatValue(previousReferenceValue)}</small><small>Stored inputs are coupled according to the backend method.</small><pre>{candidateTransition ? JSON.stringify(candidateTransition, null, 2) : 'Run a counterfactual to calculate this transition.'}</pre><small>Next value: {candidateTransition ? formatValue(Number(candidateTransition.value ?? candidateTransition.return)) : 'Unavailable'} · observed candidate: {selectedCandidateValue == null ? 'Unavailable' : formatValue(selectedCandidateValue)}</small></div>
          </div> : <p className="formula-control-note">Select a path to view transition components and stored stochastic inputs.</p>}
        </Panel>

        <Panel className="formula-difference-panel">
          <PanelTitle title="Stepwise Difference" subtitle="Counterfactual value − reference value · model value units" />
          {counterfactual && pathRows.length ? <div className="formula-chart formula-difference-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={pathRows} onClick={handlePathChartClick}><CartesianGrid stroke="#1d2b31" strokeDasharray="3 5" vertical={false} /><XAxis dataKey="time" tick={{ fill: '#708488', fontSize: 8 }} label={{ value: timeGridsMatch ? (forensic?.model === 'garch' ? 'Observation period' : 'Time (years)') : 'Aligned observation index', position: 'insideBottom', offset: -3, fill: '#708488', fontSize: 8 }} /><YAxis domain={['auto', 'auto']} tick={{ fill: '#708488', fontSize: 8 }} width={68} label={{ value: 'Counterfactual − reference (model units)', angle: -90, position: 'insideLeft', fill: '#708488', fontSize: 8 }} /><Tooltip contentStyle={{ background: '#0c1417', border: '1px solid #284047', fontSize: 9 }} /><ReferenceLine y={0} stroke="#53696c" /><ReferenceLine x={activePlotTime} stroke="#a4c9d7" strokeDasharray="3 3" /><Line dataKey="difference" name="Counterfactual − reference" stroke="#55d1c3" strokeWidth={1.5} dot={false} activeDot={{ r: 5 }} isAnimationActive={false} />{pathRows[activePlotIndex]?.difference != null && <ReferenceDot x={activePlotTime} y={pathRows[activePlotIndex].difference} r={6} fill="#55d1c3" stroke="#e4fff6" strokeWidth={2} />}</LineChart></ResponsiveContainer></div> : <div className="formula-comparison-empty">No valid counterfactual is available yet. Select a stored path and run a counterfactual to calculate stepwise differences.</div>}
        </Panel>

        <Panel className="formula-summary-panel">
          <PanelTitle title="Summary comparison metrics" subtitle={counterfactual ? 'Single-path counterfactual results · conditioned on the selected stored path' : 'Metrics appear after a valid counterfactual is calculated.'} />
          {counterfactual ? <>
            <p className="formula-coupling-note">{counterfactual.exact_common_innovations ? 'Common stored stochastic inputs were reused where valid.' : 'Exact common-innovation replay is unavailable for this comparison.'} {counterfactual.coupling_method}{counterfactual.warnings.length > 0 ? ` ${counterfactual.warnings.join(' ')}` : ''}</p>
            {!timeGridsMatch && <p className="formula-warning">The reference and candidate time grids differ. Both paths are aligned by observation index; each selected time is shown in the Formula Explorer.</p>}
            {submittedDifferences.length > 0 && <div className="formula-difference-table"><div className="formula-difference-head"><span>Changed parameter</span><span>Reference</span><span>Counterfactual</span></div>{submittedDifferences.map((key) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><code>{String(forensic?.parameters[key] ?? 'default')}</code><code>{String(counterfactual.parameters[key] ?? 'default')}</code></div>)}</div>}
            <div className="formula-reference-facts">
              <span>REFERENCE TERMINAL <b>{formatValue(counterfactual.reference_path.at(-1) ?? NaN)}</b></span><span>COUNTERFACTUAL TERMINAL <b>{formatValue(counterfactual.candidate_path.at(-1) ?? NaN)}</b></span><span>PATH RMSE <b>{formatValue(displayedRmse)}</b></span><span>TERMINAL CHANGE <b>{formatValue((counterfactual.candidate_path.at(-1) ?? 0) - (counterfactual.reference_path.at(-1) ?? 0))}</b></span>
              <span>REFERENCE RETURN <b>{metricValue(counterfactual.reference_metrics, 'total_simple_return') == null ? 'Unavailable' : formatPercent(metricValue(counterfactual.reference_metrics, 'total_simple_return')!)}</b></span><span>COUNTERFACTUAL RETURN <b>{metricValue(counterfactual.candidate_metrics, 'total_simple_return') == null ? 'Unavailable' : formatPercent(metricValue(counterfactual.candidate_metrics, 'total_simple_return')!)}</b></span><span>RETURN DIFFERENCE <b>{metricValue(counterfactual.candidate_metrics, 'total_simple_return') == null || metricValue(counterfactual.reference_metrics, 'total_simple_return') == null ? 'Unavailable' : formatPercent(metricValue(counterfactual.candidate_metrics, 'total_simple_return')! - metricValue(counterfactual.reference_metrics, 'total_simple_return')!)}</b></span>
              <span>REFERENCE MAX DRAWDOWN <b>{metricValue(counterfactual.reference_metrics, 'maximum_drawdown') == null ? 'Unavailable' : formatPercent(metricValue(counterfactual.reference_metrics, 'maximum_drawdown')!)}</b></span><span>COUNTERFACTUAL MAX DRAWDOWN <b>{metricValue(counterfactual.candidate_metrics, 'maximum_drawdown') == null ? 'Unavailable' : formatPercent(metricValue(counterfactual.candidate_metrics, 'maximum_drawdown')!)}</b></span><span>DRAWDOWN DIFFERENCE <b>{metricValue(counterfactual.candidate_metrics, 'maximum_drawdown') == null || metricValue(counterfactual.reference_metrics, 'maximum_drawdown') == null ? 'Unavailable' : formatPercent(metricValue(counterfactual.candidate_metrics, 'maximum_drawdown')! - metricValue(counterfactual.reference_metrics, 'maximum_drawdown')!)}</b></span>
            </div>
            <p className="formula-control-note">These are counterfactual model results conditioned on one stored path, not real-world predictions or guaranteed profitable settings. The original reference remains unchanged.</p>
          </> : <p className="formula-control-note">Terminal value, return, drawdown, and RMSE comparisons will be shown here after a valid counterfactual run.</p>}
        </Panel>
        <div className="formula-ensemble-toggle"><label><input type="checkbox" checked={showEnsemble} onChange={(event) => setShowEnsemble(event.target.checked)} /> Show optional ensemble simulation workspace</label></div>
        {showEnsemble && <>
        <Panel className="formula-equation-panel">
          <PanelTitle title="Implemented model equations" subtitle={`${FORMULA_MODEL_TITLES[model]} · equations match the selected backend simulator`} action={<span className="formula-model-pill">{model.toUpperCase()}</span>} />
          <pre className="formula-equation">{FORMULA_TEXT[model]}</pre>
          <div className="formula-values">{fields.filter((field) => draft[field.key] !== undefined).map((field) => <div key={field.key}><span>{field.symbol} · {field.label}</span><strong>{draft[field.key] === '' ? 'model default' : `${draft[field.key]} ${field.unit}`}</strong></div>)}</div>
          {model === 'jump_diffusion' && <div className="formula-equation-note">The compensator κ = E[J]−1 offsets expected jump growth so μ remains the expected proportional price growth rate. Each time step uses a Poisson count and the simulator’s aggregate normal jump draw.</div>}
          {model === 'garch' && <div className="formula-equation-note">ω &gt; 0 and α, β ≥ 0 keep finite-step variances positive. Covariance stationarity additionally requires α + β &lt; 1; the existing simulator permits nonstationary parameters for finite horizons.</div>}
          {model === 'ou' && <div className="formula-equation-note">The discrete transition is exact for the OU process. The diffusion factor uses its θ = 0 limit when mean reversion is zero.</div>}
        </Panel>

        {run ? <>
          {dirty && <div className="formula-stale-banner"><strong>Configuration modified · not simulated</strong><span>Charts and metrics below still describe the last completed run. Run simulation to update them.</span></div>}
          {model === 'garch' && garchView === 'price' && (!Number.isFinite(Number(displayInitialPrice)) || Number(displayInitialPrice) <= 0 || Number(displayInitialPrice) > 1e9) && <div className="formula-warning">Enter a finite starting price in (0, 1,000,000,000] to display the derived GARCH index.</div>}
          <Panel className="formula-output-panel"><PanelTitle title={formulaTrajectoryLabel(run.model, run.data.quantity_kind, garchView)} subtitle={`${run.data.trajectories.length} backend paths · ${horizon} · showing ${primaryPaths?.selected.length ?? 0} representative paths plus ensemble quantiles`} action={<span className="formula-model-pill">SEED {run.data.parameters.seed ?? 'UNSEEDED'}</span>} />
            {primaryPaths && <div className="formula-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={primaryPaths.rows} margin={{ top: 9, right: 12, bottom: 0, left: -8 }}>
              <CartesianGrid stroke="#1d2b31" strokeDasharray="3 5" vertical={false} /><XAxis dataKey="time" tick={{ fill: '#708488', fontSize: 8 }} tickLine={false} axisLine={false} minTickGap={30} /><YAxis domain={['auto', 'auto']} tick={{ fill: '#708488', fontSize: 8 }} tickLine={false} axisLine={false} width={56} /><Tooltip contentStyle={{ background: '#0c1417', border: '1px solid #284047', fontSize: 9 }} labelFormatter={(time) => `${run.data.time_label}: ${Number(time).toPrecision(4)}`} />
              <Area dataKey="bandRange" name="5th–95th percentile band" stroke="none" fill="#45b7bd" fillOpacity={.13} isAnimationActive={false} />
              {primaryPaths.selected.map((pathIndex, index) => <Line key={pathIndex} dataKey={`currentpath${pathIndex}`} name={`Path ${pathIndex + 1}`} stroke={colors[index % colors.length]} strokeOpacity={.52} strokeWidth={1} dot={false} isAnimationActive={false} connectNulls={false} />)}
              <Line dataKey="p05" name="P05" stroke="#658d8c" strokeWidth={1} dot={false} isAnimationActive={false} /><Line dataKey="p50" name="Ensemble median" stroke="#f2bd59" strokeWidth={2} dot={false} isAnimationActive={false} /><Line dataKey="p95" name="P95" stroke="#658d8c" strokeWidth={1} dot={false} isAnimationActive={false} />
            </LineChart></ResponsiveContainer></div>}
            <div className="formula-chart-foot">{run.model === 'garch' && garchView === 'price' ? 'Deterministic price index derived by compounding backend GARCH returns; the backend still defines and reports risk on returns.' : `Y = ${run.data.quantity_label} · X = ${run.data.time_label}. Path lines are individual stochastic scenarios; amber is the ensemble median.`}</div>
          </Panel>

          <div className="formula-analytics-grid">
            <Panel><PanelTitle title="Terminal outcome distribution" subtitle={`${selectedModelTerminal?.label ?? 'Terminal model quantity'} · ${outcomes.length} paths`} />
              {currentHistogram.length ? <div className="formula-histogram"><ResponsiveContainer width="100%" height="100%"><BarChart data={currentHistogram} margin={{ top: 5, right: 5, bottom: 0, left: -14 }}><CartesianGrid stroke="#1d2b31" vertical={false} /><XAxis dataKey="x" tick={{ fill: '#708488', fontSize: 8 }} tickLine={false} axisLine={false} tickFormatter={(value) => selectedModelTerminal?.label.toLowerCase().includes('return') ? `${(Number(value) * 100).toFixed(1)}%` : Number(value).toPrecision(3)} /><YAxis tick={{ fill: '#708488', fontSize: 8 }} tickLine={false} axisLine={false} width={40} /><Tooltip contentStyle={{ background: '#0c1417', border: '1px solid #284047', fontSize: 9 }} formatter={(value) => [`${(Number(value) * 100).toFixed(2)}% of paths`, 'Empirical frequency']} labelFormatter={(value) => `Outcome ${selectedModelTerminal?.label.toLowerCase().includes('return') ? `${(Number(value) * 100).toFixed(3)}%` : Number(value).toPrecision(5)}`} /><Bar dataKey="count" name="Empirical frequency" fill="#52bac5" fillOpacity={.78} isAnimationActive={false} /></BarChart></ResponsiveContainer></div> : <p className="formula-control-note">The API did not return a terminal outcome sample suitable for this distribution.</p>}
            </Panel>
            <Panel><PanelTitle title="Run summary & risk" subtitle={currentRisk?.available ? `${(currentRisk.confidence_level! * 100).toFixed(0)}% confidence · ${horizon}` : 'Risk only shown when defined by the simulated quantity'} />
              <div className="formula-metric-grid">
                <FormulaMetric label={`Terminal mean (${run.model === 'garch' && garchView === 'price' ? 'price units' : quantityUnit})`} value={terminalMean == null ? '—' : formatValue(terminalMean)} />
                <FormulaMetric label="Terminal P05 / P50 / P95" value={outcomes.length ? `${formatValue(quantileLinear(outcomes, .05))} / ${formatValue(quantileLinear(outcomes, .5))} / ${formatValue(quantileLinear(outcomes, .95))}` : 'Unavailable'} note="Empirical linearly interpolated quantiles." />
                <FormulaMetric label="Terminal sample SD" value={terminalSd == null ? '—' : formatValue(terminalSd)} note="Across terminal outcomes; not annualized." />
                <FormulaMetric label="Model volatility scale" value={modelVolatility(currentRun!)} note={modelVolatilityLabel(currentRun!)} />
                <FormulaMetric label="Mean cumulative simple return" value={currentRisk?.mean_total_simple_return == null ? 'Unavailable' : formatPercent(currentRisk.mean_total_simple_return)} note="Full simulated horizon." />
                <FormulaMetric label={`VaR ${currentRisk?.confidence_level == null ? '' : `${(currentRisk.confidence_level * 100).toFixed(0)}%`}`} value={currentRisk?.var == null ? 'Unavailable' : formatPercent(currentRisk.var)} note="Quantile of terminal loss, L = −R." />
                <FormulaMetric label="Expected Shortfall" value={currentRisk?.expected_shortfall == null ? 'Unavailable' : formatPercent(currentRisk.expected_shortfall)} note={currentRisk ? `Worst-tail mean · ${currentRisk.tail_scenario_count} scenarios.` : 'Not defined for this output.'} />
                <FormulaMetric label="Median maximum drawdown" value={currentRisk?.maximum_drawdown_quantiles?.['0.5'] == null ? 'Unavailable' : formatPercent(currentRisk.maximum_drawdown_quantiles['0.5'])} note="Across positive wealth paths." />
              </div>
              {currentRisk?.warnings.map((warning) => <small className="formula-warning" key={warning}>{warning}</small>)}
              {!currentRisk?.available && <small className="formula-warning">{currentRisk?.unavailable_reason ?? 'This quantity does not define terminal return risk.'}</small>}
              <p className="formula-risk-note">VaR and ES use the backend’s terminal-horizon simple-return loss convention and empirical quantiles ({currentRisk?.quantile_method ?? 'unavailable'}; {currentRisk?.tail_convention ?? 'tail convention unavailable'}). GARCH risk applies to the cumulative return horizon; OU state paths do not define asset loss risk.</p>
            </Panel>
          </div>
          <Panel className="formula-baseline-panel"><PanelTitle title="Baseline versus scenario" subtitle="Session-only comparison; no persistence is created." action={<Button type="button" variant="secondary" disabled={!run || Boolean(dirty)} onClick={saveBaseline}><Save size={13} /> SAVE COMPLETED RUN AS BASELINE</Button>} />
            {!baselineRun ? <p className="formula-control-note">Run a simulation, then save it as a baseline. Change one or more assumptions and run again to compare the resulting backend output.</p> : !currentRun ? <p className="formula-control-note">Baseline retained for this page session. Run a scenario to compare it.</p> : <>
              <div className="formula-comparison-settings"><span>BASELINE · {FORMULA_MODEL_TITLES[baselineRun.model]} · seed {baselineRun.data.parameters.seed ?? 'unseeded'} · {baselineRun.data.trajectories.length} paths</span><span>SCENARIO · {FORMULA_MODEL_TITLES[currentRun.model]} · seed {currentRun.data.parameters.seed ?? 'unseeded'} · {currentRun.data.trajectories.length} paths</span></div>
              {differences.length ? <div className="formula-difference-table"><div className="formula-difference-head"><span>Parameter</span><span>Baseline</span><span>Scenario</span></div>{differences.map((difference) => <div key={difference.key}><span>{difference.key.replaceAll('_', ' ')}</span><code>{String(difference.baseline ?? 'default')}</code><code>{String(difference.scenario ?? 'default')}</code></div>)}</div> : <p className="formula-control-note">Model parameters are equal; differences in results can still arise if either run was unseeded.</p>}
              {comparisonCompatible && comparisonChart ? <div className="formula-compare-grid">
                <div><strong>Median trajectory comparison</strong><div className="formula-chart formula-compare-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={comparisonChart.baseline.times.map((time, index) => ({ time,
                  baseline: comparisonChart.baseline.values[index]?.p50, scenario: comparisonChart.scenario.values[index]?.p50 }))}>
                  <CartesianGrid stroke="#1d2b31" strokeDasharray="3 5" vertical={false} /><XAxis dataKey="time" tick={{ fill: '#708488', fontSize: 8 }} /><YAxis domain={['auto', 'auto']} tick={{ fill: '#708488', fontSize: 8 }} width={50} /><Tooltip contentStyle={{ background: '#0c1417', border: '1px solid #284047', fontSize: 9 }} /><Legend />
                  <Line dataKey="baseline" name="Baseline median" stroke="#728cff" dot={false} isAnimationActive={false} /><Line dataKey="scenario" name="Scenario median" stroke="#f2bd59" dot={false} isAnimationActive={false} />
                </LineChart></ResponsiveContainer></div></div>
                <div><strong>{terminalOutcomes(baselineRun).label} comparison</strong><div className="formula-histogram formula-compare-hist"><ResponsiveContainer width="100%" height="100%"><BarChart data={comparisonHistogram}><CartesianGrid stroke="#1d2b31" vertical={false} /><XAxis dataKey="x" tick={{ fill: '#708488', fontSize: 7 }} tickFormatter={(value) => Number(value).toPrecision(3)} /><YAxis tick={{ fill: '#708488', fontSize: 8 }} /><Tooltip contentStyle={{ background: '#0c1417', border: '1px solid #284047', fontSize: 9 }} /><Legend /><Bar dataKey="baseline" name="Baseline frequency" fill="#728cff" fillOpacity={.55} isAnimationActive={false} /><Bar dataKey="scenario" name="Scenario frequency" fill="#f2bd59" fillOpacity={.72} isAnimationActive={false} /></BarChart></ResponsiveContainer></div></div>
              </div> : <p className="formula-warning">Path/distribution overlays require the same model, output kind, display convention, and time grid. Summary risk values remain tied to each run’s returned metrics.</p>}
              <div className="formula-risk-comparison"><FormulaMetric label="Baseline terminal P05 / P50 / P95" value={quantileSummary(terminalOutcomes(baselineRun).values)} /><FormulaMetric label="Scenario terminal P05 / P50 / P95" value={quantileSummary(terminalOutcomes(currentRun).values)} /><FormulaMetric label="Baseline mean cumulative return" value={baselineRun.data.terminal_risk.mean_total_simple_return == null ? 'Unavailable' : formatPercent(baselineRun.data.terminal_risk.mean_total_simple_return)} /><FormulaMetric label="Scenario mean cumulative return" value={currentRun.data.terminal_risk.mean_total_simple_return == null ? 'Unavailable' : formatPercent(currentRun.data.terminal_risk.mean_total_simple_return)} /><FormulaMetric label="Baseline terminal VaR" value={baselineRun.data.terminal_risk.var == null ? 'Unavailable' : formatPercent(baselineRun.data.terminal_risk.var)} /><FormulaMetric label="Scenario terminal VaR" value={currentRun.data.terminal_risk.var == null ? 'Unavailable' : formatPercent(currentRun.data.terminal_risk.var)} /><FormulaMetric label="Baseline expected shortfall" value={baselineRun.data.terminal_risk.expected_shortfall == null ? 'Unavailable' : formatPercent(baselineRun.data.terminal_risk.expected_shortfall)} /><FormulaMetric label="Scenario expected shortfall" value={currentRun.data.terminal_risk.expected_shortfall == null ? 'Unavailable' : formatPercent(currentRun.data.terminal_risk.expected_shortfall)} /><FormulaMetric label="Baseline median max drawdown" value={baselineRun.data.terminal_risk.maximum_drawdown_quantiles?.['0.5'] == null ? 'Unavailable' : formatPercent(baselineRun.data.terminal_risk.maximum_drawdown_quantiles['0.5'])} /><FormulaMetric label="Scenario median max drawdown" value={currentRun.data.terminal_risk.maximum_drawdown_quantiles?.['0.5'] == null ? 'Unavailable' : formatPercent(currentRun.data.terminal_risk.maximum_drawdown_quantiles['0.5'])} /></div>
              <small className="formula-warning">The same seed improves controlled comparisons within a model, but model implementations may consume random numbers differently. Matching seeds do not guarantee path-by-path matched innovations across models.</small>
            </>}
          </Panel>
        </> : <Panel className="formula-empty"><Sigma size={24} /><strong>Ready to explore {FORMULA_MODEL_TITLES[model]}</strong><span>Review the implemented equation, edit assumptions, then run to load actual Python-generated paths and risk metrics.</span></Panel>}
        </>}
      </section>
    </div>
    <Panel className="formula-honesty-panel"><PanelTitle title="Scientific interpretation" subtitle="Conditional model scenarios are not guaranteed market predictions." />
      <div><span>Drift and volatility are model inputs or estimates, not known future values.</span><span>Simulated outcomes depend on stochastic innovations and the model specification.</span><span>Matching historical outcomes does not establish causal truth.</span><span>Higher expected return does not imply a guaranteed profit.</span><span>Choosing parameters after repeated favourable sweeps can create selection bias.</span></div>
    </Panel>
  </div>
}

function FormulaMetric({ label, value, note }: { label: string; value: string; note?: string }) {
  return <div className="formula-metric"><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</div>
}

function formatValue(value: number): string { return Number.isFinite(value) ? value.toPrecision(5) : 'Unavailable' }
function formatPercent(value: number): string { return Number.isFinite(value) ? `${(value * 100).toFixed(3)}%` : 'Unavailable' }
function quantileSummary(values: number[]): string { return values.length ? `${formatValue(quantileLinear(values, .05))} / ${formatValue(quantileLinear(values, .5))} / ${formatValue(quantileLinear(values, .95))}` : 'Unavailable' }
function modelVolatility(run: LabRun): string {
  const parameters = run.data.parameters
  if (run.model === 'garch') {
    const values = run.data.conditional_variances?.flat() ?? []
    const meanVariance = finiteMean(values)
    return meanVariance == null ? 'Unavailable' : formatValue(Math.sqrt(Math.max(0, meanVariance)))
  }
  const sigma = parameters.volatility
  return typeof sigma === 'number' && Number.isFinite(sigma) ? formatValue(sigma) : 'Unavailable'
}
function modelVolatilityLabel(run: LabRun): string {
  if (run.model === 'garch') return 'Mean backend conditional σ / √period.'
  if (run.model === 'ou') return 'OU diffusion parameter σ in state units / √year.'
  if (run.model === 'jump_diffusion') return 'Diffusion σ / √year; jump risk adds separately.'
  return 'GBM diffusion parameter σ / √year.'
}
