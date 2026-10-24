import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildFormulaRequest, deriveGarchPricePaths, FORMULA_DEFAULTS, parameterDifferences,
  quantileLinear, representativePathIndices, createFormulaDraft, formulaTrajectoryLabel, FORMULA_TEXT,
  buildCounterfactualParameters, changedSensitivityKeys, pathDifferenceSeries, rmse,
  clampObservationIndex, incomingTransitionIndex,
  formulaObservationSnapshot,
  observationIndexFromChartEvent,
} from '../src/formula-lab-math.ts'
import type { SimulationTrajectories } from '../src/types.ts'

test('formula lab maps each model to its existing trajectory endpoint and supported request schema', () => {
  const cases = [
    ['gbm', '/simulate-gbm/paths', ['initial_price', 'drift', 'volatility', 'years', 'steps', 'paths', 'seed']],
    ['ou', '/simulate-ou/paths', ['initial_value', 'mean_level', 'mean_reversion_speed', 'volatility', 'years', 'steps', 'paths', 'seed']],
    ['jump_diffusion', '/simulate-jump-diffusion/paths', ['initial_price', 'drift', 'volatility', 'jump_intensity', 'mean_log_jump', 'jump_volatility', 'years', 'steps', 'paths', 'seed']],
    ['garch', '/simulate-garch/paths', ['mean', 'omega', 'alpha', 'beta', 'initial_variance', 'steps', 'paths', 'seed', 'return_convention']],
  ] as const
  for (const [model, endpoint, expectedKeys] of cases) {
    const result = buildFormulaRequest(model, { ...FORMULA_DEFAULTS[model] })
    assert.equal(result.error, null)
    assert.equal(result.endpoint, endpoint)
    assert.deepEqual(Object.keys(result.payload).sort(), [...expectedKeys].sort())
    assert.equal(result.payload.seed, 42)
  }
})

test('formula lab enforces backend bounds, integer grid settings, and explicit optional defaults', () => {
  const blankSeed = buildFormulaRequest('gbm', { ...FORMULA_DEFAULTS.gbm, seed: '' })
  assert.equal(blankSeed.error, null)
  assert.equal(blankSeed.payload.seed, null)
  assert.match(buildFormulaRequest('gbm', { ...FORMULA_DEFAULTS.gbm, volatility: '4' }).error!, /between 0 and 3/)
  assert.match(buildFormulaRequest('ou', { ...FORMULA_DEFAULTS.ou, steps: '2.5' }).error!, /integer/)
  assert.match(buildFormulaRequest('jump_diffusion', { ...FORMULA_DEFAULTS.jump_diffusion, jump_intensity: '-1' }).error!, /between 0 and 10/)
  const defaultGarch = buildFormulaRequest('garch', { ...FORMULA_DEFAULTS.garch })
  assert.equal(defaultGarch.error, null)
  assert.equal(defaultGarch.payload.initial_variance, null)
  assert.equal(defaultGarch.payload.return_convention, 'log')
  assert.match(buildFormulaRequest('garch', { ...FORMULA_DEFAULTS.garch, return_convention: 'bad' }).error!, /log or simple/)
})

test('model switching starts from independent defaults and uses explicit requested seed for reproducibility', () => {
  const gbmDraft = createFormulaDraft('gbm')
  gbmDraft.drift = '.25'
  assert.equal(FORMULA_DEFAULTS.gbm.drift, '.08')
  assert.equal(createFormulaDraft('ou').initial_value, '0')
  assert.equal(createFormulaDraft('garch').return_convention, 'log')
  const first = buildFormulaRequest('ou', { ...FORMULA_DEFAULTS.ou, seed: '123' })
  const second = buildFormulaRequest('ou', { ...FORMULA_DEFAULTS.ou, seed: '123' })
  assert.deepEqual(first.payload, second.payload)
  assert.equal(first.payload.seed, 123)
})

test('formula and output labels follow the implemented model quantity and units', () => {
  assert.match(FORMULA_TEXT.gbm, /− ½σ²/)
  assert.match(FORMULA_TEXT.ou, /e\^\(−θΔt\)/)
  assert.match(FORMULA_TEXT.jump_diffusion, /μ − λκ/)
  assert.match(FORMULA_TEXT.garch, /hₜ₊₁ = ω \+ α εₜ² \+ β hₜ/)
  assert.equal(formulaTrajectoryLabel('gbm', 'price'), 'Simulated price paths')
  assert.equal(formulaTrajectoryLabel('ou', 'state'), 'Simulated state paths')
  assert.equal(formulaTrajectoryLabel('garch', 'return'), 'Simulated return paths')
  assert.equal(formulaTrajectoryLabel('garch', 'return', 'price'), 'Derived GARCH price index paths')
})

function garchResult(convention: 'log' | 'simple', trajectories: number[][]): SimulationTrajectories {
  return { model: 'garch', parameters: { return_convention: convention }, quantity_kind: 'return',
    quantity_label: convention === 'log' ? 'Per-period log return' : 'Per-period simple return', time_label: 'Observation period',
    times: trajectories[0].map((_, index) => index + 1), trajectories, conditional_variances: null,
    summary: { terminal_mean: 0, terminal_variance: 0, terminal_quantiles: {}, bands: [] },
    terminal_risk: { available: false, warnings: [] } }
}

test('derived GARCH display prices compound only backend outputs and respect return convention', () => {
  const log = deriveGarchPricePaths(garchResult('log', [[Math.log(1.1), 0], [0, Math.log(.9)]]), 100)
  assert.deepEqual(log[0].map((value) => Number(value.toFixed(8))), [100, 110, 110])
  assert.deepEqual(log[1].map((value) => Number(value.toFixed(8))), [100, 100, 90])
  const simple = deriveGarchPricePaths(garchResult('simple', [[.1, -.2]]), 50)
  assert.deepEqual(simple[0].map((value) => Number(value.toFixed(8))), [50, 55, 44])
  assert.throws(() => deriveGarchPricePaths(garchResult('simple', [[-1]]), 100), /at or below/)
  assert.throws(() => deriveGarchPricePaths(garchResult('log', [[0]]), 0), /positive display/)
})

test('representative paths, empirical quantiles, and baseline parameter comparison are stable', () => {
  assert.deepEqual(representativePathIndices(5, 3), [0, 2, 4])
  assert.deepEqual(representativePathIndices(3, 8), [0, 1, 2])
  assert.deepEqual(representativePathIndices(0), [])
  assert.equal(quantileLinear([1, 2, 3, 4], .5), 2.5)
  assert.equal(quantileLinear([1, 2, 3, 4], .05), 1.15)
  assert.deepEqual(parameterDifferences({ drift: .1, seed: 42 }, { drift: .2, seed: 42, years: 2 }), [
    { key: 'drift', baseline: .1, scenario: .2 }, { key: 'years', baseline: null, scenario: 2 },
  ])
})

test('counterfactual request filters ensemble controls and keeps supported model configuration only', () => {
  const gbm = buildCounterfactualParameters('gbm', { ...FORMULA_DEFAULTS.gbm, drift: '.12' })
  assert.equal(gbm.error, null)
  assert.deepEqual(gbm.parameters, { initial_price: 100, drift: .12, volatility: .2, years: 1, steps: 160 })
  assert.equal('paths' in gbm.parameters, false)
  assert.equal('seed' in gbm.parameters, false)
  const garch = buildCounterfactualParameters('garch', { ...FORMULA_DEFAULTS.garch })
  assert.deepEqual(garch.parameters, { omega: .000001, alpha: .05, beta: .9, mean: 0, initial_variance: null, steps: 160, return_convention: 'log' })
})

test('single-path sensitivity allows one model input change and flags multiple changes', () => {
  const baseline = { initial_price: 100, drift: .08, volatility: .2, years: 1, steps: 160 }
  assert.deepEqual(changedSensitivityKeys('gbm', baseline, { ...baseline, drift: .1 }), ['drift'])
  assert.deepEqual(changedSensitivityKeys('gbm', baseline, { ...baseline, drift: .1, volatility: .3 }), ['drift', 'volatility'])
  assert.deepEqual(changedSensitivityKeys('gbm', baseline, { ...baseline, years: 2 }), ['years'])
})

test('path-level differences and RMSE use aligned stored observations', () => {
  assert.deepEqual(pathDifferenceSeries([2, 4, 6], [1, 5, 9]), [-1, 1, 3])
  assert.equal(rmse([2, 4, 6], [1, 5, 9]), Math.sqrt(11 / 3))
  assert.deepEqual(pathDifferenceSeries([1], [1, 2]), [])
  assert.equal(Number.isNaN(rmse([], [])), true)
})

test('observation selection clamps boundaries and maps initial values to no incoming transition', () => {
  assert.equal(clampObservationIndex(2, 5), 2)
  assert.equal(clampObservationIndex(50, 5), 4)
  assert.equal(clampObservationIndex(-4, 5), 0)
  assert.equal(incomingTransitionIndex('gbm', 0), null)
  assert.equal(incomingTransitionIndex('ou', 2), 1)
  assert.equal(incomingTransitionIndex('jump_diffusion', 3), 2)
  assert.equal(incomingTransitionIndex('garch', 0), 0)
})

test('live formula snapshot keeps displayed values, times, candidate difference, and transition on one observation', () => {
  const snapshot = formulaObservationSnapshot('gbm', [100, 110, 99], [0, .5, 1], 1, [100, 112, 95], [0, .5, 1])!
  assert.equal(snapshot.index, 1)
  assert.equal(snapshot.referenceTime, .5)
  assert.equal(snapshot.currentValue, 110)
  assert.equal(snapshot.previousValue, 100)
  assert.equal(snapshot.absoluteChange, 10)
  assert.ok(Math.abs(snapshot.percentageChange! - .1) < 1e-12)
  assert.equal(snapshot.candidateValue, 112)
  assert.equal(snapshot.difference, 2)
  assert.equal(snapshot.transitionIndex, 0)
  assert.equal(formulaObservationSnapshot('gbm', [10, 11], [0, 1], 0)!.transitionIndex, null)
  assert.equal(formulaObservationSnapshot('garch', [.01, -.02], [1, 2], 1)!.transitionIndex, 1)
  assert.equal(formulaObservationSnapshot('gbm', [1, 2], [0, 1], 1, [1], [0])!.index, 0)
})

test('graph click payloads select the same bounded observation used by slider and formula panel', () => {
  assert.equal(observationIndexFromChartEvent({ activeTooltipIndex: 2 }, 4), 2)
  assert.equal(observationIndexFromChartEvent({ index: 3 }, 4), 3)
  assert.equal(observationIndexFromChartEvent({ activeTooltipIndex: '99' }, 4), 3)
  assert.equal(observationIndexFromChartEvent({ payload: {} }, 4), null)
})
