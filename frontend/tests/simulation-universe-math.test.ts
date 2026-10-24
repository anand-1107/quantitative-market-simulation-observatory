import assert from 'node:assert/strict'
import test from 'node:test'
import { estimateDensitySurface, inspectPath, pathAppearance, pathJourneySnapshot, performanceFrame, playbackAdvance, playbackPause, playbackReset, playbackStart, scenarioComparisonPaths, visiblePathVertexCount } from '../src/simulation-universe-math.ts'
import type { PathForensics, SimulationTrajectories } from '../src/types.ts'

test('Gaussian KDE uses a common grid and integrates to approximately one for each time cross-section', () => {
  const paths = Array.from({ length: 80 }, (_, index) => {
    const centered = (index - 39.5) / 20
    return [centered, centered * .5 + .25]
  })
  const estimate = estimateDensitySurface(paths, 72)
  assert.equal(estimate.densityByTime.length, 2)
  assert.equal(estimate.densityByTime[0].length, 72)
  assert.equal(estimate.valueGrid[0] < estimate.valueGrid.at(-1)!, true)
  for (const densities of estimate.densityByTime) {
    const area = densities.slice(1).reduce((sum, value, index) => sum + (densities[index] + value) * .5 * (estimate.valueGrid[index + 1] - estimate.valueGrid[index]), 0)
    assert.ok(area > .95 && area < 1.05, `KDE integrates to ${area}`)
  }
})

test('KDE remains finite for constant and near-constant samples', () => {
  const constant = estimateDensitySurface([[5, 5], [5, 5], [5, 5]], 32)
  assert.ok(constant.densityMax > 0 && Number.isFinite(constant.densityMax))
  assert.ok(constant.valueMax > constant.valueMin)
  const near = estimateDensitySurface([[1, 1], [1 + 1e-12, 1 - 1e-12], [1 - 1e-12, 1 + 1e-12]], 32)
  assert.ok(near.densityByTime.flat().every(Number.isFinite))
})

test('KDE rejects insufficient, ragged, non-finite and invalid-grid inputs', () => {
  assert.throws(() => estimateDensitySurface([[1, 2]]), /At least two/)
  assert.throws(() => estimateDensitySurface([[1], [1, 2]]), /same non-zero/)
  assert.throws(() => estimateDensitySurface([[1], [Infinity]]), /finite/)
  assert.throws(() => estimateDensitySurface([[1], [2]], 4), /between 32 and 128/)
})

test('scenario comparison aligns candidate replay with the selected unchanged reference path', () => {
  const reference = [100, 97, 101, 95]
  const candidate = [100, 96, 103, 92]
  const comparison = scenarioComparisonPaths(reference, candidate)!
  assert.deepEqual(comparison.reference, reference)
  assert.deepEqual(comparison.candidate, candidate)
  comparison.reference[0] = -1
  assert.equal(reference[0], 100)
  assert.equal(scenarioComparisonPaths([100], [99]), null)
})

const sample: SimulationTrajectories = {
  model: 'gbm', parameters: { seed: 17 }, quantity_kind: 'price', quantity_label: 'Simulated price', time_label: 'Time (years)',
  times: [0, .5, 1], trajectories: [[100, 110, 105], [100, 95, 90]], conditional_variances: null,
  summary: { terminal_mean: 97.5, terminal_variance: 112.5, terminal_quantiles: { '0.05': 90, '0.5': 97.5, '0.95': 105 },
    bands: [{ time: 0, mean: 100, p05: 100, p50: 100, p95: 100 }, { time: .5, mean: 102.5, p05: 95, p50: 102.5, p95: 110 }, { time: 1, mean: 97.5, p05: 90, p50: 97.5, p95: 105 }] },
  terminal_risk: { available: true, warnings: [], path_metrics: [{ total_simple_return: .05, maximum_drawdown: -.04545 }, { total_simple_return: -.1, maximum_drawdown: -.1 }] },
}

test('every path, density time slice, performance chart, and inspector share the selected time index', () => {
  const surface = estimateDensitySurface(sample.trajectories)
  const currentIndex = 1
  const chart = performanceFrame(sample, 1, currentIndex)
  const path = inspectPath(sample, 1, currentIndex)
  assert.equal(surface.densityByTime.length, sample.times.length)
  assert.equal(chart.length, currentIndex + 1)
  assert.equal(chart.at(-1)?.time, sample.times[currentIndex])
  assert.equal(chart.at(-1)?.selected, sample.trajectories[1][currentIndex])
  assert.equal(chart.at(-1)?.path0, sample.trajectories[0][currentIndex])
  assert.equal(chart.at(-1)?.path1, sample.trajectories[1][currentIndex])
  assert.equal(visiblePathVertexCount(sample.trajectories.length, currentIndex), currentIndex * sample.trajectories.length * 2)
  assert.equal(path.currentValue, chart.at(-1)?.selected)
  assert.equal(path.time, chart.at(-1)?.time)
  assert.equal(path.pathId, 2)
  assert.equal(path.totalReturn, -.1)
})

test('all paths remain visually distinguished while selected path receives a clear highlight', () => {
  const other = pathAppearance(0, 1)
  const selected = pathAppearance(1, 1)
  assert.equal(other.selected, false)
  assert.notEqual(other.color, selected.color)
  assert.ok(other.opacity < selected.opacity)
  assert.ok(other.width < selected.width)
  assert.equal(selected.selected, true)
})

test('playback advances, pauses, resets, and replays deterministically', () => {
  let state = playbackStart(playbackReset(), 3)
  assert.deepEqual(state, { index: 0, playing: true })
  state = playbackAdvance(state, 3)
  assert.deepEqual(state, { index: 1, playing: true })
  state = playbackPause(state)
  assert.deepEqual(playbackAdvance(state, 3), { index: 1, playing: false })
  assert.deepEqual(playbackReset(), { index: 0, playing: false })
  state = playbackStart({ index: 3, playing: false }, 3)
  assert.deepEqual(state, { index: 0, playing: true })
  state = playbackAdvance(state, 3)
  state = playbackAdvance(state, 3)
  state = playbackAdvance(state, 3)
  assert.deepEqual(state, { index: 3, playing: false })
})

function forensic(model: PathForensics['model'], path: number[], transitions: PathForensics['transitions'], parameters: Record<string, unknown> = {}): PathForensics {
  return { run_id: 'run', path_id: 1, model, model_version: 'test', created_at: 'now', parameters,
    initial_conditions: {}, times: path.map((_, index) => index), original_path: path, innovations: {}, conditional_variance: null,
    transitions, terminal_metrics: {}, reconstruction: { path: [...path], maximum_absolute_error: 0,
      verification_tolerance: 1e-12, verified: true, method: 'stored innovations' } }
}

test('journey cursor selects retained GBM values, time, return, drawdown, and signed log contributions', () => {
  const data = forensic('gbm', [100, 110, 99], [
    { time: 1, standard_normal: 1, drift_log_component: .01, diffusion_component: .0853, log_return_increment: .0953, value: 110 },
    { time: 2, standard_normal: -1, drift_log_component: .01, diffusion_component: -.1153, log_return_increment: -.1053, value: 99 },
  ], { seed: 1 })
  const up = pathJourneySnapshot(data, 1)!
  assert.equal(up.currentValue, 110)
  assert.equal(up.previousValue, 100)
  assert.equal(up.absoluteChange, 10)
  assert.ok(Math.abs(up.percentageChange! - .1) < 1e-12)
  assert.ok(Math.abs(up.cumulativeReturn! - .1) < 1e-12)
  assert.equal(up.maximumDrawdown, 0)
  assert.deepEqual(up.components.map(({ label, value, unit }) => [label, value, unit]), [
    ['Drift log component', .01, 'log return'], ['Diffusion component', .0853, 'log return'],
  ])
  const down = pathJourneySnapshot(data, 2)!
  assert.equal(down.currentValue, 99)
  assert.equal(down.absoluteChange, -11)
  assert.ok(down.components.find((item) => item.label === 'Diffusion component')!.value < 0)
  assert.ok(Math.abs(down.maximumDrawdown! + .1) < 1e-12)
})

test('OU and Merton journey contributions preserve model-specific units and stored jump details', () => {
  const ou = forensic('ou', [-2, -1.5], [{ time: 1, current_value: -2, conditional_mean: -1.7,
    mean_reversion_component: .3, standard_normal: .5, diffusion_component: .2, value: -1.5 }])
  const ouStep = pathJourneySnapshot(ou, 1)!
  assert.equal(ouStep.percentageChange, null)
  assert.deepEqual(ouStep.components.map((component) => [component.value, component.unit]), [[.3, 'state units'], [.2, 'state units']])
  const merton = forensic('jump_diffusion', [100, 80], [{ time: 1, poisson_jump_count: 1,
    jump_aggregate_standard_normal: 0, drift_log_term: .01, diffusion_component: -.02,
    jump_log_sum: -.213, log_price_increment: -.223, value: 80 }])
  const jumpStep = pathJourneySnapshot(merton, 1)!
  assert.deepEqual(jumpStep.components.map((component) => component.unit), ['log return', 'log return', 'log return'])
  assert.equal(jumpStep.transition?.poisson_jump_count, 1)
  assert.match(jumpStep.modelNote, /individual jump sizes are not/)
})

test('GARCH journey separates return and variance units and respects final-period recursion boundary', () => {
  const garch = forensic('garch', [.02, -.01], [
    { time: 0, conditional_variance: .04, standard_normal: .5, innovation: .1, return: .02,
      variance_intercept_term: .01, arch_variance_term: .001, garch_variance_term: .02, next_conditional_variance: .031 },
    { time: 1, conditional_variance: .031, standard_normal: -1, innovation: -.176, return: -.01,
      variance_intercept_term: null, arch_variance_term: null, garch_variance_term: null, next_conditional_variance: null },
  ], { mean: -.08, return_convention: 'simple' })
  const first = pathJourneySnapshot(garch, 0)!
  assert.equal(first.transition?.return, .02)
  assert.deepEqual(first.components.map((component) => component.group), ['return', 'return', 'variance', 'variance', 'variance'])
  assert.equal(first.components[0].value, -.08)
  assert.equal(first.components.at(-1)?.unit, 'return²')
  assert.equal(pathJourneySnapshot(garch, 1)!.components.filter((item) => item.group === 'variance').length, 0)
  assert.equal(pathJourneySnapshot(garch, 1)!.percentageChange, null)
  assert.ok(Math.abs(first.cumulativeReturn! - .02) < 1e-12)
})

test('journey clamps boundary selection and handles empty stored paths', () => {
  const data = forensic('gbm', [50], [])
  assert.equal(pathJourneySnapshot(data, 99)?.index, 0)
  assert.equal(pathJourneySnapshot({ ...data, original_path: [] }, 0), null)
  const initial = pathJourneySnapshot(data, 0)!
  assert.equal(initial.transition, null)
  assert.equal(initial.previousValue, null)
  assert.equal(initial.percentageChange, null)
})
