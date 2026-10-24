import test from 'node:test'
import assert from 'node:assert/strict'
import { advanceProgressive, cumulativeLogReturnPaths, dateAxisTicks, densityDifference, displayRange, formatAxisValue, initialProgressivePlayback, linearAxisTicks, mapDataToWorld, mapWorldToData, pauseProgressive, playProgressive, progressiveIndex, resetProgressive, seekProgressive, researchModels, researchState, RESEARCH_AXIS_TITLES, validRiskSurface, windowWorldTransform, toggleResearchModelVisibility, PLAYBACK_SCALE } from '../src/model-tournament-research.ts'
import type { ForecastDistributionSurface, ForecastRiskSurface, SimulationTrajectories, TournamentModelResult } from '../src/types.ts'

const surface = (offset = 0): ForecastDistributionSurface => ({
  evaluation_dates: ['2024-01-02', '2024-01-03'], origin_indices: [1, 2],
  log_return_grid: [-.1, 0, .1], density_by_origin: [[1 + offset, 2 + offset, 1 + offset], [1 + offset, 2 + offset, 1 + offset]],
  density_method: 'Analytical one-step predictive density', unit: 'natural_log_return_per_observation_interval',
})
const risk: ForecastRiskSurface = { evaluation_dates: ['2024-01-02', '2024-01-03'], origin_indices: [1, 2],
  confidence_levels: [.9, .95], var_loss_fraction_by_origin: [[.01, .02], [.02, .03]],
  expected_shortfall_95_loss_fraction_by_origin: [.03, .04], probability_negative_return_by_origin: [.45, .48],
  loss_definition: 'L=1-exp(r)' }
const model = (key: 'gbm' | 'ou' | 'jump_diffusion' | 'garch', status: 'completed' | 'unavailable' = 'completed'): TournamentModelResult => ({
  model: key, status, parameters: {}, metrics: {}, warnings: [], assumptions: [], limitation: '',
  forecast_distribution_surface: status === 'completed' ? surface(key === 'ou' ? .5 : 0) : null,
  forecast_risk_surface: status === 'completed' ? risk : null,
})

test('3D research filters unavailable models and exposes explicit loading, error, empty, and ready states', () => {
  assert.deepEqual(researchModels([model('gbm'), model('ou', 'unavailable')]).map((item) => item.model), ['gbm'])
  assert.equal(researchState(0, 0, false, ''), 'empty')
  assert.equal(researchState(1, 1, true, ''), 'loading')
  assert.equal(researchState(1, 1, false, 'provider error'), 'error')
  assert.equal(researchState(1, 1, false, ''), 'ready')
})

test('progressive playback begins empty, reveals prefixes, pauses, resumes, and completes without losing position', () => {
  const empty = initialProgressivePlayback()
  assert.equal(progressiveIndex(empty.progress, 64, empty.started), -1)
  let state = playProgressive(empty)
  assert.deepEqual(state, { progress: 0, started: true, playing: true })
  assert.equal(progressiveIndex(state.progress, 64, state.started), 0)
  state = advanceProgressive(state, 64)
  assert.ok(state.progress > 0 && state.progress < PLAYBACK_SCALE)
  const beforePause = state.progress
  state = pauseProgressive(state)
  assert.equal(advanceProgressive(state, 64).progress, beforePause)
  state = playProgressive(state)
  while (state.playing) state = advanceProgressive(state, 64)
  assert.equal(state.progress, PLAYBACK_SCALE)
  assert.equal(progressiveIndex(state.progress, 64, state.started), 63)
})

test('reset, replay, seeking, and model visibility share one normalized timeline without regenerating data', () => {
  let state = seekProgressive(initialProgressivePlayback(), 500)
  assert.deepEqual(state, { progress: 500, started: true, playing: false })
  assert.equal(progressiveIndex(state.progress, 64, state.started), 32)
  assert.equal(progressiveIndex(state.progress, 513, state.started), 256)
  let visibleModels: ('gbm' | 'ou')[] = ['gbm']
  visibleModels = toggleResearchModelVisibility(visibleModels, 'ou')
  assert.deepEqual(visibleModels, ['gbm', 'ou'])
  assert.equal(progressiveIndex(state.progress, 64, state.started), 32)
  visibleModels = toggleResearchModelVisibility(visibleModels, 'gbm')
  assert.deepEqual(visibleModels, ['ou'])
  assert.equal(progressiveIndex(state.progress, 64, state.started), 32)
  state = resetProgressive()
  assert.deepEqual(state, { progress: 0, started: false, playing: false })
  assert.equal(progressiveIndex(state.progress, 513, state.started), -1)
  const replay = playProgressive(state)
  assert.deepEqual(replay, { progress: 0, started: true, playing: true })
})

test('timeline seeking clamps invalid/out-of-range positions and remains synchronized across models', () => {
  const negative = seekProgressive(initialProgressivePlayback(), -10)
  const aboveRange = seekProgressive(negative, 5000)
  assert.equal(negative.progress, 0)
  assert.equal(aboveRange.progress, PLAYBACK_SCALE)
  assert.equal(progressiveIndex(aboveRange.progress, 64, true), 63)
  assert.equal(progressiveIndex(aboveRange.progress, 513, true), 512)
  assert.equal(progressiveIndex(250, 64, true), 16)
})

test('density difference uses aligned same-unit grids, origins, and dates', () => {
  const result = densityDifference(surface(), surface(.25))
  assert.equal(result.compatible, true)
  if (result.compatible) assert.deepEqual(result.value.rows[0], [-.25, -.25, -.25])
  assert.equal(densityDifference(surface(), { ...surface(), evaluation_dates: ['2024-01-03', '2024-01-04'] }).compatible, false)
  assert.equal(densityDifference(surface(), { ...surface(), unit: 'ou_state_level' }).compatible, false)
  assert.equal(densityDifference(surface(), null).compatible, false)
})

test('risk surface validation rejects mismatched or non-finite risk arrays', () => {
  assert.equal(validRiskSurface(risk), true)
  assert.equal(validRiskSurface({ ...risk, var_loss_fraction_by_origin: [[.1]] }), false)
  assert.equal(validRiskSurface({ ...risk, probability_negative_return_by_origin: [Number.NaN, .4] }), false)
})

test('numeric ticks use readable nice intervals, preserve endpoints, and include zero when requested', () => {
  const ticks = linearAxisTicks({ min: -.037, max: .081 }, { min: -2.55, max: 2.55 }, (value) => formatAxisValue(value), 5, true)
  assert.equal(ticks[0]?.value, -.037)
  assert.equal(ticks.at(-1)?.value, .081)
  assert.ok(ticks.some((tick) => tick.value === 0))
  assert.ok(ticks.every((tick) => tick.position >= -2.55 && tick.position <= 2.55))
  const flatTicks = linearAxisTicks({ min: .025, max: .025 }, { min: -1, max: 1 })
  assert.ok(flatTicks.length >= 2 && flatTicks.every((tick) => Number.isFinite(tick.position)))
  const invalidTicks = linearAxisTicks({ min: Number.NaN, max: Number.POSITIVE_INFINITY }, { min: -1, max: 1 })
  assert.ok(invalidTicks.length >= 2 && invalidTicks.every((tick) => Number.isFinite(tick.value) && Number.isFinite(tick.position)))
  assert.match(formatAxisValue(.0000123), /e-/)
  assert.equal(formatAxisValue(0), '0')
})

test('date ticks map visible forecast origins to real short dates and follow a changing time window', () => {
  const dates = ['2024-12-30', '2024-12-31', '2025-01-02', '2025-01-03', '2025-01-06']
  const window = dateAxisTicks(dates, 1, 4, { min: -5, max: 5 }, 3)
  assert.deepEqual(window.map((tick) => tick.value), [1, 3, 4])
  assert.ok(Math.abs((window[0]?.position ?? 0) + 5) < 1e-12)
  assert.ok(Math.abs((window[1]?.position ?? 0) - (5 / 3)) < 1e-12)
  assert.ok(Math.abs((window[2]?.position ?? 0) - 5) < 1e-12)
  assert.match(window[0]?.label ?? '', /24/)
  assert.deepEqual(dateAxisTicks([], 0, 0), [])
  assert.equal(dateAxisTicks(dates, 2, 2)[0]?.position, 0)
  const shifted = dateAxisTicks(dates, 3, 4)
  assert.notDeepEqual(shifted.map((tick) => tick.label), window.map((tick) => tick.label))
})

test('data/world transforms are inverse and remain finite for constant or invalid ranges', () => {
  const range = displayRange(-.04, .08), world = { min: -2.55, max: 2.55 }
  const position = mapDataToWorld(.02, range, world)
  assert.ok(Math.abs(mapWorldToData(position, range, world) - .02) < 1e-12)
  const flat = displayRange(.01, .01)
  assert.ok(flat.max > flat.min)
  assert.ok(Number.isFinite(mapDataToWorld(.01, flat, world)))
  assert.deepEqual(displayRange(Number.NaN, Number.POSITIVE_INFINITY), { min: 0, max: 1 })
  const transform = windowWorldTransform(20, 40, 100)
  assert.ok(Math.abs(transform.offset + (-5 + 20 / 99 * 10) * transform.scale + 5) < 1e-10)
  assert.ok(Math.abs(transform.scale * (-5 + 40 / 99 * 10) + transform.offset - 5) < 1e-10)
})

test('all observatory modes expose semantically distinct axis titles and units', () => {
  assert.match(RESEARCH_AXIS_TITLES.distribution.x, /date/i)
  assert.match(RESEARCH_AXIS_TITLES.distribution.y, /log return/i)
  assert.match(RESEARCH_AXIS_TITLES.distribution.z, /density/i)
  assert.match(RESEARCH_AXIS_TITLES.paths.x, /years/i)
  assert.match(RESEARCH_AXIS_TITLES.paths.z, /categorical/i)
  assert.match(RESEARCH_AXIS_TITLES.risk.y, /confidence/i)
  assert.match(RESEARCH_AXIS_TITLES.risk.z, /VaR/i)
  assert.match(RESEARCH_AXIS_TITLES.disagreement.z, /density A − density B/i)
})

test('existing price, OU state, and GARCH return paths map to a compatible cumulative log-return display', () => {
  const asSimulation = (value: object) => value as SimulationTrajectories
  const prices = cumulativeLogReturnPaths(asSimulation({ quantity_kind: 'price', times: [0, .5, 1],
    trajectories: [[100, 110, 90], [100, 100, 100]], parameters: {} }))
  assert.deepEqual(prices.paths[0], [0, Math.log(1.1), Math.log(.9)])
  const ou = cumulativeLogReturnPaths(asSimulation({ quantity_kind: 'state', times: [0, .5, 1],
    trajectories: [[.2, .01, -.02]], parameters: {} }))
  assert.deepEqual(ou.paths[0], [0, .01, -.01])
  assert.match(ou.unit, /OU return states/)
  const garch = cumulativeLogReturnPaths(asSimulation({ quantity_kind: 'return', times: [1, 2],
    trajectories: [[.1, -.05]], parameters: { return_convention: 'log' } }))
  assert.deepEqual(garch.times, [0, .5, 1])
  assert.deepEqual(garch.paths[0], [0, .1, .05])
  const simple = cumulativeLogReturnPaths(asSimulation({ quantity_kind: 'return', times: [1, 2],
    trajectories: [[.1, -.2]], parameters: { return_convention: 'simple' } }))
  assert.ok(Math.abs(simple.paths[0][2] - Math.log(.88)) < 1e-14)
  assert.throws(() => cumulativeLogReturnPaths(asSimulation({ quantity_kind: 'return', times: [1],
    trajectories: [[-1]], parameters: { return_convention: 'simple' } })), /at or below -100%/)
})
