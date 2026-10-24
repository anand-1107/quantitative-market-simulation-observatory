import assert from 'node:assert/strict'
import test from 'node:test'
import { DETECTION_NUMERIC_DEFAULTS, isMarketEventAnalysis, marketEventAnalysisViewState, marketEventErrorMessage, numericControlWithinBounds } from '../src/market-events-view.ts'

test('event analysis view clearly distinguishes initial, loading, failure and completed result states', () => {
  assert.equal(marketEventAnalysisViewState(false, '', false), 'idle')
  assert.equal(marketEventAnalysisViewState(true, '', false), 'loading')
  assert.equal(marketEventAnalysisViewState(false, 'Could not reach backend', false), 'error')
  assert.equal(marketEventAnalysisViewState(false, '', true), 'results')
})

test('an empty candidate array is still a completed successful analysis state', () => {
  const response = { ticker: 'AAPL', candidate_events: [], bars: [{ date: '2025-10-20', close: 100 }] }
  assert.equal(marketEventAnalysisViewState(false, '', Array.isArray(response.candidate_events)), 'results')
  assert.equal(response.candidate_events.length, 0)
})

test('representative successful analysis response with zero candidates is a renderable result', () => {
  const response = {
    ticker: 'AAPL', interval: '1d', estimation_period: { start: '2025-10-20', end_exclusive: '2026-10-24' },
    observation_count: 3, source_observation_count: 3, excluded_observation_count: 0, downsampled: false,
    bars: [
      { date: '2025-10-20T00:00:00', open: 250, high: 251, low: 249, close: 250.5, volume: 1000 },
      { date: '2025-10-21T00:00:00', open: 250.5, high: 252, low: 250, close: 251, volume: 1100 },
      { date: '2025-10-22T00:00:00', open: 251, high: 252, low: 250.5, close: 251.5, volume: null },
    ],
    candidate_events: [], detection_configuration: { absolute_log_return: 0.04 },
    impact_window_observations_each_side: 10, warnings: [],
    data_convention: 'Raw OHLC conventions.', causality_note: 'Descriptive only.',
    benchmark_ticker: 'SPY', chart_note: 'Display bars may be downsampled.', analysis_cache: 'miss',
  }
  assert.equal(isMarketEventAnalysis(response), true)
  assert.equal(marketEventAnalysisViewState(false, '', isMarketEventAnalysis(response)), 'results')
  assert.equal(response.candidate_events.length, 0)
  assert.equal(isMarketEventAnalysis({ ...response, warnings: null }), false)
})

test('collapsed detection thresholds keep valid defaults while preserving min and max validation', () => {
  // Before the fix, these defaults failed HTML stepMismatch checks because the step grid was
  // anchored at min=0.0001/1.01. Collapsed details cannot be focused by Safari during submit.
  assert.deepEqual(DETECTION_NUMERIC_DEFAULTS.map(({ step }) => step), ['any', 'any', 'any', 'any', 'any'])
  for (const { value, min, max } of DETECTION_NUMERIC_DEFAULTS) assert.equal(numericControlWithinBounds(value, min, max), true)
  assert.equal(numericControlWithinBounds(0, 0.0001, 1), false)
  assert.equal(numericControlWithinBounds(101, 1.01, 100), false)
})

test('backend validation errors and object rejections become readable UI messages', () => {
  assert.equal(marketEventErrorMessage({ detail: [{ loc: ['body', 'start'], msg: 'Invalid date' }] }, 'fallback'), 'Invalid date')
  assert.equal(marketEventErrorMessage({ error: { message: 'Provider unavailable' } }, 'fallback'), 'Provider unavailable')
  assert.equal(marketEventErrorMessage({ arbitrary: true }, 'Historical event analysis failed.'), 'Historical event analysis failed.')
})
