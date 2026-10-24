import test from 'node:test'
import assert from 'node:assert/strict'
import { metricNumber, restoreSavedTournamentGroup, tournamentResultFromSaved, tournamentViewState, validTournamentPeriods } from '../src/model-tournament-view.ts'

test('chronological date window requires strict train/test boundaries', () => {
  assert.equal(validTournamentPeriods('2020-01-01', '2021-01-01', '2021-01-02', '2022-01-01'), true)
  assert.equal(validTournamentPeriods('2020-01-01', '2021-01-01', '2021-01-01', '2022-01-01'), false)
  assert.equal(validTournamentPeriods('2020-01-01', '2021-01-01', '2020-12-31', '2022-01-01'), false)
  assert.equal(validTournamentPeriods('2020-02-30', '2021-01-01', '2021-02-01', '2022-01-01'), false)
})

test('tournament panel exposes loading, failure, idle, and completed states', () => {
  assert.equal(tournamentViewState(false, '', false), 'idle')
  assert.equal(tournamentViewState(true, '', false), 'loading')
  assert.equal(tournamentViewState(false, 'network error', false), 'error')
  assert.equal(tournamentViewState(false, '', true), 'results')
})

test('saved experiment rows restore model metrics and optional diagnostics safely', () => {
  const rows = restoreSavedTournamentGroup({ models: [
    { model: 'gbm', parameters: { drift: 0.05 }, metrics: { tournament_metrics: { sample_size: 20 } },
      metadata: { warnings: ['short sample'], assumptions: ['Gaussian increments'] }, experiment_id: 'e1' },
    { model: 'garch', parameters: null, metrics: null, metadata: {} },
    null,
  ] })
  assert.equal(rows.length, 2)
  assert.equal(rows[0].parameters?.drift, 0.05)
  assert.deepEqual(rows[0].warnings, ['short sample'])
  assert.deepEqual(rows[1].metrics, {})
  assert.equal(metricNumber(rows[0].metrics, 'sample_size'), 20)
  assert.equal(metricNumber(rows[1].metrics, 'missing'), null)
})

test('saved grouped runs restore experiment metadata from their persisted model rows', () => {
  const result = tournamentResultFromSaved({ tournament_id: 't1', ticker: 'SYN', interval: '1d',
    training_period: { start: '2020-01-01' }, test_period: { end_exclusive: '2022-01-01' },
    models: [{ model: 'gbm', parameters: {}, metrics: { tournament_metrics: {} }, metadata: {
      model_version: '0.1.0', random_seed: 23, paths: 500, comparison_method: 'train only',
    } }] })
  assert.equal(result.model_version, '0.1.0')
  assert.equal(result.reproducibility.seed, 23)
  assert.equal(result.reproducibility.paths, 500)
  assert.equal(result.comparison_method, 'train only')
})

test('saved tournament restores unavailable models and their calibration diagnostic', () => {
  const rows = restoreSavedTournamentGroup({ models: [{ model: 'ou', status: 'unavailable', parameters: {}, metrics: null,
    error: 'OU mean reversion is not identified: fitted AR(1) coefficient must lie strictly between 0 and 1',
    metadata: { assumptions: ['Exact sampled OU transition'] }, experiment_id: 'failed-ou' }] })
  assert.equal(rows[0]?.status, 'unavailable')
  assert.match(rows[0]?.error ?? '', /AR\(1\) coefficient/)
  assert.deepEqual(rows[0]?.assumptions, ['Exact sampled OU transition'])
})
