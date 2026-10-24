import type { ModelTournamentResult, TournamentModelResult } from './types'

export type TournamentViewState = 'idle' | 'loading' | 'error' | 'results'

export function tournamentViewState(busy: boolean, error: string, hasResult: boolean): TournamentViewState {
  if (busy) return 'loading'
  if (error) return 'error'
  return hasResult ? 'results' : 'idle'
}

export function validTournamentPeriods(trainStart: string, trainEnd: string, testStart: string, testEnd: string): boolean {
  const isDate = (value: string) => {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false
    const parsed = new Date(`${value}T00:00:00Z`)
    return Number.isFinite(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value
  }
  return isDate(trainStart) && isDate(trainEnd) && isDate(testStart) && isDate(testEnd)
    && trainStart < trainEnd && testStart > trainEnd && testStart < testEnd
}

export function restoreSavedTournament(saved: Record<string, unknown>): TournamentModelResult {
  const metadata = (saved.metadata && typeof saved.metadata === 'object' ? saved.metadata : {}) as Record<string, unknown>
  const metrics = (saved.metrics && typeof saved.metrics === 'object' ? saved.metrics : {}) as Record<string, unknown>
  return {
    model: saved.model as TournamentModelResult['model'], status: saved.status === 'unavailable' || saved.status === 'failed' ? 'unavailable' : 'completed',
    parameters: (saved.parameters && typeof saved.parameters === 'object' ? saved.parameters : {}) as Record<string, unknown>,
    metrics: (metrics.tournament_metrics && typeof metrics.tournament_metrics === 'object' ? metrics.tournament_metrics : {}) as Record<string, unknown>,
    warnings: Array.isArray(metadata.warnings) ? metadata.warnings.filter((item): item is string => typeof item === 'string') : [],
    assumptions: Array.isArray(metadata.assumptions) ? metadata.assumptions.filter((item): item is string => typeof item === 'string') : [],
    limitation: typeof metadata.limitation === 'string' ? metadata.limitation : '',
    error: typeof saved.error === 'string' ? saved.error : undefined,
    forecast_distribution_surface: saved.forecast_distribution_surface && typeof saved.forecast_distribution_surface === 'object' ? saved.forecast_distribution_surface as TournamentModelResult['forecast_distribution_surface'] : null,
    forecast_risk_surface: saved.forecast_risk_surface && typeof saved.forecast_risk_surface === 'object' ? saved.forecast_risk_surface as TournamentModelResult['forecast_risk_surface'] : null,
    research_path: saved.research_path && typeof saved.research_path === 'object' ? saved.research_path as Record<string, number> : null,
    experiment_id: typeof saved.experiment_id === 'string' ? saved.experiment_id : undefined,
  }
}

export function restoreSavedTournamentGroup(saved: Record<string, unknown>): TournamentModelResult[] {
  return Array.isArray(saved.models) ? saved.models.filter((item): item is Record<string, unknown> => !!item && typeof item === 'object').map(restoreSavedTournament) : []
}

export function metricNumber(metrics: Record<string, unknown> | null, key: string): number | null {
  const value = metrics?.[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export function tournamentResultFromSaved(saved: Record<string, unknown>): ModelTournamentResult {
  const savedRows = Array.isArray(saved.models) ? saved.models : []
  const firstRow = savedRows.find((item): item is Record<string, unknown> => !!item && typeof item === 'object')
  const metadataValue = saved.metadata ?? firstRow?.metadata
  const metadata = (metadataValue && typeof metadataValue === 'object' ? metadataValue : {}) as Record<string, unknown>
  const results = restoreSavedTournamentGroup(saved)
  const asRecord = (value: unknown) => value && typeof value === 'object' ? value as Record<string, unknown> : {}
  return {
    ticker: String(saved.ticker ?? metadata.ticker ?? ''), model_version: typeof metadata.model_version === 'string' ? metadata.model_version : undefined,
    interval: String(saved.interval ?? metadata.interval ?? ''),
    training_period: asRecord(saved.training_period ?? metadata.training_period),
    test_period: asRecord(saved.test_period ?? metadata.test_period),
    comparison_method: String(metadata.comparison_method ?? 'Saved train-only chronological tournament.'),
    directly_comparable_metrics: ['mean_negative_log_score_nats_per_observation', 'pit_ks_statistic', 'volatility_qlike_mean', 'one_period_95pct_var_breach_rate'],
    model_specific_metrics: [], reproducibility: { seed: Number(metadata.random_seed ?? 0), paths: Number(metadata.paths ?? 0), same_runtime_seeded_replay: true },
    results,
  }
}
