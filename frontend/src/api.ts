import type {
  DownloadMetadata, GARCHCalibration, HealthResponse, MarketSummary, MertonCalibration, ModelDescription, OHLCVSeries, OUCalibration,
  GBMCalibration, GBMEvaluation, PriceSeries, SimulationResult, SimulationTrajectories, ValidationReport,
  ExperimentComparison, ExperimentRecord, PathForensics, ScenarioMatchingResult, RetainedForensicsRun, CounterfactualPathResult,
  MarketEventAnalysis, MarketEventNews, NewsProviderStatus, ModelTournamentResult, TournamentModel,
} from './types'
import { isMarketEventAnalysis } from './market-events-view'

export const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

export class ApiError extends Error {
  status?: number
  constructor(message: string, status?: number) { super(message); this.name = 'ApiError'; this.status = status }
}

function ensure<T>(value: T, valid: boolean, message: string): T {
  if (!valid) throw new ApiError(message)
  return value
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: 'application/json', ...(init.body ? { 'Content-Type': 'application/json' } : {}), ...init.headers },
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError(`Could not reach the backend at ${API_BASE}. Check that the API is running and CORS is configured.`)
  }
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const detailValue = body && typeof body === 'object' && 'detail' in body ? body.detail : null
    const detail = typeof detailValue === 'string' ? detailValue
      : Array.isArray(detailValue) ? detailValue.map((item) => {
        if (!item || typeof item !== 'object') return typeof item === 'string' ? item : ''
        const entry = item as Record<string, unknown>
        const location = Array.isArray(entry.loc) ? `${entry.loc.join('.')}: ` : ''
        return `${location}${typeof entry.msg === 'string' ? entry.msg : typeof entry.message === 'string' ? entry.message : ''}`
      }).filter(Boolean).join('; ')
      : detailValue && typeof detailValue === 'object' && 'message' in detailValue && typeof detailValue.message === 'string' ? detailValue.message
      : response.statusText
    throw new ApiError(detail || `Backend returned HTTP ${response.status}`, response.status)
  }
  if (response.status === 204) return null as T
  if (!body || typeof body !== 'object') throw new ApiError('Backend returned an unexpected response format.', response.status)
  return body as T
}

function query(values: Record<string, string | number | undefined>): string {
  const params = new URLSearchParams()
  for (const [key, value] of Object.entries(values)) if (value !== undefined && value !== '') params.set(key, String(value))
  const encoded = params.toString()
  return encoded ? `?${encoded}` : ''
}

export const api = {
  health: async (signal?: AbortSignal) => {
    const value = await request<HealthResponse>('/health', { signal })
    return ensure(value, typeof value.status === 'string' && typeof value.service === 'string', 'Backend returned an invalid health response.')
  },
  models: async (signal?: AbortSignal) => {
    const value = await request<{ models: ModelDescription[] }>('/models', { signal })
    return ensure(value, Array.isArray(value.models) && value.models.every((model) => typeof model.name === 'string' && Array.isArray(model.parameters)), 'Backend returned an invalid model registry.')
  },
  validate: async (ticker: string, values: Record<string, string>, signal?: AbortSignal) => {
    const value = await request<ValidationReport>(`/market-data/validate/${encodeURIComponent(ticker)}${query(values)}`, { signal })
    return ensure(value, typeof value.valid === 'boolean' && typeof value.row_count === 'number' && Array.isArray(value.missing_columns) && Array.isArray(value.warnings), 'Backend returned an invalid market-data validation report.')
  },
  summary: async (ticker: string, values: Record<string, string>, signal?: AbortSignal) => {
    const value = await request<MarketSummary>(`/market-data/summary/${encodeURIComponent(ticker)}${query(values)}`, { signal })
    return ensure(value, typeof value.ticker === 'string' && typeof value.row_count === 'number' && typeof value.latest_close === 'number', 'Backend returned an invalid market-data summary.')
  },
  prices: async (ticker: string, values: Record<string, string | number>, signal?: AbortSignal) => {
    const value = await request<PriceSeries>(`/market-data/prices/${encodeURIComponent(ticker)}${query(values)}`, { signal })
    return ensure(value, typeof value.ticker === 'string' && Array.isArray(value.observations) && value.observations.every((item) => typeof item.date === 'string' && typeof item.price === 'number'), 'Backend returned invalid price observations.')
  },
  ohlcv: async (ticker: string, values: Record<string, string | number>, signal?: AbortSignal) => {
    const value = await request<OHLCVSeries>(`/market-data/ohlcv/${encodeURIComponent(ticker)}${query(values)}`, { signal })
    return ensure(value, typeof value.ticker === 'string' && Array.isArray(value.observations) &&
      value.observations.every((bar) => typeof bar.date === 'string' && ['open', 'high', 'low', 'close'].every((key) => Number.isFinite(bar[key as keyof typeof bar]))),
      'Backend returned invalid OHLCV observations.')
  },
  analyzeMarketEvents: async (payload: Record<string, string | number>, signal?: AbortSignal) => {
    const value = await request<MarketEventAnalysis>('/market-events/analyze', { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, isMarketEventAnalysis(value),
      'Backend returned an invalid historical event analysis.')
  },
  marketEventNewsProvider: async (signal?: AbortSignal) => {
    const value = await request<NewsProviderStatus>('/market-events/news/provider', { signal })
    return ensure(value, typeof value.provider === 'string' && typeof value.configured === 'boolean', 'Backend returned invalid news-provider configuration.')
  },
  marketEventNews: async (payload: { ticker: string; start: string; end: string; events: Array<{ event_id: string; timestamp: string; category: string; timestamp_resolution: 'session_date' | 'intraday_observation_utc' }> }, signal?: AbortSignal) => {
    const value = await request<MarketEventNews>('/market-events/news', { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, typeof value.configured === 'boolean' && Array.isArray(value.articles) && Array.isArray(value.event_matches), 'Backend returned invalid historical-news evidence.')
  },
  download: async (payload: { ticker: string; start?: string; end?: string; interval: string }, signal?: AbortSignal) => {
    const value = await request<DownloadMetadata>('/market-data/download', { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, typeof value.ticker === 'string' && typeof value.row_count === 'number', 'Backend returned invalid download metadata.')
  },
  simulate: async (endpoint: string, payload: Record<string, number | string | null>, signal?: AbortSignal) => {
    const value = await request<SimulationResult>(endpoint, { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, typeof value.model === 'string', 'Backend returned an invalid simulation summary.')
  },
  simulateTrajectories: async (endpoint: string, payload: Record<string, unknown>, signal?: AbortSignal) => {
    const value = await request<SimulationTrajectories>(endpoint, { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, Array.isArray(value.times) && Array.isArray(value.trajectories) &&
      value.trajectories.every((path) => Array.isArray(path) && path.length === value.times.length) &&
      value.summary !== null && Array.isArray(value.summary?.bands) && value.summary.bands.length === value.times.length &&
      value.terminal_risk !== null && typeof value.terminal_risk?.available === 'boolean',
      'Backend returned invalid simulation trajectory dimensions or summary.')
  },
  pathForensics: async (runId: string, pathId: number, signal?: AbortSignal) => {
    const value = await request<PathForensics>(`/path-forensics/${encodeURIComponent(runId)}/paths/${pathId}`, { signal })
    return ensure(value, value.run_id === runId && value.path_id === pathId && Array.isArray(value.original_path) &&
      Array.isArray(value.transitions) && value.reconstruction && Array.isArray(value.reconstruction.path) &&
      typeof value.reconstruction.verified === 'boolean', 'Backend returned invalid path-forensics data.')
  },
  retainedForensicsRuns: async (signal?: AbortSignal) => {
    const value = await request<{ runs: RetainedForensicsRun[] }>('/path-forensics/runs', { signal })
    return ensure(value, Array.isArray(value.runs) && value.runs.every((run) => typeof run.run_id === 'string' &&
      ['gbm', 'ou', 'jump_diffusion', 'garch'].includes(run.model) && Number.isInteger(run.path_count) &&
      run.path_count >= 0 && Number.isInteger(run.step_count)), 'Backend returned invalid retained forensic runs.')
  },
  counterfactualPath: async (payload: { run_id: string; path_id: number; candidate_parameters: Record<string, number | string | null> }, signal?: AbortSignal) => {
    const value = await request<CounterfactualPathResult>('/path-forensics/counterfactual', { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, value.run_id === payload.run_id && value.path_id === payload.path_id &&
      Array.isArray(value.times) && Array.isArray(value.reference_path) && Array.isArray(value.candidate_path) &&
      Array.isArray(value.differences) && value.reference_path.length === value.candidate_path.length &&
      value.differences.length === value.reference_path.length && Number.isFinite(value.raw_path_rmse) &&
      Array.isArray(value.warnings) && Array.isArray(value.transitions), 'Backend returned invalid counterfactual path data.')
  },
  matchScenarios: async (payload: { run_id: string; path_id: number; candidate_parameters: Record<string, number>[]; terminal_return_weight?: number; drawdown_weight?: number }, signal?: AbortSignal) => {
    const value = await request<ScenarioMatchingResult>('/path-forensics/scenario-matches', { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, value.reference_run_id === payload.run_id && value.reference_path_id === payload.path_id &&
      Array.isArray(value.candidates) && value.candidates.every((candidate) => typeof candidate.objective === 'number'),
      'Backend returned invalid scenario-match results.')
  },
  calibrateGBM: async (payload: { ticker: string; start: string; end: string; interval: string }, signal?: AbortSignal) => {
    const value = await request<GBMCalibration>('/calibration/gbm', { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, typeof value.drift === 'number' && typeof value.volatility === 'number' &&
      typeof value.return_count === 'number' && Array.isArray(value.warnings), 'Backend returned an invalid GBM calibration.')
  },
  calibrateOU: async (payload: { ticker: string; start: string; end: string; interval: string; price_field: string; input_series: string; confirm_mean_reverting_price_series: boolean }, signal?: AbortSignal) => {
    const value = await request<OUCalibration>('/calibration/ou', { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, value.model === 'ou' && typeof value.mean_reversion_speed === 'number' && Array.isArray(value.warnings), 'Backend returned an invalid OU calibration.')
  },
  calibrateMerton: async (payload: { ticker: string; start: string; end: string; interval: string; price_field: string }, signal?: AbortSignal) => {
    const value = await request<MertonCalibration>('/calibration/jump-diffusion', { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, value.model === 'jump_diffusion' && typeof value.jump_intensity === 'number' && typeof value.optimizer_success === 'boolean', 'Backend returned an invalid Merton calibration.')
  },
  calibrateGARCH: async (payload: { ticker: string; start: string; end: string; interval: string; price_field: string }, signal?: AbortSignal) => {
    const value = await request<GARCHCalibration>('/calibration/garch', { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, value.model === 'garch' && typeof value.omega === 'number' && Array.isArray(value.conditional_volatility), 'Backend returned an invalid GARCH calibration.')
  },
  evaluateGBM: async (payload: { ticker: string; estimation_start: string; estimation_end: string; evaluation_start: string; evaluation_end: string; interval: string; paths: number; seed: number }, signal?: AbortSignal) => {
    const value = await request<GBMEvaluation>('/evaluation/gbm', { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, value.evaluation?.comparison_type === 'out_of_sample_chronological' &&
      typeof value.evaluation.metrics?.historical_annualized_volatility === 'number', 'Backend returned invalid GBM evaluation metrics.')
  },
  runModelTournament: async (payload: { ticker: string; models: TournamentModel[]; train_start: string; train_end: string; test_start: string; test_end: string; interval: string; paths: number; seed: number; save: boolean; name?: string }, signal?: AbortSignal) => {
    const value = await request<ModelTournamentResult>('/model-tournaments', { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, typeof value.ticker === 'string' && Array.isArray(value.results) &&
      Array.isArray(value.directly_comparable_metrics) && value.results.every((item) => typeof item.model === 'string' && ['completed', 'unavailable'].includes(item.status)),
      'Backend returned an invalid model-tournament result.')
  },
  modelTournaments: async (signal?: AbortSignal) => {
    const value = await request<{ tournaments: Array<Record<string, unknown>> }>('/model-tournaments', { signal })
    return ensure(value, Array.isArray(value.tournaments), 'Backend returned an invalid saved-tournament list.')
  },
  createExperiment: async (payload: { name: string; description?: string | null; model: SimulationTrajectories['model']; ticker?: string; parameters: Record<string, unknown> }, signal?: AbortSignal) => {
    const value = await request<{ experiment: ExperimentRecord; simulation: SimulationTrajectories }>('/experiments', { method: 'POST', body: JSON.stringify(payload), signal })
    return ensure(value, typeof value.experiment?.id === 'string' && value.experiment.status === 'completed' &&
      value.experiment.metrics !== null && Array.isArray(value.simulation?.trajectories),
      'Backend did not return a completed experiment and its generated trajectories.')
  },
  experiments: async (signal?: AbortSignal) => {
    const value = await request<{ experiments: ExperimentRecord[]; count: number }>('/experiments', { signal })
    return ensure(value, Array.isArray(value.experiments) && value.experiments.every((item) => typeof item.id === 'string'), 'Backend returned an invalid experiment list.')
  },
  experiment: async (id: string, signal?: AbortSignal) => request<ExperimentRecord>(`/experiments/${encodeURIComponent(id)}`, { signal }),
  experimentSummary: async (id: string, signal?: AbortSignal) => request<{ id: string; status: string; metrics: Record<string, unknown> | null; error: string | null }>(`/experiments/${encodeURIComponent(id)}/summary`, { signal }),
  compareExperiments: async (ids: string[], signal?: AbortSignal) => {
    const query = new URLSearchParams()
    ids.forEach((id) => query.append('experiment_ids', id))
    const value = await request<ExperimentComparison>(`/experiments/compare?${query}`, { signal })
    return ensure(value, Array.isArray(value.experiments) && typeof value.comparable === 'boolean', 'Backend returned an invalid experiment comparison.')
  },
  deleteExperiment: async (id: string, signal?: AbortSignal) => {
    await request<unknown>(`/experiments/${encodeURIComponent(id)}`, { method: 'DELETE', signal })
  },
}
