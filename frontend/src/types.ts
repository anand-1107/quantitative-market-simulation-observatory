export type Page = 'overview' | 'market' | 'models' | 'formula' | 'universe' | 'risk' | 'experiments' | 'events' | 'tournament'
export type TournamentModel = 'gbm' | 'ou' | 'jump_diffusion' | 'garch'
export interface TournamentModelResult {
  model: TournamentModel
  status: 'completed' | 'unavailable'
  parameters: Record<string, unknown> | null
  metrics: Record<string, unknown> | null
  warnings: string[]
  assumptions: string[]
  limitation: string
  error?: string
  experiment_id?: string
  forecast_distribution_surface?: ForecastDistributionSurface | null
  forecast_risk_surface?: ForecastRiskSurface | null
  research_path?: Record<string, number> | null
}
export interface ForecastDistributionSurface {
  evaluation_dates: string[]
  origin_indices: number[]
  log_return_grid: number[]
  density_by_origin: number[][]
  density_method: string
  unit: string
}
export interface ForecastRiskSurface {
  evaluation_dates: string[]
  origin_indices: number[]
  confidence_levels: number[]
  var_loss_fraction_by_origin: number[][]
  expected_shortfall_95_loss_fraction_by_origin: number[]
  probability_negative_return_by_origin: number[]
  loss_definition: string
}
export interface ModelTournamentResult {
  ticker: string
  model_version?: string
  interval: string
  training_period: Record<string, unknown>
  test_period: Record<string, unknown>
  comparison_method: string
  directly_comparable_metrics: string[]
  model_specific_metrics: string[]
  reproducibility: { seed: number; paths: number; same_runtime_seeded_replay: boolean }
  results: TournamentModelResult[]
  tournament_id?: string
}

export interface ExperimentRecord {
  id: string
  name: string
  description: string | null
  created_at: string
  updated_at: string
  model: SimulationTrajectories['model']
  parameters: Record<string, unknown>
  metadata: Record<string, unknown>
  metrics: Record<string, unknown> | null
  status: 'pending' | 'running' | 'completed' | 'failed'
  error: string | null
}

export interface ExperimentComparison {
  experiments: ExperimentRecord[]
  comparable: boolean
  comparison_reason: string | null
  common_metrics: Record<string, Record<string, number | null>>
}

export interface ModelDescription {
  name: string
  description: string
  parameters: string[]
  output_convention: string
  calibration_status?: 'implemented_historical_moments' | 'implemented_conditional_least_squares' | 'implemented_compound_poisson_mle' | 'implemented_conditional_mle' | 'manual_only' | 'experimental'
}

export interface HealthResponse {
  status: string
  service: string
}

export interface ValidationReport {
  valid: boolean
  row_count: number
  missing_columns: string[]
  duplicate_dates: number
  missing_values: Record<string, number>
  invalid_prices: number
  invalid_ohlc_rows: number
  date_range: { start: string | null; end: string | null }
  warnings: string[]
}

export interface MarketSummary {
  ticker: string
  row_count: number
  first_date: string
  last_date: string
  latest_close: number
  mean_daily_log_return: number | null
  daily_volatility: number | null
  annualized_volatility: number | null
  maximum_drawdown: number | null
  interval?: string
  annualization_periods?: number
  latest_timestamp?: string
  historical_period_risk?: HistoricalPeriodRisk | null
  historical_period_risk_unavailable_reason?: string | null
  drawdown_price_field?: 'adj_close' | 'close'
}

export interface HistoricalPeriodRisk {
  measure: 'historical_one_period'
  interval: string
  confidence_level: number
  var: number
  expected_shortfall: number
  tail_observation_count: number
  sample_size: number
  unit: 'simple_return_fraction'
  loss_definition: string
  quantile_method: string
  tail_convention: string
  warnings: string[]
}

export interface PriceObservation {
  date: string
  price: number
}

export interface PriceSeries {
  ticker: string
  price_column: string
  interval: string
  observation_count: number
  source_observation_count: number
  downsampled: boolean
  observations: PriceObservation[]
}

export interface OHLCVObservation {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number | null
}

export interface OHLCVSeries {
  ticker: string
  interval: string
  observation_count: number
  source_observation_count: number
  downsampled: boolean
  timestamp_convention: string
  observations: OHLCVObservation[]
}

export interface HistoricalMarketEvent {
  event_id: string
  ticker: string
  timestamp: string
  timestamp_resolution: 'session_date' | 'intraday_observation_utc'
  date: string
  category: string
  method: string
  evidence: Record<string, number | string>
  close: number
  event_log_return: number | null
  impact: Record<string, number | string | boolean | null>
  model_comparison: Record<string, unknown>
}
export interface MarketEventAnalysis {
  ticker: string
  interval: string
  estimation_period: { start: string; end_exclusive: string }
  observation_count: number
  source_observation_count: number
  excluded_observation_count: number
  downsampled: boolean
  bars: OHLCVObservation[]
  candidate_events: HistoricalMarketEvent[]
  detection_configuration: Record<string, number>
  impact_window_observations_each_side: number
  warnings: string[]
  data_convention: string
  causality_note: string
  benchmark_ticker: string | null
  chart_note: string
}
export interface HistoricalNewsArticle {
  headline: string
  publisher: string
  original_url: string
  publication_timestamp: string | null
  publication_timezone: string | null
  retrieved_timestamp: string
  relevant_instrument: string | null
  provider_query_ticker: string
  excerpt: string | null
  category: string
  source_provider: string
}
export interface NewsProviderStatus {
  provider: string
  configured: boolean
  setup_message: string | null
  credentials_location: string
  historical_coverage: string
}
export interface EventNewsMatch {
  event_id: string
  category: string
  ranked_articles: Array<{ rank: number; score: number; article: HistoricalNewsArticle; signals: Record<string, number | string> }>
}
export interface MarketEventNews {
  provider: string
  configured: boolean
  searched: boolean
  cached: boolean
  articles: HistoricalNewsArticle[]
  event_matches: EventNewsMatch[]
  message?: string
  matching_note?: string
  retrieved_at: string | null
}

export interface DownloadMetadata {
  ticker: string
  downloaded_at?: string
  stored_at?: string
  start: string
  end: string
  interval: string
  row_count: number
}

export interface QuantileSummary {
  terminal_mean?: number
  terminal_variance?: number
  terminal_quantiles?: Record<string, number>
}

export interface TrajectoryBandPoint {
  time: number
  mean: number
  p05: number
  p50: number
  p95: number
}

export interface TrajectorySummary {
  terminal_mean: number
  terminal_variance: number
  terminal_quantiles: Record<string, number>
  bands: TrajectoryBandPoint[]
}

export interface SimulationTrajectories {
  model: 'gbm' | 'ou' | 'jump_diffusion' | 'garch'
  parameters: Record<string, number | string | null> & { calibration?: SimulationCalibrationProvenance | null }
  quantity_kind: 'price' | 'state' | 'return'
  quantity_label: string
  time_label: string
  times: number[]
  trajectories: number[][]
  conditional_variances: number[][] | null
  summary: TrajectorySummary
  terminal_risk: TerminalRiskMetrics
  forensics_run_id?: string | null
}

export interface PathForensics {
  run_id: string
  path_id: number
  model: SimulationTrajectories['model']
  model_version: string
  created_at: string
  parameters: Record<string, unknown>
  initial_conditions: Record<string, unknown>
  times: number[]
  original_path: number[]
  innovations: Record<string, number[]>
  conditional_variance: number[] | null
  transitions: Array<Record<string, number | boolean | null>>
  terminal_metrics: Record<string, unknown>
  reconstruction: {
    path: number[]
    maximum_absolute_error: number
    verification_tolerance: number
    verified: boolean
    method: string
  }
}

export interface RetainedForensicsRun {
  run_id: string
  model: SimulationTrajectories['model']
  created_at: string
  path_count: number
  step_count: number
  parameters: Record<string, unknown>
}

export interface CounterfactualPathResult {
  run_id: string
  path_id: number
  model: SimulationTrajectories['model']
  parameters: Record<string, unknown>
  times: number[]
  reference_path: number[]
  candidate_path: number[]
  differences: number[]
  reference_metrics: Record<string, unknown>
  candidate_metrics: Record<string, unknown>
  raw_path_rmse: number
  exact_common_innovations: boolean
  coupling_method: string
  warnings: string[]
  transitions: Array<Record<string, unknown>>
  innovations: Record<string, number[]>
}

export interface ScenarioMatchingResult {
  reference_run_id: string
  reference_path_id: number
  method: string
  warnings: string[]
  candidates: Array<{
    rank: number
    model: SimulationTrajectories['model']
    parameters: Record<string, unknown>
    objective: number
    normalized_path_rmse: number
    terminal_return_difference: number | null
    drawdown_difference: number | null
    terminal_value: number
    candidate_path: number[]
    total_simple_return: number | null
    maximum_drawdown: number | null
    exactly_replayable: boolean
    replay_provenance: Record<string, unknown>
  }>
}

export interface TerminalRiskMetrics {
  available: boolean
  measure?: 'terminal_horizon' | null
  confidence_level?: number | null
  unit?: string | null
  loss_definition?: string | null
  scenario_count?: number | null
  mean_total_simple_return?: number | null
  total_simple_return_quantiles?: Record<string, number> | null
  maximum_drawdown_quantiles?: Record<string, number> | null
  var?: number | null
  expected_shortfall?: number | null
  tail_scenario_count?: number | null
  quantile_method?: string | null
  tail_convention?: string | null
  warnings: string[]
  path_metrics?: Array<{ total_simple_return: number; maximum_drawdown: number }> | null
  unavailable_reason?: string | null
  horizon_steps?: number | null
  horizon_years?: number | null
  horizon_unit?: 'years' | 'observation_periods' | null
}

export interface SimulationCalibrationProvenance {
  model: 'gbm' | 'ou' | 'jump_diffusion' | 'garch'
  ticker: string
  estimation_start: string
  estimation_end: string
  interval: string
  drift?: number
  volatility?: number
  mean_level?: number
  mean_reversion_speed?: number
  initial_value?: number
  jump_intensity?: number
  mean_log_jump?: number
  jump_volatility?: number
  initial_price?: number
  omega?: number
  alpha?: number
  beta?: number
  mean?: number
  initial_variance?: number
  return_convention?: 'log'
  calibrated_at: string
  method: string
  observation_count: number
  return_count: number
}

export interface GBMCalibration {
  ticker: string
  estimation_start: string
  estimation_end: string
  interval: string
  price_field: string
  observation_count: number
  return_count: number
  annualization_periods: number
  annualization_convention: string
  drift: number
  volatility: number
  mean_period_log_return: number
  sample_std_period_log_return: number
  calibrated_at: string
  warnings: string[]
  method: string
  assumptions: string[]
  limitations: string
}

export interface GBMEvaluation {
  ticker: string
  calibration: GBMCalibration
  evaluation: {
    comparison_type: string
    estimation_period: { start: string; end: string; interval: string; observations: number; returns: number }
    evaluation_period: { start: string; end: string; interval: string; observations: number; returns: number }
    metrics: Record<string, number | Record<string, number> | null>
    methodology: Record<string, string>
  }
}

export interface OUCalibration extends SimulationCalibrationProvenance {
  model: 'ou'
  input_series: 'log_return' | 'log_price' | 'price'
  input_units: string
  observation_count: number
  return_count: number
  mean_level: number
  mean_reversion_speed: number
  volatility: number
  initial_value: number
  ar1_coefficient: number
  optimizer_success: boolean
  warnings: string[]
  assumptions: string[]
}

export interface MertonCalibration extends SimulationCalibrationProvenance {
  model: 'jump_diffusion'
  observation_count: number
  return_count: number
  drift: number
  volatility: number
  jump_intensity: number
  mean_log_jump: number
  jump_volatility: number
  initial_price: number
  estimated_expected_jumps_in_sample: number
  posterior_expected_jump_count_in_sample: number
  optimizer_success: boolean
  optimizer_message: string
  warnings: string[]
  assumptions: string[]
  limitations: string
}

export interface GARCHCalibration extends SimulationCalibrationProvenance {
  model: 'garch'
  observation_count: number
  return_count: number
  mean: number
  omega: number
  alpha: number
  beta: number
  initial_variance: number
  return_convention: 'log'
  stationary: boolean
  conditional_volatility: { date: string; value: number }[]
  optimizer_success: boolean
  optimizer_message: string
  warnings: string[]
  assumptions: string[]
  limitations: string
}

export type SimulationResult = Record<string, unknown> & {
  model: string
  parameters?: Record<string, unknown>
  paths_simulated?: number
  seed?: number | null
  terminal_price?: { mean: number; median: number; std: number; min: number; max: number }
  theoretical_terminal_mean?: number
  mean_standard_error?: number | null
  terminal_mean?: number
  terminal_variance?: number
  terminal_quantiles?: Record<string, number>
  price_terminal?: QuantileSummary
  return_mean?: number
  return_variance?: number
  return_lag1_autocorrelation?: number | null
  squared_return_lag1_autocorrelation?: number | null
  stationary_parameterization?: boolean
}
