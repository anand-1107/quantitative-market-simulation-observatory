"""Bounded request and typed response contracts for trajectory simulations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class _PathGrid(BaseModel):
    """Interactive path responses are capped to 128 paths by 512 steps."""

    steps: int = Field(128, ge=1, le=512, description="Number of time intervals / return observations")
    paths: int = Field(64, ge=1, le=128, description="Number of independent simulated trajectories")
    seed: int | None = Field(42, ge=0, le=2**32 - 1)
    retain_forensics: bool = Field(False, description="Persist path innovations for exact path reconstruction")


class GBMTrajectoryRequest(_PathGrid):
    initial_price: float = Field(100, gt=0, le=1_000_000_000)
    drift: float = Field(.08, ge=-2, le=2, description="Expected proportional growth rate per year")
    volatility: float = Field(.2, ge=0, le=3, description="Diffusion volatility per square-root year")
    years: float = Field(1, gt=0, le=10)
    calibration: "CalibrationProvenance | None" = None


class CalibrationProvenance(BaseModel):
    model: Literal["gbm"] = "gbm"
    ticker: str = Field(..., min_length=1, max_length=32, pattern=r"^[A-Za-z0-9^=._-]+$")
    estimation_start: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    estimation_end: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    interval: Literal["1d", "5d", "1wk", "1mo", "3mo"]
    drift: float = Field(..., ge=-2, le=2)
    volatility: float = Field(..., gt=0, le=3)
    calibrated_at: str
    method: str = "historical_gbm_moments_v1"
    observation_count: int = Field(..., ge=3)
    return_count: int = Field(..., ge=2)


class GBMCalibrationRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=32, pattern=r"^[A-Za-z0-9^=._-]+$")
    start: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    interval: str = Field("1d", pattern=r"^(1m|2m|5m|15m|30m|60m|90m|1h|1d|5d|1wk|1mo|3mo)$")


class GBMCalibrationResponse(BaseModel):
    ticker: str
    estimation_start: str
    estimation_end: str
    interval: str
    price_field: str
    observation_count: int
    return_count: int
    annualization_periods: int
    annualization_convention: str
    drift: float
    volatility: float
    mean_period_log_return: float
    sample_std_period_log_return: float
    calibrated_at: str
    warnings: list[str]
    method: str
    assumptions: list[str]
    limitations: str


class GBMEvaluationRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=32, pattern=r"^[A-Za-z0-9^=._-]+$")
    estimation_start: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    estimation_end: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    evaluation_start: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    evaluation_end: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    interval: str = Field("1d", pattern=r"^(1m|2m|5m|15m|30m|60m|90m|1h|1d|5d|1wk|1mo|3mo)$")
    paths: int = Field(256, ge=16, le=5000)
    seed: int = Field(42, ge=0, le=2**32 - 1)


class GBMEvaluationResponse(BaseModel):
    ticker: str
    calibration: GBMCalibrationResponse
    evaluation: dict[str, Any]


class ModelTournamentRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=32, pattern=r"^[A-Za-z0-9^=._-]+$")
    models: list[Literal["gbm", "ou", "jump_diffusion", "garch"]] = Field(..., min_length=1, max_length=4)
    train_start: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    train_end: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Exclusive")
    test_start: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    test_end: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="Exclusive")
    interval: Literal["1d", "5d", "1wk", "1mo", "3mo"] = "1d"
    paths: int = Field(1000, ge=100, le=5000)
    seed: int = Field(42, ge=0, le=2**32 - 1)
    save: bool = False
    name: str | None = Field(None, min_length=1, max_length=100)


class TournamentPeriod(BaseModel):
    start: str
    end_exclusive: str
    first_observation: str
    last_observation: str
    observation_count: int = Field(..., ge=1)
    return_count: int | None = Field(None, ge=0)


class TournamentModelResult(BaseModel):
    model: Literal["gbm", "ou", "jump_diffusion", "garch"]
    status: Literal["completed", "unavailable"]
    parameters: dict[str, Any] | None
    metrics: dict[str, Any] | None
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitation: str = ""
    error: str | None = None
    price_field: str | None = None
    experiment_id: str | None = None
    forecast_distribution_surface: dict[str, Any] | None = None
    forecast_risk_surface: dict[str, Any] | None = None
    research_path: dict[str, Any] | None = None


class ModelTournamentResponse(BaseModel):
    ticker: str
    model_version: str | None = None
    interval: str
    training_period: TournamentPeriod
    test_period: TournamentPeriod
    comparison_method: str
    directly_comparable_metrics: list[str]
    model_specific_metrics: list[str]
    reproducibility: dict[str, Any]
    results: list[TournamentModelResult]
    tournament_id: str | None = None


class SavedTournamentModel(BaseModel):
    model: Literal["gbm", "ou", "jump_diffusion", "garch"]
    status: str
    parameters: dict[str, Any]
    metrics: dict[str, Any] | None
    error: str | None = None
    metadata: dict[str, Any]
    experiment_id: str
    forecast_distribution_surface: dict[str, Any] | None = None
    forecast_risk_surface: dict[str, Any] | None = None
    research_path: dict[str, Any] | None = None


class SavedTournament(BaseModel):
    tournament_id: str
    name: str
    ticker: str
    interval: str
    training_period: TournamentPeriod
    test_period: TournamentPeriod
    created_at: str
    models: list[SavedTournamentModel]


class SavedTournamentsResponse(BaseModel):
    tournaments: list[SavedTournament]


class OHLCVObservation(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float | None


class OHLCVSeriesResponse(BaseModel):
    ticker: str
    interval: str
    observation_count: int
    source_observation_count: int
    downsampled: bool
    timestamp_convention: str
    observations: list[OHLCVObservation]


class EmpiricalLossRisk(BaseModel):
    confidence_level: float
    var: float
    expected_shortfall: float
    tail_observation_count: int
    sample_size: int
    unit: Literal["simple_return_fraction"] = "simple_return_fraction"
    loss_definition: str = "L = -R"
    quantile_method: str
    tail_convention: str
    warnings: list[str] = []


class HistoricalPeriodRisk(EmpiricalLossRisk):
    measure: Literal["historical_one_period"] = "historical_one_period"
    interval: str


class MarketSummaryResponse(BaseModel):
    ticker: str
    row_count: int
    first_date: str
    last_date: str
    latest_timestamp: str
    latest_close: float
    interval: str
    annualization_periods: int
    mean_daily_log_return: float | None
    daily_volatility: float | None
    annualized_volatility: float | None
    maximum_drawdown: float | None
    drawdown_price_field: Literal["adj_close", "close"]
    historical_period_risk: HistoricalPeriodRisk | None = None
    historical_period_risk_unavailable_reason: str | None = None


class HistoricalCalibrationRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=32, pattern=r"^[A-Za-z0-9^=._-]+$")
    start: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    interval: Literal["1d", "5d", "1wk", "1mo", "3mo"] = "1d"
    price_field: Literal["auto", "adj_close", "close"] = "auto"


class OUCalibrationRequest(HistoricalCalibrationRequest):
    input_series: Literal["log_return", "log_price", "price"] = "log_return"
    confirm_mean_reverting_price_series: bool = False


class MertonCalibrationRequest(HistoricalCalibrationRequest):
    pass


class GARCHCalibrationRequest(HistoricalCalibrationRequest):
    pass


class MarketEventAnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=32, pattern=r"^[A-Za-z0-9^=._-]+$")
    start: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    interval: Literal["15m", "30m", "60m", "1h", "1d", "1wk", "1mo"] = "1d"
    benchmark: str | None = Field(None, max_length=32, pattern=r"^[A-Za-z0-9^=._-]+$")
    absolute_log_return_threshold: float = Field(.04, gt=0, le=1, allow_inf_nan=False)
    volume_multiple_threshold: float = Field(2.5, gt=1, le=100, allow_inf_nan=False)
    opening_gap_threshold: float = Field(.03, gt=0, le=1, allow_inf_nan=False)
    volatility_multiple_threshold: float = Field(2, gt=1, le=20, allow_inf_nan=False)
    sustained_log_return_threshold: float = Field(.08, gt=0, le=2, allow_inf_nan=False)
    sustained_periods: int = Field(5, ge=2, le=60)
    impact_window_observations: int = Field(10, ge=2, le=60)


class NewsMatchEvent(BaseModel):
    event_id: str = Field(..., max_length=160)
    timestamp: str = Field(..., max_length=64)
    category: str = Field(..., max_length=64)
    timestamp_resolution: Literal["session_date", "intraday_observation_utc"] = "session_date"


class MarketEventNewsRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=32, pattern=r"^[A-Za-z0-9^=._-]+$")
    start: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    events: list[NewsMatchEvent] = Field(default_factory=list, max_length=200)


class NewsProviderStatusResponse(BaseModel):
    provider: str
    configured: bool
    setup_message: str | None = None
    credentials_location: str
    historical_coverage: str


class HistoricalNewsArticleResponse(BaseModel):
    headline: str
    publisher: str
    original_url: str
    publication_timestamp: str | None = None
    publication_timezone: str | None = None
    retrieved_timestamp: str
    relevant_instrument: str | None = None
    provider_query_ticker: str
    excerpt: str | None = None
    category: str
    source_provider: str


class RankedNewsArticleResponse(BaseModel):
    rank: int
    score: float
    article: HistoricalNewsArticleResponse
    signals: dict[str, int | float | str]


class EventNewsMatchResponse(BaseModel):
    event_id: str
    category: str
    ranked_articles: list[RankedNewsArticleResponse]


class MarketEventNewsResponse(BaseModel):
    provider: str
    configured: bool
    searched: bool
    cached: bool
    articles: list[HistoricalNewsArticleResponse]
    event_matches: list[EventNewsMatchResponse]
    message: str | None = None
    retrieved_at: str | None = None
    publication_timestamps_are_distinct_from_retrieval_timestamps: bool | None = None
    matching_note: str | None = None


class OUCalibrationResponse(BaseModel):
    model: Literal["ou"]
    ticker: str
    estimation_start: str
    estimation_end: str
    interval: str
    price_field: str
    input_series: str
    input_units: str
    observation_count: int
    return_count: int
    dt_years: float
    mean_level: float
    mean_reversion_speed: float
    volatility: float
    initial_value: float
    ar1_coefficient: float
    conditional_innovation_variance: float
    calibrated_at: str
    method: str
    optimizer_success: bool
    warnings: list[str]
    assumptions: list[str]


class MertonCalibrationResponse(BaseModel):
    model: Literal["jump_diffusion"]
    ticker: str
    estimation_start: str
    estimation_end: str
    interval: str
    price_field: str
    input_series: str
    observation_count: int
    return_count: int
    dt_years: float
    drift: float
    volatility: float
    jump_intensity: float
    mean_log_jump: float
    jump_volatility: float
    initial_price: float
    estimated_expected_jumps_in_sample: float
    posterior_expected_jump_count_in_sample: float
    log_likelihood: float
    aic: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_iterations: int
    calibrated_at: str
    method: str
    warnings: list[str]
    assumptions: list[str]
    limitations: str


class ConditionalVolatilityPoint(BaseModel):
    date: str
    value: float


class GARCHCalibrationResponse(BaseModel):
    model: Literal["garch"]
    ticker: str
    estimation_start: str
    estimation_end: str
    interval: str
    price_field: str
    input_series: str
    return_convention: Literal["log"]
    observation_count: int
    return_count: int
    mean: float
    omega: float
    alpha: float
    beta: float
    stationary: bool
    initial_variance: float
    conditional_volatility: list[ConditionalVolatilityPoint]
    negative_log_likelihood: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_iterations: int
    calibrated_at: str
    method: str
    warnings: list[str]
    assumptions: list[str]
    limitations: str


class OUCalibrationProvenance(BaseModel):
    model: Literal["ou"] = "ou"
    ticker: str
    estimation_start: str
    estimation_end: str
    interval: Literal["1d", "5d", "1wk", "1mo", "3mo"]
    input_series: str
    mean_level: float
    mean_reversion_speed: float
    volatility: float
    initial_value: float
    calibrated_at: str
    method: str
    observation_count: int


class MertonCalibrationProvenance(BaseModel):
    model: Literal["jump_diffusion"] = "jump_diffusion"
    ticker: str
    estimation_start: str
    estimation_end: str
    interval: Literal["1d", "5d", "1wk", "1mo", "3mo"]
    drift: float
    volatility: float
    jump_intensity: float
    mean_log_jump: float
    jump_volatility: float
    initial_price: float
    calibrated_at: str
    method: str
    observation_count: int
    return_count: int


class GARCHCalibrationProvenance(BaseModel):
    model: Literal["garch"] = "garch"
    ticker: str
    estimation_start: str
    estimation_end: str
    interval: Literal["1d", "5d", "1wk", "1mo", "3mo"]
    mean: float
    omega: float
    alpha: float
    beta: float
    initial_variance: float
    return_convention: Literal["log"] = "log"
    calibrated_at: str
    method: str
    observation_count: int
    return_count: int


class OUTrajectoryRequest(_PathGrid):
    initial_value: float = Field(0, ge=-1_000_000_000, le=1_000_000_000)
    mean_level: float = Field(0, ge=-1_000_000_000, le=1_000_000_000)
    mean_reversion_speed: float = Field(1, ge=0, le=100_000, description="Per year")
    volatility: float = Field(1, ge=0, le=1_000_000, description="Level units per square-root year")
    years: float = Field(1, gt=0, le=10)
    calibration: OUCalibrationProvenance | None = None


class JumpDiffusionTrajectoryRequest(_PathGrid):
    initial_price: float = Field(100, gt=0, le=1_000_000_000)
    drift: float = Field(.08, ge=-2, le=2, description="Compensated expected proportional growth per year")
    volatility: float = Field(.2, ge=0, le=3, description="Diffusion volatility per square-root year")
    jump_intensity: float = Field(.5, ge=0, le=10, description="Expected jumps per year")
    mean_log_jump: float = Field(-.1, ge=-2, le=2)
    jump_volatility: float = Field(.2, ge=0, le=2, description="Log jump-size standard deviation")
    years: float = Field(1, gt=0, le=10)
    calibration: MertonCalibrationProvenance | None = None


class GARCHTrajectoryRequest(_PathGrid):
    omega: float = Field(1e-6, gt=0, le=1)
    alpha: float = Field(.05, ge=0, le=10)
    beta: float = Field(.9, ge=0, le=10)
    mean: float = Field(0, ge=-10, le=10, description="Mean return per observation period")
    initial_variance: float | None = Field(None, gt=0, le=100)
    return_convention: Literal["log", "simple"] = "log"
    calibration: GARCHCalibrationProvenance | None = None


class TrajectoryBandPoint(BaseModel):
    time: float
    mean: float
    p05: float
    p50: float
    p95: float


class TrajectorySummary(BaseModel):
    terminal_mean: float
    terminal_variance: float
    terminal_quantiles: dict[str, float]
    bands: list[TrajectoryBandPoint]


class SimulationTrajectoriesResponse(BaseModel):
    model: Literal["gbm", "ou", "jump_diffusion", "garch"]
    parameters: dict[str, Any]
    quantity_kind: Literal["price", "state", "return"]
    quantity_label: str
    time_label: str
    times: list[float]
    trajectories: list[list[float]]
    conditional_variances: list[list[float]] | None = None
    summary: TrajectorySummary
    terminal_risk: "TerminalRiskMetrics"
    forensics_run_id: str | None = None


class PathForensicsResponse(BaseModel):
    run_id: str
    path_id: int
    model: Literal["gbm", "ou", "jump_diffusion", "garch"]
    model_version: str
    created_at: str
    parameters: dict[str, Any]
    initial_conditions: dict[str, Any]
    times: list[float]
    original_path: list[float]
    innovations: dict[str, list[float]]
    conditional_variance: list[float] | None = None
    transitions: list[dict[str, Any]]
    terminal_metrics: dict[str, Any]
    reconstruction: dict[str, Any]


class ScenarioMatchingRequest(BaseModel):
    run_id: str = Field(..., pattern=r"^[0-9a-fA-F-]{36}$")
    path_id: int = Field(..., ge=1, le=128)
    candidate_parameters: list[dict[str, Any]] = Field(..., min_length=1, max_length=32)
    terminal_return_weight: float = Field(.25, ge=0, le=10, allow_inf_nan=False)
    drawdown_weight: float = Field(.25, ge=0, le=10, allow_inf_nan=False)


class ScenarioMatchCandidate(BaseModel):
    rank: int
    model: Literal["gbm", "ou", "jump_diffusion", "garch"]
    parameters: dict[str, Any]
    objective: float
    normalized_path_rmse: float
    terminal_return_difference: float | None
    drawdown_difference: float | None
    terminal_value: float
    candidate_path: list[float]
    total_simple_return: float | None
    maximum_drawdown: float | None
    exactly_replayable: bool
    replay_provenance: dict[str, Any]


class ScenarioMatchingResponse(BaseModel):
    reference_run_id: str
    reference_path_id: int
    method: str
    warnings: list[str]
    candidates: list[ScenarioMatchCandidate]


class RetainedForensicsRun(BaseModel):
    run_id: str
    created_at: str
    model: Literal["gbm", "ou", "jump_diffusion", "garch"]
    parameters: dict[str, Any]
    path_count: int
    step_count: int


class RetainedForensicsRunsResponse(BaseModel):
    runs: list[RetainedForensicsRun]


class PathCounterfactualRequest(BaseModel):
    run_id: str = Field(..., pattern=r"^[0-9a-fA-F-]{36}$")
    path_id: int = Field(..., ge=1, le=128)
    candidate_parameters: dict[str, Any]


class PathCounterfactualResponse(BaseModel):
    run_id: str
    path_id: int
    model: Literal["gbm", "ou", "jump_diffusion", "garch"]
    parameters: dict[str, Any]
    times: list[float]
    reference_path: list[float]
    candidate_path: list[float]
    differences: list[float]
    reference_metrics: dict[str, Any]
    candidate_metrics: dict[str, Any]
    raw_path_rmse: float
    exact_common_innovations: bool
    coupling_method: str
    warnings: list[str]
    transitions: list[dict[str, Any]]
    innovations: dict[str, list[float]]


class ExperimentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str | None = Field(None, max_length=2000)
    model: Literal["gbm", "ou", "jump_diffusion", "garch"]
    ticker: str | None = Field(None, min_length=1, max_length=32, pattern=r"^[A-Za-z0-9^=._-]+$")
    parameters: dict[str, Any]


class ExperimentRecord(BaseModel):
    id: str
    name: str
    description: str | None
    created_at: str
    updated_at: str
    model: Literal["gbm", "ou", "jump_diffusion", "garch"]
    parameters: dict[str, Any]
    metadata: dict[str, Any]
    metrics: dict[str, Any] | None
    status: Literal["pending", "running", "completed", "failed"]
    error: str | None


class ExperimentRunResponse(BaseModel):
    experiment: ExperimentRecord
    simulation: SimulationTrajectoriesResponse


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentRecord]
    count: int


class ExperimentComparisonRequest(BaseModel):
    experiment_ids: list[str] = Field(..., min_length=2, max_length=10)


class ExperimentComparisonResponse(BaseModel):
    experiments: list[ExperimentRecord]
    comparable: bool
    comparison_reason: str | None
    common_metrics: dict[str, dict[str, float | None]]


class SimulatedPathRisk(BaseModel):
    total_simple_return: float
    maximum_drawdown: float


class TerminalRiskMetrics(BaseModel):
    available: bool
    measure: Literal["terminal_horizon"] | None = None
    confidence_level: float | None = None
    unit: str | None = None
    loss_definition: str | None = None
    scenario_count: int | None = None
    mean_total_simple_return: float | None = None
    total_simple_return_quantiles: dict[str, float] | None = None
    maximum_drawdown_quantiles: dict[str, float] | None = None
    var: float | None = None
    expected_shortfall: float | None = None
    tail_scenario_count: int | None = None
    quantile_method: str | None = None
    tail_convention: str | None = None
    warnings: list[str] = []
    path_metrics: list[SimulatedPathRisk] | None = None
    unavailable_reason: str | None = None
    horizon_steps: int | None = None
    horizon_years: float | None = None
    horizon_unit: Literal["years", "observation_periods"] | None = None
