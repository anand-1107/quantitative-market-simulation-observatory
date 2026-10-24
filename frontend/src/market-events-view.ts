export type MarketEventAnalysisViewState = 'idle' | 'loading' | 'error' | 'results'

export const DETECTION_NUMERIC_DEFAULTS = [
  { label: 'Absolute log-return threshold', value: 0.04, min: 0.0001, max: 1, step: 'any' },
  { label: 'Volume multiple vs. prior median', value: 2.5, min: 1.01, max: 100, step: 'any' },
  { label: 'Opening gap threshold', value: 0.03, min: 0.0001, max: 1, step: 'any' },
  { label: 'Volatility multiple', value: 2, min: 1.01, max: 20, step: 'any' },
  { label: 'Sustained log return', value: 0.08, min: 0.0001, max: 2, step: 'any' },
] as const

/** Convert API and browser rejection values into a useful message without `[object Object]`. */
export function marketEventErrorMessage(reason: unknown, fallback: string): string {
  if (typeof reason === 'string' && reason.trim()) return reason
  if (reason instanceof Error && reason.message.trim()) return reason.message
  if (reason && typeof reason === 'object') {
    const value = reason as Record<string, unknown>
    for (const key of ['message', 'detail', 'error']) {
      const candidate = value[key]
      if (typeof candidate === 'string' && candidate.trim()) return candidate
      if (Array.isArray(candidate)) {
        const messages = candidate.map((item) => {
          if (!item || typeof item !== 'object') return typeof item === 'string' ? item : ''
          const entry = item as Record<string, unknown>
          return typeof entry.msg === 'string' ? entry.msg : typeof entry.message === 'string' ? entry.message : ''
        }).filter(Boolean)
        if (messages.length) return messages.join('; ')
      }
      if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
        const nested = candidate as Record<string, unknown>
        if (typeof nested.message === 'string' && nested.message.trim()) return nested.message
      }
    }
  }
  return fallback
}

export function numericControlWithinBounds(value: number, min: number, max: number): boolean {
  return Number.isFinite(value) && value >= min && value <= max
}

/** Validate every field consumed by the completed-analysis view before it enters React state. */
export function isMarketEventAnalysis(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false
  const result = value as Record<string, unknown>
  const period = result.estimation_period
  const validPeriod = !!period && typeof period === 'object'
    && typeof (period as Record<string, unknown>).start === 'string'
    && typeof (period as Record<string, unknown>).end_exclusive === 'string'
  const validBars = Array.isArray(result.bars) && result.bars.every((item) => {
    if (!item || typeof item !== 'object') return false
    const bar = item as Record<string, unknown>
    return typeof bar.date === 'string' && ['open', 'high', 'low', 'close'].every((key) => typeof bar[key] === 'number' && Number.isFinite(bar[key]))
      && (bar.volume === null || (typeof bar.volume === 'number' && Number.isFinite(bar.volume)))
  })
  const validEvents = Array.isArray(result.candidate_events) && result.candidate_events.every((item) => {
    if (!item || typeof item !== 'object') return false
    const event = item as Record<string, unknown>
    return typeof event.event_id === 'string' && typeof event.timestamp === 'string' && typeof event.date === 'string'
      && typeof event.category === 'string' && typeof event.method === 'string' && Number.isFinite(event.close)
      && !!event.evidence && typeof event.evidence === 'object'
      && !!event.impact && typeof event.impact === 'object'
      && !!event.model_comparison && typeof event.model_comparison === 'object'
      && (event.event_log_return === null || (typeof event.event_log_return === 'number' && Number.isFinite(event.event_log_return)))
  })
  return typeof result.ticker === 'string' && typeof result.interval === 'string' && validPeriod
    && Number.isFinite(result.observation_count) && Number.isFinite(result.source_observation_count)
    && Number.isFinite(result.excluded_observation_count) && typeof result.downsampled === 'boolean'
    && validBars && validEvents && !!result.detection_configuration && typeof result.detection_configuration === 'object'
    && Number.isFinite(result.impact_window_observations_each_side)
    && Array.isArray(result.warnings) && result.warnings.every((warning) => typeof warning === 'string')
    && typeof result.data_convention === 'string' && typeof result.causality_note === 'string'
    && (result.benchmark_ticker === null || typeof result.benchmark_ticker === 'string')
    && typeof result.chart_note === 'string'
}

export function marketEventAnalysisViewState(loading: boolean, error: string, hasResults: boolean): MarketEventAnalysisViewState {
  if (loading) return 'loading'
  if (error) return 'error'
  if (hasResults) return 'results'
  return 'idle'
}
