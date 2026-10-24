import type { ForecastDistributionSurface, ForecastRiskSurface, TournamentModel, TournamentModelResult } from './types'
import type { SimulationTrajectories } from './types'

export type ResearchMode = 'distribution' | 'paths' | 'risk' | 'disagreement'

export type ProgressivePlayback = { progress: number; started: boolean; playing: boolean }
export const PLAYBACK_SCALE = 1000
export const initialProgressivePlayback = (): ProgressivePlayback => ({ progress: 0, started: false, playing: false })
export function playProgressive(state: ProgressivePlayback): ProgressivePlayback {
  return { progress: state.progress >= PLAYBACK_SCALE ? 0 : state.progress, started: true, playing: true }
}
export function pauseProgressive(state: ProgressivePlayback): ProgressivePlayback {
  return { ...state, playing: false }
}
export function resetProgressive(): ProgressivePlayback { return initialProgressivePlayback() }
export function seekProgressive(state: ProgressivePlayback, progress: number): ProgressivePlayback {
  return { progress: Math.max(0, Math.min(PLAYBACK_SCALE, Math.round(Number.isFinite(progress) ? progress : 0))),
    started: state.started || Number.isFinite(progress), playing: false }
}
export function advanceProgressive(state: ProgressivePlayback, originCount: number): ProgressivePlayback {
  if (!state.playing) return state
  const increment = Math.max(1, Math.ceil(PLAYBACK_SCALE / Math.max(1, originCount - 1)))
  const progress = Math.min(PLAYBACK_SCALE, state.progress + increment)
  return { progress, started: true, playing: progress < PLAYBACK_SCALE }
}
export function progressiveIndex(progress: number, dataCount: number, started: boolean): number {
  if (!started || dataCount <= 0) return -1
  return Math.max(0, Math.min(dataCount - 1, Math.floor(Math.max(0, Math.min(PLAYBACK_SCALE, progress)) / PLAYBACK_SCALE * dataCount)))
}
export function toggleResearchModelVisibility(visible: TournamentModel[], model: TournamentModel): TournamentModel[] {
  return visible.includes(model) ? visible.filter((item) => item !== model) : [...visible, model]
}

export type NumericRange = { min: number; max: number }
export type AxisTick = { value: number; position: number; label: string }
export type AxisTitles = { x: string; y: string; z: string }
export const RESEARCH_AXIS_TITLES: Record<ResearchMode, AxisTitles> = {
  distribution: { x: 'Evaluation origin date', y: 'One-step log return / observation interval', z: 'Conditional predictive density / log-return unit' },
  paths: { x: 'Simulation horizon (years)', y: 'Cumulative log return', z: 'Model lane (categorical)' },
  risk: { x: 'Evaluation origin date', y: 'Forecast confidence level', z: 'One-step loss VaR fraction' },
  disagreement: { x: 'Evaluation origin date', y: 'One-step log return / observation interval', z: 'Density A − density B / log-return unit' },
}

export function displayRange(min: number, max: number): NumericRange {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return { min: 0, max: 1 }
  const low = Math.min(min, max), high = Math.max(min, max)
  if (high > low) return { min: low, max: high }
  const pad = Math.max(Math.abs(low) * .05, 1e-6)
  return { min: low - pad, max: high + pad }
}

export function mapDataToWorld(value: number, data: NumericRange, world: NumericRange): number {
  const safe = displayRange(data.min, data.max)
  if (!Number.isFinite(value)) return (world.min + world.max) / 2
  return world.min + (value - safe.min) / (safe.max - safe.min) * (world.max - world.min)
}

export function mapWorldToData(value: number, data: NumericRange, world: NumericRange): number {
  const safe = displayRange(data.min, data.max)
  if (!Number.isFinite(value) || world.max === world.min) return (safe.min + safe.max) / 2
  return safe.min + (value - world.min) / (world.max - world.min) * (safe.max - safe.min)
}

export function formatAxisValue(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return '—'
  if (value === 0) return '0'
  const magnitude = Math.abs(value)
  return magnitude < .001 || magnitude >= 10000 ? value.toExponential(1) : Number(value.toPrecision(digits)).toString()
}

export function formatShortDate(value: string, includeYear = false): string {
  const date = new Date(`${value.slice(0, 10)}T00:00:00Z`)
  if (!Number.isFinite(date.getTime())) return value
  return new Intl.DateTimeFormat('en-US', { month: 'short', day: '2-digit', ...(includeYear ? { year: '2-digit' as const } : {}), timeZone: 'UTC' }).format(date)
}

export function linearAxisTicks(range: NumericRange, world: NumericRange, formatter = formatAxisValue,
  targetCount = 5, includeZero = false): AxisTick[] {
  const safe = displayRange(range.min, range.max)
  const span = safe.max - safe.min
  const raw = span / Math.max(1, targetCount - 1)
  const power = 10 ** Math.floor(Math.log10(raw || 1))
  const fraction = raw / power
  const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 2.5 ? 2.5 : fraction <= 5 ? 5 : 10
  const step = niceFraction * power
  const values = [safe.min, safe.max]
  for (let value = Math.ceil(safe.min / step) * step; value < safe.max; value += step) values.push(value)
  if (includeZero && safe.min < 0 && safe.max > 0) values.push(0)
  return [...new Set(values.map((value) => Number(value.toPrecision(12))))].sort((a, b) => a - b).map((value) => ({
    value, position: mapDataToWorld(value, safe, world), label: formatter(value),
  }))
}

export function dateAxisTicks(dates: string[], firstIndex: number, lastIndex: number,
  world: NumericRange = { min: -5, max: 5 }, targetCount = 5): AxisTick[] {
  if (!dates.length) return []
  const first = Math.max(0, Math.min(dates.length - 1, Math.floor(Math.min(firstIndex, lastIndex))))
  const last = Math.max(0, Math.min(dates.length - 1, Math.ceil(Math.max(firstIndex, lastIndex))))
  const count = Math.max(1, Math.min(targetCount, last - first + 1))
  const indexes = count === 1 ? [first] : Array.from({ length: count }, (_, index) => Math.round(first + index * (last - first) / (count - 1)))
  const spansYears = Number(dates[last]?.slice(0, 4)) !== Number(dates[first]?.slice(0, 4))
  return [...new Set(indexes)].map((index) => ({
    value: index,
    position: first === last ? (world.min + world.max) / 2 : world.min + (index - first) / (last - first) * (world.max - world.min),
    label: formatShortDate(dates[index], spansYears),
  }))
}

export function windowWorldTransform(firstIndex: number, lastIndex: number, count: number,
  world: NumericRange = { min: -5, max: 5 }): { scale: number; offset: number } {
  const original = (index: number) => count <= 1 ? (world.min + world.max) / 2 : mapDataToWorld(index, { min: 0, max: count - 1 }, world)
  if (lastIndex <= firstIndex || count <= 1) return { scale: 1, offset: (world.min + world.max) / 2 - original(firstIndex) }
  const sourceSpan = original(lastIndex) - original(firstIndex)
  const scale = (world.max - world.min) / sourceSpan
  return { scale, offset: world.min - original(firstIndex) * scale }
}

export function validForecastSurface(surface: ForecastDistributionSurface | null | undefined): boolean {
  return Boolean(surface && surface.evaluation_dates.length > 0
    && surface.origin_indices.length === surface.evaluation_dates.length
    && surface.origin_indices.every((value, index) => Number.isInteger(value) && (index === 0 || value > (surface.origin_indices[index - 1] ?? -Infinity)))
    && surface.log_return_grid.length >= 2 && surface.log_return_grid.every((value, index) => Number.isFinite(value)
      && (index === 0 || value > surface.log_return_grid[index - 1]))
    && surface.density_by_origin.length === surface.evaluation_dates.length
    && surface.density_by_origin.every((row) => row.length === surface.log_return_grid.length && row.every((value) => Number.isFinite(value) && value >= 0)))
}

export function researchModels(results: TournamentModelResult[]): TournamentModelResult[] {
  return results.filter((item) => item.status === 'completed' && item.forecast_distribution_surface
    && validForecastSurface(item.forecast_distribution_surface) && validRiskSurface(item.forecast_risk_surface) && item.parameters)
}

export function researchState(resultCount: number, availableCount: number, loading: boolean, error: string): 'loading' | 'error' | 'empty' | 'ready' {
  if (loading) return 'loading'
  if (error) return 'error'
  if (!resultCount || !availableCount) return 'empty'
  return 'ready'
}

function sameNumbers(a: number[], b: number[], tolerance = 1e-12): boolean {
  return a.length === b.length && a.every((value, index) => Number.isFinite(value)
    && Number.isFinite(b[index]) && Math.abs(value - b[index]) <= tolerance * Math.max(1, Math.abs(value), Math.abs(b[index])))
}

export type DensityDifference = { grid: number[]; dates: string[]; originIndices: number[]; rows: number[][] }
export function densityDifference(a: ForecastDistributionSurface | null | undefined,
  b: ForecastDistributionSurface | null | undefined): { compatible: true; value: DensityDifference } | { compatible: false; reason: string } {
  if (!a || !b || !validForecastSurface(a) || !validForecastSurface(b)) return { compatible: false, reason: 'Both models need completed, valid out-of-sample predictive distributions.' }
  if (a.unit !== b.unit || a.unit !== 'natural_log_return_per_observation_interval')
    return { compatible: false, reason: 'The selected models do not share a log-return target and units.' }
  if (!sameNumbers(a.log_return_grid, b.log_return_grid) || !sameNumbers(a.origin_indices, b.origin_indices)
    || a.evaluation_dates.length !== b.evaluation_dates.length
    || a.evaluation_dates.some((date, index) => date !== b.evaluation_dates[index]))
    return { compatible: false, reason: 'The models do not share the same return grid and evaluation origins.' }
  if (a.density_by_origin.length !== b.density_by_origin.length
    || a.density_by_origin.some((row, index) => row.length !== a.log_return_grid.length || b.density_by_origin[index]?.length !== b.log_return_grid.length
      || row.some((value, column) => !Number.isFinite(value) || !Number.isFinite(b.density_by_origin[index][column]))))
    return { compatible: false, reason: 'A model density surface has invalid dimensions.' }
  return { compatible: true, value: { grid: a.log_return_grid, dates: a.evaluation_dates,
    originIndices: a.origin_indices, rows: a.density_by_origin.map((row, index) => row.map((value, column) => value - b.density_by_origin[index][column])) } }
}

export function validRiskSurface(surface: ForecastRiskSurface | null | undefined): boolean {
  return Boolean(surface && surface.evaluation_dates.length > 0
    && surface.confidence_levels.length >= 2 && surface.confidence_levels.every((value, index) => Number.isFinite(value)
      && value > 0 && value < 1 && (index === 0 || value > (surface.confidence_levels[index - 1] ?? -Infinity)))
    && surface.origin_indices.length === surface.evaluation_dates.length
    && surface.var_loss_fraction_by_origin.length === surface.evaluation_dates.length
    && surface.var_loss_fraction_by_origin.every((row) => row.length === surface.confidence_levels.length && row.every(Number.isFinite))
    && surface.expected_shortfall_95_loss_fraction_by_origin.length === surface.evaluation_dates.length
    && surface.expected_shortfall_95_loss_fraction_by_origin.every(Number.isFinite)
    && surface.probability_negative_return_by_origin.length === surface.evaluation_dates.length
    && surface.probability_negative_return_by_origin.every((value) => Number.isFinite(value) && value >= 0 && value <= 1))
}

export function cumulativeLogReturnPath(data: SimulationTrajectories): { times: number[]; values: number[]; unit: string } {
  const paths = cumulativeLogReturnPaths(data)
  return { times: paths.times, values: paths.paths[0], unit: paths.unit }
}

export function cumulativeLogReturnPaths(data: SimulationTrajectories): { times: number[]; paths: number[][]; unit: string } {
  const source = data.trajectories[0]
  if (!source?.length || !data.times.length) throw new Error('This simulation returned no trajectory observations.')
  if (data.quantity_kind === 'price') {
    const paths = data.trajectories.map((path) => {
      const start = path[0]
      if (!(start > 0) || path.some((value) => !(value > 0) || !Number.isFinite(value))) throw new Error('Price paths must remain finite and positive.')
      return path.map((value) => Math.log(value / start))
    })
    return { times: data.times, paths, unit: 'cumulative log return' }
  }
  if (data.quantity_kind === 'state') {
    if (data.trajectories.some((path) => path.length !== data.times.length || path.some((value) => !Number.isFinite(value)))) throw new Error('OU state path dimensions are invalid.')
    const paths = data.trajectories.map((path) => { let total = 0; return path.map((value, index) => { if (index > 0) total += value; return total }) })
    return { times: data.times, paths, unit: 'cumulative sum of per-interval OU return states' }
  }
  if (data.trajectories.some((path) => path.length !== data.times.length || path.some((value) => !Number.isFinite(value)))) throw new Error('GARCH return path dimensions are invalid.')
  const simple = data.parameters.return_convention === 'simple'
  const paths = data.trajectories.map((path) => {
    let total = 0, wealth = 1
    return [0, ...path.map((value) => {
      if (simple) { if (value <= -1) throw new Error('Simple returns at or below -100% cannot form a comparable wealth path.'); wealth *= 1 + value; total = Math.log(wealth) }
      else total += value
      return total
    })]
  })
  const times = [0, ...data.times.map((_, index) => (index + 1) / Math.max(1, data.times.length))]
  return { times, paths, unit: simple ? 'cumulative log return derived from simple returns' : 'cumulative log return' }
}

export const TOURNAMENT_MODEL_ORDER: TournamentModel[] = ['gbm', 'ou', 'jump_diffusion', 'garch']
