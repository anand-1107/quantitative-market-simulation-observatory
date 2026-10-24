import type { PathForensics, SimulationTrajectories } from './types'

export const SIMULATION_PATH_PALETTE = [
  '#38cbe0', '#5a96ff', '#9278f2', '#40c8a0', '#e6ad59', '#5dd3c7', '#b17ce0', '#529bd0',
  '#28aabe', '#78acff', '#ad92fa', '#5fdaa9', '#c99748', '#35b7a0', '#8a62ca', '#397fb2',
  '#62d8e6', '#3c7de0', '#795bdf', '#279977', '#f0c77b', '#76c9d7', '#bd92d8', '#62a8c4',
] as const

export function pathAppearance(pathIndex: number, selectedPath: number) {
  return pathIndex === selectedPath
    ? { color: '#ffc45e', opacity: 1, width: 2.5, selected: true }
    : { color: SIMULATION_PATH_PALETTE[pathIndex % SIMULATION_PATH_PALETTE.length], opacity: .54, width: .8, selected: false }
}

export function visiblePathVertexCount(pathCount: number, currentIndex: number): number {
  return Math.max(0, currentIndex) * Math.max(0, pathCount) * 2
}

/** Align a same-run counterfactual to its retained reference without mutating either source path. */
export function scenarioComparisonPaths(reference: number[], candidate: number[]): { reference: number[]; candidate: number[] } | null {
  const length = Math.min(reference.length, candidate.length)
  if (length < 2) return null
  return { reference: reference.slice(0, length), candidate: candidate.slice(0, length) }
}

export interface DensitySurface {
  valueGrid: number[]
  densityByTime: number[][]
  valueMin: number
  valueMax: number
  densityMax: number
  bandwidths: number[]
  method: string
}

/** Gaussian KDE on a shared value grid; bandwidth has a half-grid resolution floor. */
export function estimateDensitySurface(paths: number[][], binCount = 40): DensitySurface {
  if (!Array.isArray(paths) || paths.length < 2) throw new Error('At least two simulated paths are required to estimate a distribution.')
  if (!Number.isInteger(binCount) || binCount < 32 || binCount > 128) throw new Error('Density grid must contain between 32 and 128 value points.')
  const pointCount = paths[0]?.length ?? 0
  if (pointCount < 1 || paths.some((path) => path.length !== pointCount)) throw new Error('Simulated paths must have the same non-zero number of time points.')
  let rawMin = Infinity, rawMax = -Infinity
  for (const path of paths) for (const value of path) {
    if (!Number.isFinite(value)) throw new Error('Density estimation requires finite simulated values.')
    rawMin = Math.min(rawMin, value); rawMax = Math.max(rawMax, value)
  }
  const rawSpan = rawMax - rawMin
  const scale = Math.max(Math.abs(rawMin), Math.abs(rawMax), 1)
  if (!Number.isFinite(rawSpan)) throw new Error('The simulated value spread is too large for density estimation.')
  const sampleStats = paths[0].map((_, timeIndex) => {
    const sample = paths.map((path) => path[timeIndex])
    const mean = sample.reduce((sum, value) => sum + value / sample.length, 0)
    const variance = sample.reduce((sum, value) => sum + ((value - mean) / Math.sqrt(sample.length - 1)) ** 2, 0)
    const sd = Math.sqrt(variance)
    const bandwidth = Math.max(1.06 * sd * sample.length ** (-.2), scale * 1e-4)
    return { sample, bandwidth }
  })
  const maxBandwidth = Math.max(...sampleStats.map((stats) => stats.bandwidth))
  let margin = Math.max(rawSpan * .02, maxBandwidth * 4, scale * 1e-12)
  let binWidth = 0
  for (let iteration = 0; iteration < 8; iteration += 1) {
    binWidth = (rawSpan + 2 * margin) / (binCount - 1)
    const gridBandwidth = Math.max(maxBandwidth, binWidth * .5)
    const expanded = Math.max(rawSpan * .02, gridBandwidth * 4, scale * 1e-12)
    if (expanded <= margin * (1 + 1e-10)) break
    margin = expanded
  }
  const valueMin = rawMin - margin
  const valueMax = rawMax + margin
  if (!Number.isFinite(valueMin) || !Number.isFinite(valueMax) || valueMax <= valueMin) throw new Error('The simulated value range cannot be represented for density estimation.')
  binWidth = (valueMax - valueMin) / (binCount - 1)
  const valueGrid = Array.from({ length: binCount }, (_, index) => valueMin + index * binWidth)
  const densityByTime: number[][] = Array.from({ length: pointCount }, () => Array(binCount).fill(0))
  const bandwidths: number[] = []
  let densityMax = 0
  const gaussianScale = 1 / Math.sqrt(2 * Math.PI)
  for (let timeIndex = 0; timeIndex < pointCount; timeIndex += 1) {
    const { sample, bandwidth: scottBandwidth } = sampleStats[timeIndex]
    const bandwidth = Math.max(scottBandwidth, binWidth * .5)
    bandwidths.push(bandwidth)
    for (let bin = 0; bin < binCount; bin += 1) {
      const x = valueGrid[bin]
      let sum = 0
      for (const value of sample) {
        const u = (x - value) / bandwidth
        if (Math.abs(u) < 9) sum += Math.exp(-.5 * u * u)
      }
      const density = gaussianScale * sum / (sample.length * bandwidth)
      if (!Number.isFinite(density)) throw new Error('Estimated density exceeded finite numeric range.')
      densityByTime[timeIndex][bin] = density
      densityMax = Math.max(densityMax, density)
    }
  }
  if (!(densityMax > 0) || !Number.isFinite(densityMax)) throw new Error('Could not estimate a finite probability density.')
  return { valueGrid, densityByTime, valueMin, valueMax, densityMax, bandwidths,
    method: 'Gaussian KDE on a shared value grid; Scott bandwidth (1.06 × sample SD × n⁻¹⁄⁵) with a half-grid-spacing resolution floor at each time.' }
}

export interface PlaybackState { index: number; playing: boolean }
export function playbackStart(state: PlaybackState, finalIndex: number): PlaybackState {
  return { index: state.index >= finalIndex ? 0 : state.index, playing: true }
}
export function playbackPause(state: PlaybackState): PlaybackState { return { ...state, playing: false } }
export function playbackReset(): PlaybackState { return { index: 0, playing: false } }
export function playbackAdvance(state: PlaybackState, finalIndex: number): PlaybackState {
  if (!state.playing || state.index >= finalIndex) return { index: Math.min(state.index, finalIndex), playing: false }
  const index = state.index + 1
  return { index, playing: index < finalIndex }
}

export function performanceFrame(data: SimulationTrajectories, pathIndex: number, currentIndex: number) {
  const path = data.trajectories[pathIndex]
  if (!path) throw new Error('Selected simulation path does not exist.')
  const last = Math.min(currentIndex, data.times.length - 1, path.length - 1)
  return data.times.slice(0, last + 1).map((time, index) => ({
    time, selected: path[index],
    ...Object.fromEntries(data.trajectories.map((trajectory, trajectoryIndex) => [`path${trajectoryIndex}`, trajectory[index]])),
    p05: data.summary.bands[index]?.p05,
    p50: data.summary.bands[index]?.p50,
    p95: data.summary.bands[index]?.p95,
    band: (data.summary.bands[index]?.p95 ?? 0) - (data.summary.bands[index]?.p05 ?? 0),
  }))
}

export function inspectPath(data: SimulationTrajectories, pathIndex: number, currentIndex: number) {
  const path = data.trajectories[pathIndex]
  if (!path) throw new Error('Selected simulation path does not exist.')
  const index = Math.max(0, Math.min(currentIndex, path.length - 1))
  const metrics = data.terminal_risk.path_metrics?.[pathIndex]
  return { pathId: pathIndex + 1, currentValue: path[index], time: data.times[index], finalValue: path.at(-1)!,
    totalReturn: metrics?.total_simple_return ?? null, maximumDrawdown: metrics?.maximum_drawdown ?? null }
}

export interface JourneyComponent { label: string; value: number; unit: string; group: 'state' | 'log' | 'return' | 'variance' }
export interface PathJourneySnapshot {
  index: number; time: number | null; currentValue: number; previousValue: number | null
  absoluteChange: number | null; percentageChange: number | null; cumulativeReturn: number | null
  maximumDrawdown: number | null; transition: Record<string, number | boolean | null> | null
  components: JourneyComponent[]; modelNote: string
}

/** Build a view from retained path and transition values; this never samples or replays a model. */
export function pathJourneySnapshot(data: PathForensics, requestedIndex: number): PathJourneySnapshot | null {
  const path = data.original_path
  if (!path.length || !data.times.length || path.some((value) => !Number.isFinite(value))) return null
  const index = Math.max(0, Math.min(path.length - 1, Math.trunc(requestedIndex)))
  const currentValue = path[index]
  const previousValue = index > 0 ? path[index - 1] : null
  const absoluteChange = previousValue == null ? null : currentValue - previousValue
  const transitionIndex = data.model === 'garch' ? index : index - 1
  const transition = transitionIndex >= 0 ? data.transitions[transitionIndex] ?? null : null
  const time = Number.isFinite(data.times[index]) ? data.times[index] : null
  const percentageChange = (data.model === 'gbm' || data.model === 'jump_diffusion') && previousValue != null && previousValue > 0
    ? currentValue / previousValue - 1 : null
  let cumulativeReturn: number | null = null
  let maximumDrawdown: number | null = null
  if (data.model === 'gbm' || data.model === 'jump_diffusion') {
    const initial = path[0]
    if (initial > 0) {
      cumulativeReturn = currentValue / initial - 1
      const wealth = path.slice(0, index + 1).map((value) => value / initial)
      maximumDrawdown = drawdownToDate(wealth)
    }
  } else if (data.model === 'garch') {
    const convention = data.parameters.return_convention === 'simple' ? 'simple' : 'log'
    let wealth = 1
    const wealthPath: number[] = []
    for (const value of path.slice(0, index + 1)) {
      wealth = convention === 'simple' ? wealth * (1 + value) : wealth * Math.exp(value)
      wealthPath.push(wealth)
    }
    if (wealthPath.every(Number.isFinite) && wealthPath.every((value) => value > 0)) {
      cumulativeReturn = wealth - 1
      maximumDrawdown = drawdownToDate([1, ...wealthPath])
    }
  }

  const components: JourneyComponent[] = []
  if (transition) {
    if (data.model === 'gbm') {
      addComponent(components, 'Drift log component', transition.drift_log_component, 'log return', 'log')
      addComponent(components, 'Diffusion component', transition.diffusion_component, 'log return', 'log')
    } else if (data.model === 'ou') {
      addComponent(components, 'Mean-reversion contribution', transition.mean_reversion_component, 'state units', 'state')
      addComponent(components, 'Diffusion contribution', transition.diffusion_component, 'state units', 'state')
    } else if (data.model === 'jump_diffusion') {
      addComponent(components, 'Drift log term', transition.drift_log_term, 'log return', 'log')
      addComponent(components, 'Diffusion contribution', transition.diffusion_component, 'log return', 'log')
      addComponent(components, 'Aggregate jump log sum', transition.jump_log_sum, 'log return', 'log')
    } else {
      const mean = finiteParameter(data.parameters.mean)
      if (mean != null) components.push({ label: 'Conditional mean', value: mean, unit: 'return', group: 'return' })
      addComponent(components, 'Innovation', transition.innovation, 'return', 'return')
      addComponent(components, 'Variance intercept ω', transition.variance_intercept_term, 'return²', 'variance')
      addComponent(components, 'ARCH term αε²', transition.arch_variance_term, 'return²', 'variance')
      addComponent(components, 'GARCH term βh', transition.garch_variance_term, 'return²', 'variance')
    }
  }
  const modelNote = data.model === 'gbm'
    ? 'Log-return components add: next price = previous price × exp(total log-return increment). Bars are log returns, not dollar changes.'
    : data.model === 'ou'
      ? 'State contributions add directly: next state = previous state + mean-reversion contribution + diffusion contribution.'
      : data.model === 'jump_diffusion'
        ? 'Log terms add: next price = previous price × exp(drift + diffusion + aggregate jump log sum). Jump count and aggregate draw are retained; individual jump sizes are not.'
        : 'Return = conditional mean + innovation. The variance recursion terms are in return² and sum to the next conditional variance when a next period exists.'
  return { index, time, currentValue, previousValue, absoluteChange, percentageChange, cumulativeReturn,
    maximumDrawdown, transition, components, modelNote }
}

function drawdownToDate(wealth: number[]): number {
  let peak = -Infinity
  let worst = 0
  for (const value of wealth) { peak = Math.max(peak, value); if (peak > 0) worst = Math.min(worst, value / peak - 1) }
  return worst
}

function finiteParameter(value: unknown): number | null { return typeof value === 'number' && Number.isFinite(value) ? value : null }
function addComponent(target: JourneyComponent[], label: string, value: unknown, unit: string, group: JourneyComponent['group']): void {
  if (typeof value === 'number' && Number.isFinite(value)) target.push({ label, value, unit, group })
}
