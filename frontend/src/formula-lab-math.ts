import type { SimulationTrajectories } from './types'

export type FormulaModel = SimulationTrajectories['model']
export type FormulaDraft = Record<string, string>
export type FormulaField = { key: string; label: string; symbol: string; unit: string; description: string; min: number; max: number; step?: string; optional?: boolean }

export const FORMULA_ENDPOINTS: Record<FormulaModel, string> = {
  gbm: '/simulate-gbm/paths', ou: '/simulate-ou/paths',
  jump_diffusion: '/simulate-jump-diffusion/paths', garch: '/simulate-garch/paths',
}

export const FORMULA_DEFAULTS: Record<FormulaModel, FormulaDraft> = {
  gbm: { initial_price: '100', drift: '.08', volatility: '.2', years: '1', steps: '160', paths: '64', seed: '42' },
  ou: { initial_value: '0', mean_level: '0', mean_reversion_speed: '1', volatility: '1', years: '1', steps: '160', paths: '64', seed: '42' },
  jump_diffusion: { initial_price: '100', drift: '.08', volatility: '.2', jump_intensity: '.5', mean_log_jump: '-.1', jump_volatility: '.2', years: '1', steps: '160', paths: '64', seed: '42' },
  garch: { omega: '.000001', alpha: '.05', beta: '.9', mean: '0', initial_variance: '', steps: '160', paths: '64', seed: '42', return_convention: 'log' },
}

const COMMON: FormulaField[] = [
  { key: 'steps', label: 'Time steps', symbol: 'N', unit: 'steps', description: 'Discrete simulation intervals (GARCH: return observations).', min: 1, max: 512, step: '1' },
  { key: 'paths', label: 'Independent paths', symbol: 'M', unit: 'paths', description: 'Number of backend-generated scenarios.', min: 1, max: 128, step: '1' },
  { key: 'seed', label: 'Random seed', symbol: 's', unit: 'integer or blank', description: 'A fixed seed reproduces a run. Blank requests an unseeded run.', min: 0, max: 4294967295, step: '1', optional: true },
]

export const FORMULA_FIELDS: Record<FormulaModel, FormulaField[]> = {
  gbm: [
    { key: 'initial_price', label: 'Initial price', symbol: 'S₀', unit: 'price units', description: 'Positive starting price.', min: Number.MIN_VALUE, max: 1e9 },
    { key: 'drift', label: 'Expected growth rate', symbol: 'μ', unit: 'per year', description: 'Annualized expected proportional price growth.', min: -2, max: 2 },
    { key: 'volatility', label: 'Diffusion volatility', symbol: 'σ', unit: '√year⁻¹', description: 'Annualized diffusion scale; must be nonnegative.', min: 0, max: 3 },
    { key: 'years', label: 'Horizon', symbol: 'T', unit: 'years', description: 'Positive simulation duration.', min: Number.MIN_VALUE, max: 10 }, ...COMMON,
  ],
  ou: [
    { key: 'initial_value', label: 'Initial state', symbol: 'X₀', unit: 'state units', description: 'Starting level of the modeled spread or state.', min: -1e9, max: 1e9 },
    { key: 'mean_level', label: 'Long-run mean', symbol: 'μ', unit: 'state units', description: 'Mean level toward which the process reverts.', min: -1e9, max: 1e9 },
    { key: 'mean_reversion_speed', label: 'Mean-reversion speed', symbol: 'θ', unit: 'per year', description: 'Nonnegative speed of pull toward the long-run mean.', min: 0, max: 1e5 },
    { key: 'volatility', label: 'Diffusion volatility', symbol: 'σ', unit: 'state units / √year', description: 'Nonnegative state-level diffusion scale.', min: 0, max: 1e6 },
    { key: 'years', label: 'Horizon', symbol: 'T', unit: 'years', description: 'Positive simulation duration.', min: Number.MIN_VALUE, max: 10 }, ...COMMON,
  ],
  jump_diffusion: [
    { key: 'initial_price', label: 'Initial price', symbol: 'S₀', unit: 'price units', description: 'Positive starting price.', min: Number.MIN_VALUE, max: 1e9 },
    { key: 'drift', label: 'Compensated expected growth', symbol: 'μ', unit: 'per year', description: 'Annualized expected proportional price growth after jump compensation.', min: -2, max: 2 },
    { key: 'volatility', label: 'Diffusion volatility', symbol: 'σ', unit: '√year⁻¹', description: 'Annualized continuous diffusion scale, separate from jump risk.', min: 0, max: 3 },
    { key: 'jump_intensity', label: 'Jump intensity', symbol: 'λ', unit: 'expected jumps / year', description: 'Poisson arrival rate; nonnegative.', min: 0, max: 10 },
    { key: 'mean_log_jump', label: 'Mean log jump', symbol: 'm', unit: 'log return', description: 'Mean of the normal log jump size.', min: -2, max: 2 },
    { key: 'jump_volatility', label: 'Jump-size volatility', symbol: 'σJ', unit: 'log return', description: 'Standard deviation of normal log jump size.', min: 0, max: 2 },
    { key: 'years', label: 'Horizon', symbol: 'T', unit: 'years', description: 'Positive simulation duration.', min: Number.MIN_VALUE, max: 10 }, ...COMMON,
  ],
  garch: [
    { key: 'mean', label: 'Mean return', symbol: 'μ', unit: 'per observation period', description: 'Conditional mean of each log or simple return observation.', min: -10, max: 10 },
    { key: 'omega', label: 'Variance intercept', symbol: 'ω', unit: 'return²', description: 'Strictly positive constant in the variance recursion.', min: Number.MIN_VALUE, max: 1 },
    { key: 'alpha', label: 'ARCH coefficient', symbol: 'α', unit: 'dimensionless', description: 'Nonnegative response to the previous squared innovation.', min: 0, max: 10 },
    { key: 'beta', label: 'GARCH coefficient', symbol: 'β', unit: 'dimensionless', description: 'Nonnegative persistence from the previous conditional variance.', min: 0, max: 10 },
    { key: 'initial_variance', label: 'Initial variance', symbol: 'h₀', unit: 'return² or model default', description: 'Blank uses the simulator default: ω/(1−α−β) if stationary, otherwise ω.', min: Number.MIN_VALUE, max: 100, optional: true },
    { key: 'steps', label: 'Observation periods', symbol: 'N', unit: 'periods', description: 'Discrete return observations; GARCH parameters are per period.', min: 1, max: 512, step: '1' },
    { key: 'paths', label: 'Independent paths', symbol: 'M', unit: 'paths', description: 'Number of backend-generated scenarios.', min: 1, max: 128, step: '1' },
    { key: 'seed', label: 'Random seed', symbol: 's', unit: 'integer or blank', description: 'A fixed seed reproduces a run. Blank requests an unseeded run.', min: 0, max: 4294967295, step: '1', optional: true },
  ],
}

export const FORMULA_TEXT: Record<FormulaModel, string> = {
  gbm: 'dSₜ = μ Sₜ dt + σ Sₜ dWₜ\nΔt = T / N,   Zₜ ~ N(0,1)\nSₜ₊₁ = Sₜ exp[(μ − ½σ²) Δt + σ √Δt Zₜ]',
  ou: 'dXₜ = θ(μ − Xₜ) dt + σ dWₜ\nXₜ₊₁ = μ + (Xₜ − μ)e^(−θΔt) + σ √[(1 − e^(−2θΔt))/(2θ)] Zₜ\nθ = 0 limit: Xₜ₊₁ = Xₜ + σ√Δt Zₜ',
  jump_diffusion: 'dSₜ/Sₜ = (μ − λκ)dt + σ dWₜ + (J − 1)dNₜ\nκ = E[J] − 1 = exp(m + ½σJ²) − 1\nΔlog S = (μ − λκ − ½σ²)Δt + σ√Δt Z + N m + √N σJ ZJ,   N ~ Poisson(λΔt)',
  garch: 'εₜ = √hₜ zₜ,   rₜ = μ + εₜ,   zₜ ~ N(0,1)\nhₜ₊₁ = ω + α εₜ² + β hₜ\nh₀ = supplied value, else ω/(1−α−β) when α+β<1, otherwise ω',
}

export const FORMULA_MODEL_TITLES: Record<FormulaModel, string> = {
  gbm: 'Geometric Brownian Motion', ou: 'Ornstein–Uhlenbeck',
  jump_diffusion: 'Merton jump-diffusion', garch: 'GARCH(1,1)',
}

export function createFormulaDraft(model: FormulaModel): FormulaDraft { return { ...FORMULA_DEFAULTS[model] } }

export function formulaTrajectoryLabel(model: FormulaModel, quantity: SimulationTrajectories['quantity_kind'], garchView: 'returns' | 'price' = 'returns'): string {
  if (model === 'garch' && garchView === 'price') return 'Derived GARCH price index paths'
  if (quantity === 'price') return 'Simulated price paths'
  if (quantity === 'state') return 'Simulated state paths'
  return 'Simulated return paths'
}

export function buildFormulaRequest(model: FormulaModel, draft: FormulaDraft): { endpoint: string; payload: Record<string, unknown>; error: string | null } {
  const payload: Record<string, unknown> = {}
  for (const field of FORMULA_FIELDS[model]) {
    const value = draft[field.key] ?? ''
    if (value === '' && field.optional) { payload[field.key] = null; continue }
    const number = Number(value)
    if (!Number.isFinite(number)) return { endpoint: FORMULA_ENDPOINTS[model], payload: {}, error: `${field.label} must be a finite number${field.optional ? ' or blank' : ''}.` }
    if (number < field.min || number > field.max) return { endpoint: FORMULA_ENDPOINTS[model], payload: {}, error: `${field.label} must be between ${field.min} and ${field.max}.` }
    if (field.step === '1' && !Number.isInteger(number)) return { endpoint: FORMULA_ENDPOINTS[model], payload: {}, error: `${field.label} must be an integer.` }
    payload[field.key] = number
  }
  if (model === 'garch') {
    const convention = draft.return_convention || 'log'
    if (convention !== 'log' && convention !== 'simple') return { endpoint: FORMULA_ENDPOINTS[model], payload: {}, error: 'Return convention must be log or simple.' }
    payload.return_convention = convention
  }
  return { endpoint: FORMULA_ENDPOINTS[model], payload, error: null }
}

export function representativePathIndices(pathCount: number, maximum = 16): number[] {
  if (!Number.isInteger(pathCount) || pathCount <= 0 || !Number.isInteger(maximum) || maximum <= 0) return []
  const count = Math.min(pathCount, maximum)
  if (count === pathCount) return Array.from({ length: pathCount }, (_, index) => index)
  return Array.from({ length: count }, (_, index) => Math.round(index * (pathCount - 1) / (count - 1)))
}

export function deriveGarchPricePaths(data: SimulationTrajectories, initialPrice: number): number[][] {
  if (data.model !== 'garch' || !Number.isFinite(initialPrice) || initialPrice <= 0) throw new Error('A positive display starting price and GARCH return paths are required.')
  const convention = data.parameters.return_convention
  return data.trajectories.map((returns) => {
    let price = initialPrice
    const path = [price]
    for (const value of returns) {
      if (convention === 'simple') {
        if (value <= -1) throw new Error('Cannot convert a simple return at or below −100% into a positive price index.')
        price *= 1 + value
      } else price *= Math.exp(value)
      if (!Number.isFinite(price)) throw new Error('Derived GARCH price index exceeded finite numeric range.')
      path.push(price)
    }
    return path
  })
}

export function quantileLinear(values: number[], probability: number): number {
  if (!values.length || !Number.isFinite(probability) || probability < 0 || probability > 1) return NaN
  const sorted = [...values].sort((a, b) => a - b)
  const rank = probability * (sorted.length - 1)
  const lower = Math.floor(rank), upper = Math.ceil(rank)
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - lower)
}

export function parameterDifferences(baseline: Record<string, unknown>, scenario: Record<string, unknown>): Array<{ key: string; baseline: unknown; scenario: unknown }> {
  return [...new Set([...Object.keys(baseline), ...Object.keys(scenario)])]
    .filter((key) => baseline[key] !== scenario[key])
    .map((key) => ({ key, baseline: baseline[key] ?? null, scenario: scenario[key] ?? null }))
}

const COUNTERFACTUAL_KEYS: Record<FormulaModel, string[]> = {
  gbm: ['initial_price', 'drift', 'volatility', 'years', 'steps'],
  ou: ['initial_value', 'mean_level', 'mean_reversion_speed', 'volatility', 'years', 'steps'],
  jump_diffusion: ['initial_price', 'drift', 'volatility', 'jump_intensity', 'mean_log_jump', 'jump_volatility', 'years', 'steps'],
  garch: ['omega', 'alpha', 'beta', 'mean', 'initial_variance', 'steps', 'return_convention'],
}
const SENSITIVITY_KEYS: Record<FormulaModel, string[]> = {
  gbm: ['initial_price', 'drift', 'volatility', 'years'],
  ou: ['initial_value', 'mean_level', 'mean_reversion_speed', 'volatility', 'years'],
  jump_diffusion: ['initial_price', 'drift', 'volatility', 'jump_intensity', 'mean_log_jump', 'jump_volatility', 'years'],
  garch: ['omega', 'alpha', 'beta', 'mean', 'initial_variance', 'return_convention'],
}

/** Keep run controls and random seeds out of a path-level parameter experiment. */
export function buildCounterfactualParameters(model: FormulaModel, draft: FormulaDraft): { parameters: Record<string, number | string | null>; error: string | null } {
  const request = buildFormulaRequest(model, draft)
  if (request.error) return { parameters: {}, error: request.error }
  const parameters: Record<string, number | string | null> = {}
  for (const key of COUNTERFACTUAL_KEYS[model]) {
    if (Object.hasOwn(request.payload, key)) parameters[key] = request.payload[key] as number | string | null
  }
  return { parameters, error: null }
}

export function changedSensitivityKeys(model: FormulaModel, baseline: Record<string, unknown>, candidate: Record<string, number | string | null>): string[] {
  return SENSITIVITY_KEYS[model].filter((key) => {
    const before = baseline[key] ?? null
    const after = candidate[key] ?? null
    return typeof before === 'number' && typeof after === 'number'
      ? Math.abs(before - after) > Math.max(1, Math.abs(before), Math.abs(after)) * 1e-12
      : before !== after
  })
}

export function pathDifferenceSeries(reference: number[], candidate: number[]): number[] {
  if (reference.length !== candidate.length || reference.some((value) => !Number.isFinite(value)) || candidate.some((value) => !Number.isFinite(value))) return []
  return candidate.map((value, index) => value - reference[index])
}

export function rmse(reference: number[], candidate: number[]): number {
  if (reference.length !== candidate.length || !reference.length) return NaN
  return Math.sqrt(reference.reduce((sum, value, index) => sum + (value - candidate[index]) ** 2, 0) / reference.length)
}

export function clampObservationIndex(requested: number, observationCount: number): number {
  if (!Number.isFinite(requested) || observationCount <= 1) return 0
  return Math.max(0, Math.min(observationCount - 1, Math.trunc(requested)))
}

/** Read Recharts chart-click or point-click payloads and return a bounded observation index. */
export function observationIndexFromChartEvent(event: unknown, observationCount: number): number | null {
  if (!event || typeof event !== 'object') return null
  const source = event as { activeTooltipIndex?: unknown; index?: unknown }
  const value = source.activeTooltipIndex ?? source.index
  const requested = typeof value === 'number' && Number.isInteger(value) ? value
    : typeof value === 'string' && /^\d+$/.test(value) ? Number(value) : null
  return requested == null ? null : clampObservationIndex(requested, observationCount)
}

/** Map a plotted observation (including initial state) to its incoming model transition. */
export function incomingTransitionIndex(model: FormulaModel, observationIndex: number): number | null {
  const transitionIndex = model === 'garch' ? observationIndex : observationIndex - 1
  return transitionIndex < 0 ? null : transitionIndex
}

export function formulaObservationSnapshot(model: FormulaModel, referencePath: number[], referenceTimes: number[],
  requestedIndex: number, candidatePath?: number[], candidateTimes?: number[]) {
  const length = Math.min(referencePath.length, referenceTimes.length,
    candidatePath?.length ?? referencePath.length, candidateTimes?.length ?? referenceTimes.length)
  if (length < 1) return null
  const index = clampObservationIndex(requestedIndex, length)
  const currentValue = referencePath[index]
  const previousValue = index > 0 ? referencePath[index - 1] : null
  const absoluteChange = previousValue == null ? null : currentValue - previousValue
  const percentageChange = (model === 'gbm' || model === 'jump_diffusion') && previousValue != null && previousValue > 0
    ? currentValue / previousValue - 1 : null
  const candidateValue = candidatePath?.[index] ?? null
  const referenceTime = referenceTimes[index]
  const candidateTime = candidateTimes?.[index] ?? null
  const timeGridsMatch = Boolean(candidateTimes && candidateTimes.length === referenceTimes.length &&
    referenceTimes.every((time, i) => Math.abs(time - candidateTimes[i]) <= 1e-12))
  return { index, length, currentValue, previousValue, absoluteChange, percentageChange,
    candidateValue, difference: candidateValue == null ? null : candidateValue - currentValue,
    referenceTime, candidateTime, timeGridsMatch,
    transitionIndex: incomingTransitionIndex(model, index) }
}
