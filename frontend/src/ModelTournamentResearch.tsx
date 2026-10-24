import { useEffect, useMemo, useRef, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { Html, Line, OrbitControls } from '@react-three/drei'
import * as THREE_NAMESPACE from 'three'
import { ArrowLeftRight, Pause, Play, RotateCcw, ScanLine } from 'lucide-react'
import { api } from './api'
import { Button, ErrorNotice } from './components'
import { dateAxisTicks, displayRange, formatAxisValue, linearAxisTicks, mapDataToWorld, mapWorldToData, cumulativeLogReturnPaths, densityDifference, researchModels, researchState, validRiskSurface, RESEARCH_AXIS_TITLES, initialProgressivePlayback, playProgressive, pauseProgressive, resetProgressive, seekProgressive, advanceProgressive, progressiveIndex, toggleResearchModelVisibility, PLAYBACK_SCALE } from './model-tournament-research'
import type { AxisTick, DensityDifference, NumericRange, ResearchMode } from './model-tournament-research'
import type { ForecastDistributionSurface, ModelTournamentResult, SimulationTrajectories, TournamentModel, TournamentModelResult } from './types'

const MODES: { id: ResearchMode; label: string }[] = [
  { id: 'distribution', label: 'Distribution Evolution' }, { id: 'paths', label: 'Path Comparison' },
  { id: 'risk', label: 'Risk Landscape' }, { id: 'disagreement', label: 'Model Disagreement' },
]
const LABEL: Record<TournamentModel, string> = { gbm: 'GBM', ou: 'OU', jump_diffusion: 'Merton', garch: 'GARCH' }
const COLOR: Record<TournamentModel, string> = { gbm: '#31c9dc', ou: '#5d91ff', jump_diffusion: '#efb758', garch: '#b178e8' }
const ROUTE: Record<TournamentModel, string> = { gbm: '/simulate-gbm/paths', ou: '/simulate-ou/paths', jump_diffusion: '/simulate-jump-diffusion/paths', garch: '/simulate-garch/paths' }
const PERIODS: Record<string, number> = { '1d': 252, '5d': 50, '1wk': 52, '1mo': 12, '3mo': 4 }
const number = (value: unknown, fallback = 0) => typeof value === 'number' && Number.isFinite(value) ? value : fallback
type PathResult = { model: TournamentModel; times: number[]; paths: number[][]; unit: string; horizonYears: number }
const X_WORLD: NumericRange = { min: -5, max: 5 }
const Y_WORLD: NumericRange = { min: -2.55, max: 2.55 }
const DATE_FORMATTER = (date: string, firstDate: string, lastDate: string) => {
  const includeYear = date.slice(0, 4) !== firstDate.slice(0, 4) || date.slice(0, 4) !== lastDate.slice(0, 4)
  return includeYear ? new Date(`${date}T00:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: '2-digit', timeZone: 'UTC' })
    : new Date(`${date}T00:00:00Z`).toLocaleDateString('en-US', { month: 'short', day: '2-digit', timeZone: 'UTC' })
}

type AxisDefinition = { title: string; ticks: AxisTick[]; world: NumericRange }
function QuantAxes({ axes }: { axes: { x: AxisDefinition; y: AxisDefinition; z: AxisDefinition } }) {
  const x0 = axes.x.world.min, x1 = axes.x.world.max
  const y0 = axes.y.world.min, y1 = axes.y.world.max
  const z0 = axes.z.world.min, z1 = axes.z.world.max
  return <>
    <Line points={[[x0, y0, z0], [x1, y0, z0]]} color="#82a9c3" lineWidth={1.2} />
    <Line points={[[x0, y0, z0], [x0, y1, z0]]} color="#82a9c3" lineWidth={1.2} />
    <Line points={[[x0, y0, z0], [x0, y0, z1]]} color="#82a9c3" lineWidth={1.2} />
    {axes.x.ticks.map((tick, index) => <group key={`x-${index}`}>
      <Line points={[[tick.position, y0, z0], [tick.position, y0, z0 + .12]]} color="#9cb8cc" lineWidth={1.1} />
      <Line points={[[tick.position, y0, z0], [tick.position, y0, z1]]} color="#27445b" lineWidth={.55} />
      <Html position={[tick.position, y0 - .2, z0]} center distanceFactor={13}><span className="research-axis-tick">{tick.label}</span></Html>
    </group>)}
    {axes.y.ticks.map((tick, index) => <group key={`y-${index}`}>
      <Line points={[[x0, tick.position, z0], [x0 + .12, tick.position, z0]]} color="#9cb8cc" lineWidth={1.1} />
      <Line points={[[x0, tick.position, z0], [x1, tick.position, z0]]} color="#27445b" lineWidth={.55} />
      <Html position={[x0 - .28, tick.position, z0]} center distanceFactor={13}><span className="research-axis-tick">{tick.label}</span></Html>
    </group>)}
    {axes.z.ticks.map((tick, index) => <group key={`z-${index}`}>
      <Line points={[[x0, y0, tick.position], [x0 + .12, y0, tick.position]]} color="#9cb8cc" lineWidth={1.1} />
      <Line points={[[x0, y0, tick.position], [x1, y0, tick.position]]} color="#27445b" lineWidth={.55} />
      <Html position={[x0 - .3, y0 - .17, tick.position]} center distanceFactor={13}><span className="research-axis-tick">{tick.label}</span></Html>
    </group>)}
    <Html position={[(x0 + x1) / 2, y0 - .62, z0]} center distanceFactor={12}><span className="research-axis-title">{axes.x.title}</span></Html>
    <Html position={[x0 - .82, (y0 + y1) / 2, z0]} center distanceFactor={12}><span className="research-axis-title">{axes.y.title}</span></Html>
    <Html position={[x0 - .8, y0 - .65, (z0 + z1) / 2]} center distanceFactor={12}><span className="research-axis-title">{axes.z.title}</span></Html>
    <Html fullscreen><div className="research-axis-summary-3d">X {axes.x.ticks[0]?.label ?? '—'}–{axes.x.ticks.at(-1)?.label ?? '—'} · Y {axes.y.ticks[0]?.label ?? '—'}–{axes.y.ticks.at(-1)?.label ?? '—'} · Z {axes.z.ticks[0]?.label ?? '—'}–{axes.z.ticks.at(-1)?.label ?? '—'}</div></Html>
    <OrbitControls makeDefault enableDamping dampingFactor={.12} enablePan enableZoom enableRotate minDistance={7} maxDistance={28} target={[0, 0, 0]} />
  </>
}

function dateTicks(dates: string[], first: number, last: number) {
  const ticks = dateAxisTicks(dates, first, last, X_WORLD, 5)
  if (!dates.length) return ticks
  return ticks.map((tick) => ({ ...tick,
    position: mapDataToWorld(tick.value, { min: 0, max: Math.max(1, dates.length - 1) }, X_WORLD),
    label: DATE_FORMATTER(dates[tick.value] ?? '', dates[0] ?? '', dates.at(-1) ?? '') }))
}

function modelParams(item: TournamentModelResult, model: TournamentModel, years: number, pathCount: number, seed: number, steps: number) {
  const p = item.parameters ?? {}, initial = item.research_path ?? {}
  if (model === 'gbm') return { initial_price: number(initial.initial_price, 100), drift: number(p.drift), volatility: number(p.volatility), years, steps, paths: pathCount, seed }
  if (model === 'ou') return { initial_value: number(initial.initial_state), mean_level: number(p.mean_level), mean_reversion_speed: number(p.mean_reversion_speed), volatility: number(p.volatility), years, steps, paths: pathCount, seed }
  if (model === 'jump_diffusion') return { initial_price: number(initial.initial_price, 100), drift: number(p.drift), volatility: number(p.volatility), jump_intensity: number(p.jump_intensity), mean_log_jump: number(p.mean_log_jump), jump_volatility: number(p.jump_volatility), years, steps, paths: pathCount, seed }
  return { omega: number(p.omega), alpha: number(p.alpha), beta: number(p.beta), mean: number(p.mean), initial_variance: number(initial.initial_variance), steps, paths: pathCount, seed, return_convention: 'log' }
}

function SurfaceGeometry({ surface, model, color, densityMax, visibleTo, windowSize, onInspect }: {
  model: TournamentModel
  surface: ForecastDistributionSurface; color: string; densityMax: number; visibleTo: number; windowSize: number
  onInspect: (value: { model: TournamentModel; date: string; origin: number; returnValue: number; density: number }) => void
}) {
  const rows = surface.log_return_grid.length, columns = surface.density_by_origin.length
  const lastColumn = visibleTo < 0 ? columns - 1 : Math.max(0, Math.min(columns - 1, visibleTo))
  const firstColumn = visibleTo < 0 ? 0 : Math.max(0, lastColumn - windowSize + 1)
  const xTransform = { scale: 1, offset: 0 }
  const returnRange = displayRange(surface.log_return_grid[0] ?? 0, surface.log_return_grid.at(-1) ?? 0)
  const densityRange = { min: 0, max: Math.max(1e-12, densityMax) }
  const geometry = useMemo(() => {
    const positions: number[] = [], colors: number[] = [], indices: number[] = []
    const base = new THREE_NAMESPACE.Color(color)
    surface.density_by_origin.forEach((densityRow, column) => densityRow.forEach((density, row) => {
      const x = mapDataToWorld(column, { min: 0, max: Math.max(1, columns - 1) }, X_WORLD)
      const y = mapDataToWorld(surface.log_return_grid[row], returnRange, Y_WORLD)
      const z = mapDataToWorld(density, densityRange, { min: 0, max: 3.1 })
      positions.push(x, y, z)
      const shade = base.clone().multiplyScalar(.38 + .62 * Math.sqrt(Math.max(0, density / Math.max(densityMax, 1e-12))))
      colors.push(shade.r, shade.g, shade.b)
    }))
    for (let column = 0; column < columns - 1; column += 1) for (let row = 0; row < rows - 1; row += 1) {
      const a = column * rows + row, b = a + rows, c = a + 1, d = b + 1
      indices.push(a, b, c, b, d, c)
    }
    const result = new THREE_NAMESPACE.BufferGeometry()
    result.setAttribute('position', new THREE_NAMESPACE.Float32BufferAttribute(positions, 3))
    result.setAttribute('color', new THREE_NAMESPACE.Float32BufferAttribute(colors, 3))
    result.setIndex(indices); result.computeVertexNormals()
    return result
  }, [surface, color, densityMax, columns, rows, returnRange.min, returnRange.max])
  useEffect(() => () => geometry.dispose(), [geometry])
  const faceCount = (rows - 1) * 6
  const start = firstColumn * faceCount
  const end = Math.max(firstColumn, Math.min(columns - 1, visibleTo))
  geometry.setDrawRange(start, Math.max(0, end - firstColumn) * faceCount)
  const profile = visibleTo >= 0 && visibleTo === firstColumn ? surface.density_by_origin[firstColumn] ?? [] : []
  const profilePoints = profile.map((density, row) => [mapDataToWorld(firstColumn, { min: 0, max: Math.max(1, columns - 1) }, X_WORLD),
    mapDataToWorld(surface.log_return_grid[row], returnRange, Y_WORLD), mapDataToWorld(density, densityRange, { min: 0, max: 3.1 })] as [number, number, number])
  return <>
    {profilePoints.length > 1 && <Line points={profilePoints} color={color} lineWidth={2} transparent opacity={.86} />}
    <mesh geometry={geometry} position={[xTransform.offset, 0, 0]} scale={[xTransform.scale, 1, 1]} onPointerMove={(event) => {
    event.stopPropagation()
    const column = Math.max(firstColumn, Math.min(lastColumn, Math.round(mapWorldToData(event.point.x, { min: 0, max: Math.max(1, columns - 1) }, X_WORLD))))
    const value = mapWorldToData(event.point.y, returnRange, Y_WORLD)
    const row = surface.log_return_grid.reduce((best, current, index) => Math.abs(current - value) < Math.abs((surface.log_return_grid[best] ?? 0) - value) ? index : best, 0)
    onInspect({ model, date: surface.evaluation_dates[column], origin: surface.origin_indices[column], returnValue: surface.log_return_grid[row], density: surface.density_by_origin[column][row] })
  }}>
    <meshStandardMaterial vertexColors transparent opacity={.54} side={THREE_NAMESPACE.DoubleSide} roughness={.48} metalness={.06} />
    </mesh>
  </>
}
function RiskGeometry({ result, model, maxRisk, visibleTo, windowSize, onInspect }: {
  result: TournamentModelResult; model: TournamentModel; visibleTo: number; windowSize: number
  maxRisk: number
  onInspect: (value: { model: TournamentModel; date: string; origin: number; confidence: number; varLoss: number; expectedShortfall: number; probabilityNegative: number }) => void
}) {
  const surface = result.forecast_risk_surface!
  const confidenceCount = surface.confidence_levels.length, count = surface.evaluation_dates.length
  const lastColumn = Math.max(0, Math.min(count - 1, visibleTo))
  const firstColumn = Math.max(0, lastColumn - windowSize + 1)
  const xTransform = { scale: 1, offset: 0 }
  const confidenceRange = displayRange(surface.confidence_levels[0] ?? 0, surface.confidence_levels.at(-1) ?? 1)
  const geometry = useMemo(() => {
    const positions: number[] = [], colors: number[] = [], indices: number[] = []
    surface.var_loss_fraction_by_origin.forEach((row, column) => row.forEach((risk, confidenceIndex) => {
      positions.push(mapDataToWorld(column, { min: 0, max: Math.max(1, count - 1) }, X_WORLD),
        mapDataToWorld(surface.confidence_levels[confidenceIndex], confidenceRange, Y_WORLD),
        mapDataToWorld(risk, { min: -maxRisk, max: maxRisk }, { min: -3, max: 3 }))
      const shade = new THREE_NAMESPACE.Color(COLOR[model]).multiplyScalar(.45 + .55 * (confidenceIndex + 1) / confidenceCount)
      colors.push(shade.r, shade.g, shade.b)
    }))
    for (let column = 0; column < count - 1; column += 1) for (let row = 0; row < confidenceCount - 1; row += 1) {
      const a = column * confidenceCount + row, b = a + confidenceCount, c = a + 1, d = b + 1
      indices.push(a, b, c, b, d, c)
    }
    const resultGeometry = new THREE_NAMESPACE.BufferGeometry()
    resultGeometry.setAttribute('position', new THREE_NAMESPACE.Float32BufferAttribute(positions, 3))
    resultGeometry.setAttribute('color', new THREE_NAMESPACE.Float32BufferAttribute(colors, 3))
    resultGeometry.setIndex(indices); resultGeometry.computeVertexNormals()
    return resultGeometry
  }, [surface, count, confidenceCount, maxRisk, model, confidenceRange.min, confidenceRange.max])
  useEffect(() => () => geometry.dispose(), [geometry])
  const faceCount = (confidenceCount - 1) * 6
  geometry.setDrawRange(firstColumn * faceCount, Math.max(0, Math.min(count - 1, visibleTo) - firstColumn) * faceCount)
  const profile = visibleTo >= 0 && visibleTo === firstColumn ? surface.var_loss_fraction_by_origin[firstColumn] ?? [] : []
  const profilePoints = profile.map((risk, confidenceIndex) => [mapDataToWorld(firstColumn, { min: 0, max: Math.max(1, count - 1) }, X_WORLD),
    mapDataToWorld(surface.confidence_levels[confidenceIndex], confidenceRange, Y_WORLD), mapDataToWorld(risk, { min: -maxRisk, max: maxRisk }, { min: -3, max: 3 })] as [number, number, number])
  return <>
    {profilePoints.length > 1 && <Line points={profilePoints} color={COLOR[model]} lineWidth={2} transparent opacity={.86} />}
    <mesh geometry={geometry} position={[xTransform.offset, 0, 0]} scale={[xTransform.scale, 1, 1]} onPointerMove={(event) => {
    event.stopPropagation()
    const column = Math.max(firstColumn, Math.min(lastColumn, Math.round(mapWorldToData(event.point.x, { min: 0, max: Math.max(1, count - 1) }, X_WORLD))))
    const confidenceValue = mapWorldToData(event.point.y, confidenceRange, Y_WORLD)
    const confidenceIndex = surface.confidence_levels.reduce((best, value, index) => Math.abs(value - confidenceValue) < Math.abs((surface.confidence_levels[best] ?? 0) - confidenceValue) ? index : best, 0)
    onInspect({ model, date: surface.evaluation_dates[column], origin: surface.origin_indices[column], confidence: surface.confidence_levels[confidenceIndex],
      varLoss: surface.var_loss_fraction_by_origin[column][confidenceIndex],
      expectedShortfall: surface.expected_shortfall_95_loss_fraction_by_origin[column],
      probabilityNegative: surface.probability_negative_return_by_origin[column] })
  }}><meshStandardMaterial vertexColors transparent opacity={.7} side={THREE_NAMESPACE.DoubleSide} roughness={.5} /></mesh>
  </>
}

function DensityScene({ models, axisModel, domainModels, progress, started, windowSize, onInspect }: { models: TournamentModelResult[]; axisModel: TournamentModelResult | null; domainModels: TournamentModelResult[]; progress: number; started: boolean; windowSize: number; onInspect: (value: { model: TournamentModel; date: string; origin: number; returnValue: number; density: number }) => void }) {
  const maxDensity = Math.max(1e-12, ...domainModels.flatMap((item) => item.forecast_distribution_surface?.density_by_origin.flat() ?? []))
  const axesSurface = models[0]?.forecast_distribution_surface ?? axisModel?.forecast_distribution_surface
  const dates = axesSurface?.evaluation_dates ?? []
  const grid = axesSurface?.log_return_grid ?? [0, 1]
  const axisVisibleTo = progressiveIndex(progress, dates.length, started)
  const last = axisVisibleTo < 0 ? dates.length - 1 : axisVisibleTo, first = axisVisibleTo < 0 ? 0 : Math.max(0, last - windowSize + 1)
  const returnRange = displayRange(grid[0] ?? 0, grid.at(-1) ?? 0)
  const densityRange = { min: 0, max: maxDensity }
  const axes = { x: { title: RESEARCH_AXIS_TITLES.distribution.x, ticks: dateTicks(dates, first, last), world: X_WORLD },
    y: { title: RESEARCH_AXIS_TITLES.distribution.y, ticks: linearAxisTicks(returnRange, Y_WORLD, (value) => formatAxisValue(value), 5, true), world: Y_WORLD },
    z: { title: RESEARCH_AXIS_TITLES.distribution.z, ticks: linearAxisTicks(densityRange, { min: 0, max: 3.1 }, formatAxisValue, 5), world: { min: 0, max: 3.1 } } }
  return <Canvas dpr={[1, 1.5]} camera={{ position: [8.4, 6.4, 10.8], fov: 40 }} gl={{ antialias: true, alpha: false, powerPreference: 'low-power' }} fallback={<div className="research-empty">WebGL is unavailable. Use the 2D reference view for model disagreement.</div>}>
    <color attach="background" args={['#050b17']} /><ambientLight intensity={.8} /><directionalLight position={[3, 7, 6]} intensity={1} color="#d9edff" />
    <QuantAxes axes={axes} />
    {models.map((item) => item.forecast_distribution_surface && <SurfaceGeometry key={item.model} surface={item.forecast_distribution_surface} model={item.model} color={COLOR[item.model]} densityMax={maxDensity} visibleTo={progressiveIndex(progress, item.forecast_distribution_surface.density_by_origin.length, started)} windowSize={windowSize} onInspect={onInspect} />)}
  </Canvas>
}

function RiskScene({ models, axisModel, domainModels, progress, started, windowSize, onInspect }: { models: TournamentModelResult[]; axisModel: TournamentModelResult | null; domainModels: TournamentModelResult[]; progress: number; started: boolean; windowSize: number; onInspect: (value: { model: TournamentModel; date: string; origin: number; confidence: number; varLoss: number; expectedShortfall: number; probabilityNegative: number }) => void }) {
  const maxRisk = Math.max(.001, ...domainModels.flatMap((item) => item.forecast_risk_surface?.var_loss_fraction_by_origin.flat().map(Math.abs) ?? []))
  const riskSurface = models[0]?.forecast_risk_surface ?? axisModel?.forecast_risk_surface
  const dates = riskSurface?.evaluation_dates ?? [], levels = riskSurface?.confidence_levels ?? [0, 1]
  const axisVisibleTo = progressiveIndex(progress, dates.length, started)
  const last = axisVisibleTo < 0 ? dates.length - 1 : axisVisibleTo, first = axisVisibleTo < 0 ? 0 : Math.max(0, last - windowSize + 1)
  const confidenceRange = displayRange(levels[0] ?? 0, levels.at(-1) ?? 1)
  const lossRange = { min: -maxRisk, max: maxRisk }
  const confidenceTicks = levels.map((value) => ({ value, position: mapDataToWorld(value, confidenceRange, Y_WORLD), label: `${(value * 100).toFixed(1)}%` }))
  const axes = { x: { title: RESEARCH_AXIS_TITLES.risk.x, ticks: dateTicks(dates, first, last), world: X_WORLD },
    y: { title: RESEARCH_AXIS_TITLES.risk.y, ticks: confidenceTicks, world: Y_WORLD },
    z: { title: RESEARCH_AXIS_TITLES.risk.z, ticks: linearAxisTicks(lossRange, { min: -3, max: 3 }, (value) => `${formatAxisValue(value * 100)}%`, 5, true), world: { min: -3, max: 3 } } }
  return <Canvas dpr={[1, 1.5]} camera={{ position: [8.4, 6.4, 10.8], fov: 40 }} gl={{ antialias: true, alpha: false, powerPreference: 'low-power' }} fallback={<div className="research-empty">WebGL unavailable. Predictive VaR and ES remain in the 2D comparison diagnostics.</div>}>
    <color attach="background" args={['#050b17']} /><ambientLight intensity={.8} /><directionalLight position={[3, 7, 6]} intensity={1} />
    <QuantAxes axes={axes} />
    {models.map((item) => item.forecast_risk_surface && <RiskGeometry key={item.model} result={item} model={item.model} maxRisk={maxRisk} visibleTo={progressiveIndex(progress, item.forecast_risk_surface.evaluation_dates.length, started)} windowSize={windowSize} onInspect={onInspect} />)}
  </Canvas>
}

function DifferenceScene({ difference, visibleTo, windowSize, onInspect }: { difference: DensityDifference; visibleTo: number; windowSize: number; onInspect: (value: { date: string; origin: number; returnValue: number; difference: number }) => void }) {
  const columns = difference.dates.length, rows = difference.grid.length
  const returnRange = displayRange(difference.grid[0] ?? 0, difference.grid.at(-1) ?? 0)
  const maxAbs = Math.max(1e-12, ...difference.rows.flat().map(Math.abs))
  const lastColumn = visibleTo < 0 ? columns - 1 : Math.max(0, Math.min(columns - 1, visibleTo))
  const firstColumn = visibleTo < 0 ? 0 : Math.max(0, lastColumn - windowSize + 1)
  const xTransform = { scale: 1, offset: 0 }
  const axes = { x: { title: RESEARCH_AXIS_TITLES.disagreement.x, ticks: dateTicks(difference.dates, firstColumn, lastColumn), world: X_WORLD },
    y: { title: RESEARCH_AXIS_TITLES.disagreement.y, ticks: linearAxisTicks(returnRange, Y_WORLD, formatAxisValue, 5, true), world: Y_WORLD },
    z: { title: RESEARCH_AXIS_TITLES.disagreement.z, ticks: linearAxisTicks({ min: -maxAbs, max: maxAbs }, { min: -2.8, max: 2.8 }, formatAxisValue, 5, true), world: { min: -2.8, max: 2.8 } } }
  const geometry = useMemo(() => {
    const positions: number[] = [], colors: number[] = [], indices: number[] = []
    difference.rows.forEach((row: number[], column: number) => row.forEach((value: number, rowIndex: number) => {
      positions.push(mapDataToWorld(column, { min: 0, max: Math.max(1, columns - 1) }, X_WORLD),
        mapDataToWorld(difference.grid[rowIndex], returnRange, Y_WORLD),
        mapDataToWorld(value, { min: -maxAbs, max: maxAbs }, { min: -2.8, max: 2.8 }))
      const color = new THREE_NAMESPACE.Color(value >= 0 ? '#36d2c5' : '#e66b72').multiplyScalar(.5 + .5 * Math.abs(value) / maxAbs)
      colors.push(color.r, color.g, color.b)
    }))
    for (let column = 0; column < columns - 1; column += 1) for (let row = 0; row < rows - 1; row += 1) {
      const a = column * rows + row, b = a + rows, c = a + 1, d = b + 1
      indices.push(a, b, c, b, d, c)
    }
    const result = new THREE_NAMESPACE.BufferGeometry()
    result.setAttribute('position', new THREE_NAMESPACE.Float32BufferAttribute(positions, 3))
    result.setAttribute('color', new THREE_NAMESPACE.Float32BufferAttribute(colors, 3))
    result.setIndex(indices); result.computeVertexNormals()
    return result
  }, [difference, columns, rows, maxAbs, returnRange.min, returnRange.max])
  useEffect(() => () => geometry.dispose(), [geometry])
  const faceCount = (rows - 1) * 6
  geometry.setDrawRange(firstColumn * faceCount, Math.max(0, Math.min(columns - 1, visibleTo) - firstColumn) * faceCount)
  const profile = visibleTo >= 0 && visibleTo === firstColumn ? difference.rows[firstColumn] ?? [] : []
  const profilePoints = profile.map((value, rowIndex) => [mapDataToWorld(firstColumn, { min: 0, max: Math.max(1, columns - 1) }, X_WORLD),
    mapDataToWorld(difference.grid[rowIndex], returnRange, Y_WORLD), mapDataToWorld(value, { min: -maxAbs, max: maxAbs }, { min: -2.8, max: 2.8 })] as [number, number, number])
  return <group><QuantAxes axes={axes} /><mesh geometry={geometry} position={[xTransform.offset, 0, 0]} scale={[xTransform.scale, 1, 1]} onPointerMove={(event) => {
    event.stopPropagation()
    const column = Math.max(firstColumn, Math.min(lastColumn, Math.round(mapWorldToData(event.point.x, { min: 0, max: Math.max(1, columns - 1) }, X_WORLD))))
    const value = mapWorldToData(event.point.y, returnRange, Y_WORLD)
    const row = difference.grid.reduce((best, current, index) => Math.abs(current - value) < Math.abs((difference.grid[best] ?? 0) - value) ? index : best, 0)
    onInspect({ date: difference.dates[column], origin: difference.originIndices[column], returnValue: difference.grid[row], difference: difference.rows[column][row] })
  }}><meshStandardMaterial vertexColors transparent opacity={.72} side={THREE_NAMESPACE.DoubleSide} roughness={.5} /></mesh>
    {profilePoints.length > 1 && <Line points={profilePoints} color="#36d2c5" lineWidth={2} transparent opacity={.86} />}
  </group>
}

function PathScene({ groups, progress, started, onInspect }: { groups: PathResult[]; progress: number; started: boolean; onInspect: (value: { model: TournamentModel; path: number; time: number; value: number }) => void }) {
  const all = groups.flatMap((group) => group.paths.flat())
  const min = Math.min(0, ...all), max = Math.max(0, ...all), range = displayRange(min, max)
  const laneWorld: NumericRange = { min: -2.4, max: 2.4 }
  const laneTicks = groups.map((group, index) => ({ value: index + 1,
    position: groups.length === 1 ? 0 : laneWorld.min + index / (groups.length - 1) * (laneWorld.max - laneWorld.min),
    label: `${index + 1} ${LABEL[group.model]}` }))
  const maxHorizon = Math.max(...groups.map((group) => group.horizonYears), 1e-6)
  const axes = { x: { title: RESEARCH_AXIS_TITLES.paths.x, ticks: linearAxisTicks({ min: 0, max: maxHorizon }, X_WORLD, (value) => `${formatAxisValue(value)}y`), world: X_WORLD },
    y: { title: RESEARCH_AXIS_TITLES.paths.y, ticks: linearAxisTicks(range, Y_WORLD, formatAxisValue, 5, true), world: Y_WORLD },
    z: { title: RESEARCH_AXIS_TITLES.paths.z, ticks: laneTicks, world: laneWorld } }
  return <Canvas dpr={[1, 1.5]} camera={{ position: [8.4, 6.4, 10.8], fov: 40 }} gl={{ antialias: true, alpha: false, powerPreference: 'low-power' }}>
    <color attach="background" args={['#050b17']} /><ambientLight intensity={.85} />
    <QuantAxes axes={axes} />
    {groups.map((group, groupIndex) => group.paths.map((path, pathIndex) => {
      const visibleTo = progressiveIndex(progress, path.length, started)
      const limit = Math.max(0, Math.min(path.length - 1, visibleTo))
      const center = groups.length === 1 ? 0 : laneWorld.min + groupIndex / (groups.length - 1) * (laneWorld.max - laneWorld.min)
      const laneSpacing = groups.length === 1 ? 1.2 : Math.min(1.2, (laneWorld.max - laneWorld.min) / (groups.length - 1) * .55)
      const withinLane = path.length < 2 ? 0 : (pathIndex / (group.paths.length - 1) - .5) * laneSpacing
      const points = visibleTo < 0 ? [] : path.slice(0, limit + 1).map((value, index) => [
        mapDataToWorld(group.times[index] * group.horizonYears, { min: 0, max: maxHorizon }, X_WORLD),
        mapDataToWorld(value, range, Y_WORLD),
        center + withinLane,
      ] as [number, number, number])
      return points.length > 0 ? <Line key={`${group.model}-${pathIndex}`} points={points} color={COLOR[group.model]} lineWidth={1.15}
        transparent opacity={.54} onPointerMove={(event) => { event.stopPropagation(); const fraction = Math.max(0, Math.min(1, (event.point.x - X_WORLD.min) / (X_WORLD.max - X_WORLD.min))); const index = Math.max(0, Math.min(path.length - 1, Math.round(fraction * (path.length - 1)))); onInspect({ model: group.model, path: pathIndex + 1, time: group.times[index], value: path[index] }) }} />
        : null
    }))}
  </Canvas>
}

function ReferenceChart({ mode, model, difference, origin }: { mode: ResearchMode; model: TournamentModelResult | null; difference: ReturnType<typeof densityDifference>; origin: number }) {
  const width = 760, height = 220, left = 48, right = 14, top = 14, bottom = 48
  let values: { x: number; y: number }[] = [], color = '#31c9dc', label = ''
  if (mode === 'disagreement' && difference.compatible) {
    const row = difference.value.rows[origin] ?? []
    values = difference.value.grid.map((x, index) => ({ x, y: row[index] }))
    color = '#37d1c4'; label = 'Density A − density B · per log-return unit'
  } else if (mode === 'distribution' && model?.forecast_distribution_surface) {
    const surface = model.forecast_distribution_surface, row = surface.density_by_origin[origin] ?? []
    values = surface.log_return_grid.map((x, index) => ({ x, y: row[index] }))
    color = COLOR[model.model]; label = `${LABEL[model.model]} one-step predictive density`
  } else if (mode === 'risk' && model?.forecast_risk_surface) {
    const risk = model.forecast_risk_surface
    values = risk.evaluation_dates.map((_, index) => ({ x: index, y: risk.var_loss_fraction_by_origin[index]?.[1] ?? 0 }))
    color = COLOR[model.model]; label = '95% forecast loss VaR · fraction'
  }
  if (!values.length) return <div className="research-empty">No compatible data for the 2D reference view.</div>
  const rangeX = displayRange(Math.min(...values.map((point) => point.x)), Math.max(...values.map((point) => point.x)))
  const rangeY = displayRange(Math.min(...values.map((point) => point.y)), Math.max(...values.map((point) => point.y)))
  const xWorld = { min: left, max: width - right }, yWorld = { min: height - bottom, max: top }
  const xTicks = linearAxisTicks(rangeX, xWorld, formatAxisValue, 5, true), yTicks = linearAxisTicks(rangeY, yWorld, formatAxisValue, 5, true)
  const px = (x: number) => mapDataToWorld(x, rangeX, xWorld)
  const py = (y: number) => mapDataToWorld(y, rangeY, yWorld)
  const d = values.map((point, index) => `${index ? 'L' : 'M'}${px(point.x)},${py(point.y)}`).join(' ')
  return <svg className="research-reference-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label}>
    {xTicks.map((tick) => <g key={`x-${tick.value}`}><line className="research-reference-grid" x1={tick.position} x2={tick.position} y1={top} y2={height-bottom}/><text x={tick.position} y={height-bottom+13} textAnchor="middle">{tick.label}</text></g>)}
    {yTicks.map((tick) => <g key={`y-${tick.value}`}><line className="research-reference-grid" x1={left} x2={width-right} y1={tick.position} y2={tick.position}/><text x={left-5} y={tick.position+3} textAnchor="end">{tick.label}</text></g>)}
    <line x1={left} x2={width - right} y1={height - bottom} y2={height - bottom} /><line x1={left} x2={left} y1={top} y2={height - bottom} />
    <path d={d} fill="none" stroke={color} strokeWidth="2" />
    {mode === 'disagreement' && <line x1={left} x2={width-right} y1={py(0)} y2={py(0)} className="research-zero" />}
    <text className="research-reference-axis-title" x={(left+width-right)/2} y={height-2} textAnchor="middle">One-step log return / observation interval</text>
    <text className="research-reference-axis-title" x={9} y={(top+height-bottom)/2} textAnchor="middle" transform={`rotate(-90 9 ${(top+height-bottom)/2})`}>Density A − density B / log-return unit</text>
  </svg>
}

export function ModelTournamentResearch({ result }: { result: ModelTournamentResult }) {
  const [mode, setMode] = useState<ResearchMode>('distribution')
  const [visible, setVisible] = useState<TournamentModel[]>([])
  const [playback, setPlayback] = useState(initialProgressivePlayback)
  const [windowSize, setWindowSize] = useState(32)
  const [hover, setHover] = useState<Record<string, string> | null>(null)
  const [pairA, setPairA] = useState<TournamentModel | ''>('')
  const [pairB, setPairB] = useState<TournamentModel | ''>('')
  const [twoD, setTwoD] = useState(false)
  const [pathCount, setPathCount] = useState(8)
  const [pathYears, setPathYears] = useState(1)
  const [pathGroups, setPathGroups] = useState<PathResult[]>([])
  const [pathBusy, setPathBusy] = useState(false)
  const [pathError, setPathError] = useState('')
  const pathRequestRef = useRef(0)
  const available = useMemo(() => researchModels(result.results), [result.results])
  const keys = available.map((item) => item.model)
  const availableKey = JSON.stringify([result.ticker, result.interval, result.training_period, result.test_period,
    result.reproducibility.seed, result.reproducibility.paths, result.model_version,
    available.map((item) => [item.model, item.parameters])])
  useEffect(() => { pathRequestRef.current += 1; setPathBusy(false); setPathError(''); setVisible(keys); setPairA(keys[0] ?? ''); setPairB(keys[1] ?? ''); setPlayback(resetProgressive()); setPathGroups([]); return () => { pathRequestRef.current += 1 } }, [availableKey])
  const originCount = Math.max(1, ...available.map((item) => item.forecast_distribution_surface?.evaluation_dates.length ?? 0))
  const originIndex = progressiveIndex(playback.progress, originCount, playback.started)
  useEffect(() => { setHover(null) }, [playback.progress])
  const selectedModels = available.filter((item) => visible.includes(item.model))
  const selectedSingle = selectedModels.find((item) => item.model === (visible[0] ?? keys[0])) ?? selectedModels[0] ?? available[0] ?? null
  const modelA = available.find((item) => item.model === pairA) ?? null
  const modelB = available.find((item) => item.model === pairB) ?? null
  const rawDifference = densityDifference(modelA?.forecast_distribution_surface, modelB?.forecast_distribution_surface)
  const difference = pairA === pairB ? { compatible: false as const, reason: 'Choose two distinct models to inspect disagreement.' } : rawDifference
  const originDates = (mode === 'disagreement' ? modelA : selectedSingle)?.forecast_distribution_surface?.evaluation_dates ?? []
  const displayedOriginIndex = progressiveIndex(playback.progress, originDates.length, playback.started)
  const differenceIndex = difference.compatible ? progressiveIndex(playback.progress, difference.value.dates.length, playback.started) : -1
  const state = researchState(result.results.length, available.length, pathBusy && mode === 'paths', pathError && mode === 'paths' ? pathError : '')
  const allSelected = selectedModels.length > 0
  useEffect(() => {
    if (!playback.playing) return
    const timer = window.setInterval(() => setPlayback((current) => advanceProgressive(current, originCount)), 150)
    return () => window.clearInterval(timer)
  }, [playback.playing, originCount])
  const finalPathIndex = Math.max(1, ...pathGroups.map((group) => group.times.length - 1))
  const pathIndex = progressiveIndex(playback.progress, finalPathIndex + 1, playback.started)
  const togglePlayback = () => setPlayback((current) => current.playing ? pauseProgressive(current) : playProgressive(current))
  const replayPlayback = () => setPlayback(playProgressive(resetProgressive()))
  const toggleVisible = (key: TournamentModel) => { setHover(null); setVisible((current) => toggleResearchModelVisibility(current, key)) }
  const runPaths = async () => {
    const requestId = ++pathRequestRef.current
    setPathBusy(true); setPathError(''); setPathGroups([])
    const selected = selectedModels
    const steps = Math.max(1, Math.min(512, Math.round((PERIODS[result.interval] ?? 252) * pathYears)))
    try {
      const settled = await Promise.allSettled(selected.map(async (item) => {
        const payload = modelParams(item, item.model, pathYears, pathCount, result.reproducibility.seed, steps)
        const data = await api.simulateTrajectories(ROUTE[item.model], payload as Record<string, unknown>) as SimulationTrajectories
        const normalized = cumulativeLogReturnPaths(data)
        const sourceTimes = data.quantity_kind === 'return' ? normalized.times : data.times
        const times = sourceTimes.map((value) => value / (data.quantity_kind === 'return' ? 1 : pathYears))
        return { model: item.model, times, paths: normalized.paths, unit: normalized.unit, horizonYears: pathYears }
      }))
      const groups = settled.flatMap((entry) => entry.status === 'fulfilled' ? [entry.value] : [])
      const errors = settled.flatMap((entry, index) => entry.status === 'rejected' ? [`${LABEL[selected[index].model]}: ${entry.reason instanceof Error ? entry.reason.message : 'simulation failed'}`] : [])
      if (!groups.length) throw new Error(errors.join('; ') || 'No path simulations were returned.')
      if (requestId !== pathRequestRef.current) return
      setPathGroups(groups); setPlayback(resetProgressive())
      if (errors.length) setPathError(errors.join('; '))
    } catch (error) { if (requestId === pathRequestRef.current) setPathError(error instanceof Error ? error.message : 'Path simulation failed.') }
    finally { if (requestId === pathRequestRef.current) setPathBusy(false) }
  }
  const hoverText = hover ? Object.entries(hover).map(([key, value]) => `${key}: ${value}`).join(' · ') : ''

  return <section className="tournament-research">
    <header className="research-heading"><div><span>3D RESEARCH VIEW</span><h2>Model Comparison Observatory</h2><p>Interactive visualization of rolling out-of-sample forecast distributions and seeded model simulations.</p></div><span className="research-origin-badge">{available.length} CALIBRATED MODEL OUTPUTS</span></header>
    <nav className="research-tabs" aria-label="3D research visualization modes">{MODES.map((item) => <button type="button" key={item.id} className={mode === item.id ? 'active' : ''} onClick={() => { setMode(item.id); setHover(null) }}>{item.label}</button>)}</nav>
    {state === 'loading' && <div className="research-state">Generating paths from existing simulation engines…</div>}
    {state === 'error' && <ErrorNotice>{pathError}</ErrorNotice>}
    {(state === 'empty' || !available.length) && <div className="research-empty">{result.results.length ? 'No selected model completed calibration with the data required for out-of-sample visual research.' : 'Run or restore a completed tournament to load research outputs.'}</div>}
    {available.length > 0 && <>
      <div className="research-controls">
        <div className="research-model-checks"><span>Models / visibility</span>{available.map((item) => <label key={item.model}><input type="checkbox" checked={visible.includes(item.model)} onChange={() => toggleVisible(item.model)} /><i style={{ background: COLOR[item.model] }} />{LABEL[item.model]}</label>)}</div>
        {mode === 'disagreement' && <div className="research-pair"><label>Model A<select value={pairA} onChange={(event) => { setHover(null); setPairA(event.target.value as TournamentModel) }}>{available.map((item) => <option value={item.model} key={item.model}>{LABEL[item.model]}</option>)}</select></label><ArrowLeftRight size={15}/><label>Model B<select value={pairB} onChange={(event) => { setHover(null); setPairB(event.target.value as TournamentModel) }}>{available.map((item) => <option value={item.model} key={item.model}>{LABEL[item.model]}</option>)}</select></label><Button type="button" variant="quiet" disabled={!difference.compatible} onClick={() => { setHover(null); setTwoD((value) => !value) }}><ScanLine size={13}/>{twoD ? '3D surface' : '2D reference'}</Button></div>}
        {mode !== 'paths' && <label className="research-window">Time window · {windowSize} forecast origins<input type="range" min="1" max={originCount} value={Math.min(windowSize, originCount)} onChange={(event) => { setHover(null); setWindowSize(Number(event.target.value)) }}/></label>}
        {mode === 'paths' && <div className="research-path-settings"><label>Paths / model<select value={pathCount} onChange={(event) => setPathCount(Number(event.target.value))}>{[1,2,4,8,12,16].map((value) => <option key={value} value={value}>{value}</option>)}</select></label><label>Horizon<select value={pathYears} onChange={(event) => setPathYears(Number(event.target.value))}><option value={.5}>0.5 years</option><option value={1}>1 year</option><option value={2}>2 years</option></select></label><Button type="button" variant="secondary" busy={pathBusy} disabled={!allSelected} onClick={() => void runPaths()}><Play size={13}/>RUN SEEDED PATHS</Button></div>}
      </div>
      <div className="research-playback">
        <Button type="button" variant="quiet" onClick={togglePlayback}>{playback.playing ? <Pause size={13}/> : <Play size={13}/>} {playback.playing ? 'Pause' : 'Play'}</Button>
        <Button type="button" variant="quiet" onClick={() => setPlayback(resetProgressive())}><RotateCcw size={13}/> Reset</Button>
        <Button type="button" variant="quiet" onClick={replayPlayback}><Play size={13}/> Replay</Button>
        <input aria-label="Playback progress" type="range" min="0" max={PLAYBACK_SCALE} value={playback.progress} onChange={(event) => { setHover(null); setPlayback((current) => seekProgressive(current, Number(event.target.value))) }}/>
        <span>{mode === 'paths' && pathGroups.length ? `Path step ${playback.started ? pathIndex + 1 : 0} / ${finalPathIndex + 1}` : `Forecast origins ${playback.started ? originIndex + 1 : 0} / ${originCount}`}</span>
        <span>{playback.started ? (mode === 'paths' ? `${(Math.max(0, pathIndex) / finalPathIndex * pathYears).toFixed(3)} years` : originDates[displayedOriginIndex] ?? 'No origin') : 'Ready to simulate — press Play'}</span>
      </div>
      {mode === 'disagreement' && !difference.compatible && <div className="research-empty">Density difference disabled: {difference.reason}</div>}
      {mode === 'risk' && selectedModels.some((item) => !validRiskSurface(item.forecast_risk_surface)) && <ErrorNotice>A selected model has incomplete forecast risk data and is omitted from this view.</ErrorNotice>}
      {mode === 'paths' && pathError && state !== 'error' && <ErrorNotice>{pathError}</ErrorNotice>}
      {mode === 'paths' && !pathGroups.length && !pathBusy && !pathError && <div className="research-empty">Choose calibrated models and run bounded seeded simulations. Paths show possible outcomes, not predictions of actual future prices.</div>}
      {mode === 'paths' && pathGroups.length > 0 && pathGroups.some((group) => visible.includes(group.model)) && <div className="research-scene-wrap"><PathScene groups={pathGroups.filter((group) => visible.includes(group.model))} progress={playback.progress} started={playback.started} onInspect={(point) => { const group = pathGroups.find((item) => item.model === point.model); setHover({ Model: LABEL[point.model], Path: String(point.path), 'Horizon years': (point.time * (group?.horizonYears ?? pathYears)).toFixed(3), 'Cumulative log return': point.value.toPrecision(5), Inspection: 'Nearest simulated observation' }) }}/>{!playback.started && <div className="research-scene-prompt">Ready to simulate — press Play</div>}<div className="research-scene-caption">Possible outcomes · normalized cumulative log return from existing simulation APIs · not forecasts</div></div>}
      {mode === 'paths' && pathGroups.length > 0 && !pathGroups.some((group) => visible.includes(group.model)) && <div className="research-empty">All path models are hidden. Enable at least one model in the visibility controls.</div>}
      {mode === 'distribution' && <div className="research-scene-wrap"><DensityScene models={selectedModels.filter((item) => item.forecast_distribution_surface)} axisModel={available[0] ?? null} domainModels={available} progress={playback.progress} started={playback.started} windowSize={windowSize} onInspect={(point) => setHover({ Model: LABEL[point.model], Date: point.date, 'Forecast origin index': String(point.origin), 'Observation interval': result.interval, 'Log return / interval': point.returnValue.toPrecision(5), 'Conditional density / log-return unit': point.density.toPrecision(5), Inspection: 'Nearest stored surface grid point' })}/>{!playback.started && <div className="research-scene-prompt">Ready to simulate — press Play</div>}{!allSelected && <div className="research-scene-prompt">Enable a model to reveal its predictive density</div>}<div className="research-scene-caption">X: held-out forecast-origin dates · Y: one-step natural-log return per observation interval · Z: conditional predictive density per log-return unit</div></div>}
      {mode === 'risk' && <div className="research-scene-wrap"><RiskScene models={selectedModels.filter((item) => validRiskSurface(item.forecast_risk_surface))} axisModel={available[0] ?? null} domainModels={available} progress={playback.progress} started={playback.started} windowSize={windowSize} onInspect={(point) => setHover({ Model: LABEL[point.model], Date: point.date, 'Forecast origin index': String(point.origin), 'Observation interval': result.interval, Confidence: `${(point.confidence * 100).toFixed(1)}%`, 'Predicted loss VaR': `${(point.varLoss * 100).toFixed(3)}%`, 'Predicted 95% ES': `${(point.expectedShortfall * 100).toFixed(3)}%`, 'Probability return below zero': `${(point.probabilityNegative * 100).toFixed(2)}%`, Inspection: 'Nearest stored surface grid point' })}/>{!playback.started && <div className="research-scene-prompt">Ready to simulate — press Play</div>}{!allSelected && <div className="research-scene-prompt">Enable a model to reveal its forecast risk</div>}<div className="research-scene-caption">X: held-out forecast-origin date · Y: confidence · Z: conditional one-step loss VaR fraction, L=1−exp(r). These are forecasts, not realized losses.</div></div>}
      {mode === 'risk' && allSelected && playback.started && <div className="research-legend">{selectedModels.map((item) => { const surface = item.forecast_risk_surface!; const modelOrigin = progressiveIndex(playback.progress, surface.evaluation_dates.length, playback.started); return <span key={item.model}><i style={{ background: COLOR[item.model] }}/>{LABEL[item.model]} · P(loss): {((surface.probability_negative_return_by_origin[modelOrigin] ?? 0) * 100).toFixed(2)}% · ES95: {((surface.expected_shortfall_95_loss_fraction_by_origin[modelOrigin] ?? 0) * 100).toFixed(3)}%</span> })}</div>}
      {mode === 'disagreement' && difference.compatible && <div className="research-scene-wrap">{twoD ? (!playback.started ? <div className="research-empty">Ready to simulate — press Play to reveal the aligned density difference.</div> : <ReferenceChart mode={mode} model={null} difference={difference} origin={differenceIndex}/>) : <Canvas dpr={[1, 1.5]} camera={{ position: [8.4, 6.4, 10.8], fov: 40 }} gl={{ antialias: true, alpha: false, powerPreference: 'low-power' }} fallback={<div className="research-empty">WebGL unavailable. Switch to the 2D reference view.</div>}><color attach="background" args={['#050b17']}/><ambientLight intensity={.8}/><DifferenceScene difference={difference.value} visibleTo={differenceIndex} windowSize={windowSize} onInspect={(point) => setHover({ Comparison: `${LABEL[pairA as TournamentModel]} − ${LABEL[pairB as TournamentModel]}`, Date: point.date, 'Forecast origin index': String(point.origin), 'Log return / interval': point.returnValue.toPrecision(5), 'Density difference / log-return unit': point.difference.toPrecision(5), Inspection: 'Nearest stored surface grid point' })}/>{!playback.started && <Html center><span className="research-axis-tick">Ready to simulate — press Play</span></Html>}</Canvas>}<div className="research-scene-caption">{LABEL[pairA as TournamentModel]} − {LABEL[pairB as TournamentModel]} · same origins, natural-log return grid, interval and analytical density method. Cyan: positive; red: negative.</div></div>}
      {mode === 'distribution' && selectedSingle && <div className="research-distribution-readout">Selected model: <b>{LABEL[selectedSingle.model]}</b> · {selectedSingle.forecast_distribution_surface?.density_method}</div>}
      {hoverText && <div className="research-hover" role="status">{hoverText}</div>}
      {mode === 'disagreement' && difference.compatible && twoD && playback.started && <div className="research-reference-note">2D reference slice at origin {difference.value.dates[differenceIndex] ?? difference.value.dates.at(-1)}. Difference is density A minus density B, in inverse log-return units.</div>}
      {mode === 'paths' && pathGroups.length > 0 && <div className="research-legend">{pathGroups.filter((group) => visible.includes(group.model)).map((group) => <span key={group.model}><i style={{ background: COLOR[group.model] }}/>{LABEL[group.model]} · {group.paths.length} paths · {group.unit}</span>)}</div>}
    </>}
  </section>
}
