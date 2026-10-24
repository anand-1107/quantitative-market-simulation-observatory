import { memo, useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { BufferGeometry, Color, DoubleSide, Float32BufferAttribute, Line as ThreeLine, LineBasicMaterial, Uint32BufferAttribute } from 'three'
import { Canvas } from '@react-three/fiber'
import { Grid, Html, Line as DreiLine, OrbitControls } from '@react-three/drei'
import type { DensitySurface } from './simulation-universe-math'
import { pathAppearance, performanceFrame, SIMULATION_PATH_PALETTE, visiblePathVertexCount } from './simulation-universe-math'
import type { SimulationTrajectories } from './types'

function densityGeometry(density: DensitySurface, times: number[]) {
  const geometry = new BufferGeometry()
  const rows = density.valueGrid.length, columns = density.densityByTime.length
  const first = times[0] ?? 0, duration = (times.at(-1) ?? 1) - first || 1
  const positions: number[] = [], colors: number[] = [], indices: number[] = []
  for (let column = 0; column < columns; column += 1) {
    const x = -5.2 + (((times[column] ?? first) - first) / duration) * 10.4
    for (let row = 0; row < rows; row += 1) {
      const y = -2.55 + ((density.valueGrid[row] - density.valueMin) / (density.valueMax - density.valueMin)) * 5.1
      const d = density.densityByTime[column][row]
      const intensity = Math.sqrt(Math.max(0, d / density.densityMax))
      const color = new Color().setHSL(.73 - .47 * intensity, .82, .34 + .2 * intensity)
      positions.push(x, y, 2.8 * d / density.densityMax)
      colors.push(color.r, color.g, color.b)
    }
  }
  for (let column = 0; column < columns - 1; column += 1) for (let row = 0; row < rows - 1; row += 1) {
    const a = column * rows + row, b = a + rows, c = a + 1, d = b + 1
    indices.push(a, b, c, b, d, c)
  }
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
  geometry.setAttribute('color', new Float32BufferAttribute(colors, 3))
  geometry.setIndex(new Uint32BufferAttribute(indices, 1))
  geometry.computeVertexNormals()
  return geometry
}

function densityAt(density: DensitySurface, timeIndex: number, value: number): number {
  const coordinate = Math.max(0, Math.min(density.valueGrid.length - 1,
    (value - density.valueMin) / (density.valueMax - density.valueMin) * (density.valueGrid.length - 1)))
  const lower = Math.floor(coordinate), upper = Math.min(density.valueGrid.length - 1, lower + 1)
  const fraction = coordinate - lower
  const row = density.densityByTime[timeIndex]
  return (row?.[lower] ?? 0) * (1 - fraction) + (row?.[upper] ?? 0) * fraction
}

function pathGeometry(data: SimulationTrajectories, density: DensitySurface) {
  const geometry = new BufferGeometry()
  const pathCount = data.trajectories.length, timeCount = data.times.length
  const first = data.times[0] ?? 0, duration = (data.times.at(-1) ?? 1) - first || 1
  const positions = new Float32Array(Math.max(0, (timeCount - 1) * pathCount * 2 * 3))
  const colors = new Float32Array(positions.length)
  const colorObjects = SIMULATION_PATH_PALETTE.map((value) => new Color(value))
  let offset = 0
  // Segment-major order lets drawRange reveal one synchronized time slice for every path.
  for (let step = 0; step < timeCount - 1; step += 1) for (let pathIndex = 0; pathIndex < pathCount; pathIndex += 1) {
    const values = data.trajectories[pathIndex]
    const color = colorObjects[pathIndex % colorObjects.length]
    for (const pointIndex of [step, step + 1]) {
      const value = values[pointIndex]
      const x = -5.2 + (((data.times[pointIndex] ?? first) - first) / duration) * 10.4
      const y = -2.55 + ((value - density.valueMin) / (density.valueMax - density.valueMin)) * 5.1
      const z = 2.8 * densityAt(density, pointIndex, value) / density.densityMax + .04
      positions[offset] = x; positions[offset + 1] = y; positions[offset + 2] = z
      colors[offset] = color.r; colors[offset + 1] = color.g; colors[offset + 2] = color.b
      offset += 3
    }
  }
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
  geometry.setAttribute('color', new Float32BufferAttribute(colors, 3))
  geometry.setDrawRange(0, 0)
  return geometry
}

function selectedPathGeometry(data: SimulationTrajectories, density: DensitySurface, selectedPath: number) {
  const geometry = new BufferGeometry()
  const values = data.trajectories[selectedPath] ?? []
  const first = data.times[0] ?? 0, duration = (data.times.at(-1) ?? 1) - first || 1
  const positions: number[] = []
  values.forEach((value, index) => {
    const x = -5.2 + (((data.times[index] ?? first) - first) / duration) * 10.4
    const y = -2.55 + ((value - density.valueMin) / (density.valueMax - density.valueMin)) * 5.1
    const z = 2.8 * densityAt(density, index, value) / density.densityMax + .08
    positions.push(x, y, z)
  })
  geometry.setAttribute('position', new Float32BufferAttribute(positions, 3))
  geometry.setDrawRange(0, 0)
  return geometry
}

function AnimatedDensity({ data, density, currentIndex, selectedPath }: {
  data: SimulationTrajectories; density: DensitySurface; currentIndex: number; selectedPath: number
}) {
  const surfaceGeometry = useMemo(() => densityGeometry(density, data.times), [density, data.times])
  const paths = useMemo(() => pathGeometry(data, density), [data, density])
  const selected = useMemo(() => selectedPathGeometry(data, density, selectedPath), [data, density, selectedPath])
  const selectedLine = useMemo(() => new ThreeLine(selected, new LineBasicMaterial({ color: '#ffc45e', transparent: true, opacity: .98, depthWrite: false })), [selected])
  const indexRef = useRef(currentIndex)
  const densityIndexCount = (density.valueGrid.length - 1) * 6
  useLayoutEffect(() => { indexRef.current = currentIndex }, [currentIndex])
  useEffect(() => () => { selected.dispose(); (selectedLine.material as LineBasicMaterial).dispose() }, [selected, selectedLine])
  useFrame(() => {
    const visibleStep = Math.max(0, Math.min(indexRef.current, data.times.length - 1))
    const surfaceCount = visibleStep * densityIndexCount
    surfaceGeometry.setDrawRange(0, surfaceCount)
    paths.setDrawRange(0, visiblePathVertexCount(data.trajectories.length, visibleStep))
    selected.setDrawRange(0, visibleStep + 1)
  })
  const first = data.times[0] ?? 0, duration = (data.times.at(-1) ?? 1) - first || 1
  const progressX = -5.2 + (((data.times[currentIndex] ?? first) - first) / duration) * 10.4
  return <>
    <mesh geometry={surfaceGeometry}>
      <meshStandardMaterial vertexColors side={DoubleSide} roughness={.39} metalness={.12} />
    </mesh>
    <mesh geometry={surfaceGeometry}>
      <meshBasicMaterial color="#9bd9ff" wireframe transparent opacity={.105} depthWrite={false} polygonOffset polygonOffsetFactor={-1} />
    </mesh>
    <lineSegments geometry={paths}>
      <lineBasicMaterial vertexColors transparent opacity={.48} depthWrite={false} />
    </lineSegments>
    <primitive object={selectedLine} />
    <mesh position={[progressX, 0, 0]}>
      <boxGeometry args={[.018, 5.2, .018]} />
      <meshBasicMaterial color="#f2c36a" transparent opacity={.72} depthWrite={false} />
    </mesh>
  </>
}

const SimulationSurface = memo(function SimulationSurface({ data, density, selectedPath, currentIndex }: {
  data: SimulationTrajectories; density: DensitySurface; selectedPath: number; currentIndex: number
}) {
  const densityFormatter = new Intl.NumberFormat(undefined, { maximumSignificantDigits: 3 })
  return <Canvas dpr={[1, 1.5]} camera={{ position: [8.8, 7, 11], fov: 39 }}
    gl={{ antialias: true, alpha: false, powerPreference: 'low-power' }}
    fallback={<div className="universe-fallback">WebGL is unavailable. The simulated paths and statistics remain available below.</div>}>
    <color attach="background" args={['#050b17']} />
    <ambientLight intensity={.72} />
    <directionalLight position={[3, 8, 6]} intensity={1.15} color="#cce7ff" />
    <Grid position={[0, -2.62, 0]} args={[10.8, 5.8]} cellSize={.5} cellThickness={.32}
      sectionSize={2} sectionThickness={.8} sectionColor="#255072" cellColor="#142b43" fadeDistance={18} fadeStrength={1.4} />
    <AnimatedDensity data={data} density={density} selectedPath={selectedPath} currentIndex={currentIndex} />
    <DreiLine points={[[-5.4, -2.62, -2.9], [5.5, -2.62, -2.9]]} color="#4c7692" lineWidth={1} />
    <DreiLine points={[[-5.4, -2.62, -2.9], [-5.4, 2.8, -2.9]]} color="#4c7692" lineWidth={1} />
    <DreiLine points={[[-5.4, -2.62, -2.9], [-5.4, -2.62, 2.9]]} color="#4c7692" lineWidth={1} />
    <Html position={[5.3, -2.85, -2.9]} center distanceFactor={10}><div className="universe-axis-label">{data.time_label.toUpperCase()}</div></Html>
    <Html position={[-5.55, 2.9, -2.9]} center distanceFactor={10}><div className="universe-axis-label">{data.quantity_label.toUpperCase()}<small>{density.valueMin.toPrecision(4)} … {density.valueMax.toPrecision(4)}</small></div></Html>
    <Html position={[-5.55, -2.85, 3]} center distanceFactor={10}><div className="universe-axis-label">PROBABILITY DENSITY<small>0 … {densityFormatter.format(density.densityMax)} / {data.quantity_kind === 'price' ? 'price unit' : data.quantity_kind === 'return' ? 'return unit' : 'state unit'}</small></div></Html>
    <OrbitControls makeDefault enableDamping dampingFactor={.11} enablePan enableZoom enableRotate minDistance={7} maxDistance={24} target={[0, 0, 0]} />
  </Canvas>
})

export const DistributionSurface = memo(SimulationSurface)

export const SynchronizedPerformanceChart = memo(function SynchronizedPerformanceChart({ data, density, selectedPath, currentIndex }: {
  data: SimulationTrajectories; density?: DensitySurface | null; selectedPath: number; currentIndex: number
}) {
  const chartData = useMemo(() => performanceFrame(data, selectedPath, currentIndex), [data, selectedPath, currentIndex])
  const limits = useMemo(() => {
    let min = density?.valueMin ?? Infinity, max = density?.valueMax ?? -Infinity
    for (const path of data.trajectories) for (const value of path) { min = Math.min(min, value); max = Math.max(max, value) }
    const pad = (max - min || Math.max(Math.abs(max) * .01, 1e-6)) * .06
    return [min - pad, max + pad] as [number, number]
  }, [data, density])
  const unit = data.quantity_kind === 'price' ? 'SIMULATED PRICE' : data.quantity_kind === 'return' ? 'PER-PERIOD RETURN' : 'SIMULATED STATE'
  return <div className="universe-chart-wrap">
    <div className="universe-chart-legend"><span className="legend-all-paths">ALL {data.trajectories.length} SIMULATED PATHS</span><span className="legend-selected">PATH {selectedPath + 1} · SELECTED</span><span className="legend-median">ENSEMBLE MEDIAN</span><span className="legend-band">5–95% PATH BAND</span><span className="legend-progress">CURRENT STEP</span></div>
    <div className="universe-performance-chart"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={chartData} margin={{ top: 12, right: 20, left: 10, bottom: 4 }}>
      <CartesianGrid stroke="#172a40" strokeDasharray="3 5" vertical />
      <XAxis dataKey="time" type="number" domain={[data.times[0] ?? 0, data.times.at(-1) ?? 1]} tick={{ fill: '#7991a8', fontSize: 9 }} axisLine={{ stroke: '#34516b' }} tickLine={false} />
      <YAxis domain={limits} tick={{ fill: '#7991a8', fontSize: 9 }} axisLine={{ stroke: '#34516b' }} tickLine={false} width={66} tickFormatter={(value: number) => Number(value).toPrecision(4)} />
      <Tooltip contentStyle={{ background: '#081321', border: '1px solid #2a4965', color: '#d7e8f4', fontSize: 10 }} labelFormatter={(value) => `${data.time_label}: ${value}`} formatter={(value, name) => [Number(value ?? 0).toPrecision(6), String(name)]} />
      <Area dataKey="p05" stackId="quantile-band" stroke="none" fill="transparent" isAnimationActive={false} />
      <Area dataKey="band" stackId="quantile-band" name="5–95% path band" stroke="none" fill="#22b8d1" fillOpacity={.14} isAnimationActive={false} />
      {data.trajectories.map((_, pathIndex) => pathIndex !== selectedPath && <Line key={`path-${pathIndex}`} dataKey={`path${pathIndex}`} name={`Path ${pathIndex + 1}`} type="linear" stroke={pathAppearance(pathIndex, selectedPath).color} strokeOpacity={pathAppearance(pathIndex, selectedPath).opacity} strokeWidth={pathAppearance(pathIndex, selectedPath).width} dot={false} activeDot={false} isAnimationActive={false} />)}
      <Line dataKey="p50" name="Ensemble median" type="linear" stroke="#8ba7ff" strokeWidth={1.5} dot={false} isAnimationActive={false} />
      <Line dataKey={`path${selectedPath}`} name={`Selected path ${selectedPath + 1}`} type="linear" stroke={pathAppearance(selectedPath, selectedPath).color} strokeWidth={pathAppearance(selectedPath, selectedPath).width} dot={false} isAnimationActive={false} />
      <ReferenceLine x={data.times[currentIndex]} stroke="#f1d384" strokeDasharray="4 3" strokeWidth={1.2} />
    </ComposedChart></ResponsiveContainer></div>
    <div className="universe-chart-axis-label"><span>{data.time_label.toUpperCase()}</span><span>{unit} · {data.quantity_label}</span></div>
  </div>
})
