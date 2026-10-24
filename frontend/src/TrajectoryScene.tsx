import { memo, useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { Grid, Html, Line, OrbitControls } from '@react-three/drei'
import type { SimulationTrajectories } from './types'

type Props = {
  data: SimulationTrajectories
  selectedIndex: number | null
  visiblePathCount: number
  flat: boolean
  resetToken: number
  onSelect: (index: number) => void
}

type Point3 = [number, number, number]

function TrajectoryLines({ data, selectedIndex, visiblePathCount, flat, onSelect }: Omit<Props, 'resetToken'>) {
  const geometry = useMemo(() => {
    const allValues = data.trajectories.flat()
    const low = Math.min(...allValues)
    const high = Math.max(...allValues)
    const span = high - low || Math.max(Math.abs(low) * 0.01, 1e-8)
    const firstTime = data.times[0] ?? 0
    const lastTime = data.times.at(-1) ?? 1
    const duration = lastTime - firstTime || 1
    const count = data.trajectories.length
    const limit = Math.min(count, visiblePathCount)
    const indices = new Set(Array.from({ length: limit }, (_, index) => index))
    if (selectedIndex !== null && selectedIndex < count) indices.add(selectedIndex)
    const paths = [...indices].sort((a, b) => a - b).map((index) => {
      const values = data.trajectories[index]
      const points: Point3[] = values.map((value, step) => [
        -5.4 + ((data.times[step] - firstTime) / duration) * 10.8,
        -2.8 + ((value - low) / span) * 5.6,
        flat || count < 2 ? 0 : -3.1 + (index / (count - 1)) * 6.2,
      ])
      const selected = index === selectedIndex
      const color = selected ? '#ffb84a' : data.quantity_kind === 'state' ? '#64c5cb'
        : data.quantity_kind === 'return' ? '#8a9cba'
          : values.at(-1)! >= values[0] ? '#4cbf8b' : '#bf6666'
      return { index, points, selected, color }
    })
    return { paths, low, high, count }
  }, [data, selectedIndex, visiblePathCount, flat])

  return <>
    <Grid position={[0, -2.85, 0]} args={[11, 7]} cellSize={1} cellThickness={0.45}
      sectionSize={4} sectionThickness={0.8} sectionColor="#4a4639" cellColor="#272e2e"
      fadeDistance={18} fadeStrength={1.4} />
    <Line points={[[-5.4, -2.82, -3.45], [5.45, -2.82, -3.45]]} color="#75694f" lineWidth={1} />
    <Line points={[[-5.4, -2.82, -3.45], [-5.4, 2.9, -3.45]]} color="#75694f" lineWidth={1} />
    {!flat && <Line points={[[-5.4, -2.82, -3.45], [-5.4, -2.82, 3.45]]} color="#75694f" lineWidth={1} />}
    {geometry.paths.map(({ index, points, selected, color }) => <Line key={index} points={points}
      color={color} lineWidth={selected ? 2.5 : 1.05} transparent opacity={selected ? 1 : .36}
      onClick={(event) => { event.stopPropagation(); onSelect(index) }} />)}
    <Html position={[5.15, -3.18, -3.35]} center distanceFactor={10}>
      <div className="scene-axis-label">{data.time_label.toUpperCase()}</div>
    </Html>
    <Html position={[-5.4, 3.12, -3.35]} center distanceFactor={10}>
      <div className="scene-axis-label scene-axis-y">{data.quantity_label.toUpperCase()}<small>{geometry.low.toPrecision(4)} — {geometry.high.toPrecision(4)}</small></div>
    </Html>
    {!flat && <Html position={[-5.35, -3.18, 3.35]} center distanceFactor={10}>
      <div className="scene-axis-label">PATH INDEX / 0—{geometry.count - 1}</div>
    </Html>}
  </>
}

function TrajectorySceneView(props: Props) {
  return <div className="trajectory-canvas" aria-label="Interactive three-dimensional simulation trajectory view">
    <Canvas key={`${props.resetToken}-${props.flat}`} dpr={[1, 1.5]} camera={{ position: [8.2, 6.2, 10.5], fov: 42 }}
      gl={{ antialias: true, alpha: false, powerPreference: 'low-power' }} fallback={<div className="webgl-fallback">WebGL is unavailable in this browser. Simulation summaries remain accessible in the inspector.</div>}>
      <color attach="background" args={['#090d0f']} />
      <ambientLight intensity={0.75} />
      <TrajectoryLines {...props} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.12} enablePan enableZoom
        enableRotate={!props.flat} minDistance={6} maxDistance={24} target={[0, 0, 0]} />
    </Canvas>
    <div className="scene-range"><span>t₀ {props.data.times[0]?.toPrecision(3)}</span><span>{props.data.quantity_label}: actual model units</span><span>tₙ {props.data.times.at(-1)?.toPrecision(3)}</span></div>
  </div>
}

export default memo(TrajectorySceneView)
