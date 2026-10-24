export type ChartRange = [number, number]
export type ChartDrag = { pointerId: number; lastX: number }

const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value))

export function fitChartRange(total: number): ChartRange {
  return [0, Math.max(0, total - 1)]
}

export function isPrimaryChartDrag(button: number, isPrimary: boolean): boolean {
  return button === 0 && isPrimary
}

export function panChartRange(range: ChartRange, total: number, deltaPixels: number, plotWidth: number): ChartRange {
  if (total < 2 || plotWidth <= 0) return [0, Math.max(0, total - 1)]
  const visible = Math.min(total, Math.max(2, range[1] - range[0] + 1))
  const deltaObservations = Math.round(-deltaPixels / plotWidth * visible)
  const start = clamp(range[0] + deltaObservations, 0, total - visible)
  return [start, start + visible - 1]
}

/** Wheel delta > 0 zooms out; anchorRatio is measured over the plot area, 0..1. */
export function zoomChartRange(range: ChartRange, total: number, deltaY: number, anchorRatio: number): ChartRange {
  if (total < 2 || deltaY === 0) return [...range]
  const visible = Math.min(total, Math.max(2, range[1] - range[0] + 1))
  const minimumVisible = Math.min(12, total)
  const nextVisible = clamp(Math.round(visible * Math.exp(clamp(deltaY, -1200, 1200) * 0.0015)), minimumVisible, total)
  const ratio = clamp(anchorRatio, 0, 1)
  const anchorIndex = range[0] + ratio * (visible - 1)
  const start = clamp(Math.round(anchorIndex - ratio * (nextVisible - 1)), 0, total - nextVisible)
  return [start, start + nextVisible - 1]
}

export function chartObservationAtPointer(range: ChartRange, total: number, clientX: number, rectLeft: number, rectWidth: number): number {
  if (total < 1 || rectWidth <= 0) return 0
  const plotStart = rectLeft + rectWidth * (72 / 1000)
  const plotWidth = rectWidth * ((1000 - 72 - 18) / 1000)
  const ratio = clamp((clientX - plotStart) / plotWidth, 0, 1)
  const index = Math.round(range[0] + ratio * (range[1] - range[0]))
  return clamp(index, 0, total - 1)
}

export function endChartDrag(drag: ChartDrag | null, pointerId: number): ChartDrag | null {
  return drag?.pointerId === pointerId ? null : drag
}
