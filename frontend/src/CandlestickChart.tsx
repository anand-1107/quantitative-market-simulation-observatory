import { useEffect, useMemo, useRef, useState } from 'react'
import type { OHLCVObservation } from './types'
import { chartObservationAtPointer, endChartDrag, fitChartRange, isPrimaryChartDrag, panChartRange, zoomChartRange, type ChartDrag } from './chart-navigation'

const WIDTH = 1000
const HEIGHT = 470
const LEFT = 72
const RIGHT = 18
const TOP = 18
const PRICE_BOTTOM = 325
const VOL_TOP = 350
const VOL_BOTTOM = 414
const formatBarTime = (value: string) => value.includes('T')
  ? `${value.slice(5, 10)} ${value.slice(11, 16)}` : value.slice(0, 10)

function movingAverage(bars: OHLCVObservation[], period: number, exponential = false): (number | null)[] {
  const result: (number | null)[] = Array(bars.length).fill(null)
  if (exponential) {
    const alpha = 2 / (period + 1)
    let current: number | null = null
    bars.forEach((bar, index) => {
      current = current == null ? bar.close : alpha * bar.close + (1 - alpha) * current
      if (index >= period - 1) result[index] = current
    })
    return result
  }
  let total = 0
  bars.forEach((bar, index) => {
    total += bar.close
    if (index >= period) total -= bars[index - period].close
    if (index >= period - 1) result[index] = total / period
  })
  return result
}

export function CandlestickChart({ bars, ticker, interval, downsampled, timestampConvention }: {
  bars: OHLCVObservation[]; ticker: string; interval: string; downsampled: boolean; timestampConvention: string
}) {
  const host = useRef<HTMLDivElement>(null)
  const [range, setRange] = useState<[number, number]>(() => fitChartRange(bars.length))
  const rangeRef = useRef(range)
  const [hover, setHover] = useState<{ index: number; x: number; y: number } | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const drag = useRef<ChartDrag | null>(null)
  const [scale, setScale] = useState(1)
  const [smaOn, setSmaOn] = useState(false)
  const [emaOn, setEmaOn] = useState(false)
  const averages = useMemo(() => ({ sma: movingAverage(bars, 20), ema: movingAverage(bars, 20, true) }), [bars])

  useEffect(() => {
    const node = host.current
    if (!node || bars.length < 2) return
    const onWheel = (event: WheelEvent) => {
      if (event.deltaY === 0 && event.deltaX === 0) return
      // Safari may dispatch delegated wheel events as passive. A native non-passive listener
      // lets us prevent page scrolling only while the pointer is over this chart.
      event.preventDefault()
      const rect = node.getBoundingClientRect()
      if (event.deltaY !== 0) {
        const plotStart = rect.left + rect.width * (LEFT / WIDTH)
        const plotWidth = rect.width * ((WIDTH - LEFT - RIGHT) / WIDTH)
        const anchor = plotWidth > 0 ? (event.clientX - plotStart) / plotWidth : 0.5
        const next = zoomChartRange(rangeRef.current, bars.length, event.deltaY, anchor)
        rangeRef.current = next; setRange(next)
        setHover({ index: chartObservationAtPointer(next, bars.length, event.clientX, rect.left, rect.width), x: event.clientX - rect.left, y: event.clientY - rect.top })
      }
      // Horizontal two-finger trackpad gestures pan; vertical wheel/trackpad gestures zoom.
      if (event.deltaX !== 0 && event.deltaY === 0) {
        const next = panChartRange(rangeRef.current, bars.length, -event.deltaX, rect.width * ((WIDTH - LEFT - RIGHT) / WIDTH))
        rangeRef.current = next; setRange(next)
        setHover({ index: chartObservationAtPointer(next, bars.length, event.clientX, rect.left, rect.width), x: event.clientX - rect.left, y: event.clientY - rect.top })
      }
    }
    node.addEventListener('wheel', onWheel, { passive: false })
    return () => node.removeEventListener('wheel', onWheel)
  }, [bars.length])

  if (bars.length < 2) return <div className="candle-empty">At least two historical OHLC bars are required to render a navigable chart.</div>

  const visible = Math.max(2, range[1] - range[0] + 1)
  const minIndex = Math.max(0, Math.min(range[0], bars.length - 2))
  const maxIndex = Math.max(minIndex + 1, Math.min(range[1], bars.length - 1))
  const shown = bars.slice(minIndex, maxIndex + 1)
  const low = Math.min(...shown.map((bar) => bar.low))
  const high = Math.max(...shown.map((bar) => bar.high))
  const mid = (low + high) / 2
  const half = (high - low) * 0.56 * scale || Math.max(Math.abs(mid) * 0.005, 0.01)
  const priceMin = mid - half
  const priceMax = mid + half
  const xAt = (index: number) => LEFT + ((index - minIndex) / Math.max(1, visible - 1)) * (WIDTH - LEFT - RIGHT)
  const yAt = (value: number) => TOP + (priceMax - value) / (priceMax - priceMin) * (PRICE_BOTTOM - TOP)
  const maxVolume = Math.max(...shown.map((bar) => bar.volume ?? 0), 1)
  const candleWidth = Math.max(1.1, Math.min(9, (WIDTH - LEFT - RIGHT) / visible * 0.62))
  const linePath = (values: (number | null)[]) => values.map((value, index) => value == null || index < minIndex || index > maxIndex ? '' : `${index === minIndex || values[index - 1] == null ? 'M' : 'L'}${xAt(index)},${yAt(value)}`).filter(Boolean).join(' ')

  const locate = (event: React.PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const index = chartObservationAtPointer(rangeRef.current, bars.length, event.clientX, rect.left, rect.width)
    return { index, x: event.clientX - rect.left, y: event.clientY - rect.top }
  }
  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    if (drag.current?.pointerId === event.pointerId) {
      const rect = event.currentTarget.getBoundingClientRect()
      const delta = event.clientX - drag.current.lastX
      drag.current.lastX = event.clientX
      setHover(null)
      setRange((current) => {
        const next = panChartRange(current, bars.length, delta, rect.width * ((WIDTH - LEFT - RIGHT) / WIDTH))
        rangeRef.current = next
        return next
      })
      return
    }
    setHover(locate(event))
  }
  const endPointerDrag = (event: React.PointerEvent<HTMLDivElement>, cancelled = false) => {
    if (drag.current?.pointerId !== event.pointerId) return
    drag.current = endChartDrag(drag.current, event.pointerId)
    setIsDragging(false)
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      try { event.currentTarget.releasePointerCapture(event.pointerId) } catch { /* Capture may already have been released by Safari. */ }
    }
    setHover(cancelled ? null : locate(event))
  }

  const active = hover ? bars[hover.index] : bars[maxIndex]
  return <div className="candle-frame">
    <div className="candle-toolbar">
      <div className="candle-title"><strong>{ticker}</strong><span>{interval} · HISTORICAL OHLCV · {timestampConvention.toUpperCase()}</span>{downsampled && <em>AGGREGATED TO DISPLAY LIMIT</em>}</div>
      <div className="candle-actions">
        <label><input type="checkbox" checked={smaOn} onChange={(event) => setSmaOn(event.target.checked)} /> SMA 20</label>
        <label><input type="checkbox" checked={emaOn} onChange={(event) => setEmaOn(event.target.checked)} /> EMA 20</label>
        <button onClick={() => setScale((value) => Math.max(.25, value * .82))} aria-label="Zoom price axis in">Y+</button>
        <button onClick={() => setScale((value) => Math.min(4, value * 1.22))} aria-label="Zoom price axis out">Y−</button>
        <button onClick={() => { const next = fitChartRange(bars.length); rangeRef.current = next; setRange(next); setScale(1) }}>FIT</button>
      </div>
    </div>
    <div ref={host} className={`candle-canvas${isDragging ? ' candle-dragging' : ''}`} onPointerMove={onPointerMove} onPointerLeave={() => { if (!drag.current) setHover(null) }} onPointerDown={(event) => {
      if (!isPrimaryChartDrag(event.button, event.isPrimary)) return
      if (event.pointerType === 'mouse') event.preventDefault()
      drag.current = { pointerId: event.pointerId, lastX: event.clientX }
      setIsDragging(true)
      setHover(null)
      try { event.currentTarget.setPointerCapture(event.pointerId) } catch { drag.current = null; setIsDragging(false) }
    }} onPointerUp={(event) => endPointerDrag(event)} onPointerCancel={(event) => endPointerDrag(event, true)} onLostPointerCapture={(event) => {
      if (drag.current?.pointerId === event.pointerId) {
        drag.current = endChartDrag(drag.current, event.pointerId)
        setIsDragging(false); setHover(null)
      }
    }}>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" role="img" aria-label={`${ticker} historical candlestick chart with volume`}>
        {Array.from({ length: 5 }, (_, index) => {
          const value = priceMax - (priceMax - priceMin) * index / 4
          const y = yAt(value)
          return <g key={`grid-${index}`}><line x1={LEFT} x2={WIDTH - RIGHT} y1={y} y2={y} className="candle-grid"/><text x={LEFT - 9} y={y + 3} textAnchor="end" className="candle-axis">{value.toFixed(value < 10 ? 3 : 2)}</text></g>
        })}
        {Array.from({ length: Math.min(6, shown.length) }, (_, index) => {
          const barIndex = minIndex + Math.round(index * (visible - 1) / Math.max(1, Math.min(6, shown.length) - 1))
          const x = xAt(barIndex)
          return <g key={`date-${barIndex}`}><line x1={x} x2={x} y1={TOP} y2={VOL_BOTTOM} className="candle-grid candle-grid-vertical"/><text x={x} y={HEIGHT - 19} textAnchor="middle" className="candle-axis">{formatBarTime(bars[barIndex].date)}</text></g>
        })}
        <text x={LEFT} y={VOL_TOP - 8} className="volume-label">VOLUME</text>
        {shown.map((bar, offset) => {
          const index = minIndex + offset, x = xAt(index), rising = bar.close >= bar.open
          const top = yAt(Math.max(bar.open, bar.close)), bodyHeight = Math.max(1, Math.abs(yAt(bar.open) - yAt(bar.close)))
          const volHeight = (bar.volume ?? 0) / maxVolume * (VOL_BOTTOM - VOL_TOP - 5)
          return <g key={`${bar.date}-${index}`} className={rising ? 'candle-up' : 'candle-down'}>
            <line x1={x} x2={x} y1={yAt(bar.high)} y2={yAt(bar.low)} className="candle-wick" />
            <rect x={x - candleWidth / 2} y={top} width={candleWidth} height={bodyHeight} className="candle-body" />
            <rect x={x - candleWidth / 2} y={VOL_BOTTOM - volHeight} width={Math.max(1, candleWidth)} height={volHeight} className="volume-bar" />
          </g>
        })}
        {smaOn && <path d={linePath(averages.sma)} className="indicator-sma" />}
        {emaOn && <path d={linePath(averages.ema)} className="indicator-ema" />}
        {hover && <g className="chart-crosshair"><line x1={xAt(hover.index)} x2={xAt(hover.index)} y1={TOP} y2={VOL_BOTTOM} /><line x1={LEFT} x2={WIDTH - RIGHT} y1={yAt(active.close)} y2={yAt(active.close)} /><circle cx={xAt(hover.index)} cy={yAt(active.close)} r="3" /></g>}
      </svg>
      {hover && <div className="candle-tooltip" style={{ left: Math.min(hover.x + 14, (host.current?.clientWidth ?? 500) - 190), top: Math.max(8, hover.y - 46) }}>
        <b>{active.date.replace('T', ' ')}</b><span>O {active.open.toFixed(2)}　H {active.high.toFixed(2)}</span><span>L {active.low.toFixed(2)}　C {active.close.toFixed(2)}</span><span>V {active.volume == null ? '—' : active.volume.toLocaleString()}</span>
      </div>}
    </div>
    <div className="candle-foot"><span>DRAG TO PAN · SCROLL TO ZOOM · Y+/Y− TO SCALE PRICE AXIS</span><span>{formatBarTime(bars[minIndex].date)} — {formatBarTime(bars[maxIndex].date)}</span></div>
  </div>
}
