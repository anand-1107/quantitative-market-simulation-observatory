import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'
import { chartObservationAtPointer, endChartDrag, fitChartRange, isPrimaryChartDrag, panChartRange, zoomChartRange } from '../src/chart-navigation.ts'

test('horizontal pointer drag pans a fixed-width viewport and clamps at both data boundaries', () => {
  assert.deepEqual(panChartRange([20, 39], 100, -100, 500), [24, 43])
  assert.deepEqual(panChartRange([0, 19], 100, 2000, 500), [0, 19])
  assert.deepEqual(panChartRange([80, 99], 100, -2000, 500), [80, 99])
})

test('vertical wheel zoom preserves the point under the pointer and respects sensible bounds', () => {
  const before = [20, 69] as [number, number]
  const pointerRatio = 0.35
  const anchorBefore = before[0] + pointerRatio * (before[1] - before[0])
  const zoomed = zoomChartRange(before, 200, -300, pointerRatio)
  const anchorAfter = zoomed[0] + pointerRatio * (zoomed[1] - zoomed[0])
  assert.ok(zoomed[1] - zoomed[0] < before[1] - before[0])
  assert.ok(Math.abs(anchorAfter - anchorBefore) <= 1)
  assert.deepEqual(zoomChartRange([0, 199], 200, 1200, 0.5), [0, 199])
  assert.deepEqual(zoomChartRange([0, 19], 20, -1200, 0.5), [4, 15])
})

test('FIT restores the complete observation range and only primary left pointer starts a drag', () => {
  assert.deepEqual(fitChartRange(125), [0, 124])
  assert.deepEqual(fitChartRange(0), [0, 0])
  assert.equal(isPrimaryChartDrag(0, true), true)
  assert.equal(isPrimaryChartDrag(1, true), false)
  assert.equal(isPrimaryChartDrag(0, false), false)
})

test('pointer cancellation clears only the active captured drag', () => {
  const drag = { pointerId: 7, lastX: 420 }
  assert.deepEqual(endChartDrag(drag, 7), null)
  assert.equal(endChartDrag(drag, 8), drag)
})

test('candle inspection maps to the same visible observations after zoom and pan', () => {
  const range = panChartRange(zoomChartRange([20, 99], 150, -240, 0.6), 150, -125, 600)
  const selected = chartObservationAtPointer(range, 150, 72 + 0.6 * (1000 - 72 - 18), 0, 1000)
  assert.ok(selected >= range[0] && selected <= range[1])
  assert.ok(Math.abs(selected - Math.round(range[0] + 0.6 * (range[1] - range[0]))) <= 1)
})

test('the chart prevents Safari text selection without disabling pointer and vertical page-scroll interaction', async () => {
  const css = await readFile(new URL('../src/terminal.css', import.meta.url), 'utf8')
  assert.match(css, /-webkit-user-select:none/)
  assert.match(css, /user-select:none/)
  assert.match(css, /touch-action:pan-y/)
})
