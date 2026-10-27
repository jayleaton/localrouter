import { describe, expect, it } from 'vitest'
import { FakeNotes } from '../fake.ts'
import { dragPath, STROKES } from './controller.ts'
import { NeedsPointer } from './types.ts'
import type { ScreenControl } from './types.ts'

/** Windows run, 2026-10-27: Paint's canvas was clicked twice and the task ended `stuck`, with no way to draw a line. */
describe('drag: press, move, release', () => {
  const canvas: ScreenControl = { ref: 'c', role: 'other', label: 'canvas (it draws itself)', bounds: { x: 100, y: 200, w: 400, h: 200 } }
  const bin: ScreenControl = { ref: 'b', role: 'button', label: 'Bin', bounds: { x: 600, y: 600, w: 40, h: 40 } }

  it('a stroke inside the canvas, or from one control onto another, in screen coordinates', () => {
    expect(dragPath(canvas, 'stroke_across')).toEqual([{ x: 200, y: 300 }, { x: 300, y: 300 }, { x: 400, y: 300 }])
    expect(dragPath(canvas, 'stroke_diagonal')).toEqual([{ x: 200, y: 250 }, { x: 300, y: 300 }, { x: 400, y: 350 }])
    expect(dragPath(canvas, bin)).toEqual([{ x: 300, y: 300 }, { x: 620, y: 620 }])
    expect(dragPath({ ...canvas, bounds: undefined }, 'stroke_down')).toBeNull()
    expect(Object.keys(STROKES)).toEqual(['stroke_across', 'stroke_down', 'stroke_diagonal'])
  })

  it('is always the real pointer: refused in the background, done with the pointer when allowed', async () => {
    const desktop = new FakeNotes()
    const path = dragPath(canvas, 'stroke_across')!
    await expect(desktop.act({ kind: 'drag', ref: 'c', path })).rejects.toBeInstanceOf(NeedsPointer)
    expect(await desktop.act({ kind: 'drag', ref: 'c', path }, { allowPointer: true })).toMatchObject({ mode: 'pointer', at: { x: 400, y: 300 } })
  })
})
