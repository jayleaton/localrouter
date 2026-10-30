import { describe, expect, it } from 'vitest'
import { LINGER_MS, overlayHtml, VirtualCursor } from './overlay.ts'
import type { Surface } from './overlay.ts'

function fake() {
  const calls: string[] = []
  let fire: (() => void) | null = null
  const surface: Surface = {
    showAt: (x, y, mode) => void calls.push(`show ${x},${y} ${mode}`),
    hide: () => void calls.push('hide'),
    close: () => void calls.push('close'),
  }
  const schedule = (fn: () => void) => {
    fire = fn
    return { cancel: () => void calls.push('cancel') }
  }
  return { calls, surface, schedule, lapse: () => fire?.() }
}

describe('the visible virtual cursor (JJ: show where the agent acts without touching the real pointer)', () => {
  it('shows a background action as a ring and a pointer step as a dot, at rounded screen points', () => {
    const f = fake()
    const cursor = new VirtualCursor(f.surface, { schedule: f.schedule })
    cursor.show({ x: 100.6, y: 200.4, mode: 'background' })
    cursor.show({ x: 300, y: 400, mode: 'pointer' })
    expect(f.calls).toEqual(['show 101,200 background', 'cancel', 'show 300,400 pointer'])
    expect(cursor.visible).toBe(true)
  })

  it('fades on its own after it lingers, and the task’s end closes the window', () => {
    const f = fake()
    const cursor = new VirtualCursor(f.surface, { schedule: f.schedule })
    cursor.show({ x: 1, y: 2, mode: 'background' })
    f.lapse()
    expect(f.calls).toEqual(['show 1,2 background', 'cancel', 'hide'])
    expect(cursor.visible).toBe(false)
    cursor.end()
    expect(f.calls.slice(-1)).toEqual(['close'])
    expect(LINGER_MS).toBeGreaterThan(0)
  })

  it('the overlay page is transparent and click-through, and exposes the two hooks the window calls', () => {
    const html = overlayHtml()
    expect(html).toContain('pointer-events:none')
    expect(html).toContain('background:transparent')
    expect(html).toContain('window.lrCursor=')
    expect(html).toContain('window.lrHide=')
  })

  it('a task start loads the window ahead of the first action (Windows run, 2026-10-27: no ring 167 ms in)', () => {
    const warmed: string[] = []
    const cursor = new VirtualCursor({ showAt: () => undefined, hide: () => undefined, close: () => undefined, warm: () => warmed.push('warm') })
    cursor.ready()
    expect(warmed).toEqual(['warm'])
    // A surface without warm (the fake) is fine.
    new VirtualCursor({ showAt: () => undefined, hide: () => undefined, close: () => undefined }).ready()
  })
})
