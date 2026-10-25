import { describe, expect, it } from 'vitest'
import { HELLO_BEHIND, WindowsHello } from './presence.ts'
import type { WindowsCall } from './worker.ts'

type Call = { op: string; args: Record<string, unknown> }

function worker(respond: (call: Call) => unknown): { worker: WindowsCall; calls: Call[] } {
  const calls: Call[] = []
  return {
    calls,
    worker: {
      async call(op, args = {}) {
        calls.push({ op, args })
        const out = respond({ op, args })
        if (out instanceof Error) throw out
        return out
      },
    },
  }
}

describe('Windows Hello for an approval (Windows run, 2026-10-25: the prompt opened behind LocalRouter)', () => {
  it('asks for LocalRouter’s window so the prompt opens over it, and says nothing when it is in front', async () => {
    const { worker: w, calls } = worker((c) => (c.op === 'hello_start' ? { front: true, seen: true } : { result: 'Verified' }))
    const notes: string[] = []
    expect(await new WindowsHello(w).verify('Let Jarvis type “hi” into Notepad', { window: 0x1a2b3c, waiting: (t) => void notes.push(t) })).toBe(true)
    expect(calls.map((c) => c.op)).toEqual(['hello_start', 'hello_wait'])
    expect(calls[0].args).toMatchObject({ hwnd: 0x1a2b3c, prompt: 'Let Jarvis type “hi” into Notepad' })
    expect(notes).toEqual([])
  })

  it('tells the card where to look when the Windows Security window could not come to the front', async () => {
    const { worker: w } = worker((c) => (c.op === 'hello_start' ? { front: false, seen: true } : { result: 'Canceled' }))
    const notes: string[] = []
    expect(await new WindowsHello(w).verify('x', { window: 7, waiting: (t) => void notes.push(t) })).toBe(false)
    expect(notes).toEqual([HELLO_BEHIND])
  })

  it('without a window, or when the window call fails, uses the plain request and says where to look from the start', async () => {
    const failing = worker((c) => (c.op === 'hello_start' ? new Error('LR_HELLO|activation factory failed (0x80040154)|') : { result: 'Verified' }))
    const notes: string[] = []
    expect(await new WindowsHello(failing.worker).verify('x', { window: 7, waiting: (t) => void notes.push(t) })).toBe(true)
    expect(failing.calls.map((c) => c.op)).toEqual(['hello_start', 'hello_verify'])
    expect(notes).toEqual([HELLO_BEHIND])
    const windowless = worker(() => ({ result: 'Verified' }))
    expect(await new WindowsHello(windowless.worker).verify('x')).toBe(true)
    expect(windowless.calls.map((c) => c.op)).toEqual(['hello_verify'])
  })

  it('a prompt that started and then failed is a no, never retried as a second prompt', async () => {
    const { worker: w, calls } = worker((c) => (c.op === 'hello_start' ? { front: true, seen: true } : new Error('worker restarted')))
    expect(await new WindowsHello(w).verify('x', { window: 7 })).toBe(false)
    expect(calls.map((c) => c.op)).toEqual(['hello_start', 'hello_wait'])
  })
})
