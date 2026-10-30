import { describe, expect, it } from 'vitest'
import type { SocketEvents } from '../../../desktop-helper/src/connection.ts'
import { LegacyBridge } from './legacy.ts'
import { fakeBroker } from './testing.ts'

/** A box on the other end of /v1/desktop that already granted everything on its side. */
function permissiveBox() {
  const sent: Array<Record<string, unknown>> = []
  let events: SocketEvents | null = null
  const dial = (_url: string, _token: string, e: SocketEvents) => {
    events = e
    queueMicrotask(() => e.open())
    return {
      send: (text: string) => {
        const frame = JSON.parse(text) as Record<string, unknown>
        sent.push(frame)
        if (frame.type === 'hello') e.message(JSON.stringify({ type: 'welcome', helper_id: 'mac_1', limits: { max_steps: 15, timeout_ms: 90000, approval_timeout_ms: 60000 } }))
        if (frame.type === 'authorize') e.message(JSON.stringify({ type: 'result', id: frame.id, ok: true, grant: 'approved' }))
        if (frame.type === 'decide') {
          const q = frame.questions as Record<string, unknown>
          const answers = q.next ? { next: { type: 'choice', choice: 'c3', probability: 0.9 } } : { surface: { type: 'choice', choice: 'current', probability: 0.9 } }
          e.message(JSON.stringify({ type: 'result', id: frame.id, ok: true, answers }))
        }
      },
      close: () => undefined,
    }
  }
  return { dial, sent, box: (frame: Record<string, unknown>) => events!.message(JSON.stringify(frame)) }
}

const until = async (check: () => boolean) => {
  for (let i = 0; i < 200 && !check(); i++) await new Promise((r) => setTimeout(r, 10))
  expect(check()).toBe(true)
}

describe('the legacy /v1/desktop bridge: the box asks, this computer decides', () => {
  it('a fixed control the box granted still asks here when the local rules say so', async () => {
    const { broker, mac } = fakeBroker([])
    const box = permissiveBox()
    const cards: string[] = []
    broker.on((e) => { if (e.type === 'approval') { cards.push(e.card.id); void broker.answer(e.card.id, 'deny', 'app window') } })
    const bridge = new LegacyBridge({ url: 'http://127.0.0.1:8190', token: 't', name: 'Test Mac', version: '0', broker, access: async () => ({ accessibility: true, screen_recording: true }), dial: box.dial })
    bridge.start()
    await until(() => bridge.status.connected)
    box.box({ type: 'control', id: 'b1', task_id: 'ctl_1', control: { kind: 'lock_screen' } })
    await until(() => box.sent.some((f) => f.type === 'result' && f.id === 'b1'))
    expect(cards).toHaveLength(1)
    expect(box.sent.find((f) => f.id === 'b1')).toMatchObject({ ok: false })
    expect(mac.controls.calls).toEqual([])
    bridge.close()
  })

  it('a click the box approved waits for this computer’s approval; denied here, nothing is clicked', async () => {
    const { broker, mac } = fakeBroker([])
    const box = permissiveBox()
    broker.on((e) => { if (e.type === 'approval') void broker.answer(e.card.id, 'deny', 'app window') })
    const bridge = new LegacyBridge({ url: 'http://127.0.0.1:8190', token: 't', name: 'Test Mac', version: '0', broker, access: async () => ({ accessibility: true, screen_recording: true }), dial: box.dial })
    bridge.start()
    await until(() => bridge.status.connected)
    box.box({ type: 'use.start', id: 'b2', task_id: 'cu_box', goal: 'press send', text: null })
    await until(() => box.sent.some((f) => f.type === 'finished'))
    expect(box.sent.find((f) => f.type === 'finished')).toMatchObject({ outcome: { status: 'blocked' } })
    expect(box.sent.some((f) => f.type === 'authorize')).toBe(true) // the box was asked, and said approved
    expect(mac.desktop.sent).toEqual([])
    bridge.close()
  })
})
