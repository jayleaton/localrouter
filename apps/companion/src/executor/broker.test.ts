import { describe, expect, it } from 'vitest'
import type { Grant } from '../../../desktop-helper/src/computer/controller.ts'
import type { Card } from './broker.ts'
import { DISABLED_MESSAGE, OWN_TARGET } from './broker.ts'
import { fakeBroker } from './testing.ts'

// The fake Mac (apps/desktop-helper/src/fake.ts) shows Notes: c0 Search, c1 New Note, c2 Delete, c3 Send, c4.. the notes.
const SEND_GROCERIES = ['c4', 'c3', 'done']
const app = { caller: { id: 'app', name: 'LocalRouter', kind: 'app' as const } }
const voice = (utterance: string) => ({ caller: { id: 'voice', name: 'Jarvis (voice)', kind: 'voice' as const }, utterance })

/** The next approval card the broker shows. */
function nextCard(broker: ReturnType<typeof fakeBroker>['broker']): Promise<Card> {
  return new Promise((resolve) => {
    const off = broker.on((e) => {
      if (e.type === 'approval') {
        off()
        resolve(e.card)
      }
    })
  })
}

describe('the executor broker: final authority on this computer', () => {
  it('runs benign steps at once and waits for a present person on a consequential one', async () => {
    const { broker, mac, gate, audit } = fakeBroker([...SEND_GROCERIES])
    const card = nextCard(broker)
    const started = broker.startUse(app, 'send the Groceries note', null)
    if (!started.ok) throw new Error(started.error)
    const c = await card
    expect(c.action).toBe('click “Send” in Notes')
    expect(c.why).toMatch(/sending/)
    expect(mac.desktop.sent).toEqual([]) // nothing happened yet
    expect(await broker.answer(c.id, 'approve', 'app window')).toEqual({ ok: true })
    const outcome = await started.finished
    expect(outcome.status).toBe('done')
    expect(mac.desktop.sent).toEqual(['Groceries: milk, eggs, bread'])
    expect(gate.asked.map((x) => x.id)).toEqual([c.id])
    expect(mac.brought).toContain('Notes') // the approved app was brought back after the card
    const events = audit.lines.map((l) => l.event)
    expect(events).toEqual(expect.arrayContaining(['task.start', 'grant', 'approval.request', 'approval.answer', 'approval.end', 'task.end']))
  })

  describe('injected self-approval is rejected', () => {
    it('an approval while this executor is injecting input does not count', async () => {
      const { broker, mac } = fakeBroker([...SEND_GROCERIES])
      const card = nextCard(broker)
      const started = broker.startUse(app, 'send the Groceries note', null)
      if (!started.ok) throw new Error(started.error)
      const c = await card
      // The executor's own input arriving as the "click" on Approve: the answer comes while an injected action runs.
      let answered: Awaited<ReturnType<typeof broker.answer>> | null = null
      const slow = broker.driven.desktop!
      await Promise.all([slow.act({ kind: 'scroll', direction: 'down' }), broker.answer(c.id, 'approve', 'app window').then((r) => (answered = r))])
      expect(answered).toMatchObject({ ok: false, status: 403 })
      expect(broker.approvals().find((x) => x.id === c.id)?.state).toBe('pending')
      broker.stop(undefined, 'test')
      expect((await started.finished).status).toBe('stopped')
      expect(mac.desktop.sent).toEqual([])
    })

    it('an approval without the presence gate confirming does not count; deny always works', async () => {
      const { broker, gate, mac } = fakeBroker([...SEND_GROCERIES])
      gate.answer = false
      const card = nextCard(broker)
      const started = broker.startUse(app, 'send the Groceries note', null)
      if (!started.ok) throw new Error(started.error)
      const c = await card
      expect(await broker.answer(c.id, 'approve', 'app window')).toMatchObject({ ok: false, status: 403 })
      expect(await broker.answer(c.id, 'deny', 'app window')).toEqual({ ok: true })
      const outcome = await started.finished
      expect(outcome.status).toBe('blocked')
      expect(outcome.reason).toMatch(/said no/)
      expect(mac.desktop.sent).toEqual([])
    })

    it('a click on the card as it appears (a stale or scripted click) does not count', async () => {
      const { broker } = fakeBroker([...SEND_GROCERIES], { minDwellMs: 60_000 })
      const card = nextCard(broker)
      const started = broker.startUse(app, 'send the Groceries note', null)
      if (!started.ok) throw new Error(started.error)
      const c = await card
      expect(await broker.answer(c.id, 'approve', 'app window')).toMatchObject({ ok: false, error: expect.stringMatching(/too quick/) })
      broker.stop(undefined, 'test')
      await started.finished
    })

    it('the model cannot aim at LocalRouter’s own windows', async () => {
      const { broker, mac } = fakeBroker(['c3'], { ownApps: ['Notes'] }) // pretend the fake Notes app is the companion
      const started = broker.startUse(app, 'press send', null)
      if (!started.ok) throw new Error(started.error)
      const outcome = await started.finished
      expect(outcome.status).toBe('blocked')
      expect(outcome.reason).toBe(OWN_TARGET)
      expect(mac.desktop.actions).toEqual([])
      expect((await broker.control(app, { kind: 'quit_app', app: 'Notes' })).text).toBe(OWN_TARGET)
    })
  })

  it('a stale target is rejected: the screen changed while it waited for the approval', async () => {
    const { broker, mac } = fakeBroker([...SEND_GROCERIES])
    const card = nextCard(broker)
    const started = broker.startUse(app, 'send the Groceries note', null)
    if (!started.ok) throw new Error(started.error)
    const c = await card
    mac.desktop.selected = 1 // the person (or another app) selected "Trip ideas" meanwhile: another window title
    expect(await broker.answer(c.id, 'approve', 'app window')).toEqual({ ok: true })
    const outcome = await started.finished
    expect(outcome.status).toBe('blocked')
    expect(outcome.reason).toMatch(/screen changed/)
    expect(mac.desktop.sent).toEqual([])
  })

  describe('a local deny holds even when the box says allow', () => {
    const boxAllows = async (): Promise<Grant> => ({ ok: true, grant: 'approved' })

    it('a box grant does not replace the approval on this computer', async () => {
      const { broker, mac } = fakeBroker([...SEND_GROCERIES])
      const card = nextCard(broker)
      const started = broker.startUse({ caller: { id: 'box', name: 'LocalRouter box (legacy)', kind: 'box' } }, 'send the Groceries note', null, { advice: boxAllows })
      if (!started.ok) throw new Error(started.error)
      const c = await card // asked here although the box said approved
      await broker.answer(c.id, 'deny', 'app window')
      expect((await started.finished).status).toBe('blocked')
      expect(mac.desktop.sent).toEqual([])
    })

    it('switched off here, nothing runs whatever the box says', async () => {
      const { broker, mac } = fakeBroker(['c4'])
      broker.setDisabled(true)
      const started = broker.startUse({ caller: { id: 'box', name: 'box', kind: 'box' } }, 'open the groceries', null, { advice: boxAllows })
      expect(started).toEqual({ ok: false, error: DISABLED_MESSAGE })
      expect((await broker.control({ caller: { id: 'box', name: 'box', kind: 'box' } }, { kind: 'volume', action: 'up', target: 'system' })).ok).toBe(false)
      expect(mac.controls.calls).toEqual([])
    })

    it('a refusal from the box holds too', async () => {
      const { broker, mac } = fakeBroker(['c4'])
      const started = broker.startUse(app, 'open the groceries note', null, { advice: async () => ({ ok: false, denied: 'The box said no.' }) })
      if (!started.ok) throw new Error(started.error)
      const outcome = await started.finished
      expect(outcome).toMatchObject({ status: 'blocked', reason: 'The box said no.' })
      expect(mac.desktop.actions).toEqual([])
    })
  })

  it('stop works during an approval, with no network: the approval is cancelled', async () => {
    const { broker, mac } = fakeBroker([...SEND_GROCERIES])
    const card = nextCard(broker)
    const started = broker.startUse(app, 'send the Groceries note', null)
    if (!started.ok) throw new Error(started.error)
    const c = await card
    expect(broker.stop(undefined, 'tray')).toBe(1)
    expect((await started.finished).status).toBe('stopped')
    expect(await broker.answer(c.id, 'approve', 'app window')).toMatchObject({ ok: false, status: 409 })
    expect(mac.desktop.sent).toEqual([])
  })

  it('one task at a time on this computer, whoever asks', async () => {
    const { broker } = fakeBroker([...SEND_GROCERIES])
    const card = nextCard(broker)
    const first = broker.startUse(app, 'send the Groceries note', null)
    await card
    const second = broker.startUse({ caller: { id: 'mcp:x', name: 'MCP: opencode', kind: 'mcp' } }, 'something else', null)
    expect(second).toMatchObject({ ok: false, error: expect.stringMatching(/busy/) })
    broker.stop(undefined, 'test')
    if (first.ok) await first.finished
  })

  it('budgets: a task stops within its steps', async () => {
    const { broker } = fakeBroker(Array.from({ length: 10 }, (_, i) => (i % 2 ? 'scroll_up' : 'scroll_down')), { limits: { maxSteps: 3 } })
    const started = broker.startUse(app, 'look around', null)
    if (!started.ok) throw new Error(started.error)
    const outcome = await started.finished
    expect(outcome.status).toBe('stuck')
    expect(outcome.steps.length).toBeLessThanOrEqual(4)
  })

  it('fixed controls: simple ones run, locking the screen asks first, a voice turn needs the person’s words', async () => {
    const { broker, mac } = fakeBroker([])
    expect(await broker.control(app, { kind: 'open_app', app: 'Safari' })).toMatchObject({ ok: true })
    expect((await broker.control(voice(''), { kind: 'open_app', app: 'Safari' })).text).toMatch(/no user request/)
    const card = nextCard(broker)
    const locking = broker.control(voice('lock my screen'), { kind: 'lock_screen' })
    const c = await card
    expect(c.action).toBe('lock the screen')
    await broker.answer(c.id, 'approve', 'app window')
    expect(await locking).toMatchObject({ ok: true })
    expect(mac.controls.calls.map((x) => x.kind)).toEqual(['open_app', 'lock_screen'])
  })

  it('typed text stays out of the audit: its length and a digest only', async () => {
    const { broker, audit } = fakeBroker(['type_text', 'press_enter', 'done'])
    const started = broker.startUse(app, 'search my notes for trip', 'trip')
    if (!started.ok) throw new Error(started.error)
    expect((await started.finished).status).toBe('done')
    expect(JSON.stringify(audit.lines)).not.toContain('"trip"')
  })
})
