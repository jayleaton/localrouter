import { describe, expect, it } from 'vitest'
import type { Grant } from '../../../desktop-helper/src/computer/controller.ts'
import type { Card } from './broker.ts'
import { Broker, DISABLED_MESSAGE, OWN_TARGET } from './broker.ts'
import type { Answers, Question } from '../../../desktop-helper/src/computer/decisions.ts'
import type { Observation, ScreenAction } from '../../../desktop-helper/src/computer/types.ts'
import { MemoryAudit } from './audit.ts'
import { TestGate } from './testing.ts'
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
  it('says before Approve that the OS will ask, and does not let the card run out while it does (Windows run, 2026-10-25)', async () => {
    // Windows Hello opened Windows Security after Approve; the 60 s card ran out meanwhile and typing was denied.
    const hello = { method: 'Windows Hello', asked: 0, verify: async () => { hello.asked += 1; await new Promise((r) => setTimeout(r, 250)); return true } }
    const { broker, mac } = fakeBroker([...SEND_GROCERIES], { gate: hello, limits: { approvalTimeoutMs: 100 } })
    const card = nextCard(broker)
    const started = broker.startUse(app, 'send the Groceries note', null)
    if (!started.ok) throw new Error(started.error)
    const c = await card
    expect(c.confirm).toBe('After Approve, Windows asks for your PIN, face or fingerprint (Windows Hello).')
    expect(await broker.answer(c.id, 'approve', 'app window')).toEqual({ ok: true })
    expect((await started.finished).status).toBe('done')
    expect(mac.desktop.sent).toEqual(['Groceries: milk, eggs, bread'])
    expect(hello.asked).toBe(1)
  })

  it('while the OS check waits behind other windows, the card says where to look, and the note goes when it returns', async () => {
    let release: () => void = () => undefined
    const seenDuring: string[] = []
    const hello = {
      method: 'Windows Hello',
      verify: async (_card: unknown, waiting?: (text: string) => void) => {
        waiting?.('Windows Hello is waiting: look for the Windows Security window.')
        await new Promise<void>((resolve) => (release = resolve))
        return true
      },
    }
    const { broker } = fakeBroker([...SEND_GROCERIES], { gate: hello })
    const updates: string[] = []
    broker.on((e) => void (e.type === 'approval.update' && updates.push(e.card.waiting)))
    const card = nextCard(broker)
    const started = broker.startUse(app, 'send the Groceries note', null)
    if (!started.ok) throw new Error(started.error)
    const c = await card
    expect(c.waiting).toBe('')
    const answered = broker.answer(c.id, 'approve', 'app window')
    await new Promise((resolve) => setTimeout(resolve, 20))
    seenDuring.push(broker.approvals().find((x) => x.id === c.id)?.waiting ?? '')
    release()
    expect(await answered).toEqual({ ok: true })
    expect(seenDuring).toEqual(['Windows Hello is waiting: look for the Windows Security window.'])
    expect(updates).toEqual(['Windows Hello is waiting: look for the Windows Security window.'])
    expect((await started.finished).status).toBe('done')
  })

  it('a card held open for the OS check still runs out when nobody completes it', async () => {
    const never = { method: 'Windows Hello', verify: async () => false }
    const { broker, mac } = fakeBroker([...SEND_GROCERIES], { gate: never, limits: { approvalTimeoutMs: 100 }, presenceMs: 300 })
    const card = nextCard(broker)
    const started = broker.startUse(app, 'send the Groceries note', null)
    if (!started.ok) throw new Error(started.error)
    const c = await card
    expect(await broker.answer(c.id, 'approve', 'app window')).toMatchObject({ ok: false, status: 403 })
    // Held for the check past the card's 100 ms, then it runs out by itself: the step is refused, nothing is sent.
    expect(broker.approvals().find((x) => x.id === c.id)?.state).toBe('pending')
    expect(await started.finished).toMatchObject({ status: 'blocked', reason: 'Nobody approved it in time.' })
    expect(mac.desktop.sent).toEqual([])
  })

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
      const slow = broker.driven.screen!
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

/**
 * JJ's #79 failure: he talks to Jarvis in LocalRouter's own window, so LocalRouter is in front when a task starts,
 * and "switch to Google Chrome" was refused as acting on LocalRouter's windows.
 */
function fromLocalRouter(pick: { surface: string; app?: string; next: string[] }) {
  let front = 'LocalRouter'
  const actions: ScreenAction[] = []
  const brought: string[] = []
  const screens: Record<string, Observation> = {
    LocalRouter: { backend: 'desktop', app: 'LocalRouter', title: 'Jarvis', controls: [{ ref: 'mic', role: 'button', label: 'Start a call' }] },
    'Google Chrome': { backend: 'desktop', app: 'Google Chrome', title: 'Inbox (3) - Gmail', url: 'https://mail.google.com/', controls: [{ ref: 'tab1', role: 'tab', label: 'Primary' }] },
  }
  const decider = {
    async ask(_state: unknown, q: Record<string, Question>): Promise<Answers> {
      const out: Answers = {}
      if (q.surface) out.surface = { type: 'choice', choice: pick.surface, probability: 0.9 }
      if (q.app && pick.app) {
        const c = q.app.type === 'choice' ? Object.entries(q.app.criteria).find(([, name]) => name === pick.app)?.[0] : undefined
        if (c) out.app = { type: 'choice', choice: c, probability: 0.9 }
      }
      if (q.next) out.next = { type: 'choice', choice: pick.next.shift() ?? 'stuck', probability: 0.9 }
      return out
    },
  }
  const desktop = { kind: 'desktop' as const, observe: async () => screens[front], act: async (a: ScreenAction) => void actions.push(a) }
  const dock = { list: async () => ['LocalRouter', 'Google Chrome', 'Mail'], activate: async (name: string) => { brought.push(name); if (screens[name]) front = name } }
  const broker = new Broker({ executorId: 'exec_t', name: 'JJ’s MacBook', machine: { screen: desktop, tabs: null, launcher: dock, apps: null, controls: { run: async () => ({ done: 'ok' }) } }, decider, audit: new MemoryAudit(), gate: new TestGate(), minDwellMs: 0, settleMs: 1, refocusMs: 0, ownApps: ['LocalRouter', 'Electron'] })
  return { broker, actions, brought }
}

describe('LocalRouter in front: leaving it is fine, acting on it is not', () => {
  it('switching from LocalRouter to Google Chrome is allowed, and the task works there', async () => {
    const { broker, actions, brought } = fromLocalRouter({ surface: 'app', app: 'Google Chrome', next: ['c0', 'done'] })
    const started = broker.startUse(app, 'open my Gmail primary tab in Chrome', null)
    if (!started.ok) throw new Error(started.error)
    const outcome = await started.finished
    expect(outcome).toMatchObject({ status: 'done' })
    expect(brought).toEqual(['Google Chrome'])
    expect(actions).toEqual([{ kind: 'click', ref: 'tab1' }])
  })

  it('acting on LocalRouter’s own window is still refused', async () => {
    const { broker, actions } = fromLocalRouter({ surface: 'current', next: ['c0'] })
    const started = broker.startUse(app, 'press the button', null)
    if (!started.ok) throw new Error(started.error)
    expect(await started.finished).toMatchObject({ status: 'blocked', reason: OWN_TARGET })
    expect(actions).toEqual([])
  })

  it('bringing LocalRouter itself forward is refused', async () => {
    const { broker, brought } = fromLocalRouter({ surface: 'app', app: 'LocalRouter', next: [] })
    const started = broker.startUse(app, 'go to LocalRouter', null)
    if (!started.ok) throw new Error(started.error)
    expect(await started.finished).toMatchObject({ status: 'blocked', reason: OWN_TARGET })
    expect(brought).toEqual([])
  })
})
