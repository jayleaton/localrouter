import { describe, expect, it } from 'vitest'
import type { Grant } from '../../../desktop-helper/src/computer/controller.ts'
import type { Card } from './broker.ts'
import { Broker, CONTROL_OFF, DISABLED_MESSAGE, OWN_TARGET } from './broker.ts'
import type { Answers, Question } from '../../../desktop-helper/src/computer/decisions.ts'
import type { Observation, ScreenAction } from '../../../desktop-helper/src/computer/types.ts'
import { LocalStop } from '../../../desktop-helper/src/stop.ts'
import { MemoryAudit } from './audit.ts'
import { MemoryStanding, TestGate } from './testing.ts'
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
  it('audits a decision that could not be read (attempts, redacted replies) and how each action was done', async () => {
    const { broker, audit } = fakeBroker(['type_text', 'done'], {})
    const started = broker.startUse(app, 'search my notes for groceries', 'groceries')
    if (!started.ok) throw new Error(started.error)
    await started.finished
    expect(audit.lines.filter((l) => l.event === 'act').map((l) => l.kind)).toContain('type')
    const failing = fakeBroker([], { decider: { ask: async () => { throw Object.assign(new Error('the decision was not the JSON asked for (asked 2 times)'), { attempts: 2, replies: ['(empty reply)', '(empty reply)'] }) } } })
    const second = failing.broker.startUse(app, 'search my notes for groceries', null)
    if (!second.ok) throw new Error(second.error)
    expect((await second.finished).status).toBe('stuck')
    expect(failing.audit.lines.find((l) => l.event === 'decide.failed')).toMatchObject({ attempts: 2, replies: ['(empty reply)', '(empty reply)'] })
  })

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

describe('the standing grant (JJ, 2026-10-26: auto-approved; no permission window ever blocks a task)', () => {
  /** Every card the broker shows. */
  const cards = (broker: ReturnType<typeof fakeBroker>['broker']) => {
    const shown: Card[] = []
    broker.on((e) => void (e.type === 'approval' && shown.push(e.card)))
    return shown
  }
  const run = async (broker: ReturnType<typeof fakeBroker>['broker'], goal = 'send the Groceries note') => {
    const started = broker.startUse(app, goal, null)
    if (!started.ok) throw new Error(started.error)
    return started.finished
  }
  /** A presence gate that fails the test if anything asks it: the default path never opens Touch ID, Hello or a sheet. */
  const noModal = { method: 'Touch ID', verify: async (): Promise<boolean> => { throw new Error('a modal was opened during a task') } }

  it('without the grant a task is refused at once, with an honest reply and a notification event, never a card', async () => {
    const standing = new MemoryStanding()
    const { broker, mac, audit } = fakeBroker([...SEND_GROCERIES], { standing, gate: noModal })
    const shown = cards(broker)
    const needed: string[] = []
    broker.on((e) => void (e.type === 'consent.needed' && needed.push(e.goal)))
    const started = broker.startUse(app, 'send the Groceries note', null)
    expect(started).toEqual({ ok: false, error: CONTROL_OFF })
    expect(CONTROL_OFF).toMatch(/Turn it on in LocalRouter → Settings/)
    expect(needed).toEqual(['send the Groceries note'])
    expect(shown).toEqual([])
    expect(mac.desktop.actions).toEqual([])
    expect(broker.status().tasks).toEqual([]) // nothing waits
    // Locking the screen (a step that would have asked) is refused the same way; a volume change still runs.
    expect(await broker.control(app, { kind: 'lock_screen' })).toEqual({ ok: false, text: CONTROL_OFF })
    expect((await broker.control(app, { kind: 'volume', action: 'up', target: 'system' })).ok).toBe(true)
    expect(shown).toEqual([])
    expect(audit.lines.filter((l) => l.event === 'deny')).toHaveLength(2)
  })

  it('granted (onboarding, or migrated from an install already set up): no card, no modal, for any step, the pointer included', async () => {
    const standing = new MemoryStanding()
    const { broker, mac, audit } = fakeBroker([...SEND_GROCERIES], { standing, gate: noModal })
    broker.grantStanding('migrated')
    broker.grantStanding('onboarding') // already granted: nothing more
    expect(audit.lines.filter((l) => l.event === 'consent.granted')).toEqual([expect.objectContaining({ grant: 'standing', source: 'migrated' })])
    mac.desktop.pointerOnly.add('send') // Send needs the real pointer: it still runs without a card
    const shown = cards(broker)
    expect((await run(broker)).status).toBe('done')
    expect((await broker.control(app, { kind: 'lock_screen' })).ok).toBe(true)
    expect(shown).toEqual([])
    expect(mac.desktop.sent).toEqual(['Groceries: milk, eggs, bread'])
    expect(mac.desktop.modes).toContain('pointer')
    expect(audit.lines.filter((l) => l.event === 'act').every((l) => l.grant === 'standing')).toBe(true)
    expect(audit.lines.find((l) => l.event === 'grant' && (l.action as { label?: string }).label === 'Send')).toMatchObject({ grant: 'standing', would_ask: expect.stringMatching(/send/) })
  })

  it('Windows run 2026-10-27: a background-only task is never stopped by the person moving and clicking their own mouse', async () => {
    let n = 0
    // The person's mouse: somewhere new on every reading, and just clicked.
    const pointer = async () => {
      n += 1
      return { x: 300 + ((n * 97) % 800), y: 200 + ((n * 53) % 500), sinceClickS: 0 }
    }
    // Time moves on between readings (a real task's steps take hundreds of milliseconds).
    let clock = 1_000_000
    const now = () => (clock += 300)
    const watch = (abort: (reason: string) => void) => new LocalStop({ pointer, now, schedule: () => () => undefined }).watch(abort)
    const { broker, mac } = fakeBroker([...SEND_GROCERIES], { standing: new MemoryStanding({ granted: true, askEach: false }), watch })
    const outcome = await run(broker)
    expect(outcome).toMatchObject({ status: 'done' })
    expect(mac.desktop.modes.every((m) => m === 'background')).toBe(true)
  })

  it('kept across a restart or an update (a new broker over the same stored grant): no card at all', async () => {
    const kept = { granted: false, askEach: false }
    fakeBroker([], { standing: new MemoryStanding(kept) }).broker.grantStanding('onboarding')
    const { broker, mac, gate } = fakeBroker([...SEND_GROCERIES], { standing: new MemoryStanding(kept) })
    const shown = cards(broker)
    expect((await run(broker)).status).toBe('done')
    expect(shown).toEqual([])
    expect(gate.asked).toEqual([])
    expect(mac.desktop.sent).toEqual(['Groceries: milk, eggs, bread'])
  })

  it('revoking turns computer control off again: the next task is refused at once', async () => {
    const standing = new MemoryStanding({ granted: true, askEach: false })
    const { broker } = fakeBroker([...SEND_GROCERIES], { standing })
    broker.revokeStanding('settings')
    expect(standing.kept.granted).toBe(false)
    expect(broker.approvalMode()).toBe('consent')
    expect(broker.startUse(app, 'send the Groceries note', null)).toEqual({ ok: false, error: CONTROL_OFF })
  })

  it('"Ask before each action" (opt-in) brings the per-step cards back, the presence check with them', async () => {
    const { broker, mac, gate } = fakeBroker([...SEND_GROCERIES], { standing: new MemoryStanding({ granted: false, askEach: true }) })
    expect(broker.approvalMode()).toBe('each')
    const shown = cards(broker)
    broker.on((e) => void (e.type === 'approval' && setTimeout(() => void broker.answer(e.card.id, 'approve', 'app window'), 1)))
    expect((await run(broker)).status).toBe('done')
    expect(shown.map((c) => c.action)).toEqual(['click “Send” in Notes'])
    expect(gate.asked).toHaveLength(1)
    expect(mac.desktop.sent).toEqual(['Groceries: milk, eggs, bread'])
  })
})

describe('a stop or another answer dismisses the presence check still open for that approval (Mac run, 2026-10-26)', () => {
  it('⌘⇧⎋ while Touch ID or the confirmation sheet is open: its signal aborts, the audit says so, nothing is done', async () => {
    let seen: AbortSignal | undefined
    const gate = {
      method: 'Touch ID',
      verify: (_card: unknown, _waiting?: (text: string) => void, signal?: AbortSignal) =>
        new Promise<boolean>((resolve) => {
          seen = signal
          signal?.addEventListener('abort', () => resolve(false), { once: true })
        }),
    }
    const { broker, mac, audit } = fakeBroker([...SEND_GROCERIES], { gate })
    const card = nextCard(broker)
    const started = broker.startUse(app, 'send the Groceries note', null)
    if (!started.ok) throw new Error(started.error)
    const c = await card
    const answered = broker.answer(c.id, 'approve', 'app window')
    await new Promise((resolve) => setTimeout(resolve, 10))
    expect(seen?.aborted).toBe(false)
    broker.stop(undefined, 'the keyboard')
    expect(await answered).toMatchObject({ ok: false, status: 409 })
    expect(seen?.aborted).toBe(true)
    expect((await started.finished).status).toBe('stopped')
    expect(mac.desktop.sent).toEqual([])
    expect(audit.lines.find((l) => l.event === 'approval.verify')).toMatchObject({ approval_id: c.id, method: 'Touch ID' })
    expect(audit.lines.find((l) => l.event === 'approval.dismissed')).toMatchObject({ approval_id: c.id, method: 'Touch ID', outcome: 'cancelled' })
  })

  it('the card names the fallback and why the OS check is not used', async () => {
    const gate = { method: 'a confirmation on this computer', why: 'Touch ID is not available: no fingerprint is enrolled', verify: async () => true }
    const { broker } = fakeBroker([...SEND_GROCERIES], { gate })
    const card = nextCard(broker)
    const started = broker.startUse(app, 'send the Groceries note', null)
    if (!started.ok) throw new Error(started.error)
    const c = await card
    expect(c.confirm).toBe('After Approve, LocalRouter asks you to confirm it once more in its own window (Touch ID is not available: no fingerprint is enrolled).')
    broker.stop(undefined, 'test')
    await started.finished
  })
})
