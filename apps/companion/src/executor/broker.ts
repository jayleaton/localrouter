import { randomBytes } from 'node:crypto'
import { ComputerUse, clampLimit, MAX_STEPS, TIMEOUT_MS } from '../../../desktop-helper/src/computer/controller.ts'
import type { AuthorizeAction, AuthorizeRequest, ComputerOutcome, Deps, Grant, Authorize, TaskWatch } from '../../../desktop-helper/src/computer/controller.ts'
import type { Answers, Decider, Question } from '../../../desktop-helper/src/computer/decisions.ts'
import type { ComputerControl } from '../../../desktop-helper/src/computer/macControls.ts'
import type { BrowserTabs, Observation, ScreenBackend } from '../../../desktop-helper/src/computer/types.ts'
import { silentLogger } from '../../../desktop-helper/src/log.ts'
import type { Logger } from '../../../desktop-helper/src/log.ts'
import { Store } from './approvals.ts'
import type { Approval, Binding, Taken } from './approvals.ts'
import { clip, textDigest } from './audit.ts'
import type { Audit } from './audit.ts'
import * as policy from './policy.ts'

/**
 * The executor's broker: the final authority over this computer, whoever asks. Every caller (our own harness, a voice
 * session's tool call, a third-party harness over MCP, the box's legacy `/v1/desktop` bridge) goes through it, and it
 * enforces, beneath every API: one task at a time on this computer (the lease), the step and working-time budgets,
 * the policy (`policy.ts`, the box's rules), the decision provider's judgment where the rules ask for one, a person's
 * approval on this computer for consequential steps, the target revalidated after that approval, the local stop and
 * switch, and the audit. Remote advice (a box's grant, a model's confidence) can add a refusal, never permission.
 *
 * Approvals are answered only through `answer`, which the trusted app UI calls; no caller channel (MCP, the box, the
 * model) has an operation for it. An approval needs the presence gate (Touch ID on a Mac that has it), is refused while
 * this executor is itself injecting input, and is refused within `minDwellMs` of the card appearing (a stale click).
 */
export type CallerKind = 'app' | 'voice' | 'mcp' | 'box'
export type Caller = { id: string; name: string; kind: CallerKind }
/** Who asks, and the person's own words when the request came from a conversation turn. */
export type Requester = { caller: Caller; utterance?: string }

/** What the executor drives: the real Mac, or the fake one. */
export type Machine = {
  desktop: ScreenBackend | null
  tabs: BrowserTabs | null
  dock: Deps['dock']
  apps: Deps['apps']
  controls: { run(control: ComputerControl): Promise<{ done: string }>; notify?(title: string, text: string): Promise<void> }
}

export type Card = {
  id: string
  /** The sentence: what exactly would happen, where. */
  action: string
  kind: string
  text: string
  app: string
  title: string
  url: string
  why: string
  goal: string
  requester: string
  expires_in_s: number
  state: Approval['state']
}

/** A person's presence, checked by the OS where it can be (Touch ID), for one approval. */
export interface PresenceGate {
  readonly method: string
  verify(card: Card): Promise<boolean>
}

export type Limits = { maxSteps: number; timeoutMs: number; approvalTimeoutMs: number }

export type BrokerOptions = {
  /** This installation's stable id: approvals bind to it. */
  executorId: string
  /** The computer's name, for cards and status ("JJ's MacBook Pro"). */
  name: string
  machine: Machine
  /** Advisory decisions (the Spark's System One); null: computer use is unavailable and judged clicks ask. */
  decider: Decider | null
  audit: Audit
  gate: PresenceGate
  /** The local stop for one task (the pointer corner / takeover watch). */
  watch?: (abort: (reason: string) => void) => TaskWatch | null
  /** App names that are this companion itself: never observed as a target, never acted on. */
  ownApps?: string[]
  limits?: Partial<Limits>
  minDwellMs?: number
  /** Time for the screen to settle after an action (the controller's 600 ms by default). */
  settleMs?: number
  now?: () => number
  log?: Logger
}

export type TaskStatus = 'running' | ComputerOutcome['status']
export type TaskView = {
  task_id: string
  kind: 'control' | 'use'
  computer: string
  requester: string
  goal: string
  status: TaskStatus
  reason: string
  steps: string[]
  waiting_for_approval: string | null
  ms: number
}

export type BrokerEvent =
  | { type: 'approval'; card: Card }
  | { type: 'approval.end'; id: string; outcome: Taken }
  | { type: 'task'; task: TaskView }

export type Started = { ok: true; task_id: string; finished: Promise<ComputerOutcome> } | { ok: false; error: string }

export const DISABLED_MESSAGE = 'Computer control is switched off on this computer'
export const OWN_TARGET = 'LocalRouter’s own windows are never a target.'
const MAX_DECISIONS_PER_STEP = 2

type Task = {
  id: string
  kind: 'control' | 'use'
  requester: Requester
  goal: string
  status: TaskStatus
  reason: string
  steps: string[]
  started: number
  waited: number
  authorized: number
  decided: number
  approvalId: string | null
  stopping: boolean
  controller: AbortController
  wake: (() => void) | null
  ended: number | null
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error))

export class Broker {
  readonly executorId: string
  readonly name: string
  readonly limits: Limits
  private readonly options: BrokerOptions
  private readonly store: Store
  private readonly now: () => number
  private readonly log: Logger
  private readonly ownApps: string[]
  private readonly minDwellMs: number
  private readonly listeners = new Set<(event: BrokerEvent) => void>()
  private readonly waiters = new Map<string, () => void>()
  private readonly tasks: Task[] = []
  private active: Task | null = null
  private injecting = 0
  private disabledFlag = false
  private readonly machine: Machine

  constructor(options: BrokerOptions) {
    this.options = options
    this.executorId = options.executorId
    this.name = options.name
    this.now = options.now ?? Date.now
    this.log = options.log ?? silentLogger
    this.limits = {
      maxSteps: clampLimit(options.limits?.maxSteps, MAX_STEPS),
      timeoutMs: clampLimit(options.limits?.timeoutMs, TIMEOUT_MS),
      approvalTimeoutMs: clampLimit(options.limits?.approvalTimeoutMs, 60_000),
    }
    this.store = new Store(this.limits.approvalTimeoutMs)
    this.ownApps = (options.ownApps ?? ['LocalRouter']).map((n) => n.toLowerCase())
    this.minDwellMs = options.minDwellMs ?? 800
    this.machine = this.guarded(options.machine)
  }

  on(listener: (event: BrokerEvent) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  get disabled() {
    return this.disabledFlag
  }

  /** The local switch: off refuses every request and stops what runs. */
  setDisabled(disabled: boolean) {
    this.disabledFlag = disabled
    this.options.audit.record(disabled ? 'executor.disabled' : 'executor.enabled')
    if (disabled) this.stop(undefined, 'this computer', `${DISABLED_MESSAGE}.`)
  }

  /** Whether this executor is moving the pointer or typing right now (an approval cannot come during that). */
  get inputActive() {
    return this.injecting > 0
  }

  /** The machine as this broker drives it (every input counted); tests and the revalidation read it. */
  get driven(): Machine {
    return this.machine
  }

  get capabilities() {
    return {
      executor: this.executorId,
      computer: this.name,
      controls: [...policy.CONTROL_KINDS],
      computer_use: Boolean(this.options.machine.desktop) && this.options.decider !== null,
      presence: this.options.gate.method,
      limits: { max_steps: this.limits.maxSteps, timeout_ms: this.limits.timeoutMs, approval_timeout_ms: this.limits.approvalTimeoutMs },
      disabled: this.disabledFlag,
    }
  }

  // ------------------------------------------------------------------ fixed controls

  /** A fixed control: the policy, the person's approval when it asks for one, then the machine runs it. */
  async control(who: Requester, ctl: ComputerControl, describe: (c: ComputerControl) => string = describeControl): Promise<{ ok: boolean; text: string; task_id?: string }> {
    const refused = this.admit(who)
    if (refused) return { ok: false, text: refused }
    const d = policy.control(ctl.kind)
    if (d.verdict === 'refused') return { ok: false, text: d.why }
    if (ctl.kind === 'open_app' || ctl.kind === 'quit_app') {
      if (this.isOwn(ctl.app)) return { ok: false, text: OWN_TARGET }
    }
    if (this.active) return { ok: false, text: `${this.name} is busy with another task; stop it first` }
    const summary = describe(ctl)
    const task = this.newTask('control', who, summary)
    this.options.audit.record('task.start', { task_id: task.id, kind: 'control', control: ctl.kind, goal: clip(summary), requester: who.caller.name, caller: who.caller.kind })
    try {
      if (d.verdict === 'tap') {
        const g = await this.awaitApproval(task, { step: 0, app: '', title: '', url: '', action: { kind: 'control', name: ctl.kind }, why: d.why })
        if (!g.ok) return this.end(task, 'blocked', g.denied)
      }
      if (task.stopping) return this.end(task, 'stopped', task.reason || 'It was stopped.')
      const { done } = await this.machine.controls.run(ctl)
      return this.end(task, 'done', done)
    } catch (error) {
      return this.end(task, 'failed', messageOf(error))
    }
  }

  // ------------------------------------------------------------------ computer use

  /**
   * Starts a bounded computer-use task; it runs here and `finished` gives its outcome. `advice` is a remote grant
   * (the box's `authorize`, for the legacy bridge): it may refuse a step, never permit one the local rules do not.
   */
  startUse(who: Requester, goal: string, text: string | null, extra: { decider?: Decider; advice?: Authorize; limits?: { maxSteps?: number; timeoutMs?: number }; onStep?: (step: string) => void; taskId?: string } = {}): Started {
    const refused = this.admit(who)
    if (refused) return { ok: false, error: refused }
    const g = goal.trim()
    if (!g || g.length > 1000) return { ok: false, error: 'goal is required (at most 1000 characters)' }
    if (text !== null && (!text || Buffer.byteLength(text, 'utf8') > policy.MAX_TEXT)) return { ok: false, error: 'text is at most 4000 characters' }
    const decider = extra.decider ?? this.options.decider
    if (!decider) return { ok: false, error: 'computer use needs the decision service on the Spark, which is not connected' }
    if (!this.machine.desktop) return { ok: false, error: `${this.name} cannot read its screen: give LocalRouter Accessibility access in System Settings` }
    if (this.active) return { ok: false, error: `${this.name} is busy with another task; stop it first` }
    const task = this.newTask('use', who, g, extra.taskId)
    this.options.audit.record('task.start', { task_id: task.id, kind: 'use', goal: clip(g), text: text === null ? null : textDigest(text), requester: who.caller.name, caller: who.caller.kind })
    const watch = this.options.watch?.((reason) => this.stop(task.id, 'this computer', reason)) ?? null
    const counted: Decider = { ask: (state, questions) => this.decideFor(task, decider, state, questions) }
    const computer = new ComputerUse({
      decider: counted,
      authorize: (request) => this.authorizeStep(task, request, extra.advice),
      onStep: (step) => {
        task.steps.push(step)
        this.emitTask(task)
        extra.onStep?.(step)
      },
      tabs: this.machine.tabs,
      desktop: this.machine.desktop,
      dock: this.machine.dock,
      apps: this.machine.apps,
      log: this.log,
      now: this.now,
      ...(this.options.settleMs !== undefined ? { settleMs: this.options.settleMs } : {}),
    })
    void this.machine.controls.notify?.('LocalRouter', `Jarvis is using ${this.name}: ${g}. Move the pointer to the top-left corner to stop.`).catch(() => undefined)
    const finished = computer
      .run({ goal: g, text }, { signal: task.controller.signal, limits: { maxSteps: Math.min(this.limits.maxSteps, extra.limits?.maxSteps ?? MAX_STEPS), timeoutMs: Math.min(this.limits.timeoutMs, extra.limits?.timeoutMs ?? TIMEOUT_MS) }, ...(watch ? { watch } : {}) })
      .then((outcome) => {
        watch?.end()
        this.end(task, outcome.status, outcome.reason)
        return outcome
      })
    return { ok: true, task_id: task.id, finished }
  }

  /** Stops a task (every running one when `taskId` is empty). Works with no network and no model. */
  stop(taskId: string | undefined, by: string, reason = 'You stopped it.'): number {
    let n = 0
    for (const task of this.tasks) {
      if (task.status !== 'running' || (taskId && task.id !== taskId)) continue
      task.stopping = true
      task.reason = reason
      task.controller.abort(reason)
      this.store.cancelTask(task.id)
      task.wake?.()
      this.options.audit.record('task.stop', { task_id: task.id, by })
      n += 1
    }
    return n
  }

  /** Waits until the task ends or `ms` passes; its state then. */
  async wait(taskId: string, ms: number): Promise<TaskView | null> {
    const deadline = this.now() + ms
    for (;;) {
      const task = this.tasks.find((t) => t.id === taskId)
      if (!task) return null
      if (task.status !== 'running' || this.now() >= deadline) return this.view(task)
      await new Promise((resolve) => setTimeout(resolve, Math.min(100, Math.max(1, deadline - this.now()))))
    }
  }

  task(taskId: string): TaskView | null {
    const task = this.tasks.find((t) => t.id === taskId)
    return task ? this.view(task) : null
  }

  /** The shape the `/voice` page's computer cards read (`/v1/computer/status`), served by this executor. */
  status() {
    const now = this.now()
    const cards = this.store.list(now)
    return {
      computers: [{ helper_id: this.executorId, name: this.name, controls: [...policy.CONTROL_KINDS], computer_use: this.capabilities.computer_use, accessibility: Boolean(this.options.machine.desktop), busy: this.active?.id ?? null, disabled: this.disabledFlag, placement: 'device_executor' }],
      tasks: [...this.tasks].reverse().map((t) => this.view(t)),
      pending_approvals: cards.filter((c) => c.state === 'pending').length,
    }
  }

  approvals(): Card[] {
    const now = this.now()
    return this.store.list(now).map((x) => this.card(x, now))
  }

  audit(limit = 100) {
    return this.options.audit.tail(Math.max(1, Math.min(1000, limit)))
  }

  /**
   * The person's answer from the trusted app UI. Deny always works. Approve needs the presence gate, and is refused
   * while this executor is injecting input or within `minDwellMs` of the card appearing.
   */
  async answer(id: string, decision: 'approve' | 'deny', by: string): Promise<{ ok: true } | { ok: false; error: string; status: number }> {
    const now = this.now()
    const x = this.store.get(id, now)
    if (!x) return { ok: false, error: 'no such approval', status: 404 }
    if (x.state !== 'pending') return { ok: false, error: `that approval is ${x.state}`, status: 409 }
    if (decision === 'deny') {
      this.store.decide(id, false, by, now)
      this.options.audit.record('approval.answer', { approval_id: id, decision, by })
      this.waiters.get(id)?.()
      return { ok: true }
    }
    if (this.inputActive) return this.refuseAnswer(id, by, 'LocalRouter is moving the pointer or typing right now, so an approval cannot come from this computer at this moment.')
    if (now - x.created_ms < this.minDwellMs) return this.refuseAnswer(id, by, 'That was too quick to be a deliberate approval: read the card, then approve.')
    let present = false
    try {
      present = await this.options.gate.verify(this.card(x, now))
    } catch (error) {
      this.log.warn('approval.gate_failed', { error: messageOf(error) })
    }
    if (!present) return this.refuseAnswer(id, by, `Not approved: ${this.options.gate.method} did not confirm it was you.`)
    if (this.inputActive) return this.refuseAnswer(id, by, 'LocalRouter moved the pointer or typed while you approved, so it did not count.')
    const r = this.store.decide(id, true, `${by} (${this.options.gate.method})`, this.now())
    if (r !== 'ok') return { ok: false, error: r === 'expired' ? 'that approval expired' : 'that approval was already answered', status: 409 }
    this.options.audit.record('approval.answer', { approval_id: id, decision, by, presence: this.options.gate.method })
    this.waiters.get(id)?.()
    return { ok: true }
  }

  // ------------------------------------------------------------------ the rules

  private refuseAnswer(id: string, by: string, error: string): { ok: false; error: string; status: number } {
    this.options.audit.record('approval.refused', { approval_id: id, by, reason: error })
    return { ok: false, error, status: 403 }
  }

  private admit(who: Requester): string | null {
    if (this.disabledFlag) return DISABLED_MESSAGE
    if (who.caller.kind === 'voice' && !(who.utterance ?? '').trim()) return 'no user request in this voice turn'
    return null
  }

  private isOwn(name: string | undefined) {
    const n = (name ?? '').trim().toLowerCase()
    return n.length > 0 && this.ownApps.some((own) => n === own || n.startsWith(`${own} `) || n.startsWith(`${own}:`))
  }

  private async decideFor(task: Task, decider: Decider, state: unknown, questions: Record<string, Question>): Promise<Answers> {
    if (task.stopping || task.status !== 'running') throw new Error('the task was stopped')
    if (task.decided >= MAX_DECISIONS_PER_STEP * this.limits.maxSteps + 4) throw new Error('the task used up its decisions')
    if (this.worked(task) > this.limits.timeoutMs) throw new Error('It was taking too long.')
    task.decided += 1
    const answers = await decider.ask(state, questions)
    this.options.audit.record('decide', { task_id: task.id, answers: Object.fromEntries(Object.entries(answers).map(([k, a]) => [k, a.type === 'choice' ? { choice: a.choice, p: a.probability } : { p: a.probability }])) })
    return answers
  }

  private worked(task: Task) {
    return this.now() - task.started - task.waited
  }

  /** One on-screen step, as the controller asks for it: the budget, the target, the rules, then approval. */
  async authorizeStep(task: Task, request: AuthorizeRequest, advice?: Authorize): Promise<Grant> {
    const act = flatten(request.action)
    const deny = (reason: string): Grant => {
      this.options.audit.record('deny', { task_id: task.id, step: request.step, action: actionLog(act), reason })
      return { ok: false, denied: reason }
    }
    if (task.stopping || task.status !== 'running') return deny('It was stopped.')
    if (this.disabledFlag) return deny(DISABLED_MESSAGE)
    if (task.authorized >= this.limits.maxSteps + 1) return deny('It did not finish within its steps.') // the surface, then the steps
    if (this.worked(task) > this.limits.timeoutMs) return deny('It was taking too long.')
    task.authorized += 1
    const obs = request.observation
    if (this.isOwn(obs.app) || (act.kind === 'surface' && this.isOwn(act.name))) return deny(OWN_TARGET)
    let d = policy.screen(act)
    let judged: number | null = null
    if (d.verdict === 'judge') {
      judged = await this.judge(task, request)
      d = policy.afterJudge(judged)
    }
    if (d.verdict === 'refused') return deny(d.why)
    if (advice) {
      // The box's own grant first (it may wait for its own approval). A refusal there holds; an allow is not ours.
      let remote: Grant
      try {
        remote = await advice(request)
      } catch (error) {
        return deny(`The box did not allow the next step: ${messageOf(error)}`)
      }
      if (!remote.ok) return deny(remote.denied || 'The box did not allow that.')
      if (task.stopping) return deny('It was stopped.')
    }
    if (d.verdict === 'auto') {
      this.options.audit.record('grant', { task_id: task.id, step: request.step, action: actionLog(act), app: clip(obs.app), judged })
      return { ok: true, grant: 'allow' }
    }
    const g = await this.awaitApproval(task, { step: request.step, app: obs.app, title: obs.title, url: obs.url ?? '', action: act, why: d.why })
    if (!g.ok) return deny(g.denied)
    const stale = await this.revalidate(request)
    if (stale) return deny(stale)
    this.options.audit.record('grant', { task_id: task.id, step: request.step, action: actionLog(act), app: clip(obs.app), approved: true })
    return { ok: true, grant: 'approved' }
  }

  /** Could this step send, buy, delete, post or submit? The provider's yes probability, or null when it cannot say. */
  private async judge(task: Task, request: AuthorizeRequest): Promise<number | null> {
    const decider = this.options.decider
    if (!decider || request.action.kind !== 'click') return null
    try {
      const state = { goal: task.goal, screen: request.observation, action: { kind: 'click', role: request.action.role, label: request.action.label } }
      const answers = await decider.ask(state, { consequential: { type: 'noul', instructions: 'If the program does this action on this screen, could it send a message or email, post or publish something, buy or pay for something, delete or remove something, or submit a form? Answer yes if it could, or if you cannot tell.' } })
      const a = answers.consequential
      return a?.type === 'noul' ? a.probability : null
    } catch {
      return null
    }
  }

  /**
   * After an approval time has passed: the screen must still be the one approved. Same app, window title and page,
   * and the exact control (ref, role, label) still on it; for Enter, the field it was typed into. Else nothing runs.
   */
  private async revalidate(request: AuthorizeRequest): Promise<string | null> {
    const action = request.action
    if (action.kind === 'scroll' || action.kind === 'surface') return null
    const desktop = this.machine.desktop
    if (!desktop) return 'The screen cannot be read to check it again.'
    let now: Observation
    try {
      now = await desktop.observe()
    } catch (error) {
      return `The screen could not be checked again: ${messageOf(error)}`
    }
    const was = request.observation
    const changed = 'The screen changed while it waited for your OK, so it did not act.'
    if (now.app !== was.app || now.title !== was.title || (was.url ?? '') !== (now.url ?? '')) return changed
    if (this.isOwn(now.app)) return OWN_TARGET
    if (action.kind === 'click' || action.kind === 'type') {
      const same = now.controls.find((c) => c.ref === action.ref)
      if (!same || same.role !== action.role || same.label !== action.label) return changed
    }
    if (action.kind === 'press' && action.field) {
      const field = action.field
      if (!now.controls.some((c) => c.role === field.role && c.label === field.label && c.focused !== false)) return changed
    }
    return null
  }

  private async awaitApproval(task: Task, b: Omit<Binding, 'executor' | 'task_id' | 'goal' | 'requester'>): Promise<Grant> {
    if (task.stopping) return { ok: false, denied: 'It was stopped.' }
    const binding: Binding = { executor: this.executorId, task_id: task.id, goal: task.goal, requester: task.requester.caller.name, ...b }
    const id = `appr_${randomBytes(12).toString('hex')}`
    const start = this.now()
    let x: Approval
    try {
      x = this.store.create(binding, start, id)
    } catch (error) {
      return { ok: false, denied: messageOf(error) }
    }
    task.approvalId = id
    this.options.audit.record('approval.request', { task_id: task.id, approval_id: id, step: b.step, action: actionLog(b.action), app: clip(b.app), why: b.why })
    this.emit({ type: 'approval', card: this.card(x, start) })
    this.emitTask(task)
    const outcome = await new Promise<Taken>((resolve) => {
      let timer: ReturnType<typeof setTimeout> | null = null
      const settle = (t: Taken) => {
        if (timer) clearTimeout(timer)
        this.waiters.delete(id)
        task.wake = null
        resolve(t)
      }
      const check = () => {
        const t = this.store.take(id, x.digest, this.now())
        if (t !== 'pending') return settle(t)
        if (task.stopping || task.status !== 'running') {
          this.store.cancelTask(task.id)
          return settle('cancelled')
        }
      }
      this.waiters.set(id, check)
      task.wake = check
      timer = setTimeout(check, this.store.ttlMs + 5)
      check() // an answer (or a stop) that came while the card was being shown
    })
    task.waited += this.now() - start
    task.approvalId = null
    this.options.audit.record('approval.end', { task_id: task.id, approval_id: id, outcome })
    this.emit({ type: 'approval.end', id, outcome })
    this.emitTask(task)
    switch (outcome) {
      case 'approved':
        return { ok: true, grant: 'approved' }
      case 'denied':
        return { ok: false, denied: 'You said no on the approval card.' }
      case 'expired':
        return { ok: false, denied: 'Nobody approved it in time.' }
      case 'cancelled':
        return { ok: false, denied: 'It was stopped.' }
      default:
        return { ok: false, denied: 'The approval did not match this step.' }
    }
  }

  // ------------------------------------------------------------------ bookkeeping

  private newTask(kind: Task['kind'], who: Requester, goal: string, id?: string): Task {
    const task: Task = { id: id ?? `${kind === 'use' ? 'cu' : 'ctl'}_${randomBytes(6).toString('hex')}`, kind, requester: who, goal, status: 'running', reason: '', steps: [], started: this.now(), waited: 0, authorized: 0, decided: 0, approvalId: null, stopping: false, controller: new AbortController(), wake: null, ended: null }
    this.tasks.push(task)
    while (this.tasks.length > 50) {
      const i = this.tasks.findIndex((t) => t.status !== 'running')
      if (i < 0) break
      this.tasks.splice(i, 1)
    }
    this.active = task
    this.emitTask(task)
    return task
  }

  private end(task: Task, status: ComputerOutcome['status'], reason: string) {
    if (task.status === 'running') {
      task.status = task.stopping && status !== 'done' ? 'stopped' : status
      task.reason = task.stopping && task.reason ? task.reason : reason
      task.ended = this.now()
      this.store.cancelTask(task.id)
      if (this.active === task) this.active = null
      this.options.audit.record('task.end', { task_id: task.id, status: task.status, reason: clip(task.reason), steps: task.steps.length })
      this.emitTask(task)
    }
    return { ok: task.status === 'done', text: task.reason, task_id: task.id }
  }

  private view(t: Task): TaskView {
    return { task_id: t.id, kind: t.kind, computer: this.name, requester: t.requester.caller.name, goal: t.goal, status: t.status, reason: t.reason, steps: [...t.steps], waiting_for_approval: t.approvalId, ms: (t.ended ?? this.now()) - t.started }
  }

  private card(x: Approval, now: number): Card {
    const b = x.binding
    return { id: x.id, action: describeAction(b), kind: b.action.kind, text: b.action.text ?? '', app: b.app, title: b.title, url: b.url, why: b.why, goal: b.goal, requester: b.requester, expires_in_s: Math.max(0, Math.ceil((x.expires_ms - now) / 1000)), state: x.state }
  }

  private emit(event: BrokerEvent) {
    for (const l of this.listeners) {
      try {
        l(event)
      } catch (error) {
        this.log.warn('broker.listener_failed', { error: messageOf(error) })
      }
    }
  }

  private emitTask(task: Task) {
    this.emit({ type: 'task', task: this.view(task) })
  }

  /** The machine with every input it injects counted, so an approval is never taken while it does. */
  private guarded(m: Machine): Machine {
    const during = async <T>(work: () => Promise<T>): Promise<T> => {
      this.injecting += 1
      try {
        return await work()
      } finally {
        this.injecting -= 1
      }
    }
    const desktop = m.desktop
    return {
      desktop: desktop
        ? { kind: desktop.kind, observe: () => desktop.observe(), act: (a) => during(() => desktop.act(a)), ...(desktop.pointer ? { pointer: () => desktop.pointer!() } : {}) }
        : null,
      tabs: m.tabs ? Object.assign(Object.create(m.tabs) as BrowserTabs, { activate: (tab: Parameters<BrowserTabs['activate']>[0]) => during(() => m.tabs!.activate(tab)), ...(m.tabs.search ? { search: (name: string) => during(() => m.tabs!.search!(name)) } : {}) }) : null,
      dock: m.dock ? { list: () => m.dock!.list(), activate: (name) => during(() => m.dock!.activate(name)) } : null,
      apps: m.apps ? { running: () => m.apps!.running(), activate: (name) => during(() => m.apps!.activate(name)) } : null,
      controls: { run: (c) => during(() => m.controls.run(c)), ...(m.controls.notify ? { notify: (t: string, x: string) => m.controls.notify!(t, x) } : {}) },
    }
  }
}

/** The controller's action as the rules read it. */
export function flatten(a: AuthorizeAction): policy.Action {
  switch (a.kind) {
    case 'click':
      return { kind: 'click', ref: a.ref, role: a.role, label: a.label }
    case 'type':
      return { kind: 'type', ref: a.ref, role: a.role, label: a.label, text: a.text }
    case 'press':
      return { kind: 'press', key: a.key, ...(a.field ? { field_role: a.field.role, field_label: a.field.label } : {}) }
    case 'scroll':
      return { kind: 'scroll', direction: a.direction }
    case 'surface':
      return { kind: 'surface', what: a.what, name: a.name }
  }
}

/** An action for the audit log: its kind and target, typed text as length and digest only. */
function actionLog(a: policy.Action) {
  return { kind: a.kind, role: a.role ?? '', label: clip(a.label ?? ''), key: a.key ?? '', name: clip(a.name ?? ''), ...(a.text ? { text: textDigest(a.text) } : {}) }
}

/** The card's sentence: what exactly would happen, where. */
export function describeAction(b: Binding): string {
  const a = b.action
  const where = b.app || 'this computer'
  switch (a.kind) {
    case 'type':
      return `type “${a.text ?? ''}” into ${a.label || 'a text box'} in ${where}`
    case 'press':
      return `press ${a.key ?? ''} in ${a.field_label || 'the focused field'} (${where})`
    case 'click':
      return `click “${a.label || a.role || ''}” in ${where}`
    case 'control':
      return b.goal
    default:
      return `${a.kind} in ${where}`
  }
}

/** A fixed control in words ("open Safari", "set the volume to 30"). */
export function describeControl(c: ComputerControl): string {
  switch (c.kind) {
    case 'open_app':
      return `open ${c.app}`
    case 'quit_app':
      return `quit ${c.app}`
    case 'volume':
      return c.action === 'set' ? `set the volume to ${c.level ?? ''}` : `volume ${c.action}`
    case 'media':
      return `${c.action} on ${c.player}`
    case 'dark_mode':
      return `dark mode ${c.action}`
    case 'lock_screen':
      return 'lock the screen'
  }
}
