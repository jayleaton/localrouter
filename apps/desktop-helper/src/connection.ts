import { DISABLED_MESSAGE, desktopSocketUrl } from './config.ts'
import type { AuthorizeRequest, ComputerUseRuns, Grant, Session } from './computer/controller.ts'
import { parseAnswers } from './computer/decisions.ts'
import type { Answers, Question } from './computer/decisions.ts'
import { CONTROL_KINDS, parseControl } from './computer/macControls.ts'
import type { ComputerControl } from './computer/macControls.ts'
import { silentLogger } from './log.ts'
import type { Logger } from './log.ts'
import { json, nullable, number, object, oneOf, optional, string } from './validate.ts'
import type { Fields } from './validate.ts'

/**
 * The helper's one connection to the box: a WebSocket to `<url>/v1/desktop` that the helper dials
 * (protocol version 1 in docs/dev/COMPUTER.md). It says hello, waits for welcome, pings every 20 s,
 * and treats 60 s of silence as a drop. A drop rejects every pending request and stops the running
 * task, and nothing is replayed after reconnecting (with backoff from 1 s to 60 s, with jitter).
 * The box asks for `control`, `use.start` and `stop`; during a task the helper asks the box
 * `decide` and `authorize`, and reports `progress`, then `finished` exactly once.
 */
export const PROTOCOL = 1
export const PING_MS = 20_000
export const SILENCE_MS = 60_000
export const TICK_MS = 5_000
export const DECIDE_TIMEOUT_MS = 30_000
/** How much longer than the box's approval timeout the helper waits for an `authorize` reply. */
export const AUTHORIZE_GRACE_MS = 15_000
export const MIN_BACKOFF_MS = 1_000
export const MAX_BACKOFF_MS = 60_000

/** What the transport tells the connection. `close` comes once, after an error too. */
export type SocketEvents = { open(): void; message(text: string): void; close(reason: string): void }

/** One WebSocket, as small as the connection needs it, so tests can use a fake. */
export interface Socket {
  send(text: string): void
  close(): void
}

/** Opens a socket to `url` with `Authorization: Bearer <token>` on the upgrade. */
export type Dial = (url: string, token: string, events: SocketEvents) => Socket

/** Node's global WebSocket (undici), which accepts a non-standard `{ headers }` second argument. */
type HeaderedWebSocket = {
  send(data: string): void
  close(): void
  onopen: (() => void) | null
  onmessage: ((event: { data: unknown }) => void) | null
  onerror: ((event: unknown) => void) | null
  onclose: ((event: { code: number; reason: string }) => void) | null
}
type HeaderedWebSocketClass = new (url: string, init: { headers: Record<string, string> }) => HeaderedWebSocket

export const nodeDial: Dial = (url, token, events) => {
  const WebSocketClass = (globalThis as unknown as { WebSocket: HeaderedWebSocketClass }).WebSocket
  const socket = new WebSocketClass(url, { headers: { Authorization: `Bearer ${token}` } })
  let closed = false
  socket.onopen = () => events.open()
  socket.onmessage = (event) => {
    if (typeof event.data === 'string') events.message(event.data)
  }
  // An error is always followed by close, which reports it.
  socket.onerror = () => undefined
  socket.onclose = (event) => {
    if (closed) return
    closed = true
    events.close(event.reason ? `${event.code} ${event.reason}` : String(event.code))
  }
  return {
    send: (text) => socket.send(text),
    close: () => {
      try {
        socket.close()
      } catch {
        // Already closed.
      }
    },
  }
}

export type HelperInfo = { name: string; platform: string; version: string }
export type HelloAccess = { accessibility: boolean; screen_recording: boolean }
export type BoxLimits = { max_steps: number; timeout_ms: number; approval_timeout_ms: number }

export type ConnectionOptions = {
  /** The box's base address (https, or http to loopback). */
  url: string
  token: string
  helper: HelperInfo
  /** This Mac's permissions, read again for each hello. */
  access: () => Promise<HelloAccess>
  controls: { run(control: ComputerControl): Promise<{ done: string }> }
  runs: ComputerUseRuns
  /** Whether the person switched computer control off on this Mac (`localrouter-desktop disable`). */
  disabled: () => boolean
  /** Each time the box welcomes the helper (at connect and after every reconnect). */
  onWelcome?: () => void
  dial?: Dial
  random?: () => number
  log?: Logger
}

type Pending = { resolve: (frame: Fields) => void; reject: (error: Error) => void; timer: ReturnType<typeof setTimeout> }

const DEFAULT_LIMITS: BoxLimits = { max_steps: 15, timeout_ms: 90_000, approval_timeout_ms: 60_000 }
const LOST = 'The connection to the box was lost, so it stopped.'
const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error))

export class Connection {
  private readonly options: ConnectionOptions
  private readonly dial: Dial
  private readonly random: () => number
  private readonly log: Logger
  private readonly endpoint: string
  private socket: Socket | null = null
  /** Changes with every socket, so callbacks and replies from an old one are ignored. */
  private generation = 0
  private ready = false
  private closed = false
  private attempts = 0
  private seq = 0
  private lastHeard = 0
  private lastPing = 0
  private ticker: ReturnType<typeof setInterval> | null = null
  private retry: ReturnType<typeof setTimeout> | null = null
  private readonly pending = new Map<string, Pending>()
  private helperId: string | null = null
  private limits: BoxLimits = DEFAULT_LIMITS

  constructor(options: ConnectionOptions) {
    this.options = options
    this.dial = options.dial ?? nodeDial
    this.random = options.random ?? Math.random
    this.log = options.log ?? silentLogger
    this.endpoint = desktopSocketUrl(options.url)
  }

  /** Connected and welcomed, with the box's id for this helper. */
  get status() {
    return { connected: this.ready, helperId: this.helperId, task: this.options.runs.current }
  }

  start() {
    this.closed = false
    this.connect()
  }

  /** Closes for good: no reconnect, and the running task stops. */
  close() {
    this.closed = true
    if (this.retry) clearTimeout(this.retry)
    this.retry = null
    this.dropped('The helper is shutting down.')
  }

  private connect() {
    if (this.closed) return
    const generation = ++this.generation
    this.ready = false
    this.lastHeard = Date.now()
    this.lastPing = Date.now()
    this.log.info('connection.dialing', { url: this.endpoint })
    try {
      this.socket = this.dial(this.endpoint, this.options.token, {
        open: () => {
          if (generation === this.generation) void this.hello(generation)
        },
        message: (text) => {
          if (generation === this.generation) this.receive(text)
        },
        close: (reason) => {
          if (generation === this.generation) this.dropped(`closed (${reason})`)
        },
      })
    } catch (error) {
      this.socket = null
      this.dropped(`could not dial: ${messageOf(error)}`)
      return
    }
    this.ticker = setInterval(() => this.tick(), TICK_MS)
  }

  private async hello(generation: number) {
    const access = await this.options.access().catch((): HelloAccess => ({ accessibility: false, screen_recording: false }))
    if (generation !== this.generation) return
    this.send({
      type: 'hello',
      protocol: PROTOCOL,
      helper: this.options.helper,
      capabilities: { controls: [...CONTROL_KINDS], computer_use: true },
      access,
    })
  }

  private tick() {
    const now = Date.now()
    if (now - this.lastHeard > SILENCE_MS) {
      this.dropped('the box was silent for 60 s')
      return
    }
    if (this.ready && now - this.lastPing >= PING_MS) {
      this.lastPing = now
      this.send({ type: 'ping' })
    }
  }

  /**
   * The socket is gone: every pending request fails, the running task stops (its `finished` cannot be
   * sent, and is not sent later), and a new socket is dialed after a backoff.
   */
  private dropped(reason: string) {
    const wasOpen = this.socket !== null
    this.generation += 1
    this.ready = false
    if (this.ticker) clearInterval(this.ticker)
    this.ticker = null
    const socket = this.socket
    this.socket = null
    socket?.close()
    for (const [id, pending] of this.pending) {
      clearTimeout(pending.timer)
      pending.reject(new Error('The connection to the box was lost.'))
      this.pending.delete(id)
    }
    this.options.runs.stop(undefined, LOST)
    if (wasOpen) this.log.warn('connection.dropped', { reason })
    if (this.closed) return
    const base = Math.min(MAX_BACKOFF_MS, MIN_BACKOFF_MS * 2 ** this.attempts)
    const delay = Math.min(MAX_BACKOFF_MS, Math.round(base + this.random() * base * 0.25))
    this.attempts += 1
    if (this.retry) clearTimeout(this.retry)
    this.retry = setTimeout(() => {
      this.retry = null
      this.connect()
    }, delay)
  }

  private send(frame: Fields) {
    if (!this.socket) return
    try {
      this.socket.send(JSON.stringify(frame))
    } catch (error) {
      this.log.warn('connection.send_failed', { type: frame.type, error: messageOf(error) })
    }
  }

  /** Sends only on the socket a task started on: after a drop, nothing of that task is sent again. */
  private sendOn(generation: number, frame: Fields) {
    if (generation === this.generation && this.ready) this.send(frame)
  }

  private reply(id: string, fields: Fields) {
    this.send({ type: 'result', id, ...fields })
  }

  private receive(text: string) {
    this.lastHeard = Date.now()
    let frame: Fields
    try {
      frame = object(json(text, 'frame'), 'frame')
    } catch (error) {
      this.log.warn('connection.bad_frame', { error: messageOf(error) })
      return
    }
    const type = frame.type
    if (type === 'ping') return this.send({ type: 'pong' })
    if (type === 'pong') return
    if (type === 'error') {
      this.log.error('connection.box_error', { error: typeof frame.error === 'string' ? frame.error : frame.message })
      return
    }
    if (!this.ready) {
      if (type === 'welcome') return this.welcome(frame)
      this.log.warn('connection.before_welcome', { type })
      return
    }
    switch (type) {
      case 'result':
        return this.settle(frame)
      case 'control':
        return void this.onControl(frame)
      case 'use.start':
        return this.onStart(frame)
      case 'stop':
        return this.onStop(frame)
      default:
        if (typeof frame.id === 'string') this.reply(frame.id, { ok: false, error: `Unknown request: ${String(type)}` })
        else this.log.warn('connection.unknown_frame', { type })
    }
  }

  private welcome(frame: Fields) {
    try {
      this.helperId = string(frame.helper_id, 'welcome.helper_id')
      const limits: Fields = optional(frame.limits, 'welcome.limits', object) ?? {}
      this.limits = {
        max_steps: optional(limits.max_steps, 'welcome.limits.max_steps', number) ?? DEFAULT_LIMITS.max_steps,
        timeout_ms: optional(limits.timeout_ms, 'welcome.limits.timeout_ms', number) ?? DEFAULT_LIMITS.timeout_ms,
        approval_timeout_ms: optional(limits.approval_timeout_ms, 'welcome.limits.approval_timeout_ms', number) ?? DEFAULT_LIMITS.approval_timeout_ms,
      }
    } catch (error) {
      this.log.error('connection.bad_welcome', { error: messageOf(error) })
      this.dropped('the welcome was malformed')
      return
    }
    this.ready = true
    this.attempts = 0
    this.log.info('connection.welcome', { helper_id: this.helperId })
    this.options.onWelcome?.()
  }

  private settle(frame: Fields) {
    const id = typeof frame.id === 'string' ? frame.id : ''
    const pending = this.pending.get(id)
    if (!pending) return this.log.warn('connection.unexpected_result', { id })
    this.pending.delete(id)
    clearTimeout(pending.timer)
    pending.resolve(frame)
  }

  /** One request to the box; resolves with its result frame (ok or not), rejects on timeout or a drop. */
  private request(type: string, fields: Fields, timeoutMs: number): Promise<Fields> {
    if (!this.ready || !this.socket) return Promise.reject(new Error('Not connected to the box.'))
    const id = `h${++this.seq}`
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`The box did not answer ${type} in time.`))
      }, timeoutMs)
      this.pending.set(id, { resolve, reject, timer })
      this.send({ type, id, ...fields })
    })
  }

  private async decide(taskId: string, state: unknown, questions: Record<string, Question>): Promise<Answers> {
    const result = await this.request('decide', { task_id: taskId, state, questions }, DECIDE_TIMEOUT_MS)
    if (result.ok !== true) throw new Error(typeof result.error === 'string' ? result.error : 'The box refused to decide.')
    return parseAnswers(result.answers)
  }

  private async authorize(taskId: string, request: AuthorizeRequest): Promise<Grant> {
    const result = await this.request('authorize', { task_id: taskId, step: request.step, observation: request.observation, action: request.action }, this.limits.approval_timeout_ms + AUTHORIZE_GRACE_MS)
    if (result.ok === true) return { ok: true, grant: oneOf(result.grant, ['allow', 'approved'] as const, 'authorize grant') }
    if (typeof result.denied === 'string') return { ok: false, denied: result.denied }
    throw new Error(typeof result.error === 'string' ? result.error : 'The box refused the request.')
  }

  private async onControl(frame: Fields) {
    if (typeof frame.id !== 'string') return this.log.warn('connection.request_without_id', { type: 'control' })
    const id = frame.id
    if (this.options.disabled()) return this.reply(id, { ok: false, error: DISABLED_MESSAGE })
    let control: ComputerControl
    try {
      string(frame.task_id, 'task_id')
      control = parseControl(frame.control)
    } catch (error) {
      return this.reply(id, { ok: false, error: messageOf(error) })
    }
    const generation = this.generation
    try {
      const { done } = await this.options.controls.run(control)
      if (generation === this.generation) this.reply(id, { ok: true, done })
    } catch (error) {
      if (generation === this.generation) this.reply(id, { ok: false, error: messageOf(error) })
    }
  }

  private onStart(frame: Fields) {
    if (typeof frame.id !== 'string') return this.log.warn('connection.request_without_id', { type: 'use.start' })
    const id = frame.id
    if (this.options.disabled()) return this.reply(id, { ok: false, error: DISABLED_MESSAGE })
    let taskId: string
    let goal: string
    let text: string | null
    let limits: Fields
    try {
      taskId = string(frame.task_id, 'task_id')
      goal = string(frame.goal, 'goal').trim()
      if (!goal) throw new Error('goal should not be empty')
      text = nullable(frame.text ?? null, 'text', string)
      limits = frame.limits === undefined || frame.limits === null ? {} : object(frame.limits, 'limits')
    } catch (error) {
      return this.reply(id, { ok: false, error: messageOf(error) })
    }
    const generation = this.generation
    const session: Session = {
      decider: { ask: (state, questions) => this.decide(taskId, state, questions) },
      authorize: (request) => this.authorize(taskId, request),
      onStep: (step) => this.sendOn(generation, { type: 'progress', task_id: taskId, step }),
    }
    const maxSteps = isNumber(limits.max_steps) ? limits.max_steps : this.limits.max_steps
    const timeoutMs = isNumber(limits.timeout_ms) ? limits.timeout_ms : this.limits.timeout_ms
    const started = this.options.runs.start(taskId, { goal, text }, session, { maxSteps, timeoutMs })
    if (!started.ok) return this.reply(id, { ok: false, error: started.error })
    this.reply(id, { ok: true })
    this.log.info('connection.task_started', { task_id: taskId })
    let sent = false
    void started.finished.then((outcome) => {
      this.log.info('connection.task_finished', { task_id: taskId, status: outcome.status })
      if (sent) return
      sent = true
      this.sendOn(generation, { type: 'finished', task_id: taskId, outcome })
    })
  }

  private onStop(frame: Fields) {
    if (typeof frame.id !== 'string') return this.log.warn('connection.request_without_id', { type: 'stop' })
    const taskId = typeof frame.task_id === 'string' ? frame.task_id : ''
    const stopped = this.options.runs.stop(taskId, 'Stopped from LocalRouter.')
    this.reply(frame.id, stopped ? { ok: true } : { ok: false, error: 'That task is not running on this Mac.' })
  }
}

const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
