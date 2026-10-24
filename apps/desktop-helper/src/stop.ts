import { chmod, rm } from 'node:fs/promises'
import { createConnection, createServer } from 'node:net'
import type { Server } from 'node:net'
import type { TaskWatch } from './computer/controller.ts'
import type { Point, PointerReading } from './computer/types.ts'
import { silentLogger } from './log.ts'
import type { Logger } from './log.ts'

/**
 * The local stop: the person at the Mac can always end a task without the box. While a task runs the
 * pointer is read every 250 ms and before each action. Moving it into the top-left corner of the main
 * display stops the task; so does a deliberate move away from where the helper last left it, or a click
 * (the person took over). Movement the helper makes itself, and movement while it waits for the box (the
 * person may tap Approve on this Mac), is not a takeover; the corner always is.
 *
 * Deliberate means: more than TAKEOVER_POINTS away, on two readings in a row (one stray reading is not a
 * move), counting only steps larger than JITTER_POINTS between readings. A hand resting on a trackpad, or
 * a mouse drifting a point or two, moves the baseline along with it instead of adding up to a takeover.
 */
export const CORNER_POINTS = 4
export const TAKEOVER_POINTS = 24
/** Steps between two readings up to this are jitter or drift: the baseline follows them. */
export const JITTER_POINTS = 6
/** Readings in a row beyond TAKEOVER_POINTS before a poll stops the task (the check before an action needs one). */
export const SUSTAINED_READINGS = 2
/** A click this long after the helper last acted is the person's (the probe and the poll run late by up to this). */
export const CLICK_MARGIN_MS = 400
export const POLL_MS = 250

export const STOPPED_IN_CORNER = 'You stopped it on the Mac.'
export const TAKEN_OVER = 'You moved the pointer, so it stopped and left the rest to you.'

/** Runs `tick` every `ms` until the returned function is called. */
export type Schedule = (tick: () => void, ms: number) => () => void

const every: Schedule = (tick, ms) => {
  const timer = setInterval(tick, ms)
  timer.unref()
  return () => clearInterval(timer)
}

export type LocalStopOptions = {
  /** Where the pointer is now, in global top-left-origin points. */
  pointer: () => Promise<PointerReading>
  schedule?: Schedule
  now?: () => number
  intervalMs?: number
  log?: Logger
}

export class LocalStop {
  private readonly options: LocalStopOptions

  constructor(options: LocalStopOptions) {
    this.options = options
  }

  /** Starts watching for one task; `abort` is called once, with the reason, when the person stops it. */
  watch(abort: (reason: string) => void): TaskWatch {
    return new PointerTaskWatch(this.options, abort)
  }
}

class PointerTaskWatch implements TaskWatch {
  private readonly pointer: () => Promise<PointerReading>
  private readonly now: () => number
  private readonly abort: (reason: string) => void
  private readonly log: Logger
  private readonly cancel: () => void
  /** Where the helper left the pointer (moved along by jitter and drift); null until the next reading sets it. */
  private expected: Point | null = null
  /** The previous reading, for the step since it. */
  private last: Point | null = null
  /** Readings in a row beyond TAKEOVER_POINTS. */
  private beyond = 0
  /** When the helper last finished touching the pointer: a click after this is the person's. */
  private leftAt: number
  private helperMoving = false
  /** Changes whenever the helper moves the pointer, so a reading taken before that is not judged after it. */
  private generation = 0
  private polling = false
  private ended = false
  private warned = false

  constructor(options: LocalStopOptions, abort: (reason: string) => void) {
    this.pointer = options.pointer
    this.now = options.now ?? Date.now
    this.leftAt = this.now()
    this.abort = abort
    this.log = options.log ?? silentLogger
    this.cancel = (options.schedule ?? every)(() => void this.poll(), options.intervalMs ?? POLL_MS)
  }

  async check(): Promise<void> {
    if (this.ended) return
    const generation = this.generation
    const point = await this.read()
    if (point) this.judge(point, generation, 1)
  }

  moving(): void {
    // A move already under way when the helper is about to act or wait is the person's: it is not forgotten.
    if (this.beyond > 0 && !this.ended && !this.helperMoving) return this.stop(TAKEN_OVER)
    this.helperMoving = true
    this.generation += 1
    this.beyond = 0
  }

  left(point?: Point | null): void {
    this.helperMoving = false
    this.generation += 1
    this.beyond = 0
    this.leftAt = this.now()
    if (point === null) this.expected = this.last = null
    else if (point) this.expected = this.last = { x: point.x, y: point.y }
  }

  end(): void {
    if (this.ended) return
    this.ended = true
    this.cancel()
  }

  private async poll() {
    if (this.polling || this.ended) return
    this.polling = true
    try {
      if (this.ended) return
      const generation = this.generation
      const point = await this.read()
      if (point) this.judge(point, generation, SUSTAINED_READINGS)
    } finally {
      this.polling = false
    }
  }

  private async read(): Promise<PointerReading | null> {
    try {
      return await this.pointer()
    } catch (error) {
      if (!this.warned) this.log.warn('local_stop.pointer_unreadable', { error: error instanceof Error ? error.message : String(error) })
      this.warned = true
      return null
    }
  }

  /** `needed`: readings in a row beyond the takeover distance before it stops (polls 2, the check before an action 1). */
  private judge(point: PointerReading, generation: number, needed: number) {
    if (this.ended) return
    if (Math.hypot(point.x, point.y) <= CORNER_POINTS) return this.stop(STOPPED_IN_CORNER)
    if (this.helperMoving || generation !== this.generation) return
    // A click since the helper last acted (its own clicks were before that) is the person taking over.
    if (point.sinceClickS !== undefined && this.now() - point.sinceClickS * 1000 > this.leftAt + CLICK_MARGIN_MS) return this.stop(TAKEN_OVER)
    const here = { x: point.x, y: point.y }
    if (!this.expected || !this.last) {
      this.expected = this.last = here
      this.beyond = 0
      return
    }
    // Jitter and drift: the baseline follows small steps, so they never add up to a move.
    if (Math.hypot(here.x - this.last.x, here.y - this.last.y) <= JITTER_POINTS) {
      this.expected = { x: this.expected.x + here.x - this.last.x, y: this.expected.y + here.y - this.last.y }
    }
    this.last = here
    if (Math.hypot(here.x - this.expected.x, here.y - this.expected.y) > TAKEOVER_POINTS) {
      this.beyond += 1
      if (this.beyond >= needed) this.stop(TAKEN_OVER)
    } else this.beyond = 0
  }

  private stop(reason: string) {
    this.end()
    this.log.info('local_stop.stopped', { reason })
    this.abort(reason)
  }
}

export const CONTROL_COMMANDS = ['stop', 'disable', 'enable', 'status'] as const
export type ControlCommand = (typeof CONTROL_COMMANDS)[number]
export type ControlReply = { ok: boolean; message: string }

const isCommand = (value: string): value is ControlCommand => (CONTROL_COMMANDS as readonly string[]).includes(value)

/**
 * The helper's local control socket (`<config dir>/control.sock`, mode 0600, in a 0700 directory):
 * one command a connection (`stop`, `disable`, `enable`, `status`, a line each), answered with one
 * JSON line `{ok, message}`. Only the user who runs the helper can reach it.
 */
export class ControlSocket {
  private readonly path: string
  private readonly handle: (command: ControlCommand) => Promise<ControlReply> | ControlReply
  private readonly log: Logger
  private server: Server | null = null

  constructor(path: string, handle: (command: ControlCommand) => Promise<ControlReply> | ControlReply, log: Logger = silentLogger) {
    this.path = path
    this.handle = handle
    this.log = log
  }

  async listen(): Promise<void> {
    if (await reachable(this.path)) throw new Error(`LocalRouter Desktop is already running (${this.path} answers).`)
    // A socket file left by a helper that did not exit cleanly.
    await rm(this.path, { force: true })
    const server = createServer((socket) => {
      let input = ''
      let answered = false
      socket.setTimeout(5_000, () => socket.destroy())
      socket.on('error', () => undefined)
      socket.on('data', (chunk: Buffer | string) => {
        if (answered) return
        input += String(chunk)
        if (input.length > 256) {
          answered = true
          socket.end(`${JSON.stringify({ ok: false, message: 'Command too long' })}\n`)
          return
        }
        const newline = input.indexOf('\n')
        if (newline < 0) return
        answered = true
        const word = input.slice(0, newline).trim().toLowerCase()
        void this.answer(word).then((reply) => socket.end(`${JSON.stringify(reply)}\n`))
      })
    })
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(this.path, () => {
        server.off('error', reject)
        resolve()
      })
    })
    await chmod(this.path, 0o600)
    this.server = server
  }

  async close(): Promise<void> {
    const server = this.server
    this.server = null
    if (server) await new Promise<void>((resolve) => server.close(() => resolve()))
    await rm(this.path, { force: true })
  }

  private async answer(word: string): Promise<ControlReply> {
    if (!isCommand(word)) return { ok: false, message: `Unknown command: use ${CONTROL_COMMANDS.join(', ')}` }
    try {
      this.log.info('control_socket.command', { command: word })
      return await this.handle(word)
    } catch (error) {
      return { ok: false, message: error instanceof Error ? error.message : String(error) }
    }
  }
}

/** Sends one command to a running helper. Rejects with code ENOENT or ECONNREFUSED when none is running. */
export function sendControl(path: string, command: ControlCommand, timeoutMs = 5_000): Promise<ControlReply> {
  return new Promise((resolve, reject) => {
    const socket = createConnection(path)
    let output = ''
    socket.setTimeout(timeoutMs, () => {
      socket.destroy()
      reject(new Error('LocalRouter Desktop did not answer.'))
    })
    socket.on('connect', () => socket.write(`${command}\n`))
    socket.on('data', (chunk: Buffer | string) => {
      output += String(chunk)
    })
    socket.on('error', reject)
    socket.on('end', () => {
      try {
        const reply: unknown = JSON.parse(output.trim())
        if (typeof reply === 'object' && reply !== null && 'ok' in reply && 'message' in reply) {
          resolve({ ok: reply.ok === true, message: String(reply.message) })
        } else reject(new Error('LocalRouter Desktop sent an unreadable answer.'))
      } catch {
        reject(new Error('LocalRouter Desktop sent an unreadable answer.'))
      }
    })
  })
}

/** Whether a helper answers on the socket now. */
function reachable(path: string): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = createConnection(path)
    socket.once('connect', () => {
      socket.destroy()
      resolve(true)
    })
    socket.once('error', () => resolve(false))
  })
}

/** Whether an error from sendControl means no helper is running. */
export function notRunning(error: unknown): boolean {
  const code = typeof error === 'object' && error !== null && 'code' in error ? error.code : undefined
  return code === 'ENOENT' || code === 'ECONNREFUSED'
}
