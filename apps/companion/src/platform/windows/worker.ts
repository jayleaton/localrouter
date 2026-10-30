import { spawn as spawnChild } from 'node:child_process'
import { join } from 'node:path'
import { silentLogger } from '../../../../desktop-helper/src/log.ts'
import type { Logger } from '../../../../desktop-helper/src/log.ts'
import { boolean, isObject, number, string } from '../../../../desktop-helper/src/validate.ts'
import { bootstrapLine } from './bootstrap.ts'

/** What the worker needs of a child process: tests hand in an EventEmitter-based fake. */
export type WorkerChild = {
  readonly pid?: number
  stdin: { write(chunk: string): unknown; end(): unknown }
  stdout: { on(event: 'data', listener: (chunk: Buffer | string) => void): unknown }
  stderr: { on(event: 'data', listener: (chunk: Buffer | string) => void): unknown }
  on(event: 'exit', listener: (code: number | null, signal: NodeJS.Signals | null) => void): unknown
  on(event: 'error', listener: (error: Error) => void): unknown
  kill(): unknown
}

export type WorkerSpawn = (file: string, args: string[]) => WorkerChild

export type CallOptions = { timeoutMs?: number; signal?: AbortSignal }

/** One request to the Windows worker; what the screen, controls and presence code depend on. */
export interface WindowsCall {
  call(op: string, args?: Record<string, unknown>, options?: CallOptions): Promise<unknown>
}

/**
 * Windows PowerShell 5.1 by its full path, so a `powershell.exe` planted in the working directory or early on PATH is
 * never the one started. It is on every Windows 10 and 11; PowerShell 7 (pwsh) is not required.
 */
export function powershellPath(env: NodeJS.ProcessEnv = process.env): string {
  return join(env.SystemRoot ?? env.windir ?? 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
}

export const POWERSHELL_ARGS = ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', '-']

const defaultSpawn: WorkerSpawn = (file, args) => spawnChild(file, args, { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true })

export type WorkerOptions = {
  spawn?: WorkerSpawn
  /** The PowerShell executable; by default Windows PowerShell's full path under `env.SystemRoot`. */
  file?: string
  env?: NodeJS.ProcessEnv
  /** The first line sent (the bootstrap); tests send their own. */
  bootstrap?: () => string
  /** How long compiling the helper may take at start (Add-Type runs csc: a few seconds on a cold machine). */
  startTimeoutMs?: number
  /** The default per-request timeout. */
  timeoutMs?: number
  /** What the person reads when a request times out ("Windows took too long to answer"). */
  slow?: string
  log?: Logger
}

/** Replies longer than this (a screenshot is a few MB of base64) mean something is wrong: the worker is restarted. */
const MAX_LINE = 64 * 1024 * 1024

type Job = {
  op: string
  args: Record<string, unknown>
  timeoutMs: number
  signal?: AbortSignal
  resolve: (value: unknown) => void
  reject: (error: Error) => void
  id?: number
  timer?: ReturnType<typeof setTimeout>
  onAbort?: () => void
}

type Live = {
  child: WorkerChild
  ready: Promise<void>
  buffer: string
  stderr: string
  exited: boolean
  onReady: (error?: Error) => void
}

const abortMessage = (signal: AbortSignal) => (typeof signal.reason === 'string' && signal.reason ? signal.reason : 'It was stopped.')

/**
 * JSON with every character outside printable ASCII escaped. Windows PowerShell reads redirected stdin in the console's
 * code page, so anything else could reach the helper changed; `\uXXXX` arrives intact whatever that page is.
 */
export function asciiJson(value: unknown): string {
  return JSON.stringify(value).replace(/[\u007f-\uffff]/g, (c) => `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`)
}

/**
 * A persistent Windows PowerShell worker speaking JSON lines: one request `{id, op, args}` a line on stdin, one reply
 * `{id, ok, result | error}` a line on stdout. Requests run one at a time, in order. A request that times out or is
 * aborted kills the process (a UI Automation call can hang inside another app, and a stop must stop now); so does a
 * crash. Either way the next request starts a fresh one, so one bad moment never leaves the worker wedged.
 */
export class PowerShellWorker implements WindowsCall {
  private readonly spawn: WorkerSpawn
  private readonly file: string
  private readonly bootstrap: () => string
  private readonly startTimeoutMs: number
  private readonly timeoutMs: number
  private readonly slow: string
  private readonly log: Logger
  private readonly queue: Job[] = []
  private live: Live | null = null
  private running: Job | null = null
  private nextId = 1
  private closed = false
  private pumping = false

  constructor(options: WorkerOptions = {}) {
    this.spawn = options.spawn ?? defaultSpawn
    this.file = options.file ?? powershellPath(options.env)
    this.bootstrap = options.bootstrap ?? (() => bootstrapLine())
    this.startTimeoutMs = options.startTimeoutMs ?? 60_000
    this.timeoutMs = options.timeoutMs ?? 15_000
    this.slow = options.slow ?? 'Windows'
    this.log = options.log ?? silentLogger
  }

  /** The running process's id, when one runs. */
  get pid(): number | undefined {
    return this.live?.child.pid
  }

  call(op: string, args: Record<string, unknown> = {}, options: CallOptions = {}): Promise<unknown> {
    if (this.closed) return Promise.reject(new Error('The Windows helper was shut down.'))
    const signal = options.signal
    if (signal?.aborted) return Promise.reject(new Error(abortMessage(signal)))
    return new Promise((resolve, reject) => {
      const job: Job = { op, args, timeoutMs: options.timeoutMs ?? this.timeoutMs, signal, resolve, reject }
      if (signal) {
        job.onAbort = () => this.abort(job)
        signal.addEventListener('abort', job.onAbort, { once: true })
      }
      this.queue.push(job)
      void this.pump()
    })
  }

  /** Ends the process; pending requests fail. */
  close(): void {
    this.closed = true
    for (const job of this.queue.splice(0)) this.settle(job, new Error('The Windows helper was shut down.'))
    if (this.running) this.settle(this.running, new Error('The Windows helper was shut down.'))
    this.running = null
    this.stop()
  }

  private async pump() {
    if (this.pumping) return
    this.pumping = true
    try {
      while (!this.running && this.queue.length && !this.closed) {
        const job = this.queue.shift()!
        this.running = job
        let live: Live
        try {
          live = await this.start()
        } catch (error) {
          if (this.running === job) {
            this.running = null
            this.settle(job, error instanceof Error ? error : new Error(String(error)))
          }
          continue
        }
        // Aborted (or shut down) while the process started.
        if (this.running !== job) continue
        job.id = this.nextId++
        job.timer = setTimeout(() => this.timeout(job), job.timeoutMs)
        live.child.stdin.write(`${asciiJson({ id: job.id, op: job.op, args: job.args })}\n`)
        return
      }
    } finally {
      this.pumping = false
    }
  }

  /** The running process, started (and its helper compiled) when there is none. */
  private start(): Promise<Live> {
    if (this.live && !this.live.exited) return this.live.ready.then(() => this.live!)
    let onReady!: (error?: Error) => void
    const ready = new Promise<void>((resolve, reject) => {
      onReady = (error) => (error ? reject(error) : resolve())
    })
    let child: WorkerChild
    try {
      child = this.spawn(this.file, POWERSHELL_ARGS)
    } catch (error) {
      return Promise.reject(new Error(`Windows PowerShell could not be started: ${error instanceof Error ? error.message : String(error)}`))
    }
    const live: Live = { child, ready, buffer: '', stderr: '', exited: false, onReady }
    this.live = live
    const timer = setTimeout(() => {
      this.log.warn('windows_worker.start_timeout', { ms: this.startTimeoutMs })
      live.onReady(new Error(`${this.slow} took too long to start its helper (timed out).`))
      this.kill(live)
    }, this.startTimeoutMs)
    ready.then(() => clearTimeout(timer), () => clearTimeout(timer))
    child.stdout.on('data', (chunk) => this.data(live, chunk))
    child.stderr.on('data', (chunk) => {
      live.stderr = (live.stderr + String(chunk)).slice(-2000)
    })
    child.on('error', (error) => this.exit(live, `Windows PowerShell could not be started: ${error.message}`))
    child.on('exit', (code) => this.exit(live, `The Windows helper stopped (exit code ${code ?? 'none'})${live.stderr.trim() ? `: ${live.stderr.trim().slice(-400)}` : '.'}`))
    child.stdin.write(`${this.bootstrap()}\n`)
    return ready.then(() => live)
  }

  private data(live: Live, chunk: Buffer | string) {
    live.buffer += String(chunk)
    if (live.buffer.length > MAX_LINE) {
      this.log.warn('windows_worker.line_too_long', { bytes: live.buffer.length })
      live.buffer = ''
      this.fail(live, 'The Windows helper sent a reply too large to read.')
      return
    }
    let newline: number
    while ((newline = live.buffer.indexOf('\n')) >= 0) {
      // A console host may write terminal control sequences (cursor-key mode) with no newline of their own.
      // eslint-disable-next-line no-control-regex
      const line = live.buffer.slice(0, newline).replace(/\r$/, '').replace(/\u001b\[[0-9;?]*[A-Za-z]/g, '')
      live.buffer = live.buffer.slice(newline + 1)
      if (line.trim()) this.line(live, line)
    }
  }

  /** One reply line. Anything that is not a reply (a stray line PowerShell printed) is logged and skipped. */
  private line(live: Live, line: string) {
    let reply: { id: number; ok: boolean; result?: unknown; error?: string }
    try {
      const raw: unknown = JSON.parse(line)
      if (!isObject(raw)) throw new Error('not an object')
      const ok = boolean(raw.ok, 'reply ok')
      reply = { id: number(raw.id, 'reply id'), ok, ...(ok ? { result: raw.result } : { error: string(raw.error, 'reply error') }) }
    } catch {
      this.log.warn('windows_worker.malformed_line', { length: line.length })
      return
    }
    if (reply.id === 0) {
      if (reply.ok) live.onReady()
      else {
        live.onReady(new Error(`The Windows helper could not start: ${reply.error}`))
        this.kill(live)
      }
      return
    }
    const job = this.running
    if (!job || job.id !== reply.id || this.live !== live) return
    this.running = null
    this.settle(job, reply.ok ? undefined : new Error(reply.error), reply.result)
    void this.pump()
  }

  private timeout(job: Job) {
    if (this.running !== job) return
    this.log.warn('windows_worker.timeout', { op: job.op, ms: job.timeoutMs })
    this.running = null
    this.settle(job, new Error(`${this.slow} took too long to answer (timed out).`))
    this.stop()
    void this.pump()
  }

  private abort(job: Job) {
    const message = new Error(abortMessage(job.signal!))
    const queued = this.queue.indexOf(job)
    if (queued >= 0) {
      this.queue.splice(queued, 1)
      this.settle(job, message)
      return
    }
    if (this.running !== job) return
    // Whatever the helper is doing for this request (typing, a hung UIA call) ends with the process.
    this.running = null
    this.settle(job, message)
    this.stop()
    void this.pump()
  }

  /** The process exited or could not start: the running request fails, the next one starts a new process. */
  private exit(live: Live, message: string) {
    if (live.exited) return
    live.exited = true
    live.onReady(new Error(message))
    if (this.live !== live) return
    this.live = null
    this.log.warn('windows_worker.exit', { message: message.slice(0, 200) })
    const job = this.running
    if (job && job.id !== undefined) {
      this.running = null
      this.settle(job, new Error(message))
    }
    void this.pump()
  }

  private fail(live: Live, message: string) {
    const job = this.running
    if (job && this.live === live && job.id !== undefined) {
      this.running = null
      this.settle(job, new Error(message))
    }
    this.kill(live)
    void this.pump()
  }

  private stop() {
    if (this.live) this.kill(this.live)
  }

  private kill(live: Live) {
    if (this.live === live) this.live = null
    live.exited = true
    try {
      live.child.stdin.end()
    } catch {
      // already closed
    }
    try {
      live.child.kill()
    } catch {
      // already gone
    }
  }

  private settle(job: Job, error: Error | undefined, value?: unknown) {
    if (job.timer) clearTimeout(job.timer)
    if (job.onAbort) job.signal?.removeEventListener('abort', job.onAbort)
    if (error) job.reject(error)
    else job.resolve(value)
  }
}
