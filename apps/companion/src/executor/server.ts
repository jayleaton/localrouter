import { createHash, randomBytes, timingSafeEqual } from 'node:crypto'
import { chmodSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { createConnection, createServer } from 'node:net'
import type { Server, Socket } from 'node:net'
import { dirname } from 'node:path'
import { parseControl } from '../../../desktop-helper/src/computer/macControls.ts'
import type { Broker, Requester } from './broker.ts'

/**
 * How other programs on this computer reach the executor: a Unix socket in the app's data folder (mode 0600, so only
 * this user), one JSON request a line, one reply a line. Each caller is enrolled by the person in the app first and
 * presents its token; the executor's lease, policy and approvals apply to it like to anyone. The operations are the
 * bounded computer-control ones and two reads of the front window (its accessibility tree, a scaled screenshot) and
 * nothing else: there is no way to answer an approval or change the policy here.
 */
export const OPS = ['hello', 'capabilities', 'control', 'use', 'status', 'stop', 'observe', 'screenshot'] as const
export type Op = (typeof OPS)[number]

export type Enrolled = { id: string; name: string; token_sha256: string; created: number }

const sha = (s: string) => createHash('sha256').update(s, 'utf8').digest('hex')

/** The callers the person allowed to use this computer (harnesses, MCP clients), as hashed tokens on disk. */
export class Enrollment {
  readonly path: string

  constructor(path: string) {
    this.path = path
  }

  list(): Enrolled[] {
    try {
      const doc = JSON.parse(readFileSync(this.path, 'utf8')) as { callers?: Enrolled[] }
      return Array.isArray(doc.callers) ? doc.callers : []
    } catch {
      return []
    }
  }

  /** A new caller; its token is returned once and only its hash kept. */
  add(name: string, now = Date.now()): { caller: Enrolled; token: string } {
    const clean = name.trim().slice(0, 80)
    if (!clean) throw new Error('a caller needs a name')
    const token = `lrx_${randomBytes(24).toString('base64url')}`
    const caller: Enrolled = { id: `caller_${randomBytes(6).toString('hex')}`, name: clean, token_sha256: sha(token), created: now }
    this.save([...this.list(), caller])
    return { caller, token }
  }

  revoke(idOrName: string): number {
    const all = this.list()
    const keep = all.filter((c) => c.id !== idOrName && c.name !== idOrName)
    this.save(keep)
    return all.length - keep.length
  }

  /** The enrolled caller a token belongs to, re-read on every check so a revoke counts at once. */
  check(token: string): Enrolled | null {
    if (!token) return null
    const want = Buffer.from(sha(token), 'hex')
    for (const c of this.list()) {
      const have = Buffer.from(c.token_sha256, 'hex')
      if (have.length === want.length && timingSafeEqual(have, want)) return c
    }
    return null
  }

  private save(callers: Enrolled[]) {
    mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 })
    const tmp = `${this.path}.${process.pid}.tmp`
    writeFileSync(tmp, `${JSON.stringify({ callers }, null, 1)}\n`, { mode: 0o600 })
    renameSync(tmp, this.path)
  }
}

export type Reply = { ok: true; result: unknown } | { ok: false; error: string }

const str = (v: unknown, what: string, max = 4000): string => {
  if (typeof v !== 'string') throw new Error(`${what} should be a string`)
  if (v.length > max) throw new Error(`${what} is too long`)
  return v
}
const waitOf = (v: unknown, dflt: number, max: number) => (typeof v === 'number' && Number.isFinite(v) && v >= 0 ? Math.min(max, v) : dflt)

/** One request from an enrolled caller, answered by the broker. */
export async function serve(broker: Broker, enrollment: Enrollment, request: unknown): Promise<Reply> {
  try {
    if (typeof request !== 'object' || request === null || Array.isArray(request)) throw new Error('a request is a JSON object')
    const r = request as Record<string, unknown>
    const op = r.op
    if (typeof op !== 'string' || !(OPS as readonly string[]).includes(op)) return { ok: false, error: `unknown op ${JSON.stringify(op)}: the executor offers ${OPS.join(', ')}` }
    const caller = enrollment.check(typeof r.token === 'string' ? r.token : '')
    if (!caller) return { ok: false, error: 'this caller is not enrolled on this computer: open LocalRouter › Settings › Harnesses, add it, and pass its token as LOCALROUTER_EXECUTOR_TOKEN' }
    const who: Requester = { caller: { id: caller.id, name: `MCP: ${caller.name}`, kind: 'mcp' } }
    const args = (typeof r.args === 'object' && r.args !== null ? r.args : {}) as Record<string, unknown>
    switch (op as Op) {
      case 'hello':
        return { ok: true, result: { caller: caller.name, ...broker.capabilities } }
      case 'capabilities':
        return { ok: true, result: broker.capabilities }
      case 'control': {
        const done = await broker.control(who, parseControl(args.control))
        return { ok: true, result: done }
      }
      case 'use': {
        const started = broker.startUse(who, str(args.goal, 'goal', 1000), args.text === undefined || args.text === null ? null : str(args.text, 'text'))
        if (!started.ok) return { ok: true, result: { ok: false, error: started.error } }
        const view = await broker.wait(started.task_id, waitOf(args.wait_s, 120, 600) * 1000)
        return { ok: true, result: view }
      }
      case 'status': {
        if (typeof args.task_id === 'string' && args.task_id) {
          const view = await broker.wait(args.task_id, waitOf(args.wait_s, 0, 600) * 1000)
          return view ? { ok: true, result: view } : { ok: false, error: 'no such task' }
        }
        const st = broker.status()
        return { ok: true, result: { computers: st.computers, tasks: st.tasks.slice(0, 10), pending_approvals: st.pending_approvals } }
      }
      case 'stop':
        return { ok: true, result: { stopped: broker.stop(typeof args.task_id === 'string' && args.task_id ? args.task_id : undefined, `MCP: ${caller.name}`) } }
      case 'observe': {
        const seen = await broker.observe(who)
        return seen.ok ? { ok: true, result: seen.observation } : { ok: false, error: seen.error }
      }
      case 'screenshot': {
        const shot = await broker.screenshot(who, typeof args.max_edge === 'number' ? args.max_edge : undefined)
        if (!shot.ok) return { ok: false, error: shot.error }
        const { png, ...meta } = shot.screenshot
        return { ok: true, result: { ...meta, png_base64: Buffer.from(png).toString('base64') } }
      }
    }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}

/** The socket server: one connection per caller, requests answered in order. */
export class ExecutorServer {
  private server: Server | null = null
  readonly path: string
  private readonly broker: Broker
  private readonly enrollment: Enrollment

  constructor(path: string, broker: Broker, enrollment: Enrollment) {
    this.path = path
    this.broker = broker
    this.enrollment = enrollment
  }

  async listen(): Promise<void> {
    // A Windows named pipe is no file: nothing to make room for, shorten or chmod.
    const pipe = this.path.startsWith('\\\\.\\pipe\\')
    if (!pipe) {
      if (Buffer.byteLength(this.path) > 103) throw new Error(`the executor's socket path is longer than a Unix socket allows (${this.path}): set LOCALROUTER_COMPANION_HOME to a shorter folder`)
      mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 })
      rmSync(this.path, { force: true })
    }
    const server = createServer((socket) => this.connection(socket))
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject)
      server.listen(this.path, () => resolve())
    })
    if (!pipe) chmodSync(this.path, 0o600)
    this.server = server
  }

  async close(): Promise<void> {
    const server = this.server
    this.server = null
    if (server) await new Promise<void>((resolve) => server.close(() => resolve()))
    if (!this.path.startsWith('\\\\.\\pipe\\')) rmSync(this.path, { force: true })
  }

  private connection(socket: Socket) {
    let buffer = ''
    let chain = Promise.resolve()
    socket.setEncoding('utf8')
    socket.on('data', (chunk: string) => {
      buffer += chunk
      if (buffer.length > 1 << 20) return socket.destroy()
      let nl: number
      while ((nl = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, nl)
        buffer = buffer.slice(nl + 1)
        if (!line.trim()) continue
        chain = chain.then(async () => {
          let id: unknown = null
          let reply: Reply
          try {
            const req = JSON.parse(line) as Record<string, unknown>
            id = req.id ?? null
            reply = await serve(this.broker, this.enrollment, req)
          } catch {
            reply = { ok: false, error: 'a request is one line of JSON' }
          }
          if (!socket.destroyed) socket.write(`${JSON.stringify({ id, ...reply })}\n`)
        })
      }
    })
    socket.on('error', () => undefined)
  }
}

/**
 * Why the executor's socket did not answer. The app starts its executor at launch, paired or not, so no socket means
 * the app is not running; builds before that started it only after pairing, which is the other case named.
 */
export function notRunning(code: string): string {
  return `LocalRouter is not running on this computer (${code}): open the LocalRouter app. If it is open, it is an older build that starts computer control only once paired with a LocalRouter server: update it, or pair it on its Connect page.`
}

/** A caller's side of the socket: one request, its reply. */
export class ExecutorClient {
  private socket: Socket | null = null
  private buffer = ''
  private seq = 0
  private readonly pending = new Map<number, (reply: Reply) => void>()
  private readonly path: string
  private readonly token: string

  constructor(path: string, token: string) {
    this.path = path
    this.token = token
  }

  async request(op: string, args: Record<string, unknown> = {}): Promise<Reply> {
    const socket = await this.connect()
    const id = ++this.seq
    return new Promise((resolve) => {
      this.pending.set(id, resolve)
      socket.write(`${JSON.stringify({ id, op, token: this.token, args })}\n`)
    })
  }

  close() {
    this.socket?.destroy()
    this.socket = null
  }

  private connect(): Promise<Socket> {
    if (this.socket && !this.socket.destroyed) return Promise.resolve(this.socket)
    return new Promise((resolve, reject) => {
      const socket = createConnection(this.path)
      socket.setEncoding('utf8')
      socket.once('connect', () => {
        this.socket = socket
        resolve(socket)
      })
      socket.once('error', (error) => {
        reject(new Error(notRunning((error as NodeJS.ErrnoException).code ?? error.message)))
      })
      socket.on('close', () => {
        for (const [, done] of this.pending) done({ ok: false, error: 'LocalRouter closed the connection' })
        this.pending.clear()
        if (this.socket === socket) this.socket = null
      })
      socket.on('data', (chunk: string) => {
        this.buffer += chunk
        let nl: number
        while ((nl = this.buffer.indexOf('\n')) >= 0) {
          const line = this.buffer.slice(0, nl)
          this.buffer = this.buffer.slice(nl + 1)
          try {
            const msg = JSON.parse(line) as { id: number } & Reply
            const done = this.pending.get(msg.id)
            this.pending.delete(msg.id)
            done?.(msg.ok ? { ok: true, result: msg.result } : { ok: false, error: msg.error })
          } catch {
            // A malformed line is dropped.
          }
        }
      })
    })
  }
}
