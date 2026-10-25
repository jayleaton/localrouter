import { EventEmitter } from 'node:events'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { asciiJson, POWERSHELL_ARGS, PowerShellWorker, powershellPath } from './worker.ts'
import type { WorkerChild } from './worker.ts'

type Sent = { id: number; op: string; args: Record<string, unknown> }

/** A child process over EventEmitters: records what is written to stdin, and says what a test makes it say. */
class FakeChild extends EventEmitter {
  readonly stdout = new EventEmitter()
  readonly stderr = new EventEmitter()
  readonly lines: string[] = []
  killed = false
  ended = false
  readonly pid: number
  readonly stdin = {
    write: (chunk: string) => {
      this.lines.push(...chunk.split('\n').filter(Boolean))
      this.onWrite?.()
      return true
    },
    end: () => {
      this.ended = true
    },
  }
  onWrite?: () => void

  constructor(pid: number) {
    super()
    this.pid = pid
  }

  get requests(): Sent[] {
    return this.lines.slice(1).map((line) => JSON.parse(line) as Sent)
  }

  say(text: string) {
    this.stdout.emit('data', Buffer.from(text))
  }

  ready() {
    this.say('{"id":0,"ok":true,"result":{"pid":1,"dpi":"per_monitor_v2"}}\n')
  }

  reply(id: number, result: unknown) {
    this.say(`${JSON.stringify({ id, ok: true, result })}\n`)
  }

  kill() {
    this.killed = true
    queueMicrotask(() => this.emit('exit', null, 'SIGTERM'))
    return true
  }
}

function harness(options: { timeoutMs?: number; startTimeoutMs?: number; autoReady?: boolean } = {}) {
  const children: FakeChild[] = []
  const spawned: { file: string; args: string[] }[] = []
  const worker = new PowerShellWorker({
    spawn: (file, args) => {
      spawned.push({ file, args })
      const child = new FakeChild(100 + children.length)
      children.push(child)
      if (options.autoReady !== false) queueMicrotask(() => child.ready())
      return child as unknown as WorkerChild
    },
    file: 'powershell.exe',
    bootstrap: () => 'BOOTSTRAP',
    timeoutMs: options.timeoutMs ?? 1_000,
    startTimeoutMs: options.startTimeoutMs ?? 1_000,
  })
  return { worker, children, spawned }
}

/** Lets queued promise callbacks and microtasks run. */
const settle = () => new Promise((resolve) => setTimeout(resolve, 0))

afterEach(() => {
  vi.useRealTimers()
})

describe('PowerShellWorker', () => {
  it('starts Windows PowerShell by its full path with stdin as the script, sends the bootstrap once, then one request a line', async () => {
    const { worker, children, spawned } = harness()
    const first = worker.call('observe', { budgetMs: 3000 })
    await settle()
    expect(spawned).toEqual([{ file: 'powershell.exe', args: ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', '-'] }])
    const child = children[0]!
    expect(child.lines[0]).toBe('BOOTSTRAP')
    expect(child.requests).toEqual([{ id: 1, op: 'observe', args: { budgetMs: 3000 } }])
    child.reply(1, { app: 'Notepad' })
    await expect(first).resolves.toEqual({ app: 'Notepad' })
    expect(POWERSHELL_ARGS).not.toContain('pwsh')
    expect(powershellPath({ SystemRoot: 'D:\\Win' })).toMatch(/^D:\\Win[\\/]System32[\\/]WindowsPowerShell[\\/]v1\.0[\\/]powershell\.exe$/)
  })

  it('sends no request before the helper says it is ready, so PowerShell never reads a request as a command', async () => {
    const { worker, children } = harness({ autoReady: false })
    const call = worker.call('pointer')
    await settle()
    expect(children[0]!.lines).toEqual(['BOOTSTRAP'])
    children[0]!.ready()
    await settle()
    expect(children[0]!.requests.map((r) => r.op)).toEqual(['pointer'])
    children[0]!.reply(1, { x: 1, y: 2 })
    await expect(call).resolves.toEqual({ x: 1, y: 2 })
  })

  it('runs requests one at a time, in order, matching replies by id', async () => {
    const { worker, children } = harness()
    const a = worker.call('a')
    const b = worker.call('b')
    const c = worker.call('c')
    await settle()
    const child = children[0]!
    expect(child.requests.map((r) => r.op)).toEqual(['a'])
    // A reply for another id (a stale one) is not this request's.
    child.reply(7, 'stale')
    child.reply(1, 'A')
    await settle()
    expect(child.requests.map((r) => r.op)).toEqual(['a', 'b'])
    child.reply(2, 'B')
    await settle()
    child.reply(3, 'C')
    expect(await Promise.all([a, b, c])).toEqual(['A', 'B', 'C'])
  })

  it('rejects with the helper error message, and keeps the same process', async () => {
    const { worker, children } = harness()
    const failing = worker.call('semantic')
    await settle()
    children[0]!.say('{"id":1,"ok":false,"error":"LR_GONE||"}\n')
    await expect(failing).rejects.toThrow('LR_GONE||')
    const next = worker.call('pointer')
    await settle()
    children[0]!.reply(2, { x: 0, y: 0 })
    await expect(next).resolves.toEqual({ x: 0, y: 0 })
    expect(children).toHaveLength(1)
  })

  it('skips lines that are not replies and reads replies split across chunks', async () => {
    const { worker, children } = harness()
    const call = worker.call('displays')
    await settle()
    const child = children[0]!
    child.say('WARNING: something PowerShell printed\n\u001b[?1h')
    child.say('{"id":1,"ok":tr')
    child.say('ue,"result":[{"x":0}]}\r\n')
    await expect(call).resolves.toEqual([{ x: 0 }])
    child.say('{"id":"x","ok":true}\n[1,2]\nnull\n')
    const again = worker.call('pointer')
    await settle()
    child.reply(2, 'ok')
    await expect(again).resolves.toBe('ok')
  })

  it('kills a request that times out and starts a fresh process for the next one', async () => {
    const { worker, children } = harness({ timeoutMs: 30 })
    const slow = worker.call('observe')
    await expect(slow).rejects.toThrow('Windows took too long to answer (timed out).')
    expect(children[0]!.killed).toBe(true)
    const next = worker.call('pointer')
    await settle()
    expect(children).toHaveLength(2)
    // A late reply from the killed process is ignored.
    children[0]!.reply(1, 'late')
    children[1]!.reply(children[1]!.requests[0]!.id, { x: 5, y: 5 })
    await expect(next).resolves.toEqual({ x: 5, y: 5 })
  })

  it('kills the process when the running request is aborted, with the stop reason as the error', async () => {
    const { worker, children } = harness()
    const controller = new AbortController()
    const typing = worker.call('type', { text: 'hello' }, { signal: controller.signal })
    await settle()
    controller.abort('You stopped it on the computer.')
    await expect(typing).rejects.toThrow('You stopped it on the computer.')
    expect(children[0]!.killed).toBe(true)
    await expect(worker.call('pointer', {}, { signal: AbortSignal.abort('Already stopped.') })).rejects.toThrow('Already stopped.')
  })

  it('drops an aborted request that has not been sent, without touching the running one', async () => {
    const { worker, children } = harness()
    const running = worker.call('observe')
    const controller = new AbortController()
    const queued = worker.call('click', {}, { signal: controller.signal })
    await settle()
    controller.abort()
    await expect(queued).rejects.toThrow('It was stopped.')
    expect(children[0]!.killed).toBe(false)
    children[0]!.reply(1, 'seen')
    await expect(running).resolves.toBe('seen')
    expect(children[0]!.requests.map((r) => r.op)).toEqual(['observe'])
  })

  it('fails the running request when the process crashes, with what it said on stderr, then restarts on the next call', async () => {
    const { worker, children } = harness()
    const call = worker.call('observe')
    await settle()
    children[0]!.stderr.emit('data', Buffer.from('Unhandled exception: boom'))
    children[0]!.emit('exit', 3, null)
    await expect(call).rejects.toThrow('The Windows helper stopped (exit code 3): Unhandled exception: boom')
    const next = worker.call('pointer')
    await settle()
    expect(children).toHaveLength(2)
    // Ids keep counting across processes, so a reply meant for the old one can never answer the new one.
    expect(children[1]!.requests[0]!.id).toBe(2)
    children[1]!.reply(2, 'fresh')
    await expect(next).resolves.toBe('fresh')
  })

  it('reports a helper that could not compile, and tries again on the next call', async () => {
    const { worker, children } = harness({ autoReady: false })
    const call = worker.call('observe')
    await settle()
    children[0]!.say('{"id":0,"ok":false,"error":"c:\\\\temp\\\\x.cs(12) : error CS1525: Invalid expression term"}\n')
    await expect(call).rejects.toThrow('The Windows helper could not start: c:\\temp\\x.cs(12) : error CS1525')
    expect(children[0]!.killed).toBe(true)
    const next = worker.call('observe')
    await settle()
    expect(children).toHaveLength(2)
    children[1]!.ready()
    await settle()
    children[1]!.reply(children[1]!.requests[0]!.id, 'ok')
    await expect(next).resolves.toBe('ok')
  })

  it('gives up on a start that never says ready', async () => {
    const { worker, children } = harness({ autoReady: false, startTimeoutMs: 20 })
    await expect(worker.call('observe')).rejects.toThrow('Windows took too long to start its helper (timed out).')
    expect(children[0]!.killed).toBe(true)
  })

  it('fails pending requests on close', async () => {
    const { worker, children } = harness()
    const a = worker.call('a')
    const b = worker.call('b')
    await settle()
    worker.close()
    await expect(a).rejects.toThrow('shut down')
    await expect(b).rejects.toThrow('shut down')
    expect(children[0]!.killed).toBe(true)
    await expect(worker.call('c')).rejects.toThrow('shut down')
  })

  it('escapes everything outside printable ASCII, so the console code page cannot change typed text', () => {
    const line = asciiJson({ text: 'Café “x” 😀\u2028' })
    expect(line).toMatch(/^[\x20-\x7e]+$/)
    expect(JSON.parse(line)).toEqual({ text: 'Café “x” 😀\u2028' })
  })
})
