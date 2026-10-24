import type { Journal } from './journal.ts'
import type { Registry, ToolEvent } from './registry.ts'
import type { Spark } from './spark.ts'

/**
 * A realtime session with the box in **client tool mode** (docs/HARNESS.md, `tool_execution: "client"`): the Spark
 * listens, thinks and speaks; this harness owns every tool. The session asks for client mode before anything else and
 * refuses to go on if the box does not echo it (so a box tool never runs by accident), offers the registry's tools,
 * and for each response runs its complete calls once, here, then returns their outputs and asks for the continuation.
 * A call the box cancels (the person spoke again) is aborted and its output never sent. Everything is journaled.
 *
 * A UI (the `/voice` page in the app) can drive the same session: its client events go through `send`, which keeps
 * client mode and this harness's tools in every `session.update`, and `pageView` says which server events it sees.
 */
export type SocketLike = {
  readonly readyState: number
  send(data: string): void
  close(code?: number, reason?: string): void
  onopen: ((e: unknown) => void) | null
  onmessage: ((e: { data: unknown }) => void) | null
  onclose: ((e: { code: number; reason: string }) => void) | null
  onerror: ((e: unknown) => void) | null
}
export type Dial = (url: string, protocols: string[]) => SocketLike

export const nodeDial: Dial = (url, protocols) => new (globalThis as unknown as { WebSocket: new (u: string, p: string[]) => SocketLike }).WebSocket(url, protocols)

export type ServerEvent = { type: string; [k: string]: unknown }

type Call = { call_id: string; name: string; arguments: string; response_id: string; item_id: string; abort: AbortController; cancelled: boolean; at: number }

export type CallRecord = { call_id: string; name: string; arguments: string; ok: boolean | null; output: string | null; cancelled: boolean; run_ms: number | null; arguments_done_at: number; output_sent_at: number | null; first_audio_after_output_ms: number | null }

export type SessionOptions = {
  spark: Spark
  registry: Registry
  journal?: Journal
  conversation?: string
  instructions?: string
  model?: string
  /** Push to talk (true: no server VAD; the caller commits) or server VAD (false, the default). */
  pushToTalk?: boolean
  dial?: Dial
  /** Every server event, after the harness handled it, plus the harness's own `localrouter.job.*` for its tools. */
  onEvent?: (event: ServerEvent) => void
  onClose?: (reason: string) => void
  now?: () => number
}

// Made during a call, shown as the page's own job card. A video on a call is a proposal (`videoHandoff`), never a job.
const GENERATION_TOOLS = new Set(['generate_image', 'edit_image'])

export class ClientToolsSession {
  readonly calls: CallRecord[] = []
  private readonly o: SessionOptions
  private socket: SocketLike | null = null
  private readonly pending = new Map<string, Call>()
  private readonly running = new Set<Promise<void>>()
  private utterance = ''
  private utteranceAt = 0
  private lastOutputAt: number | null = null
  private awaitingAudio: CallRecord[] = []
  private assistantText = new Map<string, string>()
  private ready: Promise<ServerEvent> | null = null
  private readonly now: () => number
  /** The page's own session settings, kept so tools and the mode survive its updates. */
  private pageSession: Record<string, unknown> = {}
  sessionId = ''

  constructor(options: SessionOptions) {
    this.o = options
    this.now = options.now ?? Date.now
  }

  get lastUtterance() {
    return this.utterance
  }

  /** Busy while any tool runs or calls wait for their response to end. */
  get busy() {
    return this.running.size > 0 || this.pending.size > 0
  }

  /** Opens the socket and negotiates client tool mode; rejects when the box does not take it. */
  connect(): Promise<ServerEvent> {
    if (this.ready) return this.ready
    const { url, protocols } = this.o.spark.realtime(this.o.model)
    this.ready = new Promise((resolve, reject) => {
      let settled = false
      const socket = (this.o.dial ?? nodeDial)(url, protocols)
      this.socket = socket
      socket.onopen = () => this.sendRaw({ type: 'session.update', session: this.session({}) })
      socket.onerror = () => undefined
      socket.onclose = (e) => {
        this.socket = null
        for (const c of this.pending.values()) c.abort.abort('the session closed')
        if (!settled) {
          settled = true
          reject(new Error(`the realtime session closed (${e.code}${e.reason ? ` ${e.reason}` : ''})`))
        }
        this.o.onClose?.(e.reason || String(e.code))
      }
      socket.onmessage = (m) => {
        let ev: ServerEvent
        try {
          ev = JSON.parse(String(m.data)) as ServerEvent
        } catch {
          return
        }
        if (!settled && ev.type === 'session.updated') {
          const mode = ((ev.session as Record<string, unknown> | undefined)?.localrouter as Record<string, unknown> | undefined)?.tool_execution
          settled = true
          if (mode !== 'client') {
            reject(new Error(`the box did not take client tool execution (it said ${JSON.stringify(mode)}): it needs the client-tools realtime (contract 2026-10-23 or later)`))
            socket.close()
            return
          }
          this.sessionId = String((ev.session as Record<string, unknown>).id ?? '')
          if (this.o.journal && this.o.conversation) this.o.journal.append(this.o.conversation, { type: 'session', mode: 'client', tools: this.o.registry.names, ...(this.o.model ? { model: this.o.model } : {}) })
          resolve(ev)
        } else if (!settled && ev.type === 'error') {
          settled = true
          reject(new Error(String((ev.error as Record<string, unknown> | undefined)?.message ?? 'the box refused the session')))
        }
        this.handle(ev)
      }
    })
    return this.ready
  }

  close() {
    this.socket?.close(1000, 'done')
    this.socket = null
  }

  /** A client event from a UI. `session.update` keeps client mode and this harness's tools whatever the UI sends. */
  send(event: Record<string, unknown>) {
    if (event.type === 'session.update') {
      this.pageSession = { ...this.pageSession, ...((event.session as Record<string, unknown>) ?? {}) }
      return this.sendRaw({ type: 'session.update', session: this.session(this.pageSession) })
    }
    if (event.type === 'conversation.item.create') this.noteUserItem(event.item as Record<string, unknown> | undefined)
    this.sendRaw(event)
  }

  /** A typed turn: the person's text, then a response. */
  say(text: string) {
    this.send({ type: 'conversation.item.create', item: { type: 'message', role: 'user', content: [{ type: 'input_text', text }] } })
    this.sendRaw({ type: 'response.create' })
  }

  /** A spoken turn (push to talk): 24 kHz pcm16 mono, in 40 ms chunks, then commit and respond. */
  speak(pcm: Buffer) {
    for (let at = 0; at < pcm.length; at += 1920) this.sendRaw({ type: 'input_audio_buffer.append', audio: pcm.subarray(at, Math.min(pcm.length, at + 1920)).toString('base64') })
    this.sendRaw({ type: 'input_audio_buffer.commit' })
    this.sendRaw({ type: 'response.create' })
  }

  private session(page: Record<string, unknown>): Record<string, unknown> {
    const { tools: _ignored, tool_choice: _choice, localrouter: _lr, ...rest } = page
    return {
      ...(this.o.pushToTalk ? { turn_detection: null } : {}),
      ...(this.o.instructions && !rest.instructions ? { instructions: this.o.instructions } : {}),
      ...rest,
      ...(this.o.instructions && rest.instructions ? { instructions: `${String(rest.instructions)}\n\n${this.o.instructions}` } : {}),
      tools: this.o.registry.realtime(),
      tool_choice: 'auto',
      localrouter: { tool_execution: 'client' },
    }
  }

  private sendRaw(event: Record<string, unknown>) {
    if (this.socket && this.socket.readyState === 1) this.socket.send(JSON.stringify(event))
  }

  private noteUserItem(item: Record<string, unknown> | undefined) {
    if (!item || item.type !== 'message' || item.role !== 'user') return
    const text = ((item.content as Array<Record<string, unknown>>) ?? []).map((c) => (typeof c.text === 'string' ? c.text : '')).join(' ').trim()
    if (!text) return
    this.utterance = text
    this.utteranceAt = this.now()
    if (this.o.journal && this.o.conversation) this.o.journal.append(this.o.conversation, { type: 'user', text, via: 'text' })
  }

  private emit(event: ServerEvent) {
    try {
      this.o.onEvent?.(event)
    } catch {
      // A UI's problem is not the session's.
    }
  }

  private handle(ev: ServerEvent) {
    switch (ev.type) {
      case 'conversation.item.input_audio_transcription.completed': {
        const text = String(ev.transcript ?? '').trim()
        if (text) {
          this.utterance = text
          this.utteranceAt = this.now()
          if (this.o.journal && this.o.conversation) this.o.journal.append(this.o.conversation, { type: 'user', text, via: 'voice' })
        }
        break
      }
      case 'response.audio_transcript.delta':
      case 'response.text.delta': {
        const id = String(ev.item_id ?? '')
        this.assistantText.set(id, (this.assistantText.get(id) ?? '') + String(ev.delta ?? ''))
        break
      }
      case 'response.audio.delta':
        if (this.lastOutputAt !== null) {
          const ms = this.now() - this.lastOutputAt
          for (const r of this.awaitingAudio) r.first_audio_after_output_ms = ms
          this.awaitingAudio = []
          this.lastOutputAt = null
        }
        break
      case 'response.function_call_arguments.done': {
        const call: Call = { call_id: String(ev.call_id), name: String(ev.name), arguments: String(ev.arguments ?? ''), response_id: String(ev.response_id ?? ''), item_id: String(ev.item_id ?? ''), abort: new AbortController(), cancelled: false, at: this.now() }
        if (this.pending.has(call.call_id) || this.calls.some((c) => c.call_id === call.call_id)) break // complete arguments run once
        this.pending.set(call.call_id, call)
        this.calls.push({ call_id: call.call_id, name: call.name, arguments: call.arguments, ok: null, output: null, cancelled: false, run_ms: null, arguments_done_at: call.at, output_sent_at: null, first_audio_after_output_ms: null })
        if (this.o.journal && this.o.conversation) this.o.journal.append(this.o.conversation, { type: 'tool_call', call_id: call.call_id, name: call.name, arguments: call.arguments })
        break
      }
      case 'localrouter.function_call.cancelled': {
        const id = String(ev.call_id ?? '')
        const call = this.pending.get(id) ?? [...this.inFlight.values()].find((c) => c.call_id === id)
        if (call) {
          call.cancelled = true
          call.abort.abort(String(ev.reason ?? 'cancelled'))
        }
        const rec = this.calls.find((c) => c.call_id === id)
        if (rec) rec.cancelled = true
        if (this.o.journal && this.o.conversation) this.o.journal.append(this.o.conversation, { type: 'tool_cancelled', call_id: id, reason: String(ev.reason ?? '') })
        break
      }
      case 'response.done': {
        const response = (ev.response ?? {}) as Record<string, unknown>
        const rid = String(response.id ?? '')
        for (const out of (response.output as Array<Record<string, unknown>> | undefined) ?? []) {
          if (out.type === 'message') {
            const text = this.assistantText.get(String(out.id)) ?? ''
            if (text.trim() && this.o.journal && this.o.conversation) this.o.journal.append(this.o.conversation, { type: 'assistant', text, item_id: String(out.id) })
            this.assistantText.delete(String(out.id))
          }
        }
        const calls = [...this.pending.values()].filter((c) => !rid || !c.response_id || c.response_id === rid)
        for (const c of calls) this.pending.delete(c.call_id)
        if (calls.length) this.runAll(calls)
        break
      }
    }
    this.emit(ev)
  }

  private readonly inFlight = new Set<Call>()

  /** Runs one response's calls (together), then returns their outputs and asks for the continuation once. */
  private runAll(calls: Call[]) {
    const work = (async () => {
      const results = await Promise.all(
        calls.map(async (call) => {
          this.inFlight.add(call)
          const ctx = { callId: call.call_id, utterance: this.utterance, utteranceAt: this.utteranceAt, signal: call.abort.signal, emit: (e: ToolEvent) => this.toolEvent(e) }
          const ran = await this.o.registry.run(call.name, call.arguments, ctx)
          this.inFlight.delete(call)
          const rec = this.calls.find((c) => c.call_id === call.call_id)
          if (rec) Object.assign(rec, { ok: ran.ok, output: ran.output, run_ms: ran.ms })
          // A UI's record of the call (its chat history), for the tools it does not show as a card of its own.
          if (!GENERATION_TOOLS.has(call.name)) this.emit({ type: 'localrouter.client_tool.done', call_id: call.call_id, name: call.name, arguments: call.arguments, ok: ran.ok, output: ran.output, cancelled: call.cancelled })
          if (this.o.journal && this.o.conversation) this.o.journal.append(this.o.conversation, { type: 'tool_result', call_id: call.call_id, ok: ran.ok, output: ran.output, ms: ran.ms })
          return { call, ran, rec }
        }),
      )
      const live = results.filter((r) => !r.call.cancelled)
      if (!live.length || !this.socket) return
      for (const r of live) this.sendRaw({ type: 'conversation.item.create', item: { type: 'function_call_output', call_id: r.call.call_id, output: r.ran.output } })
      this.sendRaw({ type: 'response.create' })
      this.lastOutputAt = this.now()
      this.awaitingAudio = live.flatMap((r) => (r.rec ? [r.rec] : []))
      for (const r of this.awaitingAudio) r.output_sent_at = this.lastOutputAt
    })()
    this.running.add(work)
    void work.finally(() => this.running.delete(work))
  }

  /** A generation tool's job, as the box's own `localrouter.job.*` events, so a UI's reel shows it unchanged. */
  private toolEvent(e: ToolEvent) {
    if (e.type === 'job.created') this.emit({ type: 'localrouter.job.created', call_id: e.call_id, job_id: e.job_id, kind: e.kind })
    else if (e.type === 'job.done') this.emit({ type: 'localrouter.job.done', call_id: e.call_id, job_id: e.job_id, kind: e.kind, status: e.status, urls: e.urls, error: e.error })
    else this.emit({ type: 'localrouter.companion.handoff', call_id: e.call_id, kind: e.kind, prompt: e.prompt, seconds: e.seconds })
  }
}

/**
 * Which server events the `/voice` page sees. It draws a "making a picture" card for any function call item, so
 * only generation calls reach it as items; the other tools show as the executor's approval and task cards instead.
 */
export function pageView(ev: ServerEvent): boolean {
  if (ev.type === 'response.output_item.added' || ev.type === 'response.output_item.done') {
    const item = ev.item as Record<string, unknown> | undefined
    if (item?.type === 'function_call') return GENERATION_TOOLS.has(String(item.name))
  }
  if (ev.type.startsWith('response.function_call_arguments.')) return false
  return true
}
