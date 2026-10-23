import type { ClientEvent, ServerEvent } from './protocol'

/*
 * The conversation, built from the session's events. Typed and spoken turns share it; pictures and videos are tool
 * messages in it, so the chat's cards and the generations reel are two views of the same items, as in the web
 * client. Pure TypeScript (no React Native): the headless check drives the same store.
 */

export type GenerationKind = 'image' | 'video'
export type GenerationStatus = 'working' | 'completed' | 'failed'

export type UserMessage = { id: string; role: 'user'; text: string; via: 'voice' | 'text'; partial: boolean; error: string | null; at: number }
export type AssistantMessage = { id: string; role: 'assistant'; text: string; streaming: boolean; interrupted: boolean; failed: boolean; at: number }
export type Generation = {
  id: string
  role: 'tool'
  kind: GenerationKind
  prompt: string
  callId: string
  jobId: string | null
  status: GenerationStatus
  urls: string[]
  error: string | null
  at: number
  /** When it last changed (a job id, a poll that saw progress): the watchdog's clock. */
  touchedAt: number
}
export type Message = UserMessage | AssistantMessage | Generation

export const isGeneration = (m: Message): m is Generation => m.role === 'tool'

const TOOL_KIND: Record<string, GenerationKind> = { generate_image: 'image', generate_video: 'video' }

/** At most this many recent items go into a new session. */
const SEED_ITEMS = 40

let counter = 0
const newId = () => `${Date.now().toString(36)}-${(counter++).toString(36)}`

function promptOf(args: string | undefined) {
  try {
    const parsed: unknown = JSON.parse(args ?? '{}')
    if (parsed && typeof parsed === 'object' && 'prompt' in parsed && typeof parsed.prompt === 'string') return parsed.prompt
  } catch {
    // Arguments still being written, or not JSON: no prompt to show.
  }
  return ''
}

export class Conversation {
  messages: Message[] = []
  private listeners = new Set<() => void>()
  /** The server's item and call ids → message ids, for the current connection only (a new session renumbers). */
  private byItem = new Map<string, string>()
  private byCall = new Map<string, string>()

  /** `rebase` turns a file URL from the server into one the client can fetch (see `fileUrl`). */
  constructor(private readonly rebase: (url: string) => string = (url) => url, private readonly now: () => number = Date.now) {}

  subscribe(listener: () => void) {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  private changed() {
    for (const listener of this.listeners) listener()
  }

  private find(id: string | undefined) {
    return id ? this.messages.find((m) => m.id === id) : undefined
  }

  /** Replaces one message (a new object, so React sees the change). */
  private update<M extends Message>(message: M, patch: Partial<M>) {
    const next = { ...message, ...patch }
    this.messages = this.messages.map((m) => (m.id === message.id ? next : m))
    return next
  }

  private add<M extends Message>(message: M) {
    this.messages = [...this.messages, message]
    return message
  }

  private remove(id: string) {
    this.messages = this.messages.filter((m) => m.id !== id)
  }

  /** The pictures and videos, newest first: the generations reel. */
  generations() {
    return this.messages.filter(isGeneration).reverse()
  }

  /** A typed message: the events that send it and ask for a reply. */
  typed(text: string): ClientEvent[] {
    this.add<UserMessage>({ id: newId(), role: 'user', text, via: 'text', partial: false, error: null, at: this.now() })
    this.changed()
    return [{ type: 'conversation.item.create', item: { type: 'message', role: 'user', content: [{ type: 'input_text', text }] } }, { type: 'response.create' }]
  }

  /** A new connection: per-item state belongs to the old session, and nothing of it is streaming any more. */
  newConnection() {
    this.byItem.clear()
    this.byCall.clear()
    this.messages = this.messages.flatMap((m): Message[] => {
      if (m.role === 'user' && m.partial) return m.text ? [{ ...m, partial: false }] : []
      if (m.role === 'assistant' && m.streaming) return m.text ? [{ ...m, streaming: false, interrupted: true }] : []
      return [m]
    })
    this.changed()
  }

  /**
   * The conversation for a new session. A picture or video goes in as its function_call and output: seeded as
   * text alone ("make me a cow", "here's your cow"), the model copied that and promised pictures without calling
   * the tool (VOICE.md).
   */
  seed(): ClientEvent[] {
    const said = this.messages.filter((m) => (m.role === 'tool' ? m.status !== 'working' : !(m.role === 'user' && m.partial) && m.text.trim())).slice(-SEED_ITEMS)
    const events: ClientEvent[] = []
    for (const m of said) {
      if (m.role === 'tool') {
        const call = m.callId || `call_${m.id}`
        events.push({ type: 'conversation.item.create', item: { type: 'function_call', call_id: call, name: m.kind === 'video' ? 'generate_video' : 'generate_image', arguments: JSON.stringify({ prompt: m.prompt }) } })
        const output = m.status === 'completed' ? { status: 'completed', urls: m.urls } : { status: 'failed', error: m.error ?? 'failed' }
        events.push({ type: 'conversation.item.create', item: { type: 'function_call_output', call_id: call, output: JSON.stringify(output) } })
      } else if (m.role === 'user') {
        events.push({ type: 'conversation.item.create', item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: m.text }] } })
      } else {
        events.push({ type: 'conversation.item.create', item: { type: 'message', role: 'assistant', content: [{ type: 'text', text: m.text }] } })
      }
    }
    return events
  }

  private userFor(itemId: string) {
    const known = this.find(this.byItem.get(itemId))
    if (known?.role === 'user') return known
    const m = this.add<UserMessage>({ id: newId(), role: 'user', text: '', via: 'voice', partial: true, error: null, at: this.now() })
    this.byItem.set(itemId, m.id)
    return m
  }

  private assistantFor(itemId: string) {
    const known = this.find(this.byItem.get(itemId))
    if (known?.role === 'assistant') return known
    const m = this.add<AssistantMessage>({ id: newId(), role: 'assistant', text: '', streaming: true, interrupted: false, failed: false, at: this.now() })
    this.byItem.set(itemId, m.id)
    return m
  }

  private generationFor(callId: string, jobId: string | null, kind: GenerationKind) {
    const id = this.byCall.get(callId)
    const known = (id ? this.find(id) : undefined) ?? (jobId ? this.messages.find((m) => isGeneration(m) && m.jobId === jobId) : undefined)
    if (known && isGeneration(known)) return known
    // A call whose function_call item this client missed: the card shows anyway.
    const m = this.add<Generation>({ id: newId(), role: 'tool', kind, prompt: '', callId, jobId, status: 'working', urls: [], error: null, at: this.now(), touchedAt: this.now() })
    if (callId) this.byCall.set(callId, m.id)
    return m
  }

  /** One server event. */
  apply(event: ServerEvent) {
    switch (event.type) {
      case 'input_audio_buffer.speech_started':
        this.userFor(event.item_id)
        break
      case 'localrouter.transcription.partial': {
        const m = this.userFor(event.item_id)
        if (m.partial) this.update(m, { text: event.transcript })
        break
      }
      case 'conversation.item.input_audio_transcription.completed': {
        const m = this.userFor(event.item_id)
        // Nothing was said (a cough the VAD took for speech): no message.
        if (!event.transcript.trim()) this.remove(m.id)
        else this.update(m, { text: event.transcript, partial: false })
        break
      }
      case 'conversation.item.input_audio_transcription.failed': {
        const m = this.userFor(event.item_id)
        this.update(m, { text: m.text || '(not heard)', partial: false, error: event.error?.message ?? 'transcription failed' })
        break
      }
      case 'response.output_item.added':
        if (event.item.type === 'message' && event.item.role === 'assistant' && event.item.id) this.assistantFor(event.item.id)
        break
      case 'response.audio_transcript.delta':
      case 'response.text.delta': {
        const m = this.assistantFor(event.item_id)
        this.update(m, { text: m.text + event.delta })
        break
      }
      case 'response.output_item.done': {
        const kind = event.item.name ? TOOL_KIND[event.item.name] : undefined
        if (event.item.type !== 'function_call' || !kind) break
        const m = this.generationFor(event.item.call_id ?? '', null, kind)
        this.update(m, { prompt: promptOf(event.item.arguments) || m.prompt, kind })
        break
      }
      case 'localrouter.job.created': {
        const kind = event.kind === 'video' ? 'video' : 'image'
        const m = this.generationFor(event.call_id, event.job_id, kind)
        this.update(m, { jobId: event.job_id, touchedAt: this.now() })
        break
      }
      case 'localrouter.job.done': {
        const m = this.generationFor(event.call_id, event.job_id || null, event.kind === 'video' ? 'video' : 'image')
        this.finish(m.id, event.status === 'completed' ? { status: 'completed', urls: event.urls ?? [] } : { status: 'failed', error: event.error ?? event.status })
        break
      }
      case 'response.done': {
        for (const out of event.response.output) {
          const m = this.find(out.id ? this.byItem.get(out.id) : undefined)
          if (m?.role !== 'assistant') continue
          const interrupted = m.interrupted || event.response.status === 'cancelled'
          if (!m.text && !interrupted) this.remove(m.id)
          else this.update(m, { streaming: false, interrupted, failed: event.response.status === 'failed' })
        }
        break
      }
      default:
        return
    }
    this.changed()
  }

  /** The user talked over this reply (barge-in): it is kept, marked as cut short. */
  interrupted(itemId: string) {
    const m = this.find(this.byItem.get(itemId))
    if (m?.role !== 'assistant') return
    this.update(m, { interrupted: true })
    this.changed()
  }

  /** A generation's outcome, from whichever path learnt it first (the session's job.done, a poll, a retry). */
  finish(id: string, outcome: { status: 'completed'; urls: string[] } | { status: 'failed'; error: string }) {
    const m = this.find(id)
    if (!m || !isGeneration(m)) return
    if (m.status === 'completed' && outcome.status === 'failed') return // a late failure never replaces a picture
    if (outcome.status === 'completed') this.update(m, { status: 'completed', urls: outcome.urls.map(this.rebase), error: null, touchedAt: this.now() })
    else this.update(m, { status: 'failed', error: outcome.error, touchedAt: this.now() })
    this.changed()
  }

  /** A failed generation tried again over HTTP: back to working. */
  retrying(id: string) {
    const m = this.find(id)
    if (!m || !isGeneration(m)) return
    this.update(m, { status: 'working', error: null, jobId: null, touchedAt: this.now() })
    this.changed()
  }

  touched(id: string) {
    const m = this.find(id)
    if (m && isGeneration(m)) this.update(m, { touchedAt: this.now() })
  }

  clear() {
    this.messages = []
    this.byItem.clear()
    this.byCall.clear()
    this.changed()
  }

  /** What is kept on the phone: finished messages (streaming ones are settled as on a new connection). */
  toJSON() {
    return { version: 1, messages: this.messages }
  }

  load(saved: unknown) {
    if (!saved || typeof saved !== 'object' || !('messages' in saved) || !Array.isArray(saved.messages)) return
    this.messages = saved.messages as Message[]
    this.newConnection()
  }
}

/** How long a generation may go without news before the client asks the job itself, and when it gives up. */
export const GENERATION_LIMITS = { noJobMs: 45_000, imageMs: 4 * 60_000, videoMs: 25 * 60_000, pollAfterMs: 20_000 }
