import { z } from 'zod'

/*
 * LocalRouter's realtime voice protocol (docs/dev/VOICE.md in this repository): the OpenAI Realtime API's beta
 * events over `GET /v1/realtime`, PCM16 mono 24 kHz base64 both ways, with `localrouter.*` extensions. This file
 * is the wire contract only: no React Native, so the headless check (e2e/) runs it under Node.
 */

/** Both directions carry PCM16 mono at this rate; the server refuses any other format. */
export const AUDIO_RATE = 24_000

/** The wait before a reply (`silence_duration_ms`): the user's own, 0.3 to 5 s; speech within it restarts it. */
export const WAIT_MIN_MS = 300
export const WAIT_MAX_MS = 5_000
export const WAIT_DEFAULT_MS = 800

export function clampWait(ms: number) {
  if (!Number.isFinite(ms)) return WAIT_DEFAULT_MS
  return Math.round(Math.min(WAIT_MAX_MS, Math.max(WAIT_MIN_MS, ms)))
}

/** What the client sets on its session. Text-only sessions carry chat; a call adds audio and server VAD. */
export type SessionSettings = {
  audio: boolean
  /** The wait before a reply, in ms (clamped to 0.3–5 s). */
  waitMs: number
  /** Server VAD sensitivity, 0 to 1 (the page's "hands-free sensitivity"); null: the server's default. */
  threshold?: number | null
  /** The speech model and its voice; empty: the server's defaults. A voice is only sent when the model lists it. */
  speechModel?: string
  voice?: string
  chatModel?: string
  instructions?: string
}

export function sessionUpdate(settings: SessionSettings) {
  const session: Record<string, unknown> = {
    modalities: settings.audio ? ['audio', 'text'] : ['text'],
    input_audio_format: 'pcm16',
    output_audio_format: 'pcm16',
    turn_detection: settings.audio
      ? {
          type: 'server_vad',
          silence_duration_ms: clampWait(settings.waitMs),
          prefix_padding_ms: 300,
          create_response: true,
          interrupt_response: true,
          ...(settings.threshold != null ? { threshold: settings.threshold } : {}),
        }
      : null,
  }
  if (settings.speechModel) session.speech_model = settings.speechModel
  if (settings.voice) session.voice = settings.voice
  if (settings.chatModel) session.model = settings.chatModel
  if (settings.instructions?.trim()) session.instructions = settings.instructions.trim()
  return { type: 'session.update' as const, session }
}

export type ToolName = 'generate_image' | 'generate_video'

/** Client events the app sends. */
export type ClientEvent =
  | ReturnType<typeof sessionUpdate>
  | { type: 'input_audio_buffer.append'; audio: string }
  | { type: 'input_audio_buffer.commit' }
  | { type: 'input_audio_buffer.clear' }
  | { type: 'conversation.item.create'; item: ConversationItem }
  | { type: 'conversation.item.truncate'; item_id: string; content_index: 0; audio_end_ms: number }
  | { type: 'response.create' }
  | { type: 'response.cancel' }

export type ConversationItem =
  | { type: 'message'; role: 'user'; content: { type: 'input_text'; text: string }[] }
  | { type: 'message'; role: 'assistant'; content: { type: 'text'; text: string }[] }
  | { type: 'function_call'; call_id: string; name: string; arguments: string }
  | { type: 'function_call_output'; call_id: string; output: string }

export const userText = (text: string): ClientEvent => ({ type: 'conversation.item.create', item: { type: 'message', role: 'user', content: [{ type: 'input_text', text }] } })

// ---------------------------------------------------------------- server events

const item = z.object({ id: z.string().optional(), type: z.string(), role: z.string().optional(), name: z.string().optional(), call_id: z.string().optional(), arguments: z.string().optional() }).loose()

/** The server events the app acts on, validated at the boundary. Anything else is ignored, as the protocol asks. */
export const serverEventSchema = z.discriminatedUnion('type', [
  z.object({ type: z.literal('session.created') }).loose(),
  z.object({ type: z.literal('session.updated') }).loose(),
  z.object({ type: z.literal('input_audio_buffer.speech_started'), item_id: z.string() }).loose(),
  z.object({ type: z.literal('input_audio_buffer.speech_stopped'), item_id: z.string() }).loose(),
  z.object({ type: z.literal('input_audio_buffer.committed'), item_id: z.string() }).loose(),
  z.object({ type: z.literal('localrouter.transcription.partial'), item_id: z.string(), transcript: z.string().default('') }).loose(),
  z.object({ type: z.literal('conversation.item.input_audio_transcription.completed'), item_id: z.string(), transcript: z.string().default('') }).loose(),
  z.object({ type: z.literal('conversation.item.input_audio_transcription.failed'), item_id: z.string(), error: z.object({ message: z.string().optional() }).loose().optional() }).loose(),
  z.object({ type: z.literal('response.created'), response: z.object({ id: z.string() }).loose() }).loose(),
  z.object({ type: z.literal('response.output_item.added'), item }).loose(),
  z.object({ type: z.literal('response.output_item.done'), item }).loose(),
  z.object({ type: z.literal('response.audio_transcript.delta'), item_id: z.string(), delta: z.string().default('') }).loose(),
  z.object({ type: z.literal('response.text.delta'), item_id: z.string(), delta: z.string().default('') }).loose(),
  z.object({ type: z.literal('response.audio.delta'), item_id: z.string(), delta: z.string().default('') }).loose(),
  z.object({ type: z.literal('response.audio.done'), item_id: z.string() }).loose(),
  z.object({
    type: z.literal('response.done'),
    response: z.object({ id: z.string().optional(), status: z.string().default('completed'), output: z.array(item).default([]) }).loose(),
  }).loose(),
  z.object({
    type: z.literal('localrouter.response.text_done'),
    item_id: z.string(),
    tool_calls: z.boolean().default(false),
    chars: z.number().default(0),
    speech_rate: z.number().nullable().default(null),
  }).loose(),
  z.object({ type: z.literal('localrouter.response.speech_done'), item_id: z.string() }).loose(),
  z.object({ type: z.literal('localrouter.job.created'), call_id: z.string().default(''), job_id: z.string(), kind: z.string().default('image') }).loose(),
  z.object({
    type: z.literal('localrouter.job.done'),
    call_id: z.string().default(''),
    job_id: z.string().default(''),
    kind: z.string().default('image'),
    status: z.string(),
    urls: z.array(z.string()).nullable().default([]),
    error: z.string().nullable().default(null),
  }).loose(),
  z.object({ type: z.literal('localrouter.voice.paused'), model: z.string().default(''), for_model: z.string().default('') }).loose(),
  z.object({ type: z.literal('localrouter.voice.resumed') }).loose(),
  z.object({ type: z.literal('error'), error: z.object({ message: z.string().default('error'), code: z.string().nullable().optional(), type: z.string().optional() }).loose() }).loose(),
])

export type ServerEvent = z.infer<typeof serverEventSchema>

/** One server message, or null for events the app does not use (and for anything malformed). */
export function parseServerEvent(text: string): ServerEvent | null {
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch {
    return null
  }
  const parsed = serverEventSchema.safeParse(raw)
  return parsed.success ? parsed.data : null
}

/** The voice a session refused ("model 'x' has no voice 'y'"): the client drops it and uses the default. */
export function refusedVoice(message: string): { model: string; voice: string } | null {
  const match = /model '([^']+)' has no voice '([^']+)'/.exec(message)
  return match ? { model: match[1], voice: match[2] } : null
}
