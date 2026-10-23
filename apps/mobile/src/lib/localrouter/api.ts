import { z } from 'zod'

import { fileUrl } from './address'

/*
 * LocalRouter's HTTP API, as the app uses it: pairing, the models and their voices, a generation's retry and its
 * job. `fetch` is passed in, so the headless check runs this under Node exactly as the phone does.
 */

export type Fetch = (input: string, init?: { method?: string; headers?: Record<string, string>; body?: string }) => Promise<{
  ok: boolean
  status: number
  json(): Promise<unknown>
  text(): Promise<string>
  arrayBuffer(): Promise<ArrayBuffer>
}>

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly code: string | null = null) {
    super(message)
  }
}

const errorEnvelope = z.object({ error: z.object({ message: z.string(), code: z.string().nullable().optional() }).loose() }).loose()

const device = z.object({ id: z.string(), name: z.string() }).loose()
const pairResult = z.object({ token: z.string(), device, required: z.boolean().default(false) }).loose()
const pairStatus = z.object({ required: z.boolean(), paired: z.boolean(), device: device.nullable().default(null) }).loose()

const voice = z.union([z.string(), z.object({ id: z.string(), name: z.string().optional(), description: z.string().optional() }).loose()])
const model = z.object({
  id: z.string(),
  name: z.string().optional(),
  kind: z.string().optional(),
  running: z.boolean().optional(),
  capabilities: z.array(z.string()).default([]),
  default_for: z.array(z.string()).default([]),
  voices: z.array(voice).nullable().optional(),
  default_voice: z.string().nullable().optional(),
  qualified: z.boolean().optional(),
}).loose()
const models = z.object({ data: z.array(model) }).loose()

const job = z.object({
  id: z.string(),
  status: z.string(),
  error: z.unknown().optional(),
  data: z.array(z.object({ url: z.string() }).loose()).optional(),
  urls: z.array(z.string()).optional(),
}).loose()

export type Device = z.infer<typeof device>
export type PairStatus = z.infer<typeof pairStatus>
export type Model = z.infer<typeof model>
export type Voice = { id: string; name: string; description?: string }
export type SpeechEngine = { id: string; name: string; voices: Voice[]; defaultVoice: string | null; isDefault: boolean }
export type JobState = { status: 'queued' | 'running' | 'completed' | 'failed'; urls: string[]; error: string | null }

/** The speech models and their voices, exactly as /v1/models lists them (the voice picker offers only these). */
export function speechEngines(list: Model[]): SpeechEngine[] {
  return list
    .filter((m) => m.capabilities.includes('text_to_speech') && m.qualified !== false)
    .map((m) => ({
      id: m.id,
      name: m.name ?? m.id,
      voices: (m.voices ?? []).map((v) => (typeof v === 'string' ? { id: v, name: v } : { id: v.id, name: v.name ?? v.id, description: v.description })),
      defaultVoice: m.default_voice ?? null,
      isDefault: m.default_for.includes('text_to_speech'),
    }))
}

function message(error: unknown) {
  if (typeof error === 'string') return error
  const parsed = z.object({ message: z.string() }).loose().safeParse(error)
  return parsed.success ? parsed.data.message : null
}

export function createApi(options: { baseUrl: string; token?: string | null; fetch: Fetch }) {
  const { baseUrl, fetch } = options
  const headers = (json: boolean): Record<string, string> => ({
    ...(json ? { 'content-type': 'application/json' } : {}),
    ...(options.token ? { authorization: `Bearer ${options.token}` } : {}),
  })

  async function call<T>(path: string, schema: z.ZodType<T>, init: { method?: string; body?: unknown } = {}): Promise<T> {
    const response = await fetch(`${baseUrl}${path}`, {
      method: init.method ?? 'GET',
      headers: headers(init.body !== undefined),
      ...(init.body !== undefined ? { body: JSON.stringify(init.body) } : {}),
    })
    const raw: unknown = await response.json().catch(() => null)
    if (!response.ok) {
      const envelope = errorEnvelope.safeParse(raw)
      throw new ApiError(envelope.success ? envelope.data.error.message : `The box answered ${response.status}`, response.status, envelope.success ? envelope.data.error.code ?? null : null)
    }
    const parsed = schema.safeParse(raw)
    if (!parsed.success) throw new ApiError(`The box sent an unexpected answer to ${path}`, response.status)
    return parsed.data
  }

  return {
    baseUrl,

    /** Exchanges a pairing code from `localrouter pair add` for this device's token. */
    pair: (code: string, name: string) => call('/v1/pair', pairResult, { method: 'POST', body: { code, name } }),

    /** Whether the box requires pairing and whether this token is a paired device. */
    pairStatus: () => call('/v1/pair', pairStatus),

    models: async () => (await call('/v1/models', models)).data,

    /** Makes a picture again over HTTP (a failed card's Retry). */
    async image(prompt: string) {
      const result = await call('/v1/images/generations', z.object({ data: z.array(z.object({ url: z.string() }).loose()) }).loose(), {
        method: 'POST',
        body: { prompt, response_format: 'url' },
      })
      return result.data.map((d) => fileUrl(baseUrl, d.url))
    },

    /** Starts a video over HTTP; follow it with `job`. */
    async video(prompt: string) {
      return (await call('/v1/videos', z.object({ id: z.string() }).loose(), { method: 'POST', body: { prompt } })).id
    },

    /** Any job's state and outputs (`GET /v1/jobs/{id}`): a generation whose `job.done` was missed. */
    async job(id: string): Promise<JobState> {
      const j = await call(`/v1/jobs/${encodeURIComponent(id)}`, job)
      const urls = (j.urls ?? j.data?.map((d) => d.url) ?? []).map((u) => fileUrl(baseUrl, u))
      const status = j.status === 'completed' || j.status === 'succeeded' ? 'completed' : j.status === 'failed' || j.status === 'cancelled' ? 'failed' : j.status === 'queued' ? 'queued' : 'running'
      return { status, urls, error: message(j.error) }
    },

    /** Speech as WAV bytes (the headless check speaks its prompt this way). */
    async speech(input: string, voice?: string, model?: string) {
      const response = await fetch(`${baseUrl}/v1/audio/speech`, {
        method: 'POST',
        headers: headers(true),
        body: JSON.stringify({ input, response_format: 'wav', ...(voice ? { voice } : {}), ...(model ? { model } : {}) }),
      })
      if (!response.ok) throw new ApiError(`speech failed: ${response.status} ${await response.text().catch(() => '')}`, response.status)
      return response.arrayBuffer()
    },

    fileUrl: (url: string) => fileUrl(baseUrl, url),
  }
}

export type Api = ReturnType<typeof createApi>
