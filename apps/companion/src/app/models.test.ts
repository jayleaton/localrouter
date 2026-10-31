import { describe, expect, it } from 'vitest'
import { Spark, SparkError } from '../harness/spark.ts'
import { budgetView, callLoading, listedModels, loadsIn, toolEntryOf, voiceBlockers, modelNames, modelsIntro, modelsView, NO_MANAGE, oldBoxModels, serverName, settingsChange, settingsRefusal, toolsOf } from './models.ts'

const server = serverName('studio')

// The Mac Studio's lineup (its config's ids, kinds and resident bytes; load times as measured there).
const chat = { id: 'qwen3.8-27b', kind: 'chat', state: 'ready', resident_bytes: 28723101696, priority: 100, residency: 'on_demand', residency_default: 'always', idle_ttl_s: 300, idle_ttl_s_default: 600, allowed_residency: ['always', 'call', 'on_demand'], last_load_ms: 4900, loaded_since: '2026-10-31T10:00:00Z', last_error: '' }
const speech = { ...chat, id: 'fishaudio/drama-3', kind: 'speech', state: 'unloaded', resident_bytes: 30870077440, priority: 95, residency: 'call', residency_default: 'call', idle_ttl_s: 0, idle_ttl_s_default: 0, last_load_ms: null, loaded_since: null }
const image = { ...chat, id: 'qwen-image-2.1-turbo', kind: 'image', state: 'unloaded', resident_bytes: 10200547328, priority: 10, residency: 'on_demand', residency_default: 'on_demand', idle_ttl_s: 60, idle_ttl_s_default: 60, allowed_residency: ['always', 'on_demand'], last_load_ms: 2000, last_error: 'out of memory' }
/** 77.8 GiB ceiling; the chat model always loaded, KnowLine (10.0 GiB) loaded on demand. */
const body = { object: 'list', data: [chat, speech, image], budget: { ceiling_bytes: 83493421056, booked_bytes: 28723101696 + 10724056240, always_bytes: 28723101696 } }

/** A box answering on a fake fetch: what was asked, and a reply per route. */
function box(reply: (method: string, path: string, body: unknown) => { status: number; json: unknown }) {
  const asked: Array<{ method: string; path: string; body: unknown; auth: string }> = []
  const f = (async (input: string | URL | Request, init?: RequestInit) => {
    const u = new URL(String(input))
    const b = init?.body ? JSON.parse(String(init.body)) : undefined
    const method = init?.method ?? 'GET'
    asked.push({ method, path: u.pathname, body: b, auth: (init?.headers as Record<string, string>)?.authorization ?? '' })
    const r = reply(method, u.pathname, b)
    return new Response(JSON.stringify(r.json), { status: r.status, headers: { 'content-type': 'application/json' } })
  }) as typeof fetch
  return { spark: new Spark('http://127.0.0.1:9', 'tok', f), asked }
}

describe('the server client for model settings', () => {
  it('reads /v1/tools and posts one model’s settings with the token', async () => {
    const { spark, asked } = box((m, _p, b) => (m === 'GET' ? { status: 200, json: body } : { status: 200, json: { ...speech, ...(b as object) } }))
    expect(toolsOf(await spark.tools()).entries.map((e) => e.id)).toEqual(['qwen3.8-27b', 'fishaudio/drama-3', 'qwen-image-2.1-turbo'])
    // The id's slash stays a slash: the box's routes read the id up to the verb, without decoding.
    const out = await spark.toolSettings('fishaudio/drama-3', { idle_ttl_s: 0 })
    expect((out as { idle_ttl_s: number }).idle_ttl_s).toBe(0)
    expect(asked[1]).toEqual({ method: 'POST', path: '/v1/tools/fishaudio/drama-3/settings', body: { idle_ttl_s: 0 }, auth: 'Bearer tok' })
  })

  it('a 409 is a SparkError with the box’s message, shown as it is', async () => {
    const { spark } = box(() => ({ status: 409, json: { error: { message: 'always-loaded models would need 136.5 GiB of a 77.8 GiB budget' } } }))
    const error = await spark.toolSettings('fishaudio/drama-3', { residency: 'always' }).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(SparkError)
    expect((error as SparkError).status).toBe(409)
    expect(settingsRefusal(error, server)).toEqual({ message: 'always-loaded models would need 136.5 GiB of a 77.8 GiB budget', denied: false })
  })

  it('a refusal for the scope makes the card read-only; an older box and a bad value say so', () => {
    expect(settingsRefusal(new SparkError(403, 'needs manage'), server)).toEqual({ message: NO_MANAGE, denied: true })
    expect(settingsRefusal(new SparkError(404, 'no such route'), server).message).toBe(oldBoxModels('studio'))
    expect(settingsRefusal(new SparkError(400, 'idle_ttl_s out of range'), server).message).toBe('Refused by studio: idle_ttl_s out of range')
    expect(settingsRefusal(new Error('timeout'), server).message).toBe('Not changed on studio: timeout')
  })
})

describe('the Models card', () => {
  const names = modelNames([{ id: 'qwen3.8-27b', name: 'Qwen3.8 27B (TensorFold 1.1.4 Metal, MLX 4-bit + DFlash2 drafter)' }, { id: 'fishaudio/drama-3', name: 'Fish Audio Drama 3 (TensorFold native on Metal, 44.1 kHz). Built with Fish Audio' }, { id: 'qwen-image-2.1-turbo' }, 7])

  it('one row per model: name, kind, state, memory, load time, and only the residencies it allows', () => {
    const v = modelsView({ tools: toolsOf(body), names, manage: true, server, at: 1 })
    expect(v.read_only).toBe('')
    const [q, t, i] = v.rows
    expect(q).toMatchObject({ name: 'Qwen3.8 27B', kind: 'Chat', dot: 'ready', state: 'ready', memory: '26.8 GiB', loads: 'loads in ~5 s', residency: 'on_demand', idle_shown: true, idle_ttl_s: 300, is_default: false })
    expect(q.choices.map((c) => c.label)).toEqual(['Always loaded', 'Load when a call starts', 'Load when needed', 'Disabled (never runs)'])
    expect(q.default_text).toBe('Default: always loaded')
    expect(t).toMatchObject({ name: 'Fish Audio Drama 3', kind: 'Speech', dot: 'offline', state: 'not loaded', memory: '28.8 GiB', loads: 'load time not measured yet', is_default: true })
    expect(t.default_text).toBe('Default: load when a call starts, unloads right after each use')
    expect(i).toMatchObject({ name: 'qwen-image-2.1-turbo', kind: 'Pictures', memory: '9.5 GiB', dot: 'failed', state: 'failed', error: 'out of memory' })
    expect(i.choices.map((c) => c.value)).toEqual(['always', 'on_demand', 'disabled'])
  })

  it('disabled: offered for every model, dimmed with no idle field; a voice-loop model warns that calls need it', () => {
    const tools = toolsOf({ data: [{ ...chat, residency: 'disabled', allowed_residency: ['always', 'call', 'on_demand', 'disabled'] }, { ...image, residency: 'disabled', residency_default: 'disabled' }, { ...speech, residency: 'disabled' }] })
    const [c, pic, sp] = modelsView({ tools, names, manage: true, server, at: 1 }).rows
    expect(c).toMatchObject({ disabled: true, dot: 'disabled', state: 'disabled', idle_shown: false, warning: 'Voice calls that rely on this model won’t work while it’s disabled.', is_default: false })
    expect(c.choices.filter((x) => x.value === 'disabled')).toHaveLength(1)
    expect(pic).toMatchObject({ disabled: true, warning: '', is_default: true, default_text: 'Default: disabled (never runs)' })
    expect(sp.warning).not.toBe('')
    expect(toolEntryOf({ id: 'x', residency: 'disabled', allowed_residency: [] })?.allowed_residency).toEqual(['disabled'])
  })

  it('what blocks a call: every model of a voice kind disabled (another of the kind enabled, or a CUDA-only one, does not count)', () => {
    const whisper = { ...speech, id: 'whisper-large-v3-turbo', kind: 'transcription' }
    const lineup = (over: Record<string, object>) => toolsOf({ data: [chat, speech, whisper, image].map((e) => ({ ...e, ...(over[e.id] ?? {}) })) }).entries
    expect(voiceBlockers(lineup({}))).toEqual([])
    expect(voiceBlockers(lineup({ 'fishaudio/drama-3': { residency: 'disabled' }, 'qwen-image-2.1-turbo': { residency: 'disabled' } }))).toEqual(['fishaudio/drama-3'])
    expect(voiceBlockers([...lineup({ 'fishaudio/drama-3': { residency: 'disabled' } }), ...toolsOf({ data: [{ ...speech, id: 'other-tts', residency: 'call' }] }).entries])).toEqual([])
    expect(voiceBlockers([...lineup({ 'fishaudio/drama-3': { residency: 'disabled' } }), ...toolsOf({ data: [{ ...speech, id: 'cuda-tts', supported: false }] }).entries])).toEqual(['fishaudio/drama-3'])
  })

  it('only the models the server can run are listed; the rest are counted, with its reasons', () => {
    // The Spark's video model is CUDA-only; a second one without a reason is named as one the server cannot run.
    const video = { ...image, id: 'minimax-h3', kind: 'video', supported: false, unsupported_reason: 'needs NVIDIA CUDA' }
    const tools = toolsOf({ data: [chat, video, { ...chat, id: 'knowline-4b', kind: 'decision', supported: false }] })
    const v = modelsView({ tools, names, manage: true, server, at: 1 })
    expect(v.rows.map((r) => r.id)).toEqual(['qwen3.8-27b'])
    expect(v.unsupported_text).toBe('2 models not available on studio')
    expect(v.unsupported).toEqual([{ id: 'minimax-h3', name: 'minimax-h3', kind: 'Video', reason: 'needs NVIDIA CUDA' }, { id: 'knowline-4b', name: 'knowline-4b', kind: 'Decisions', reason: 'studio cannot run it' }])
    expect(modelsView({ tools: toolsOf({ data: [chat, video] }), names, manage: true, server, at: 1 }).unsupported_text).toBe('1 model not available on studio')
    expect(modelsView({ tools: toolsOf(body), names, manage: true, server, at: 1 }).unsupported_text).toBe('')
  })

  it('an always-loaded model has no idle field and is default whatever its idle time', () => {
    const v = modelsView({ tools: toolsOf({ data: [{ ...chat, residency: 'always', idle_ttl_s: 5 }] }), names, manage: true, server, at: 1 })
    expect(v.rows[0]).toMatchObject({ idle_shown: false, is_default: true })
  })

  it('the memory bar: always loaded, then what is loaded on demand, of the ceiling', () => {
    expect(budgetView(toolsOf(body).budget)).toEqual({ always_pct: 34.4, booked_pct: 12.8, text: 'Always loaded 26.8 GiB · in use now 36.7 GiB · ceiling 77.8 GiB', over: false })
    expect(budgetView({ ceiling_bytes: 10, booked_bytes: 12, always_bytes: 4 })).toMatchObject({ always_pct: 40, booked_pct: 60, over: true })
    expect(toolsOf({ data: [], budget: { ceiling_bytes: 0 } }).budget).toBeNull()
  })

  it('read-only without manage; an older box (keep_loaded, no residency) reads as always/on demand and says to update', () => {
    expect(modelsView({ tools: toolsOf(body), names, manage: false, server, at: 1 }).read_only).toBe(NO_MANAGE)
    const old = toolsOf({ data: [{ id: 'qwen3.8-27b', kind: 'chat', state: 'ready', keep_loaded: true, idle_ttl_s: 600 }, { id: 'qwen-image-2.1-turbo', kind: 'image', state: 'unloaded', keep_loaded: false }] })
    expect(old.residency).toBe(false)
    expect(old.entries.map((e) => [e.residency, e.allowed_residency])).toEqual([['always', ['always', 'on_demand', 'disabled']], ['on_demand', ['always', 'on_demand', 'disabled']]])
    expect(modelsView({ tools: old, names, manage: true, server, at: 1 }).read_only).toBe(oldBoxModels('studio'))
    expect(modelsView({ tools: null, names, manage: true, server, problem: 'offline', at: 1 })).toMatchObject({ available: false, problem: 'offline', rows: [] })
  })

  it('names the server by its machine name, or neutrally when it gives none', () => {
    expect(serverName('  ')).toBe('this LocalRouter server')
    expect(serverName(null)).toBe('this LocalRouter server')
    expect(modelsView({ tools: toolsOf(body), names, manage: true, server, at: 1 }).intro).toBe(modelsIntro('studio'))
    expect(modelsIntro('studio')).toMatch(/^Which models stay loaded on studio and/)
  })

  it('sizes from list_models: a model not loaded holds 0 bytes in /v1/tools but needs its size', () => {
    const listed = listedModels({ content: [{ type: 'text', text: JSON.stringify({ models: [{ id: 'fishaudio/drama-3', name: 'Fish Audio Drama 3 (TensorFold native on Metal, 44.1 kHz). Built with Fish Audio', needs_bytes: 30870077440 }, { id: 'x', needs_bytes: 0 }], machine: 'studio' }) }] })
    expect(listed).toEqual({ names: { 'fishaudio/drama-3': 'Fish Audio Drama 3', x: 'x' }, needs: { 'fishaudio/drama-3': 30870077440 }, machine: 'studio' })
    const v = modelsView({ tools: toolsOf({ data: [{ ...speech, resident_bytes: 0 }] }), names: listed.names, needs: listed.needs, manage: true, server, at: 1 })
    expect(v.rows[0].memory).toBe('28.8 GiB')
    expect(listedModels({ content: [{ type: 'text', text: 'not json' }] })).toEqual({ names: {}, needs: {}, machine: '' })
    expect(listedModels(null)).toEqual({ names: {}, needs: {}, machine: '' })
  })

  it('load times', () => {
    expect([loadsIn(null), loadsIn(400), loadsIn(1500), loadsIn(61_000)]).toEqual(['load time not measured yet', 'loads in under 1 s', 'loads in ~2 s', 'loads in ~61 s'])
  })

  it('what may be sent: a residency, whole seconds from 0, or a reset', () => {
    expect(settingsChange({ residency: 'call' })).toEqual({ residency: 'call' })
    expect(settingsChange({ residency: 'disabled' })).toEqual({ residency: 'disabled' })
    expect(settingsChange({ idle_ttl_s: ' 0 ' })).toEqual({ idle_ttl_s: 0 })
    expect(settingsChange({ idle_ttl_s: 90, residency: 'on_demand' })).toEqual({ residency: 'on_demand', idle_ttl_s: 90 })
    expect(settingsChange({ reset: true, residency: 'call' })).toEqual({ reset: true })
    for (const bad of [{ residency: 'sometimes' }, { idle_ttl_s: '-1' }, { idle_ttl_s: '1.5' }, { idle_ttl_s: '' }, { idle_ttl_s: 10 ** 9 }, {}, null]) expect(() => settingsChange(bad)).toThrow()
  })
})

describe('a call’s voice models loading', () => {
  const names = { 'fishaudio/drama-3': 'Fish Audio Drama 3', 'whisper-large-v3-turbo': 'Whisper large-v3-turbo' }
  const D = 'fishaudio/drama-3'
  const W = 'whisper-large-v3-turbo'

  it('a bar when every model has a last load time; ready ones count as done', () => {
    const v = callLoading({ type: 'localrouter.session.loading', models: [{ id: D, state: 'loading', elapsed_ms: 5000 }, { id: W, state: 'ready', elapsed_ms: 2300 }] }, { [D]: 12500, [W]: null }, names)
    expect(v).toEqual({ done: false, text: 'Loading voice models…', detail: 'Fish Audio Drama 3: 5 s of ~13 s · Whisper large-v3-turbo: ready', progress: 0.7 })
  })

  it('a model never measured: no bar, the seconds so far; a slow load never shows full', () => {
    expect(callLoading({ type: 'localrouter.session.loading', models: [{ id: D, state: 'loading', elapsed_ms: 3000 }] }, {}, names)).toMatchObject({ progress: null, detail: 'Fish Audio Drama 3: 3 s' })
    expect(callLoading({ type: 'localrouter.session.loading', models: [{ id: D, state: 'loading', elapsed_ms: 30_000 }] }, { [D]: 12500 }, names)?.progress).toBe(0.95)
  })

  it('a disabled voice model stops the banner with what to do, in loading or ready', () => {
    const v = callLoading({ type: 'localrouter.session.loading', models: [{ id: D, state: 'disabled', elapsed_ms: 0 }, { id: W, state: 'loading', elapsed_ms: 900 }] }, { [D]: 12500 }, names, 'studio')
    expect(v).toEqual({ done: false, stopped: true, progress: null, text: 'A voice model is disabled', detail: 'Fish Audio Drama 3 is disabled on studio, so this call can’t use it. Enable it in Settings › Models.' })
    expect(callLoading({ type: 'localrouter.session.ready', load: 'loaded', models: [{ id: D, state: 'disabled', error: 'disabled in settings' }, { id: W, state: 'disabled' }] }, {}, names)?.detail).toBe('Fish Audio Drama 3 and Whisper large-v3-turbo are disabled on this LocalRouter server, so this call can’t use them. Enable them in Settings › Models.')
  })

  it('a default voice model disabled before the call (not in the server’s events) stops it too, even when nothing loads', () => {
    expect(callLoading({ type: 'localrouter.session.ready', load: 'none', models: null }, {}, names, 'studio', [D])).toEqual({ done: false, stopped: true, progress: null, text: 'A voice model is disabled', detail: 'Fish Audio Drama 3 is disabled on studio, so this call can’t use it. Enable it in Settings › Models.' })
    expect(callLoading({ type: 'localrouter.session.loading', models: [{ id: W, state: 'loading', elapsed_ms: 900 }] }, {}, names, 'studio', [D])?.stopped).toBe(true)
    expect(callLoading({ type: 'localrouter.session.ready', load: 'loaded', models: [{ id: D, state: 'disabled' }] }, {}, names, 'studio', [D])?.detail).toMatch(/^Fish Audio Drama 3 is disabled/)
  })

  it('the server’s states while a call waits: queued, warming up, does not fit', () => {
    const v = callLoading({ type: 'localrouter.session.loading', models: [{ id: D, state: 'warming', elapsed_ms: 13000 }, { id: W, state: 'queued', elapsed_ms: 200 }] }, { [D]: 12500, [W]: 2300 }, names)
    expect(v).toEqual({ done: false, text: 'Loading voice models…', detail: 'Fish Audio Drama 3: warming up · Whisper large-v3-turbo: waiting', progress: null })
    expect(callLoading({ type: 'localrouter.session.loading', models: [{ id: D, state: 'warming' }, { id: W, state: 'ready' }] }, {}, names)?.progress).toBe(0.98)
  })

  it('ready: loaded ends it; failed says why and stops; nothing loaded shows nothing; other events are not its business', () => {
    expect(callLoading({ type: 'localrouter.session.ready', load: 'loaded', loaded_ms: 7300 }, {}, names)).toEqual({ done: true, text: 'Voice models ready', detail: 'loaded in 7.3 s', progress: 1 })
    expect(callLoading({ type: 'localrouter.session.ready', load: 'none', loaded_ms: 0, models: null }, {}, names)).toBeNull()
    expect(callLoading({ type: 'localrouter.session.ready', load: 'failed', message: 'fishaudio/drama-3 does not fit beside the models kept loaded', models: [{ id: D, state: 'no_room' }] }, {}, names)).toEqual({ done: false, stopped: true, progress: null, text: 'Voice models didn’t load', detail: 'fishaudio/drama-3 does not fit beside the models kept loaded' })
    expect(callLoading({ type: 'localrouter.session.ready', load: 'failed', models: [{ id: D, state: 'failed', error: 'weights missing' }, { id: W, state: 'no_room' }] }, {}, names)?.detail).toBe('Fish Audio Drama 3: weights missing and Whisper large-v3-turbo: does not fit')
    expect(callLoading({ type: 'response.done' }, {}, names)).toBeNull()
  })
})
