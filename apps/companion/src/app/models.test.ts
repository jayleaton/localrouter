import { describe, expect, it } from 'vitest'
import { Spark, SparkError } from '../harness/spark.ts'
import { budgetView, callLoading, loadsIn, modelNames, modelsIntro, modelsView, NO_MANAGE, oldBoxModels, serverName, settingsChange, settingsRefusal, toolsOf } from './models.ts'

const server = serverName('studio')

const GiB = 2 ** 30
const qwen = { id: 'qwen', kind: 'chat', state: 'ready', resident_bytes: 20 * GiB, priority: 0, residency: 'on_demand', residency_default: 'always', idle_ttl_s: 300, idle_ttl_s_default: 600, allowed_residency: ['always', 'call', 'on_demand'], last_load_ms: 8400, loaded_since: '2026-10-31T10:00:00Z', last_error: '' }
const tts = { ...qwen, id: 'tts', kind: 'speech', state: 'unloaded', resident_bytes: 3 * GiB, residency: 'call', residency_default: 'call', idle_ttl_s: 0, idle_ttl_s_default: 0, last_load_ms: null, loaded_since: null }
const img = { ...qwen, id: 'img', kind: 'image', state: 'unloaded', residency: 'on_demand', residency_default: 'on_demand', idle_ttl_s: 60, idle_ttl_s_default: 60, allowed_residency: ['always', 'on_demand'], last_error: 'out of memory' }
const body = { object: 'list', data: [qwen, tts, img], budget: { ceiling_bytes: 100 * GiB, booked_bytes: 30 * GiB, always_bytes: 20 * GiB } }

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

describe('the Spark client for model settings', () => {
  it('reads /v1/tools and posts one model’s settings with the token', async () => {
    const { spark, asked } = box((m, _p, b) => (m === 'GET' ? { status: 200, json: body } : { status: 200, json: { ...tts, ...(b as object) } }))
    expect(toolsOf(await spark.tools()).entries.map((e) => e.id)).toEqual(['qwen', 'tts', 'img'])
    const out = await spark.toolSettings('a b/c', { idle_ttl_s: 0 })
    expect((out as { idle_ttl_s: number }).idle_ttl_s).toBe(0)
    expect(asked[1]).toEqual({ method: 'POST', path: '/v1/tools/a%20b%2Fc/settings', body: { idle_ttl_s: 0 }, auth: 'Bearer tok' })
  })

  it('a 409 is a SparkError with the box’s message, shown as it is', async () => {
    const { spark } = box(() => ({ status: 409, json: { error: { message: 'always-loaded models would need 120.0 GiB of a 100.0 GiB budget' } } }))
    const error = await spark.toolSettings('qwen', { residency: 'always' }).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(SparkError)
    expect((error as SparkError).status).toBe(409)
    expect(settingsRefusal(error, server)).toEqual({ message: 'always-loaded models would need 120.0 GiB of a 100.0 GiB budget', denied: false })
  })

  it('a refusal for the scope makes the card read-only; an older box and a bad value say so', () => {
    expect(settingsRefusal(new SparkError(403, 'needs manage'), server)).toEqual({ message: NO_MANAGE, denied: true })
    expect(settingsRefusal(new SparkError(404, 'no such route'), server).message).toBe(oldBoxModels('studio'))
    expect(settingsRefusal(new SparkError(400, 'idle_ttl_s out of range'), server).message).toBe('Refused by studio: idle_ttl_s out of range')
    expect(settingsRefusal(new Error('timeout'), server).message).toBe('Not changed on studio: timeout')
  })
})

describe('the Models card', () => {
  const names = modelNames([{ id: 'qwen', name: 'Qwen3 32B (NVFP4, 4-bit)' }, { id: 'tts' }, 7])

  it('one row per model: name, kind, state, memory, load time, and only the residencies it allows', () => {
    const v = modelsView({ tools: toolsOf(body), names, manage: true, server, at: 1 })
    expect(v.read_only).toBe('')
    const [q, t, i] = v.rows
    expect(q).toMatchObject({ name: 'Qwen3 32B', kind: 'Chat', dot: 'ready', state: 'ready', memory: '20.0 GiB', loads: 'loads in ~8 s', residency: 'on_demand', idle_shown: true, idle_ttl_s: 300, is_default: false })
    expect(q.choices.map((c) => c.label)).toEqual(['Always loaded', 'Load when a call starts', 'Load when needed'])
    expect(q.default_text).toBe('Default: always loaded')
    expect(t).toMatchObject({ name: 'tts', kind: 'Speech', dot: 'offline', state: 'not loaded', loads: 'load time not measured yet', is_default: true })
    expect(t.default_text).toBe('Default: load when a call starts, unloads right after each use')
    expect(i).toMatchObject({ dot: 'failed', state: 'failed', error: 'out of memory' })
    expect(i.choices.map((c) => c.value)).toEqual(['always', 'on_demand'])
  })

  it('an always-loaded model has no idle field and is default whatever its idle time', () => {
    const v = modelsView({ tools: toolsOf({ data: [{ ...qwen, residency: 'always', idle_ttl_s: 5 }] }), names, manage: true, server, at: 1 })
    expect(v.rows[0]).toMatchObject({ idle_shown: false, is_default: true })
  })

  it('the memory bar: always loaded, then what is loaded on demand, of the ceiling', () => {
    expect(budgetView(toolsOf(body).budget)).toEqual({ always_pct: 20, booked_pct: 10, text: 'Always loaded 20.0 GiB · in use now 30.0 GiB · ceiling 100.0 GiB', over: false })
    expect(budgetView({ ceiling_bytes: 10, booked_bytes: 12, always_bytes: 4 })).toMatchObject({ always_pct: 40, booked_pct: 60, over: true })
    expect(toolsOf({ data: [], budget: { ceiling_bytes: 0 } }).budget).toBeNull()
  })

  it('read-only without manage; an older box (keep_loaded, no residency) reads as always/on demand and says to update', () => {
    expect(modelsView({ tools: toolsOf(body), names, manage: false, server, at: 1 }).read_only).toBe(NO_MANAGE)
    const old = toolsOf({ data: [{ id: 'q', kind: 'chat', state: 'ready', keep_loaded: true, idle_ttl_s: 0 }, { id: 'v', kind: 'video', state: 'unloaded', keep_loaded: false }] })
    expect(old.residency).toBe(false)
    expect(old.entries.map((e) => [e.residency, e.allowed_residency])).toEqual([['always', ['always', 'on_demand']], ['on_demand', ['always', 'on_demand']]])
    expect(modelsView({ tools: old, names, manage: true, server, at: 1 }).read_only).toBe(oldBoxModels('studio'))
    expect(modelsView({ tools: null, names, manage: true, server, problem: 'offline', at: 1 })).toMatchObject({ available: false, problem: 'offline', rows: [] })
  })

  it('names the server by its machine name, or neutrally when it gives none', () => {
    expect(serverName('  ')).toBe('this LocalRouter server')
    expect(serverName(null)).toBe('this LocalRouter server')
    expect(modelsView({ tools: toolsOf(body), names, manage: true, server, at: 1 }).intro).toBe(modelsIntro('studio'))
    expect(modelsIntro('studio')).toMatch(/^Which models stay loaded on studio and/)
  })

  it('load times', () => {
    expect([loadsIn(null), loadsIn(400), loadsIn(1500), loadsIn(61_000)]).toEqual(['load time not measured yet', 'loads in under 1 s', 'loads in ~2 s', 'loads in ~61 s'])
  })

  it('what may be sent: a residency, whole seconds from 0, or a reset', () => {
    expect(settingsChange({ residency: 'call' })).toEqual({ residency: 'call' })
    expect(settingsChange({ idle_ttl_s: ' 0 ' })).toEqual({ idle_ttl_s: 0 })
    expect(settingsChange({ idle_ttl_s: 90, residency: 'on_demand' })).toEqual({ residency: 'on_demand', idle_ttl_s: 90 })
    expect(settingsChange({ reset: true, residency: 'call' })).toEqual({ reset: true })
    for (const bad of [{ residency: 'sometimes' }, { idle_ttl_s: '-1' }, { idle_ttl_s: '1.5' }, { idle_ttl_s: '' }, { idle_ttl_s: 10 ** 9 }, {}, null]) expect(() => settingsChange(bad)).toThrow()
  })
})

describe('a call’s voice models loading', () => {
  const names = { tts: 'Voice', stt: 'Ears' }

  it('a bar when every model has a last load time; ready ones count as done', () => {
    const v = callLoading({ type: 'localrouter.session.loading', models: [{ id: 'tts', state: 'loading', elapsed_ms: 2000 }, { id: 'stt', state: 'ready', elapsed_ms: 900 }] }, { tts: 8000, stt: null }, names)
    expect(v).toEqual({ done: false, text: 'Loading voice models…', detail: 'Voice 2 s of ~8 s · Ears ready', progress: 0.63 })
  })

  it('a model never measured: no bar, the seconds so far; a slow load never shows full', () => {
    expect(callLoading({ type: 'localrouter.session.loading', models: [{ id: 'tts', state: 'loading', elapsed_ms: 3000 }] }, {}, names)).toMatchObject({ progress: null, detail: 'Voice 3 s' })
    expect(callLoading({ type: 'localrouter.session.loading', models: [{ id: 'tts', state: 'loading', elapsed_ms: 30_000 }] }, { tts: 1000 }, names)?.progress).toBe(0.95)
  })

  it('ready ends it; other events are not its business', () => {
    expect(callLoading({ type: 'localrouter.session.ready', loaded_ms: 7300 }, {}, names)).toEqual({ done: true, text: 'Voice models ready', detail: 'loaded in 7.3 s', progress: 1 })
    expect(callLoading({ type: 'response.done' }, {}, names)).toBeNull()
  })
})
