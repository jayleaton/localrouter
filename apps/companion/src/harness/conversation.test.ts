import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { Journal } from './journal.ts'
import { restoreVoice } from './models.ts'
import { ClientToolsSession, pageView } from './realtime.ts'
import type { SocketLike } from './realtime.ts'
import { Registry } from './registry.ts'
import type { Tool, ToolEvent } from './registry.ts'
import { voiceRegistry } from './runtime.ts'
import { Spark } from './spark.ts'
import { PROPOSAL_TTL_MS, saidYes, VideoHandoff, VIDEO_QUESTION } from './tools.ts'

const dirs: string[] = []
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
})

function clock(start = 1_000_000) {
  const c = { t: start, now: () => c.t }
  return c
}

describe('a video asked for on a call never starts on the call', () => {
  const ctx = (utterance: string, utteranceAt: number, events: ToolEvent[] = []) => ({ callId: 'call_v', utterance, utteranceAt, signal: new AbortController().signal, emit: (e: ToolEvent) => void events.push(e) })

  it('generate_video on a call only records it and returns the question; nothing is made', async () => {
    const c = clock()
    const h = new VideoHandoff(c.now)
    const [video] = h.tools()
    const out = (await video.run({ prompt: 'waves at dusk', seconds: 5 }, ctx('make me a video of waves', c.t - 1000))) as Record<string, unknown>
    expect(out).toMatchObject({ status: 'needs_confirmation', made: false, ask: VIDEO_QUESTION })
    expect(h.proposal).toMatchObject({ prompt: 'waves at dusk', seconds: 5 })
  })

  it('ending the call needs a proposal, an answer after it, and a clear yes', async () => {
    const c = clock()
    const h = new VideoHandoff(c.now)
    const [video, end] = h.tools()
    const events: ToolEvent[] = []
    expect(await end.run({}, ctx('yes', c.t, events))).toMatchObject({ status: 'refused', reason: expect.stringMatching(/no video waiting/) })
    await video.run({ prompt: 'waves' }, ctx('make a video', c.t - 5))
    // The model calls it in the same turn, before the person answered: refused.
    expect(await end.run({}, ctx('make a video', c.t - 5, events))).toMatchObject({ status: 'refused', reason: expect.stringMatching(/not answered/) })
    c.t += 4000
    expect(await end.run({}, ctx('yes please, go ahead', c.t, events))).toMatchObject({ status: 'ending_call' })
    expect(events).toEqual([{ type: 'handoff', call_id: 'call_v', kind: 'video', prompt: 'waves', seconds: null }])
    // Used once.
    expect(await end.run({}, ctx('yes', c.t + 1, events))).toMatchObject({ status: 'refused' })
    expect(events).toHaveLength(1)
  })

  it('a no (or a hedge) keeps the call and makes nothing; the proposal is dropped', async () => {
    const c = clock()
    const h = new VideoHandoff(c.now)
    const [video, end] = h.tools()
    const events: ToolEvent[] = []
    await video.run({ prompt: 'waves' }, ctx('video', c.t))
    c.t += 3000
    expect(await end.run({}, ctx('no, not now', c.t, events))).toMatchObject({ status: 'refused', reason: expect.stringMatching(/did not say yes/) })
    expect(h.proposal).toBeNull()
    expect(events).toEqual([])
  })

  it('a proposal goes stale', async () => {
    const c = clock()
    const h = new VideoHandoff(c.now)
    const [video, end] = h.tools()
    await video.run({ prompt: 'waves' }, ctx('video', c.t))
    c.t += PROPOSAL_TTL_MS + 1
    expect(await end.run({}, ctx('yes', c.t))).toMatchObject({ status: 'refused' })
  })

  it('what counts as yes', () => {
    for (const y of ['yes', 'Yeah, do it', 'sure', 'okay go ahead', 'Yes please end the call', 'no problem, go ahead']) expect(saidYes(y)).toBe(true)
    for (const n of ['no', 'nope', 'not now', 'yes but not now', "don't", 'wait', 'keep the call going', 'what?', '']) expect(saidYes(n)).toBe(false)
  })

  it('a call offers pictures and the handoff, never a real video tool', () => {
    const noop: Tool = { name: 'generate_image', description: '', parameters: { type: 'object', properties: {} }, run: async () => ({}) }
    const r = voiceRegistry([], [noop], new VideoHandoff())
    expect(r.names).toEqual(['generate_image', 'generate_video', 'end_call_and_make_video'])
    // The page draws a job card only for a picture; a video "call" on a call is a proposal, not a job.
    expect(pageView({ type: 'response.output_item.done', item: { type: 'function_call', name: 'generate_image' } })).toBe(true)
    expect(pageView({ type: 'response.output_item.done', item: { type: 'function_call', name: 'generate_video' } })).toBe(false)
  })
})

/** A box at the other end of /v1/realtime, as in harness.test.ts. */
function fakeBox() {
  const sent: Array<Record<string, unknown>> = []
  let socket!: SocketLike & { server(ev: Record<string, unknown>): void }
  const dial = () => {
    socket = {
      readyState: 1, onopen: null, onmessage: null, onclose: null, onerror: null,
      send(data: string) {
        const ev = JSON.parse(data) as Record<string, unknown>
        sent.push(ev)
        if (ev.type === 'session.update') this.server({ type: 'session.updated', session: { id: 's', localrouter: { tool_execution: 'client' } } })
      },
      close() { this.onclose?.({ code: 1000, reason: '' }) },
      server(ev: Record<string, unknown>) { queueMicrotask(() => this.onmessage?.({ data: JSON.stringify(ev) })) },
    }
    queueMicrotask(() => socket.onopen?.({}))
    return socket
  }
  return { dial, sent, box: (ev: Record<string, unknown>) => socket.server(ev) }
}
const flush = () => new Promise((r) => setTimeout(r, 30))

describe('a call’s tools reach the conversation, and the handoff reaches the page', () => {
  it('a local tool run on a call is reported to the page with its call and result', async () => {
    const box = fakeBox()
    const events: Array<Record<string, unknown>> = []
    const list: Tool = { name: 'list_files', description: '', parameters: { type: 'object', properties: {} }, run: async () => ({ entries: [] }) }
    const s = new ClientToolsSession({ spark: new Spark('http://127.0.0.1:8190', ''), registry: new Registry().add(list), dial: box.dial, onEvent: (e) => void events.push(e) })
    await s.connect()
    box.box({ type: 'response.function_call_arguments.done', call_id: 'c1', name: 'list_files', arguments: '{}', response_id: 'r1' })
    box.box({ type: 'response.done', response: { id: 'r1', output: [] } })
    await flush()
    expect(events.find((e) => e.type === 'localrouter.client_tool.done')).toMatchObject({ call_id: 'c1', name: 'list_files', ok: true, output: '{"entries":[]}' })
  })

  it('a spoken yes after the question ends the call through the handoff event, once', async () => {
    const box = fakeBox()
    const events: Array<Record<string, unknown>> = []
    const s = new ClientToolsSession({ spark: new Spark('http://127.0.0.1:8190', ''), registry: voiceRegistry([], [], new VideoHandoff()), dial: box.dial, onEvent: (e) => void events.push(e) })
    await s.connect()
    box.box({ type: 'conversation.item.input_audio_transcription.completed', transcript: 'make a video of waves' })
    box.box({ type: 'response.function_call_arguments.done', call_id: 'v1', name: 'generate_video', arguments: '{"prompt":"waves"}', response_id: 'r1' })
    box.box({ type: 'response.done', response: { id: 'r1', output: [] } })
    await flush()
    box.box({ type: 'response.function_call_arguments.done', call_id: 'v2', name: 'end_call_and_make_video', arguments: '{}', response_id: 'r2' })
    box.box({ type: 'response.done', response: { id: 'r2', output: [] } })
    await flush()
    expect(events.some((e) => e.type === 'localrouter.companion.handoff')).toBe(false) // nobody answered yet
    await new Promise((r) => setTimeout(r, 5))
    box.box({ type: 'conversation.item.input_audio_transcription.completed', transcript: 'yes, go ahead' })
    box.box({ type: 'response.function_call_arguments.done', call_id: 'v3', name: 'end_call_and_make_video', arguments: '{}', response_id: 'r3' })
    box.box({ type: 'response.done', response: { id: 'r3', output: [] } })
    await flush()
    expect(events.filter((e) => e.type === 'localrouter.companion.handoff')).toEqual([{ type: 'localrouter.companion.handoff', call_id: 'v3', kind: 'video', prompt: 'waves', seconds: null }])
  })
})

describe('one conversation in the journal, typed and spoken', () => {
  it('mirrors each finished message once, tool calls with results, across restarts', () => {
    const d = mkdtempSync(join(tmpdir(), 'lr-conv-'))
    dirs.push(d)
    const items = [
      { id: 'm1', role: 'user' as const, via: 'voice' as const, text: 'what is in my folder?' },
      { id: 'm2', role: 'tool' as const, via: 'voice' as const, call: { call_id: 'c1', name: 'list_files', arguments: '{}' }, ok: true, output: '{"entries":[]}' },
      { id: 'm3', role: 'assistant' as const, via: 'voice' as const, text: 'It is empty.' },
    ]
    const j = new Journal(d)
    expect(j.mirror('c_page/1', items)).toBe(3)
    expect(j.mirror('c_page/1', items)).toBe(0)
    const again = new Journal(d) // the app restarted
    expect(again.mirror('c_page/1', [...items, { id: 'm4', role: 'user', via: 'text', text: 'and now?' }])).toBe(1)
    const x = again.export(Journal.pageConversation('c_page/1'))
    expect(x.messages.map((m) => m.role)).toEqual(['user', 'tool_call', 'tool_result', 'assistant', 'user'])
  })
})

describe('the voice models come back after a video', () => {
  it('asks the box to load the voice models it keeps loaded that are not ready, and only those', async () => {
    const calls: Array<{ url: string; body: string | null }> = []
    const f = (async (url: string, init?: { body?: string }) => {
      calls.push({ url, body: init?.body ?? null })
      if (url.endsWith('/v1/tools')) return new Response(JSON.stringify({ data: [{ id: 'drama', kind: 'speech', state: 'unloaded', keep_loaded: true }, { id: 'whisper', kind: 'transcription', state: 'ready', keep_loaded: true }, { id: 'video', kind: 'video', state: 'ready' }] }))
      return new Response(JSON.stringify({ jsonrpc: '2.0', id: 1, result: { content: [{ type: 'text', text: '{"message":"loaded"}' }] } }))
    }) as unknown as typeof fetch
    const r = await restoreVoice(new Spark('http://127.0.0.1:8190', '', f))
    expect(r).toEqual({ ok: true, text: 'loaded again (drama)' })
    const loads = calls.filter((c) => c.body?.includes('load_model'))
    expect(loads).toHaveLength(1)
    expect(loads[0].body).toContain('"model":"drama"')
  })
})

describe('pictures: a call edits the last one unless it names another', () => {
  it('edit_image takes the last picture made in this conversation', async () => {
    const asked: Array<{ name: string; args: Record<string, unknown> }> = []
    const f = (async (_url: string, init?: { body?: string }) => {
      const rpc = JSON.parse(init?.body ?? '{}') as { params: { name: string; arguments: Record<string, unknown> } }
      asked.push({ name: rpc.params.name, args: rpc.params.arguments })
      const n = asked.length
      return new Response(JSON.stringify({ jsonrpc: '2.0', id: 1, result: { content: [{ type: 'text', text: JSON.stringify({ id: `img_${n}`, status: 'completed', files: [`https://box/v1/files/img_${n}/0.png`] }) }] } }))
    }) as unknown as typeof fetch
    const { pictureTools } = await import('./tools.ts')
    const [make, edit] = pictureTools(new Spark('http://127.0.0.1:8190', '', f))
    const ctx = { callId: 'c', utterance: '', signal: new AbortController().signal }
    await expect(edit.run({ prompt: 'make it night' }, ctx)).rejects.toThrow(/no picture to change/)
    await make.run({ prompt: 'a lighthouse' }, ctx)
    const out = (await edit.run({ prompt: 'make it night' }, ctx)) as { urls: string[] }
    expect(asked[1]).toMatchObject({ name: 'edit_image', args: { prompt: 'make it night', image: 'https://box/v1/files/img_1/0.png' } })
    expect(out.urls).toEqual(['https://box/v1/files/img_2/0.png'])
    await edit.run({ prompt: 'add snow', image: 'https://box/v1/files/img_1/0.png' }, ctx)
    expect(asked[2].args.image).toBe('https://box/v1/files/img_1/0.png')
  })
})
