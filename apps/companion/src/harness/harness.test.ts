import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { Journal } from './journal.ts'
import { ClientToolsSession, pageView, RealtimeRefused } from './realtime.ts'
import type { SocketLike } from './realtime.ts'
import { check, Registry } from './registry.ts'
import type { Tool } from './registry.ts'
import { isPrivateHost, normalizeBase, Spark } from './spark.ts'
import { fileTools, inside } from './tools.ts'

const dirs: string[] = []
const temp = () => {
  const d = mkdtempSync(join(tmpdir(), 'lr-harness-test-'))
  dirs.push(d)
  return d
}
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
})

describe('the Spark address', () => {
  it('is https, or http only to loopback or the tailnet; never with credentials', () => {
    expect(normalizeBase('jarvis.tail1234.ts.net:10443')).toBe('https://jarvis.tail1234.ts.net:10443')
    expect(normalizeBase('http://100.101.1.2:8190/voice')).toBe('http://100.101.1.2:8190')
    expect(() => normalizeBase('http://192.168.1.116:8190')).toThrow(/tailnet/)
    expect(() => normalizeBase('https://u:p@box')).toThrow(/credentials/)
    expect(isPrivateHost('100.63.0.1')).toBe(false)
  })
  it('the realtime token rides in a subprotocol, never the URL', () => {
    const r = new Spark('https://box.ts.net:10443', 'secret').realtime()
    expect(r.url).toBe('wss://box.ts.net:10443/v1/realtime')
    expect(r.protocols).toEqual(['realtime', 'openai-insecure-api-key.secret'])
  })
})

describe('tool calls are checked before they run', () => {
  const echo: Tool = { name: 'echo', description: '', parameters: { type: 'object', properties: { n: { type: 'integer', minimum: 1, maximum: 3 }, s: { type: 'string', enum: ['a', 'b'] } }, required: ['n'], additionalProperties: false }, run: async (a) => a }
  it('rejects missing, unknown and mistyped arguments', () => {
    expect(check(echo.parameters, {})).toMatch(/n is required/)
    expect(check(echo.parameters, { n: 1, x: 1 })).toMatch(/no argument x/)
    expect(check(echo.parameters, { n: 1.5 })).toMatch(/whole number/)
    expect(check(echo.parameters, { n: 9 })).toMatch(/from 1 to 3/)
    expect(check(echo.parameters, { n: 1, s: 'c' })).toMatch(/one of/)
    expect(check(echo.parameters, { n: 2, s: 'a' })).toBeNull()
  })
  it('an unknown tool or bad JSON is an error output, not a crash', async () => {
    const r = new Registry().add(echo)
    const ctx = { callId: 'c', utterance: '', signal: new AbortController().signal }
    expect(await r.run('nope', '{}', ctx)).toMatchObject({ ok: false })
    expect(await r.run('echo', '{n:', ctx)).toMatchObject({ ok: false, output: JSON.stringify({ error: 'the arguments are not JSON' }) })
    expect(await r.run('echo', '{"n":2}', ctx)).toMatchObject({ ok: true, output: '{"n":2}' })
  })
})

describe('local files: only inside the folders the person shared', () => {
  it('reads and lists inside; refuses outside, including through links and ..', async () => {
    const d = temp()
    const shared = join(d, 'shared')
    mkdirSync(join(shared, 'notes'), { recursive: true })
    writeFileSync(join(shared, 'notes', 'todo.txt'), 'buy milk')
    writeFileSync(join(d, 'secret.txt'), 'no')
    symlinkSync(join(d, 'secret.txt'), join(shared, 'link.txt'))
    const [list, read] = fileTools(() => [shared])
    const ctx = { callId: 'c', utterance: '', signal: new AbortController().signal }
    expect(await read.run({ path: 'notes/todo.txt' }, ctx)).toMatchObject({ text: 'buy milk' })
    expect(((await list.run({ path: '' }, ctx)) as { entries: Array<{ name: string }> }).entries.map((e) => e.name)).toEqual(['link.txt', 'notes'])
    await expect(read.run({ path: '../secret.txt' }, ctx)).rejects.toThrow(/outside/)
    await expect(read.run({ path: 'link.txt' }, ctx)).rejects.toThrow(/outside/)
    await expect(read.run({ path: join(d, 'secret.txt') }, ctx)).rejects.toThrow(/outside/)
    await expect(inside([], 'x')).rejects.toThrow(/no folder is shared/)
  })
})

describe('the journal', () => {
  it('keeps the conversation and exports it neutrally', () => {
    const j = new Journal(temp())
    const id = j.create()
    j.append(id, { type: 'user', text: 'hi', via: 'voice' })
    j.append(id, { type: 'tool_call', call_id: 'c1', name: 'list_files', arguments: '{}' })
    j.append(id, { type: 'tool_result', call_id: 'c1', ok: true, output: '{"entries":[]}', ms: 3 })
    j.append(id, { type: 'assistant', text: 'nothing there' })
    expect(j.list()).toEqual([id])
    const x = j.export(id)
    expect(x.format).toBe('localrouter-conversation/1')
    expect(x.messages.map((m) => m.role)).toEqual(['user', 'tool_call', 'tool_result', 'assistant'])
    expect(() => j.append('../../etc', { type: 'note', text: 'x' })).toThrow()
  })
})

/** A box on the other end of /v1/realtime, scripted by the test. */
function fakeBox(mode: 'client' | 'server' = 'client') {
  const sent: Array<Record<string, unknown>> = []
  let socket!: SocketLike & { server(ev: Record<string, unknown>): void }
  const dial = () => {
    socket = {
      readyState: 1,
      onopen: null,
      onmessage: null,
      onclose: null,
      onerror: null,
      send(data: string) {
        const ev = JSON.parse(data) as Record<string, unknown>
        sent.push(ev)
        if (ev.type === 'session.update') this.server({ type: 'session.updated', session: { id: 'sess_1', localrouter: { tool_execution: mode }, tools: (ev.session as Record<string, unknown>).tools } })
      },
      close() {
        this.onclose?.({ code: 1000, reason: '' })
      },
      server(ev: Record<string, unknown>) {
        queueMicrotask(() => this.onmessage?.({ data: JSON.stringify(ev) }))
      },
    }
    queueMicrotask(() => socket.onopen?.({}))
    return socket
  }
  return { dial, sent, box: (ev: Record<string, unknown>) => socket.server(ev) }
}

const flush = () => new Promise((r) => setTimeout(r, 20))

/** A socket the box closes before it opens, as a refused upgrade looks to a WebSocket client (1006, no reason). */
const unopened = () => {
  const socket: SocketLike = { readyState: 0, onopen: null, onmessage: null, onclose: null, onerror: null, send() {}, close() {} }
  queueMicrotask(() => socket.onclose?.({ code: 1006, reason: '' }))
  return socket
}

describe('a realtime socket the box refuses', () => {
  const refusing = (why: string | null) => Object.assign(new Spark('http://100.64.0.2:8190', 't'), { realtimeRefusal: async () => why })
  const voiceOnly = 'the voice endpoints answer this machine only; the operator can open them with voice.remote in the config'

  it('fails once with the box’s reason and is not reported as a dropped connection to retry', async () => {
    const closes: string[] = []
    const s = new ClientToolsSession({ spark: refusing(voiceOnly), registry: new Registry(), dial: unopened, onClose: (r) => void closes.push(r) })
    const err = await s.connect().catch((e: unknown) => e)
    expect(err).toBeInstanceOf(RealtimeRefused)
    expect((err as Error).message).toBe(`the Spark refused the call: ${voiceOnly}`)
    await flush()
    expect(closes).toEqual([])
  })

  it('a socket that fails without a refusal (the box restarting) still closes for the caller to retry', async () => {
    const closes: string[] = []
    const s = new ClientToolsSession({ spark: refusing(null), registry: new Registry(), dial: unopened, onClose: (r) => void closes.push(r) })
    await expect(s.connect()).rejects.toThrow(/the realtime session closed \(1006\)/)
    await flush()
    expect(closes).toEqual(['1006'])
  })

  it('reads the refusal from a plain GET of the route', async () => {
    const answers: Record<string, Response> = {
      '403': new Response(JSON.stringify({ error: { message: voiceOnly, type: 'invalid_request_error' } }), { status: 403 }),
      '401': new Response(JSON.stringify({ error: { message: 'this device is not paired', type: 'invalid_request_error' } }), { status: 401 }),
      '400': new Response(JSON.stringify({ error: { message: 'not a WebSocket upgrade' } }), { status: 400 }),
    }
    const why = async (status: string) => new Spark('http://100.64.0.2:8190', 't', async () => answers[status] as Response).realtimeRefusal()
    expect(await why('403')).toBe(voiceOnly)
    expect(await why('401')).toBe('this device is not paired')
    expect(await why('400')).toBeNull()
    expect(await new Spark('http://100.64.0.2:8190', 't', async () => { throw new Error('ECONNREFUSED') }).realtimeRefusal()).toBeNull()
  })
})

describe('realtime in client tool mode', () => {
  const ran: string[] = []
  const tool: Tool = { name: 'get_time', description: 'the time', parameters: { type: 'object', properties: {}, additionalProperties: false }, run: async (_a, ctx) => { ran.push(ctx.callId); await new Promise((r) => setTimeout(r, 5)); if (ctx.signal.aborted) throw new Error('aborted'); return { time: '10:00' } } }
  const spark = new Spark('http://127.0.0.1:8190', 't')

  it('asks for client mode first and refuses a box that does not take it', async () => {
    const box = fakeBox('server')
    const s = new ClientToolsSession({ spark, registry: new Registry().add(tool), dial: box.dial })
    await expect(s.connect()).rejects.toThrow(/did not take client tool execution/)
    expect(box.sent[0]).toMatchObject({ type: 'session.update', session: { localrouter: { tool_execution: 'client' }, tools: [{ name: 'get_time' }] } })
  })

  it('runs a complete call once after its response, returns the output, asks for the continuation', async () => {
    ran.length = 0
    const box = fakeBox()
    const s = new ClientToolsSession({ spark, registry: new Registry().add(tool), dial: box.dial })
    await s.connect()
    box.box({ type: 'response.function_call_arguments.done', call_id: 'call_1', name: 'get_time', arguments: '{}', response_id: 'r1' })
    box.box({ type: 'response.function_call_arguments.done', call_id: 'call_1', name: 'get_time', arguments: '{}', response_id: 'r1' }) // a repeat
    await flush()
    expect(ran).toEqual([]) // not before the response ends
    box.box({ type: 'response.done', response: { id: 'r1', status: 'completed', output: [] } })
    await flush()
    expect(ran).toEqual(['call_1'])
    const outs = box.sent.filter((e) => e.type === 'conversation.item.create')
    expect(outs).toEqual([{ type: 'conversation.item.create', item: { type: 'function_call_output', call_id: 'call_1', output: '{"time":"10:00"}' } }])
    expect(box.sent[box.sent.length - 1]).toEqual({ type: 'response.create' })
  })

  it('a call the box cancels is stopped and its output never sent', async () => {
    ran.length = 0
    const box = fakeBox()
    const s = new ClientToolsSession({ spark, registry: new Registry().add(tool), dial: box.dial })
    await s.connect()
    box.box({ type: 'response.function_call_arguments.done', call_id: 'call_2', name: 'get_time', arguments: '{}', response_id: 'r2' })
    box.box({ type: 'response.done', response: { id: 'r2', status: 'completed', output: [] } })
    box.box({ type: 'localrouter.function_call.cancelled', call_id: 'call_2', reason: 'turn_detected' })
    await new Promise((r) => setTimeout(r, 40))
    expect(box.sent.some((e) => e.type === 'conversation.item.create')).toBe(false)
    expect(s.calls[0]).toMatchObject({ call_id: 'call_2', cancelled: true })
  })

  it('a UI’s session.update keeps client mode and the harness’s tools', async () => {
    const box = fakeBox()
    const s = new ClientToolsSession({ spark, registry: new Registry().add(tool), dial: box.dial })
    await s.connect()
    s.send({ type: 'session.update', session: { voice: 'Mae', tools: [{ type: 'function', name: 'evil' }], localrouter: { tool_execution: 'server' } } })
    const last = box.sent[box.sent.length - 1] as { session: Record<string, unknown> }
    expect(last.session).toMatchObject({ voice: 'Mae', localrouter: { tool_execution: 'client' } })
    expect((last.session.tools as Array<{ name: string }>).map((t) => t.name)).toEqual(['get_time'])
  })

  it('the language a call hears: the app’s setting when the page names none, the page’s when it does', async () => {
    const box = fakeBox()
    const s = new ClientToolsSession({ spark, registry: new Registry().add(tool), dial: box.dial, language: 'de' })
    await s.connect()
    expect(box.sent[0]).toMatchObject({ type: 'session.update', session: { input_audio_transcription: { language: 'de' } } })
    s.send({ type: 'session.update', session: { voice: 'Mae' } })
    expect(box.sent[box.sent.length - 1]).toMatchObject({ session: { voice: 'Mae', input_audio_transcription: { language: 'de' } } })
    s.send({ type: 'session.update', session: { input_audio_transcription: { language: 'ja' } } })
    expect(box.sent[box.sent.length - 1]).toMatchObject({ session: { input_audio_transcription: { language: 'ja' } } })
    s.send({ type: 'session.update', session: { input_audio_transcription: { language: 'auto' } } })
    expect(box.sent[box.sent.length - 1]).toMatchObject({ session: { input_audio_transcription: { language: 'auto' } } })
  })

  it('without an app setting or a page language the session names none (the box’s default)', async () => {
    const box = fakeBox()
    const s = new ClientToolsSession({ spark, registry: new Registry().add(tool), dial: box.dial })
    await s.connect()
    s.send({ type: 'session.update', session: { voice: 'Mae' } })
    expect((box.sent[box.sent.length - 1] as { session: Record<string, unknown> }).session.input_audio_transcription).toBeUndefined()
  })

  it('the page sees generation calls as items, and no other tool’s', () => {
    expect(pageView({ type: 'response.output_item.done', item: { type: 'function_call', name: 'generate_image' } })).toBe(true)
    expect(pageView({ type: 'response.output_item.done', item: { type: 'function_call', name: 'computer_use' } })).toBe(false)
    expect(pageView({ type: 'response.audio.delta' })).toBe(true)
  })
})
