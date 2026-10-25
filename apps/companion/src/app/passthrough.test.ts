import { createServer } from 'node:http'
import type { IncomingHttpHeaders, Server } from 'node:http'
import type { AddressInfo } from 'node:net'
import { afterEach, describe, expect, it } from 'vitest'
import { Spark } from '../harness/spark.ts'
import { mediaAllowed, proxy } from './passthrough.ts'

type Seen = { method: string; url: string; headers: IncomingHttpHeaders; body: Buffer }
type Reply = { status?: number; type?: string; body: string | Buffer }

const servers: Server[] = []
afterEach(() => {
  for (const s of servers.splice(0)) s.close()
})

/** A fake box: records each request whole and answers with what `reply` says. */
async function box(reply: (s: Seen) => Reply): Promise<{ spark: Spark; seen: Seen[] }> {
  const seen: Seen[] = []
  const server = createServer((req, res) => {
    const parts: Buffer[] = []
    req.on('data', (c: Buffer) => parts.push(c))
    req.on('end', () => {
      const s = { method: req.method ?? '', url: req.url ?? '', headers: req.headers, body: Buffer.concat(parts) }
      seen.push(s)
      const r = reply(s)
      res.writeHead(r.status ?? 200, { 'content-type': r.type ?? 'application/json' })
      res.end(r.body)
    })
  })
  servers.push(server)
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
  return { spark: new Spark(`http://127.0.0.1:${(server.address() as AddressInfo).port}`, 'device-token'), seen }
}

const page = (path: string, init?: RequestInit) => new Request(`http://app.invalid${path}`, init)
const voice = (extra: Record<string, unknown> = {}) => JSON.stringify({ object: 'voice', id: 'draft-1a2b3c4d', name: 'draft-1a2b3c4d', model: 'fishaudio/drama-3', kind: 'cloned', status: 'draft', ...extra })

describe('the page reaches the box through the app: voice cloning', () => {
  it('a multipart POST /v1/voices arrives whole: every clip, transcript and field, its boundary, the device token', async () => {
    const { spark, seen } = await box(() => ({ body: voice({ clips: [{ seconds_in: 13, seconds_kept: 12.3 }] }) }))
    const clip1 = new Uint8Array(70_000).map((_, i) => i % 251)
    const clip2 = new Uint8Array(9_000).map((_, i) => (i * 7) % 256)
    const fd = new FormData()
    fd.append('file', new Blob([clip1], { type: 'audio/webm' }), 'take.webm')
    fd.append('file', new Blob([clip2], { type: 'audio/wav' }), 'more.wav')
    fd.append('transcript', 'The morning train was late.')
    fd.append('transcript', 'Was it worth rushing?')
    fd.append('consent', 'true')
    fd.append('model', 'fishaudio/drama-3')
    const r = await proxy(spark, page('/v1/voices', { method: 'POST', body: fd, headers: { authorization: 'Bearer page-key' } }), '/v1/voices')
    expect(r.status).toBe(200)
    expect(r.headers.get('content-type')).toBe('application/json')
    expect(await r.json()).toMatchObject({ id: 'draft-1a2b3c4d', status: 'draft' })
    expect(seen).toHaveLength(1)
    const s = seen[0]!
    expect(s.method).toBe('POST')
    expect(s.url).toBe('/v1/voices')
    expect(s.headers.authorization).toBe('Bearer device-token') // the page's own never reaches the box
    expect(s.headers['content-type']).toMatch(/^multipart\/form-data; boundary=/)
    expect(Number(s.headers['content-length'])).toBe(s.body.length)
    const got = await new Response(new Uint8Array(s.body), { headers: { 'content-type': String(s.headers['content-type']) } }).formData()
    const files = got.getAll('file') as File[]
    expect(files.map((f) => f.name)).toEqual(['take.webm', 'more.wav'])
    expect(new Uint8Array(await files[0]!.arrayBuffer())).toEqual(clip1)
    expect(new Uint8Array(await files[1]!.arrayBuffer())).toEqual(clip2)
    expect(got.getAll('transcript')).toEqual(['The morning train was late.', 'Was it worth rushing?'])
    expect(got.get('consent')).toBe('true')
    expect(got.get('model')).toBe('fishaudio/drama-3')
  })

  it("a refused clip's 400 comes back as the box said it", async () => {
    const message = 'clip 1 starts inside speech: leave a moment of silence before the first word'
    const { spark } = await box(() => ({ status: 400, body: JSON.stringify({ error: { message, type: 'invalid_request_error' } }) }))
    const fd = new FormData()
    fd.append('file', new Blob([new Uint8Array(10)]), 'take.webm')
    const r = await proxy(spark, page('/v1/voices', { method: 'POST', body: fd }), '/v1/voices')
    expect(r.status).toBe(400)
    expect(await r.json()).toEqual({ error: { message, type: 'invalid_request_error' } })
  })

  it('PATCH names a draft (JSON body) and a 409 passes back', async () => {
    const { spark, seen } = await box((s) => (JSON.parse(s.body.toString()).name === 'JJ' ? { status: 409, body: JSON.stringify({ error: { message: "the name 'JJ' is taken", type: 'invalid_request_error' } }) } : { body: voice({ id: 'Me', name: 'Me', status: 'ready' }) }))
    const ok = await proxy(spark, page('/v1/voices/draft-1a2b3c4d', { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ name: 'Me' }) }), '/v1/voices/draft-1a2b3c4d')
    expect(ok.status).toBe(200)
    expect(await ok.json()).toMatchObject({ id: 'Me', status: 'ready' })
    const taken = await proxy(spark, page('/v1/voices/draft-1a2b3c4d', { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ name: 'JJ' }) }), '/v1/voices/draft-1a2b3c4d')
    expect(taken.status).toBe(409)
    expect(seen.map((s) => [s.method, s.url, s.headers['content-type'], s.body.toString()])).toEqual([
      ['PATCH', '/v1/voices/draft-1a2b3c4d', 'application/json', '{"name":"Me"}'],
      ['PATCH', '/v1/voices/draft-1a2b3c4d', 'application/json', '{"name":"JJ"}'],
    ])
  })

  it('DELETE goes with no body', async () => {
    const { spark, seen } = await box(() => ({ body: JSON.stringify({ object: 'voice.deleted', id: 'JJ', deleted: true }) }))
    const r = await proxy(spark, page('/v1/voices/JJ', { method: 'DELETE' }), '/v1/voices/JJ')
    expect(await r.json()).toEqual({ object: 'voice.deleted', id: 'JJ', deleted: true })
    expect(seen[0]).toMatchObject({ method: 'DELETE', url: '/v1/voices/JJ' })
    expect(seen[0]!.body.length).toBe(0)
  })

  it('GET /v1/voices keeps its query; a preview comes back as the same audio bytes', async () => {
    const wav = Buffer.concat([Buffer.from('RIFF'), Buffer.alloc(4000, 7)])
    const { spark, seen } = await box((s) => (s.url.startsWith('/v1/audio/speech') ? { type: 'audio/wav', body: wav } : { body: JSON.stringify({ object: 'list', data: [] }) }))
    const list = await proxy(spark, page('/v1/voices?model=fishaudio%2Fdrama-3'), '/v1/voices?model=fishaudio%2Fdrama-3')
    expect(await list.json()).toEqual({ object: 'list', data: [] })
    const speech = await proxy(spark, page('/v1/audio/speech', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'fishaudio/drama-3', voice: 'draft-1a2b3c4d', input: 'Hello' }) }), '/v1/audio/speech')
    expect(speech.headers.get('content-type')).toBe('audio/wav')
    expect(Buffer.from(await speech.arrayBuffer())).toEqual(wav)
    expect(seen[0]).toMatchObject({ method: 'GET', url: '/v1/voices?model=fishaudio%2Fdrama-3' })
    expect(seen[0]!.body.length).toBe(0)
  })

  it('not paired is a 503; a box that does not answer is a 502', async () => {
    expect((await proxy(null, page('/v1/voices'), '/v1/voices')).status).toBe(503)
    const down = await proxy(new Spark('http://127.0.0.1:9', 't'), page('/v1/voices', { method: 'DELETE' }), '/v1/voices/JJ')
    expect(down.status).toBe(502)
    expect(((await down.json()) as { error: { message: string } }).error.message).toMatch(/did not answer/)
  })
})

describe("the window's permissions", () => {
  it('the microphone (calls, recording a voice to clone) for this app only; no camera, nothing else', () => {
    expect(mediaAllowed('lr://app/voice.html', 'media', ['audio'])).toBe(true)
    expect(mediaAllowed('lr://app', 'media')).toBe(true)
    expect(mediaAllowed('lr://app/voice.html', 'clipboard-sanitized-write')).toBe(true)
    expect(mediaAllowed('lr://app/voice.html', 'media', ['audio', 'video'])).toBe(false)
    expect(mediaAllowed('lr://app/voice.html', 'geolocation')).toBe(false)
    expect(mediaAllowed('lr://apple/voice.html', 'media', ['audio'])).toBe(false)
    expect(mediaAllowed('https://box.ts.net/voice', 'media', ['audio'])).toBe(false)
  })
})
