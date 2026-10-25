import { createServer } from 'node:http'
import type { Server as HttpServer, ServerResponse } from 'node:http'
import type { AddressInfo } from 'node:net'
import { afterEach, describe, expect, it } from 'vitest'
import { AddressError, capabilitiesOf, classify, CodeError, connectionOf, connectView, normalizeCode, parsePairLink, tailnetPeers } from './connect.ts'
import type { ConnectInput, Connection } from './connect.ts'
import { pair, Spark } from '../harness/spark.ts'
import type { Capabilities } from '../harness/spark.ts'

const granted = { microphone: 'granted', accessibility: 'granted', screen: 'granted', stale: { accessibility: false, screen: false } }
const server = { name: 'localrouter', version: '0.9.0', machine: 'gb10', contract: '2026-10-23' }
const caps: Capabilities = {
  contract: '2026-10-23',
  server: { name: 'localrouter', version: '0.9.0', machine: 'gb10' },
  auth: { pairing_required: true },
  chat: { available: true, models: ['qwen-27b'] },
  realtime: { profile: 'openai-realtime-beta', tool_execution: ['server', 'client'] },
  mcp: { tools: ['chat', 'generate_image'] },
  models: { capabilities: ['chat_completion', 'text_to_image', 'image_edit', 'text_to_speech', 'speech_to_text'] },
  media: { image_edit: true, image_to_video: false },
  decisions: { available: false },
}
const ok: Connection = connectionOf(caps, { required: true, paired: true, device: { id: 'd1', name: 'Mac', scopes: ['inference', 'media', 'read'] } }, 1)
const input = (over: Partial<ConnectInput>): ConnectInput => ({ paired: false, base: '', draft: null, connection: null, errors: {}, permissions: granted, ...over })
const statuses = (i: ConnectInput) => Object.fromEntries(connectView(i).steps.map((s) => [s.id, s.status]))

describe('the Connect steps', () => {
  it('walks address, code, connection, permissions in order', () => {
    const fresh = input({ permissions: { ...granted, accessibility: 'not-granted' } })
    expect(connectView(fresh).current).toBe('address')
    expect(statuses(fresh)).toEqual({ address: 'active', code: 'todo', connection: 'todo', permissions: 'todo' })

    const checked = input({ draft: { base: 'https://spark.ts.net:10443', server, repair: false } })
    expect(connectView(checked).current).toBe('code')
    expect(statuses(checked)).toMatchObject({ address: 'done', code: 'active', connection: 'todo' })
    expect(connectView(checked).steps[0].summary).toContain('LocalRouter 0.9.0 on gb10')

    const paired = input({ paired: true, base: 'https://spark.ts.net:10443' })
    expect(connectView(paired).current).toBe('connection')
    expect(connectView(paired).steps[2].summary).toBe('Checking…')

    const connected = input({ paired: true, base: 'https://spark.ts.net:10443', connection: ok })
    expect(connectView(connected).current).toBe('done')
    expect(statuses(connected)).toEqual({ address: 'done', code: 'done', connection: 'done', permissions: 'done' })
  })

  it('stops at the permission step until the grants are in, and flags a stale grant', () => {
    const missing = input({ paired: true, connection: ok, permissions: { ...granted, microphone: 'not-determined' } })
    expect(connectView(missing).current).toBe('permissions')
    expect(connectView(missing).steps[3]).toMatchObject({ status: 'active', summary: '2 of 3 granted' })
    const stale = input({ paired: true, connection: ok, permissions: { ...granted, accessibility: 'stale', stale: { accessibility: true, screen: false } } })
    expect(connectView(stale).steps[3].status).toBe('warn')
    expect(connectView(input({ paired: true, connection: ok, permissions: { microphone: 'n/a', accessibility: 'n/a', screen: 'n/a', stale: { accessibility: false, screen: false } } })).current).toBe('done')
  })

  it('shows a failed step as an error and keeps the steps after it waiting', () => {
    const e = { kind: 'unreachable' as const, title: 'This Mac cannot reach that host', hint: '' }
    expect(statuses(input({ errors: { address: e } }))).toMatchObject({ address: 'error', code: 'todo' })
    const wrong = { kind: 'wrong_code' as const, title: 'The Spark did not accept that code', hint: '' }
    const v = connectView(input({ draft: { base: 'https://s', server, repair: false }, errors: { code: wrong } }))
    expect(v.current).toBe('code')
    expect(v.steps[1]).toMatchObject({ status: 'error', summary: wrong.title })
    const down: Connection = { ok: false, at: 1, error: { kind: 'timeout', title: 'No answer within 8 s', hint: '' } }
    expect(connectView(input({ paired: true, connection: down })).current).toBe('connection')
  })

  it('re-pairing reopens the code step while the old token still works', () => {
    const v = connectView(input({ paired: true, base: 'https://s', connection: ok, draft: { base: 'https://s', server, repair: true } }))
    expect(v.current).toBe('code')
    expect(v.steps.map((s) => s.status)).toEqual(['done', 'active', 'todo', 'done'])
  })

  it('a connection that works without the token being known is a warning, not a stop', () => {
    const c = connectionOf({ ...caps, auth: { pairing_required: false } }, { required: false, paired: false }, 1)
    expect(c.ok && c.warning).toMatch(/re-pair/)
    expect(connectView(input({ paired: true, connection: c })).steps[2].status).toBe('warn')
    expect(connectView(input({ paired: true, connection: c })).current).toBe('done')
  })
})

describe('the connection step', () => {
  it('lists the scopes and what this token may use on the box', () => {
    expect(ok.ok && ok.scopes.map((s) => s.name)).toEqual(['inference', 'media', 'read'])
    const byId = Object.fromEntries((ok.ok ? ok.capabilities : []).map((c) => [c.id, c]))
    expect(byId.chat).toMatchObject({ ok: true, detail: 'qwen-27b' })
    expect(byId.voice.ok).toBe(true)
    expect(byId.images).toMatchObject({ ok: true, detail: 'make and edit' })
    expect(byId.video).toMatchObject({ ok: false, detail: 'not on this box' })
    expect(byId.decisions).toMatchObject({ ok: false, detail: 'not on this box' })
  })

  it('names the missing scope when the box offers a feature this token may not use', () => {
    const rows = capabilitiesOf(caps, ['inference'])
    expect(rows.find((r) => r.id === 'images')).toMatchObject({ ok: false, detail: 'needs the media scope' })
    // A device paired before scopes (voice, mcp) may use every service.
    expect(capabilitiesOf(caps, ['voice', 'mcp']).find((r) => r.id === 'images')?.ok).toBe(true)
    // With pairing off on the box, scopes do not limit anything.
    expect(capabilitiesOf({ ...caps, auth: { pairing_required: false } }, []).find((r) => r.id === 'images')?.ok).toBe(true)
  })

  it('a revoked token or an old box is an error with the fix', () => {
    const r = connectionOf(caps, { required: true, paired: false }, 1)
    expect(r.ok ? null : r.error.kind).toBe('revoked')
    const old = connectionOf({ ...caps, realtime: { tool_execution: ['server'] } }, { required: true, paired: true }, 1)
    expect(old.ok ? null : old.error.kind).toBe('too_old')
  })
})

describe('codes, links and the tailnet', () => {
  it('reads a code as the box does', () => {
    expect(normalizeCode('7kq2-m9xd')).toBe('7KQ2-M9XD')
    expect(normalizeCode(' 7KQ2 M9XD ')).toBe('7KQ2-M9XD')
    expect(normalizeCode('OIL0-0000')).toBe('0110-0000') // Crockford: O is 0, I and L are 1
    expect(normalizeCode('7KQ2-M9XU')).toBeNull() // U is not in the alphabet
    expect(normalizeCode('7KQ2-M9X')).toBeNull()
    expect(normalizeCode('7KQ2-M9XDA')).toBeNull()
  })

  it('takes the address and code from a pairing link', () => {
    expect(parsePairLink('localrouter://pair?url=https%3A%2F%2Fspark.tail1.ts.net%3A10443&code=7KQ2-M9XD')).toEqual({ url: 'https://spark.tail1.ts.net:10443', code: '7KQ2-M9XD' })
    expect(parsePairLink('https://spark.tail1.ts.net:10443')).toBeNull()
  })

  it('finds online peers with MagicDNS names, Linux first', () => {
    const status = {
      BackendState: 'Running',
      Peer: {
        a: { HostName: 'phone', DNSName: 'phone.tail1.ts.net.', OS: 'iOS', Online: true },
        b: { HostName: 'spark-01', DNSName: 'spark-01.tail1.ts.net.', OS: 'linux', Online: true },
        c: { HostName: 'old', DNSName: 'old.tail1.ts.net.', OS: 'linux', Online: false },
      },
    }
    expect(tailnetPeers(status)?.map((p) => p.dns)).toEqual(['spark-01.tail1.ts.net', 'phone.tail1.ts.net'])
    expect(tailnetPeers({ BackendState: 'Stopped' })).toBeNull()
    expect(tailnetPeers(null)).toBeNull()
  })
})

// ------------------------------------------------------------------ error states, from real requests

const servers: HttpServer[] = []
afterEach(async () => {
  await Promise.all(servers.splice(0).map((s) => new Promise((r) => s.close(r))))
})

async function serve(handler: (res: ServerResponse) => void): Promise<number> {
  const s = createServer((_req, res) => handler(res))
  servers.push(s)
  await new Promise<void>((r) => s.listen(0, '127.0.0.1', r))
  return (s.address() as AddressInfo).port
}

const caught = (p: Promise<unknown>) => p.then(() => null, (e: unknown) => e)
const timed = (ms: number): typeof fetch => (u, init) => fetch(u, { ...init, signal: AbortSignal.timeout(ms) })

describe('what went wrong, from real failures', () => {
  it('nothing listening: refused', async () => {
    const port = await serve(() => undefined)
    await new Promise((r) => servers.pop()?.close(r))
    const e = await caught(new Spark(`http://127.0.0.1:${port}`, '').capabilities())
    expect(classify(e, 'address', 8)).toMatchObject({ kind: 'refused', detail: 'ECONNREFUSED' })
  })

  it('a port fetch never connects to: named, not "no answer"', async () => {
    const e = await caught(new Spark('http://127.0.0.1:6000', '').capabilities())
    expect(classify(e, 'address', 8)).toMatchObject({ kind: 'invalid_address', title: 'This app cannot use that port' })
  })

  it('a host that never answers: timeout, naming the limit', async () => {
    const port = await serve(() => undefined) // accepts, never replies
    const e = await caught(new Spark(`http://127.0.0.1:${port}`, '', timed(200)).capabilities())
    expect(classify(e, 'address', 8)).toMatchObject({ kind: 'timeout', title: 'No answer within 8 s' })
  })

  it('https:// to a plain HTTP port: TLS', async () => {
    const port = await serve((res) => res.end('{}'))
    const e = await caught(fetch(`https://127.0.0.1:${port}/v1/capabilities`))
    expect(classify(e, 'address', 8)).toMatchObject({ kind: 'tls', title: 'That port does not speak HTTPS' })
  })

  it('certificate problems, as undici reports them', () => {
    const tls = (code: string) => new TypeError('fetch failed', { cause: Object.assign(new Error(code), { code }) })
    expect(classify(tls('CERT_HAS_EXPIRED'), 'address', 8)).toMatchObject({ kind: 'tls', title: 'The Spark’s certificate has expired' })
    expect(classify(tls('ERR_TLS_CERT_ALTNAME_INVALID'), 'address', 8)).toMatchObject({ kind: 'tls', title: 'This Mac does not trust the Spark’s certificate' })
    expect(classify(tls('DEPTH_ZERO_SELF_SIGNED_CERT'), 'connection', 8).kind).toBe('tls')
  })

  it('a name that does not resolve: unreachable', () => {
    const dns = new TypeError('fetch failed', { cause: Object.assign(new Error('getaddrinfo ENOTFOUND spark.example'), { code: 'ENOTFOUND' }) })
    expect(classify(dns, 'address', 8)).toMatchObject({ kind: 'unreachable', detail: 'ENOTFOUND' })
    // Several addresses tried at once: undici wraps them in an AggregateError.
    const agg = new TypeError('fetch failed', { cause: new AggregateError([Object.assign(new Error('x'), { code: 'EHOSTUNREACH' })]) })
    expect(classify(agg, 'connection', 8).kind).toBe('unreachable')
  })

  it('something else on that port: not LocalRouter', async () => {
    const notFound = await serve((res) => {
      res.statusCode = 404
      res.end('not found')
    })
    expect(classify(await caught(new Spark(`http://127.0.0.1:${notFound}`, '').capabilities()), 'address', 8)).toMatchObject({ kind: 'not_localrouter', detail: 'HTTP 404' })
    const html = await serve((res) => res.end('<html>router login</html>'))
    expect(classify(await caught(new Spark(`http://127.0.0.1:${html}`, '').capabilities()), 'address', 8).kind).toBe('not_localrouter')
  })

  it('a wrong or expired code: the box’s 401 on pairing', async () => {
    const port = await serve((res) => {
      res.statusCode = 401
      res.setHeader('content-type', 'application/json')
      res.end(JSON.stringify({ error: { message: 'that pairing code is wrong, expired (codes last 10 minutes) or already used', type: 'invalid_request_error', code: 'invalid_pairing_code' } }))
    })
    const e = await caught(pair(`http://127.0.0.1:${port}`, '7KQ2-M9XD', 'Mac'))
    expect(classify(e, 'code', 15)).toMatchObject({ kind: 'wrong_code', title: 'The Spark did not accept that code' })
    expect(classify(e, 'code', 15).hint).toMatch(/10 minutes/)
  })

  it('a code or address that cannot be right is caught before any request', () => {
    expect(classify(new CodeError('x'), 'code', 15).kind).toBe('code_format')
    expect(classify(new AddressError('the Spark address must not carry credentials'), 'address', 8)).toMatchObject({ kind: 'invalid_address', hint: expect.stringContaining('credentials') })
  })
})
