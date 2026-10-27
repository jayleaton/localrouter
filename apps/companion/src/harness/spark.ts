/**
 * The harness's client for the LocalRouter box on the Spark (docs/HARNESS.md): pairing, discovery, chat, MCP media
 * and decisions, job files, and the realtime socket. It holds the device token and adds it to every request; nothing
 * here decides what runs on this computer.
 */
import { assistantOf } from './assistant.ts'
import type { Assistant } from './assistant.ts'

export type Fetch = typeof fetch

export type Capabilities = {
  contract: string
  server?: { name?: string; version?: string; machine?: string }
  auth?: { pairing_required?: boolean; scopes?: string[] }
  chat?: { available?: boolean; models?: string[] }
  realtime?: { path?: string; profile?: string; tool_execution?: string[]; remote?: boolean }
  mcp?: { path?: string; tools?: string[] }
  [k: string]: unknown
}

export type Paired = { token: string; device: { id: string; name: string }; required: boolean }
export type WhoAmI = { required: boolean; paired: boolean; device?: { id: string; name: string; scopes?: string[] } }

export class SparkError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

/** `https://box.tailnet.ts.net:10443` or `http://127.0.0.1:8190`: a base URL with no path, credentials or query. */
export function normalizeBase(input: string): string {
  const raw = input.trim()
  const url = new URL(/^[a-z]+:\/\//i.test(raw) ? raw : `https://${raw}`)
  if (url.protocol !== 'https:' && url.protocol !== 'http:') throw new Error('the Spark address is https://… (or http:// on this computer or the tailnet)')
  if (url.username || url.password) throw new Error('the Spark address must not carry credentials')
  if (url.protocol === 'http:' && !isPrivateHost(url.hostname)) throw new PlainHttpError(url.hostname.replace(/^\[|\]$/g, ''), url.port || '80')
  return `${url.protocol}//${url.host}`
}

/**
 * Plain http:// to a host that is not this computer or the tailnet (a LAN IP, say): refused by design, since the pairing
 * code, the token and the screen would cross the network unencrypted. `suggestion`: the host's tailnet name, when the
 * caller found one (the Connect page asks `tailscale status`).
 */
export class PlainHttpError extends Error {
  readonly host: string
  readonly port: string
  suggestion: string | null = null
  constructor(host: string, port: string) {
    super('plain http:// only to this computer or a tailnet address; use the tailnet name or the https:// address `tailscale serve` gives the box')
    this.host = host
    this.port = port
  }
}

/** Loopback, a tailnet (100.64.0.0/10, fd7a:115c:a1e0::/48) or a MagicDNS name. */
export function isPrivateHost(host: string): boolean {
  const h = host.replace(/^\[|\]$/g, '').toLowerCase()
  if (h === 'localhost' || h === '::1' || /^127\./.test(h) || h.endsWith('.ts.net')) return true
  const m = /^100\.(\d+)\./.exec(h)
  if (m && Number(m[1]) >= 64 && Number(m[1]) <= 127) return true
  return h.startsWith('fd7a:115c:a1e0:')
}

async function errorOf(r: Response): Promise<SparkError> {
  const text = await r.text().catch(() => '')
  let message = text || `HTTP ${r.status}`
  try {
    const j = JSON.parse(text) as { error?: { message?: string } | string }
    message = typeof j.error === 'string' ? j.error : (j.error?.message ?? message)
  } catch {
    // Not JSON.
  }
  return new SparkError(r.status, message)
}

/** Trades a one-time pairing code (`deploy/jarvis/pair.sh add NAME`) for this device's token. */
export async function pair(base: string, code: string, name: string, f: Fetch = fetch): Promise<Paired> {
  const r = await f(`${normalizeBase(base)}/v1/pair`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ code: code.trim(), name }) })
  if (!r.ok) throw await errorOf(r)
  return (await r.json()) as Paired
}

export type McpResult = { content: Array<{ type: string; text?: string; uri?: string; name?: string; mimeType?: string; data?: string }>; isError?: boolean }

export class Spark {
  readonly base: string
  private token: string
  private readonly f: Fetch
  private seq = 0

  constructor(base: string, token: string, f: Fetch = fetch) {
    this.base = normalizeBase(base)
    this.token = token
    this.f = f
  }

  private headers(json = false): Record<string, string> {
    return { ...(this.token ? { authorization: `Bearer ${this.token}` } : {}), ...(json ? { 'content-type': 'application/json' } : {}) }
  }

  /** An authorized request to a path on the box (the app's proxy for the page's own calls uses this). */
  async request(path: string, init: { method?: string; body?: RequestInit['body']; headers?: Record<string, string>; signal?: AbortSignal } = {}): Promise<Response> {
    if (!path.startsWith('/')) throw new Error('a path on the box starts with /')
    return this.f(`${this.base}${path}`, { method: init.method ?? 'GET', headers: { ...(init.headers ?? {}), ...this.headers() }, body: init.body ?? null, ...(init.signal ? { signal: init.signal } : {}) })
  }

  private async json<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
    const r = await this.f(`${this.base}${path}`, { method: body === undefined ? 'GET' : 'POST', headers: this.headers(body !== undefined), ...(body !== undefined ? { body: JSON.stringify(body) } : {}), ...(signal ? { signal } : {}) })
    if (!r.ok) throw await errorOf(r)
    return (await r.json()) as T
  }

  capabilities(): Promise<Capabilities> {
    return this.json('/v1/capabilities')
  }

  whoami(): Promise<WhoAmI> {
    return this.json('/v1/pair')
  }

  /** The assistant's name on the box; null on a box from before the setting (404, 405, or 403 for the route). */
  async assistant(): Promise<Assistant | null> {
    const r = await this.request('/v1/assistant')
    if (r.status === 404 || r.status === 405 || r.status === 403) return null
    if (!r.ok) throw await errorOf(r)
    return assistantOf(await r.json())
  }

  /** Renames the assistant on the box for every device (`manage` scope); an empty name is the default again. */
  async renameAssistant(name: string): Promise<Assistant> {
    const r = await this.request('/v1/assistant', { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ name }) })
    if (!r.ok) throw await errorOf(r)
    return assistantOf(await r.json())
  }

  /** One Chat Completions request (not streamed): the whole completion. */
  chat(body: Record<string, unknown>, signal?: AbortSignal): Promise<{ choices: Array<{ message: { role: string; content: string | null; tool_calls?: Array<{ id: string; type: 'function'; function: { name: string; arguments: string } }> }; finish_reason: string }> }> {
    return this.json('/v1/chat/completions', { ...body, stream: false }, signal)
  }

  /** The assistant's text for a request: what the executor's decisions use. */
  async complete(body: Record<string, unknown>): Promise<string> {
    const r = await this.chat(body)
    return r.choices[0]?.message.content ?? ''
  }

  /** An MCP tool on the box (`POST /mcp`, stateless): its result; a JSON-RPC error throws. */
  async mcp(name: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<McpResult> {
    const j = await this.json<{ result?: McpResult; error?: { message: string } }>('/mcp', { jsonrpc: '2.0', id: ++this.seq, method: 'tools/call', params: { name, arguments: args } }, signal)
    if (j.error) throw new SparkError(400, j.error.message)
    return j.result ?? { content: [], isError: true }
  }

  /**
   * Why the box refuses this computer's realtime socket: a plain GET of the route answers its refusal (401, 403) with
   * the reason before any upgrade, which a WebSocket client never sees. Null when it does not refuse, or cannot say.
   */
  async realtimeRefusal(): Promise<string | null> {
    try {
      const r = await this.request('/v1/realtime')
      if (r.status !== 401 && r.status !== 403) return null
      return (await errorOf(r)).message
    } catch {
      return null
    }
  }

  /** `wss://…/v1/realtime` and the subprotocols that carry the token (no token in the URL). */
  realtime(model?: string): { url: string; protocols: string[] } {
    const u = new URL('/v1/realtime', this.base)
    u.protocol = u.protocol === 'https:' ? 'wss:' : 'ws:'
    if (model) u.searchParams.set('model', model)
    return { url: u.href, protocols: this.token ? ['realtime', `openai-insecure-api-key.${this.token}`] : ['realtime'] }
  }
}
