/**
 * The Connect steps of the settings window, as data: where the Spark is, the pairing code, the connection, and this
 * Mac's permissions. What went wrong, said so the person can fix it (a host that does not answer, a certificate, a
 * wrong or expired code, a revoked token). Pure, so it is tested without Electron or a box; the main process gathers
 * the facts and the page only draws what this returns.
 */
import { DEFAULT_ASSISTANT } from '../harness/assistant.ts'
import { PlainHttpError } from '../harness/spark.ts'
import type { Capabilities, WhoAmI } from '../harness/spark.ts'

export type StepId = 'address' | 'code' | 'connection' | 'assistant' | 'permissions' | 'control'
export type StepStatus = 'todo' | 'active' | 'done' | 'warn' | 'error'
export type ErrorKind = 'invalid_address' | 'unreachable' | 'refused' | 'timeout' | 'tls' | 'not_localrouter' | 'too_old' | 'code_format' | 'wrong_code' | 'revoked' | 'server'
export type ConnectError = { kind: ErrorKind; title: string; hint: string; detail?: string }
/** The step whose request failed. */
export type Stage = 'address' | 'code' | 'connection'

/** The box as `/v1/capabilities` named it, kept from the address check. */
export type Server = { name: string; version: string; machine: string; contract: string }
/** An address that answered as a LocalRouter box; `repair`: pairing again while still paired. */
export type Draft = { base: string; server: Server | null; repair: boolean }

export type Capability = { id: string; label: string; ok: boolean; detail: string }
export type Connection =
  | { ok: true; at: number; server: Server; scopes: Array<{ name: string; label: string }>; pairing_required: boolean; capabilities: Capability[]; warning?: string }
  | { ok: false; at: number; error: ConnectError }

/** One thing a platform needs (platform/types.ts Requirement), as the page shows it. */
export type Need = { id: string; label: string; granted: boolean | null; for: string; fix?: string }

/**
 * What the permissions step shows: macOS's TCC grants (microphone, accessibility, screen), or on Linux and Windows the
 * platform's own `requirements` (with `backend` naming it), where the Mac's three are 'n/a'.
 */
export type Permissions = { microphone: string; accessibility: string; screen: string; stale: { accessibility: boolean; screen: boolean; microphone?: boolean }; backend?: string; requirements?: Need[] }

export type ConnectInput = {
  paired: boolean
  base: string
  draft: Draft | null
  connection: Connection | null
  /** The last failed address check or pairing; a failed connection is in `connection`. */
  errors: Partial<Record<'address' | 'code', ConnectError>>
  permissions: Permissions
  /** The standing grant for computer control: given once here, in setup, never by a card over a task. */
  control?: { granted: boolean }
  /**
   * The assistant's name (the Spark's, as last read). `step`: the name step, asked once after this computer's first
   * pairing ('ask' until answered, then 'done'); an install paired before it has none and is not asked.
   */
  assistant?: { name: string; step?: 'ask' | 'done' }
}

export type Step = { id: StepId; status: StepStatus; summary: string }
export type ConnectView = { current: StepId | 'done'; steps: Step[] }

/**
 * Why the box will refuse calls from this computer, from its discovery document: it serves them to itself only
 * (`realtime.remote` false: it listens on the network with `voice.remote` off) and `base` is not this machine. Null when
 * a call may go ahead (also when the box does not say).
 */
export function callRefusal(caps: Capabilities, base: string): string | null {
  if (caps.realtime?.remote !== false) return null
  let host: string
  try {
    host = new URL(base).hostname.replace(/^\[|\]$/g, '')
  } catch {
    return null
  }
  if (host === 'localhost' || host.startsWith('127.') || host === '::1') return null
  return 'the Spark answers voice calls from itself only: set "voice": {"remote": true} in its config (the default for a box on the network with pairing on) and restart it'
}

/** The client tool mode this app's calls need (contract 2026-10-23). */
export function supportsApp(caps: Capabilities): boolean {
  return Boolean(caps.realtime?.tool_execution?.includes('client'))
}

export function serverOf(caps: Capabilities): Server {
  return { name: caps.server?.name ?? 'localrouter', version: caps.server?.version ?? '', machine: caps.server?.machine ?? '', contract: caps.contract ?? '' }
}

export function describeServer(s: Server): string {
  return [`LocalRouter${s.version ? ` ${s.version}` : ''}`, s.machine && `on ${s.machine}`].filter(Boolean).join(' ')
}

// ------------------------------------------------------------------ the steps

const PERMISSIONS = ['microphone', 'accessibility', 'screen'] as const

function permissionStep(p: Permissions): { status: StepStatus; summary: string } {
  if (p.requirements) {
    // Linux and Windows: what this computer's platform says it needs, in its own words; no macOS grants to ask for.
    const missing = p.requirements.filter((r) => r.granted === false)
    if (missing.length) return { status: 'warn', summary: `Missing: ${missing.map((r) => r.label).join(', ')}` }
    return { status: 'done', summary: p.backend ? `Nothing to grant: ${p.backend} is ready` : 'Nothing to grant on this computer' }
  }
  if (p.microphone === 'n/a') return { status: 'done', summary: 'Nothing to grant on this computer' }
  if (p.stale.accessibility || p.stale.screen) return { status: 'warn', summary: 'macOS kept a permission from an older LocalRouter' }
  const granted = PERMISSIONS.filter((k) => p[k] === 'granted').length
  return granted === PERMISSIONS.length ? { status: 'done', summary: 'Microphone, Accessibility and Screen Recording granted' } : { status: 'todo', summary: `${granted} of ${PERMISSIONS.length} granted` }
}

/**
 * Each step's state and the one to work on now. A step is done once what it gives exists (a checked address, a token,
 * a working connection, the grants), in error while its last attempt failed, and active when it is the first not done.
 * Re-pairing reopens the code step without dropping the working token until a new one replaces it.
 */
export function connectView(i: ConnectInput): ConnectView {
  const base = i.draft?.base ?? i.base
  const repairing = Boolean(i.draft?.repair)
  const server = i.draft?.server ?? (i.connection?.ok ? i.connection.server : null)

  let address: Step
  if (i.errors.address) address = { id: 'address', status: 'error', summary: i.errors.address.title }
  else if (i.paired || i.draft) address = { id: 'address', status: 'done', summary: server ? `${base} · ${describeServer(server)}` : base }
  else address = { id: 'address', status: 'todo', summary: 'Where your Spark answers' }

  let code: Step
  if (address.status !== 'done') code = { id: 'code', status: 'todo', summary: 'A one-time code from the Spark' }
  else if (i.errors.code) code = { id: 'code', status: 'error', summary: i.errors.code.title }
  else if (i.paired && !repairing) code = { id: 'code', status: 'done', summary: 'Paired' }
  else code = { id: 'code', status: 'todo', summary: repairing ? 'Enter a new code to replace this computer’s token' : 'A one-time code from the Spark' }

  let connection: Step
  if (!i.paired || repairing) connection = { id: 'connection', status: 'todo', summary: 'After pairing' }
  else if (!i.connection) connection = { id: 'connection', status: 'todo', summary: 'Checking…' }
  else if (!i.connection.ok) connection = { id: 'connection', status: 'error', summary: i.connection.error.title }
  else connection = { id: 'connection', status: i.connection.warning ? 'warn' : 'done', summary: i.connection.warning ?? `Connected · ${i.connection.capabilities.filter((c) => c.ok).length} of ${i.connection.capabilities.length} features available` }

  const name = i.assistant?.name || DEFAULT_ASSISTANT
  const named: Step[] = []
  if (i.assistant?.step === 'ask') named.push({ id: 'assistant', status: 'todo', summary: 'What should your assistant be called?' })
  if (i.assistant?.step === 'done') named.push({ id: 'assistant', status: 'done', summary: `Called ${name}` })
  const p = permissionStep(i.permissions)
  const control: Step = i.control?.granted
    ? { id: 'control', status: 'done', summary: `${name} may use this computer, in the background; stop it any time` }
    : { id: 'control', status: 'todo', summary: `Let ${name} use this computer: asked only here, once` }
  const steps: Step[] = [address, code, connection, ...named, { id: 'permissions', ...p }, control]
  // Warnings do not block: a connection that works with a caveat counts as done for what comes next.
  const first = steps.find((s) => s.status !== 'done' && s.status !== 'warn')
  if (first && first.status === 'todo') first.status = 'active'
  return { current: first?.id ?? 'done', steps }
}

// ------------------------------------------------------------------ what the connection offers

const SCOPES: Record<string, string> = {
  all: 'everything, including features added later',
  inference: 'chat and voice calls',
  media: 'pictures, video, speech and transcription',
  decisions: 'typed decisions',
  read: 'models and voices',
  manage: 'loading and unloading models',
  voice: 'every service (paired before scopes)',
  mcp: 'every service (paired before scopes)',
  desktop_helper: 'computer control for older clients',
  computer: 'computer tasks',
  approve: 'approvals',
}

export function scopeLabel(scope: string): string {
  return SCOPES[scope] ?? scope
}

/** Whether a token with `scopes` may use a service scope: `all` (the default), `voice` or `mcp` (or none listed) grants every one. */
function allows(scopes: string[], scope: string): boolean {
  return scopes.length === 0 || scopes.includes('all') || scopes.includes('voice') || scopes.includes('mcp') || scopes.includes(scope)
}

/**
 * What this app can use on the box, from its discovery document and the token's scopes: each feature the app relies
 * on, whether the box offers it, and whether this token may use it (scopes matter only with pairing required).
 */
export function capabilitiesOf(caps: Capabilities, scopes: string[], base = ''): Capability[] {
  const have = new Set(((caps.models as { capabilities?: string[] } | undefined)?.capabilities ?? []) as string[])
  const required = Boolean(caps.auth?.pairing_required)
  const media = (caps.media ?? {}) as { image_edit?: boolean; image_to_video?: boolean }
  const decisions = (caps.decisions ?? {}) as { available?: boolean }
  const row = (id: string, label: string, offered: boolean, scope: string, detail: string): Capability => {
    if (!offered) return { id, label, ok: false, detail: 'not on this box' }
    if (required && !allows(scopes, scope)) return { id, label, ok: false, detail: `needs the ${scope} scope` }
    return { id, label, ok: true, detail }
  }
  const chatModels = caps.chat?.models ?? []
  const refusal = base ? callRefusal(caps, base) : null
  const voice = row('voice', 'Voice calls', supportsApp(caps), 'inference', caps.realtime?.profile ?? 'realtime')
  const tools = caps.mcp?.tools ?? []
  return [
    row('chat', 'Chat', Boolean(caps.chat?.available), 'inference', chatModels.length ? chatModels.join(', ') : 'available'),
    voice.ok && refusal ? { ...voice, ok: false, detail: 'answered on the Spark itself only: voice.remote is off there' } : voice,
    row('images', 'Pictures', have.has('text_to_image'), 'media', media.image_edit ? 'make and edit' : 'make'),
    row('video', 'Video', have.has('text_to_video') || have.has('image_to_video'), 'media', [have.has('text_to_video') && 'from text', media.image_to_video && 'from a picture'].filter(Boolean).join(', ') || 'available'),
    row('speech', 'Speech', have.has('text_to_speech'), 'media', 'text to speech'),
    row('transcription', 'Transcription', have.has('speech_to_text'), 'media', 'speech to text'),
    row('decisions', 'Decisions', Boolean(decisions.available), 'decisions', 'typed decisions'),
    row('mcp', 'MCP tools', tools.length > 0, 'inference', `${tools.length} tools`),
  ]
}

/** The connection step from the two open discovery reads: `/v1/capabilities` and `GET /v1/pair` with the token. */
export function connectionOf(caps: Capabilities, who: WhoAmI, at: number, base = ''): Connection {
  const required = Boolean(caps.auth?.pairing_required ?? who.required)
  if (!supportsApp(caps)) return { ok: false, at, error: tooOld() }
  if (!who.paired && required) return { ok: false, at, error: revoked() }
  const scopes = who.device?.scopes ?? []
  return {
    ok: true,
    at,
    server: serverOf(caps),
    scopes: scopes.map((name) => ({ name, label: scopeLabel(name) })),
    pairing_required: required,
    capabilities: capabilitiesOf(caps, scopes, base),
    ...(who.paired ? {} : { warning: 'Connected, but the Spark does not know this computer’s token. It works only while pairing is off on the Spark: re-pair to keep working when it is turned on.' }),
  }
}

// ------------------------------------------------------------------ errors

const TLS_CODES = /^(CERT_|ERR_TLS_|ERR_SSL_|DEPTH_ZERO_SELF_SIGNED_CERT|SELF_SIGNED_CERT_IN_CHAIN|UNABLE_TO_(VERIFY|GET)_|HOSTNAME_MISMATCH|EPROTO)/
const UNREACHABLE_CODES = new Set(['ENOTFOUND', 'EAI_AGAIN', 'EAI_NONAME', 'EAI_FAIL', 'EHOSTUNREACH', 'ENETUNREACH', 'EHOSTDOWN', 'ENETDOWN', 'EADDRNOTAVAIL'])
const TIMEOUT_CODES = new Set(['ETIMEDOUT', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_HEADERS_TIMEOUT', 'UND_ERR_BODY_TIMEOUT'])

/** The error and everything under it: `cause` chains and an AggregateError's `errors` (undici reports both). */
function chain(error: unknown): Array<{ name?: string; code?: string; message?: string; status?: number }> {
  const out: Array<{ name?: string; code?: string; message?: string; status?: number }> = []
  const seen = new Set<unknown>()
  const walk = (e: unknown) => {
    if (!e || typeof e !== 'object' || seen.has(e) || out.length > 16) return
    seen.add(e)
    const x = e as { name?: string; code?: unknown; message?: string; status?: unknown; cause?: unknown; errors?: unknown }
    out.push({ name: x.name, code: typeof x.code === 'string' ? x.code : undefined, message: x.message, status: typeof x.status === 'number' ? x.status : undefined })
    walk(x.cause)
    if (Array.isArray(x.errors)) for (const c of x.errors) walk(c)
  }
  walk(error)
  return out
}

const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error))

export function tooOld(): ConnectError {
  return { kind: 'too_old', title: 'This Spark’s LocalRouter is too old for this app', hint: 'The app needs client tool execution (contract 2026-10-23). Update LocalRouter on the Spark, then try again.' }
}

export function revoked(): ConnectError {
  return { kind: 'revoked', title: 'The Spark no longer accepts this computer’s token', hint: 'It was revoked, or the Spark’s paired devices were reset. Re-pair with a new code from the Spark.' }
}

/**
 * What went wrong reaching or pairing with the box, for a person: which step, what it means, what to do. `stage` is
 * the step that asked; `timeoutS` the limit the caller set (named in the message).
 */
export function classify(error: unknown, stage: Stage, timeoutS: number): ConnectError {
  const all = chain(error)
  const codes = all.map((e) => e.code).filter((c): c is string => Boolean(c))
  const detail = codes[codes.length - 1] ?? messageOf(error)
  const status = all.find((e) => e.status !== undefined)?.status

  if (error instanceof PlainHttpError) return plainHttp(error)
  if (error instanceof AddressError) return { kind: 'invalid_address', title: 'That is not a Spark address', hint: `${error.message}. It looks like https://your-spark.your-tailnet.ts.net:10443.` }
  if (error instanceof CodeError) return { kind: 'code_format', title: 'That is not a pairing code', hint: 'A code is 8 letters and digits, like 7KQ2-M9XD, as `pair.sh add` printed it.' }
  if (status !== undefined) {
    if (stage === 'code' && status === 401) return { kind: 'wrong_code', title: 'The Spark did not accept that code', hint: 'It is wrong, expired (codes last 10 minutes) or already used. Make a new one on the Spark and enter it here. After five wrong codes in 10 minutes the Spark drops every pending code.' }
    if (stage === 'code' && status === 400) return { kind: 'code_format', title: 'The Spark refused that code', hint: messageOf(error) }
    if (stage === 'connection' && (status === 401 || status === 403)) return { ...revoked(), detail: messageOf(error) }
    if (status === 404 || status === 405) return { kind: 'not_localrouter', title: 'Something answered, but not LocalRouter', hint: 'Check the port: a LocalRouter Spark answers on the https:// address `tailscale serve` gives it, usually :10443.', detail: `HTTP ${status}` }
    return { kind: 'server', title: `The Spark answered with an error (HTTP ${status})`, hint: messageOf(error), detail: `HTTP ${status}` }
  }
  // fetch refuses the Fetch standard's blocked ports (9, 25, 6000...) before connecting.
  if (all.some((e) => e.message === 'bad port'))
    return { kind: 'invalid_address', title: 'This app cannot use that port', hint: 'Some ports (like 25 or 6000) are blocked for web requests. Use the port `tailscale serve` gives the Spark, usually :10443.', detail: 'bad port' }
  if (all.some((e) => e.name === 'TimeoutError' || (e.code !== undefined && TIMEOUT_CODES.has(e.code))))
    return { kind: 'timeout', title: `No answer within ${timeoutS} s`, hint: 'The Spark may be off or asleep, or this Mac is not on the tailnet right now. Check Tailscale on both, then try again.', detail }
  if (codes.some((c) => UNREACHABLE_CODES.has(c)))
    return { kind: 'unreachable', title: 'This Mac cannot reach that host', hint: 'No such name on this network. Is Tailscale running and signed in on this Mac? Check the name against `tailscale status` on the Spark.', detail }
  if (codes.includes('ECONNREFUSED'))
    return { kind: 'refused', title: 'The host answered, but nothing listens on that port', hint: 'Is LocalRouter running on the Spark, and `tailscale serve` set up? Use the address `pair.sh add` prints.', detail }
  if (codes.some((c) => c === 'ERR_SSL_WRONG_VERSION_NUMBER' || c === 'EPROTO') || all.some((e) => /wrong version number/i.test(e.message ?? '')))
    return { kind: 'tls', title: 'That port does not speak HTTPS', hint: 'It answers plain HTTP. Use the https:// address `tailscale serve` gives the Spark (usually :10443), or http:// with a tailnet address and the daemon’s own port.', detail }
  if (codes.some((c) => c === 'CERT_HAS_EXPIRED'))
    return { kind: 'tls', title: 'The Spark’s certificate has expired', hint: 'On the Spark, renew it: `tailscale cert` (or restart `tailscale serve`, which renews it), then try again.', detail }
  if (codes.some((c) => TLS_CODES.test(c)))
    return { kind: 'tls', title: 'This Mac does not trust the Spark’s certificate', hint: 'Use the Spark’s full https://….ts.net name from `tailscale serve`: an IP address or a short name does not match its certificate.', detail }
  if (all.some((e) => e.name === 'SyntaxError'))
    return { kind: 'not_localrouter', title: 'Something answered, but not LocalRouter', hint: 'The reply was not LocalRouter’s. Check the address and the port (usually :10443).', detail: messageOf(error) }
  if (codes.some((c) => c === 'ECONNRESET' || c === 'UND_ERR_SOCKET' || c.startsWith('HPE_')))
    return { kind: 'not_localrouter', title: 'The connection broke before LocalRouter answered', hint: 'Something else may be on that port, or it speaks HTTPS where you wrote http:// (or the other way round). Check the address.', detail }
  return { kind: 'unreachable', title: stage === 'code' ? 'Pairing failed' : 'The Spark did not answer', hint: messageOf(error), detail }
}

/** Why plain http:// to a LAN address is refused (by design), and the tailnet name to use instead when it is known. */
function plainHttp(e: PlainHttpError): ConnectError {
  const where = /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|169\.254\.|f[cd]|fe80:)/i.test(e.host) ? 'an address on your local network' : 'not this computer or a tailnet address'
  const use = e.suggestion ? `Use its tailnet name instead: http://${e.suggestion}:${e.port}` : `Use the Spark’s tailnet name, like http://your-spark.your-tailnet.ts.net:${e.port}`
  return {
    kind: 'invalid_address',
    title: 'Plain http:// works only over the tailnet',
    hint: `${e.host} is ${where}. LocalRouter would send it the pairing code, its login token and what it reads from your screen unencrypted, so plain http:// is allowed only to this computer or over Tailscale, which encrypts it. ${use}, or the https:// address \`tailscale serve\` gives the Spark.`,
    detail: `http://${e.host}:${e.port}`,
  }
}

export class AddressError extends Error {}
export class CodeError extends Error {}

// ------------------------------------------------------------------ input: codes, pairing links, the tailnet

const ALPHABET = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'

/**
 * A code as the box reads it (src/pair.zig `normalize`): any case, dashes and spaces ignored, O as 0 and I or L as 1.
 * XXXX-XXXX, or null when it is not 8 characters of the alphabet: checked here so a typo never costs one of the box's
 * five wrong-code attempts.
 */
export function normalizeCode(input: string): string | null {
  let out = ''
  for (const ch of input.toUpperCase()) {
    if (ch === '-' || ch === ' ') continue
    const c = ch === 'O' ? '0' : ch === 'I' || ch === 'L' ? '1' : ch
    if (!ALPHABET.includes(c) || out.length === 8) return null
    out += c
  }
  return out.length === 8 ? `${out.slice(0, 4)}-${out.slice(4)}` : null
}

/** `localrouter://pair?url=…&code=…`, the link `pair.sh add` prints with `tailscale serve`: the address and the code. */
export function parsePairLink(text: string): { url: string; code: string | null } | null {
  const m = /^localrouter:\/\/\/?pair\/?\?(.*)$/i.exec(text.trim())
  if (!m) return null
  const q = new URLSearchParams(m[1])
  const url = q.get('url')
  if (!url) return null
  return { url, code: normalizeCode(q.get('code') ?? '') }
}

export type TailnetPeer = { host: string; dns: string; os: string }

/** Online peers with a MagicDNS name, from `tailscale status --json`; null when Tailscale is not connected here. */
export function tailnetPeers(status: unknown): TailnetPeer[] | null {
  const s = status as { BackendState?: string; Peer?: Record<string, { HostName?: string; DNSName?: string; OS?: string; Online?: boolean }> } | null
  if (!s || s.BackendState !== 'Running') return null
  const peers: TailnetPeer[] = []
  for (const p of Object.values(s.Peer ?? {})) {
    const dns = (p.DNSName ?? '').replace(/\.$/, '')
    if (p.Online && dns) peers.push({ host: p.HostName ?? dns.split('.')[0], dns, os: p.OS ?? '' })
  }
  // Linux first: a Spark runs it; phones and other Macs are tried last.
  return peers.sort((a, b) => Number(b.os === 'linux') - Number(a.os === 'linux') || a.dns.localeCompare(b.dns))
}

/**
 * The MagicDNS name of the online peer that owns `host` (an IP, as in `http://192.168.1.117:8190`), from `tailscale
 * status --json`: a peer's `Addrs` and `CurAddr` are the "ip:port" endpoints it is reached at ("[ip]:port" for IPv6).
 * Null when Tailscale is not running here or no online peer has that address: never a guess.
 */
export function peerNameFor(status: unknown, host: string): string | null {
  const s = status as { BackendState?: string; Peer?: Record<string, { DNSName?: string; Online?: boolean; Addrs?: string[] | null; CurAddr?: string }> } | null
  if (!s || s.BackendState !== 'Running') return null
  const want = host.replace(/^\[|\]$/g, '').toLowerCase()
  const ipOf = (endpoint: string) => endpoint.replace(/:\d+$/, '').replace(/^\[|\]$/g, '').toLowerCase()
  for (const p of Object.values(s.Peer ?? {})) {
    const dns = (p.DNSName ?? '').replace(/\.$/, '')
    if (p.Online && dns && [...(p.Addrs ?? []), p.CurAddr ?? ''].some((a) => a && ipOf(a) === want)) return dns
  }
  return null
}

/** The addresses to try for a peer: `tailscale serve`'s LocalRouter port, then its default HTTPS port. */
export function peerBases(p: TailnetPeer): string[] {
  return [`https://${p.dns}:10443`, `https://${p.dns}`]
}

/**
 * Where the Tailscale CLI may be on this computer, in order: the app bundle or package paths for the OS, then every
 * folder on PATH. Windows installs `tailscale.exe` under Program Files (the GUI is `tailscale-ipn.exe`), so the Mac and
 * Unix paths alone said Tailscale was off on every Windows PC.
 */
export function tailscaleCandidates(platform: NodeJS.Platform, env: NodeJS.ProcessEnv): string[] {
  const win = platform === 'win32'
  const join = (dir: string, file: string) => `${dir.replace(/[\\/]+$/, '')}${win ? '\\' : '/'}${file}`
  const fixed = win
    ? [env.ProgramFiles, env['ProgramFiles(x86)'], env.ProgramW6432, env.LOCALAPPDATA].filter((d): d is string => Boolean(d)).map((d) => join(join(d, 'Tailscale'), 'tailscale.exe'))
    : platform === 'darwin'
      ? ['/Applications/Tailscale.app/Contents/MacOS/Tailscale', '/opt/homebrew/bin/tailscale', '/usr/local/bin/tailscale']
      : ['/usr/bin/tailscale', '/usr/local/bin/tailscale', '/usr/sbin/tailscale', '/snap/bin/tailscale']
  const path = (env.PATH ?? env.Path ?? '').split(win ? ';' : ':').filter(Boolean).map((d) => join(d, win ? 'tailscale.exe' : 'tailscale'))
  return [...new Set([...fixed, ...path])]
}

/** The executor's MCP launcher next to the app's resources: a shell script, or on Windows its .cmd. */
export function mcpLauncher(resources: string, platform: NodeJS.Platform): string {
  const win = platform === 'win32'
  return `${resources.replace(/[\\/]+$/, '')}${win ? '\\' : '/'}localrouter-executor-mcp${win ? '.cmd' : ''}`
}

