/**
 * Settings › Models: which models the LocalRouter server keeps loaded and which it shuts down, read from its
 * `/v1/tools` and changed with `POST /v1/tools/{id}/settings` (the server decides and keeps the setting; this app only
 * shows and asks). Only the models the server can run are listed; the rest are counted, with the server's reason. Also
 * the banner a call shows while its voice models load (`localrouter.session.loading`). Pure: main.ts does the I/O.
 */
import { SparkError } from '../harness/spark.ts'

export type Residency = 'always' | 'call' | 'on_demand' | 'disabled'
export const RESIDENCIES: readonly Residency[] = ['always', 'call', 'on_demand', 'disabled']

export const RESIDENCY_LABEL: Readonly<Record<Residency, string>> = {
  always: 'Always loaded',
  call: 'Load when a call starts',
  on_demand: 'Load when needed',
  disabled: 'Disabled (never runs)',
}

/** The kinds a voice call runs on: it listens (transcription), thinks (chat) and speaks (speech). */
const VOICE_KINDS = new Set(['chat', 'speech', 'transcription'])
export const VOICE_WARNING = 'Voice calls that rely on this model won’t work while it’s disabled.'

const KIND_LABEL: Readonly<Record<string, string>> = { chat: 'Chat', speech: 'Speech', transcription: 'Transcription', image: 'Pictures', video: 'Video', decision: 'Decisions' }
const STATE_LABEL: Readonly<Record<string, string>> = { ready: 'ready', busy: 'working', loading: 'loading', unloaded: 'not loaded', failed: 'failed' }

export type ToolEntry = {
  id: string
  kind: string
  state: string
  resident_bytes: number
  priority: number
  residency: Residency
  residency_default: Residency
  idle_ttl_s: number
  idle_ttl_s_default: number
  allowed_residency: Residency[]
  last_load_ms: number | null
  loaded_since: string | null
  last_error: string
  /** False: the server cannot run it on its hardware (`unsupported_reason` says why); it is not listed for settings. */
  supported: boolean
  unsupported_reason: string
}
export type Budget = { ceiling_bytes: number; booked_bytes: number; always_bytes: number }
/** `/v1/tools` as read. `residency`: false on a box from before model settings (no `residency` on its entries). */
export type Tools = { entries: ToolEntry[]; budget: Budget | null; residency: boolean }

const num = (v: unknown, d = 0) => (typeof v === 'number' && Number.isFinite(v) ? v : d)
const str = (v: unknown, d = '') => (typeof v === 'string' ? v : d)
const residencyOf = (v: unknown, d: Residency): Residency => (RESIDENCIES.includes(v as Residency) ? (v as Residency) : d)

/** One `/v1/tools` entry; what is missing or malformed reads as the box's defaults would. */
export function toolEntryOf(raw: unknown): ToolEntry | null {
  if (!raw || typeof raw !== 'object') return null
  const r = raw as Record<string, unknown>
  if (typeof r.id !== 'string' || !r.id) return null
  const residency = residencyOf(r.residency, r.keep_loaded === true ? 'always' : 'on_demand')
  // Disabled is offered for every model, whether or not the server lists it.
  const listed = Array.isArray(r.allowed_residency) ? RESIDENCIES.filter((x) => (r.allowed_residency as unknown[]).includes(x)) : ['always', 'on_demand']
  const allowed = listed.includes('disabled') ? listed : [...listed, 'disabled']
  return {
    id: r.id,
    kind: str(r.kind),
    state: str(r.state, 'unloaded'),
    resident_bytes: num(r.resident_bytes),
    priority: num(r.priority),
    residency,
    residency_default: residencyOf(r.residency_default, residency),
    idle_ttl_s: Math.max(0, Math.round(num(r.idle_ttl_s))),
    idle_ttl_s_default: Math.max(0, Math.round(num(r.idle_ttl_s_default, num(r.idle_ttl_s)))),
    allowed_residency: (allowed.includes(residency) ? allowed : [...allowed, residency]) as Residency[],
    last_load_ms: typeof r.last_load_ms === 'number' && r.last_load_ms >= 0 ? r.last_load_ms : null,
    loaded_since: typeof r.loaded_since === 'string' ? r.loaded_since : null,
    last_error: str(r.last_error),
    supported: r.supported !== false,
    unsupported_reason: str(r.unsupported_reason),
  }
}

/** `GET /v1/tools`' answer: its entries (`data`) and the memory budget. */
export function toolsOf(raw: unknown): Tools {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const list = Array.isArray(r.data) ? r.data : []
  const b = r.budget as Record<string, unknown> | undefined
  return {
    entries: list.map(toolEntryOf).filter((e): e is ToolEntry => e !== null),
    budget: b && typeof b === 'object' && num(b.ceiling_bytes) > 0 ? { ceiling_bytes: num(b.ceiling_bytes), booked_bytes: num(b.booked_bytes), always_bytes: num(b.always_bytes) } : null,
    residency: list.some((e) => !!e && typeof e === 'object' && 'residency' in e),
  }
}

/** A model's name from `/v1/models` (as the voice page shortens it: no parenthesised detail), else its id. */
export function modelNames(models: unknown): Record<string, string> {
  const out: Record<string, string> = {}
  if (!Array.isArray(models)) return out
  for (const m of models) {
    if (!m || typeof m !== 'object' || typeof (m as { id?: unknown }).id !== 'string') continue
    const { id, name } = m as { id: string; name?: unknown }
    out[id] = (typeof name === 'string' && name.replace(/\s*[(,;].*$/, '')) || id
  }
  return out
}

export const gib = (bytes: number) => `${(bytes / 2 ** 30).toFixed(1)} GiB`

/** "loads in ~8 s" from the model's last load; a model the box never loaded says so. */
export function loadsIn(ms: number | null): string {
  if (ms === null) return 'load time not measured yet'
  if (ms < 1000) return 'loads in under 1 s'
  return `loads in ~${Math.round(ms / 1000)} s`
}

export type ModelRow = {
  id: string
  name: string
  kind: string
  dot: 'ready' | 'busy' | 'loading' | 'failed' | 'offline' | 'disabled'
  state: string
  memory: string
  loads: string
  residency: Residency
  choices: Array<{ value: Residency; label: string }>
  idle_ttl_s: number
  /** The idle field shows for a model that is loaded and unloaded (not always loaded, not disabled). */
  idle_shown: boolean
  disabled: boolean
  /** A disabled model a voice call needs: calls that rely on it fail until it is enabled again. */
  warning: string
  is_default: boolean
  default_text: string
  error: string
}

export type BudgetView = { always_pct: number; booked_pct: number; text: string; over: boolean }

export type ModelsView = {
  available: boolean
  /** The card's first line, naming the server. */
  intro: string
  /** Why nothing is listed (not paired, not reachable, an older box); empty when the list is there. */
  problem: string
  rows: ModelRow[]
  /** The models the server cannot run, with its reason: shown on request, never changed here. */
  unsupported: Array<{ id: string; name: string; kind: string; reason: string }>
  /** "2 models not available on studio", or empty when every model is. */
  unsupported_text: string
  budget: BudgetView | null
  /** Why the pickers are off (no `manage` scope); empty when they may be changed. */
  read_only: string
  at: number
}

const pct = (part: number, whole: number) => Math.max(0, Math.min(100, Math.round((part / whole) * 1000) / 10))

export function budgetView(b: Budget | null): BudgetView | null {
  if (!b) return null
  const loadedOnDemand = Math.max(0, b.booked_bytes - b.always_bytes)
  const always = pct(b.always_bytes, b.ceiling_bytes)
  return {
    always_pct: always,
    booked_pct: Math.min(pct(loadedOnDemand, b.ceiling_bytes), Math.round((100 - always) * 10) / 10),
    text: `Always loaded ${gib(b.always_bytes)} · in use now ${gib(Math.max(b.booked_bytes, b.always_bytes))} · ceiling ${gib(b.ceiling_bytes)}`,
    over: b.booked_bytes > b.ceiling_bytes,
  }
}

const ttlText = (s: number) => (s === 0 ? 'right after each use' : `after ${s} s idle`)

export function modelRow(e: ToolEntry, names: Record<string, string>): ModelRow {
  const disabled = e.residency === 'disabled'
  const failed = !disabled && (e.state === 'failed' || (e.state === 'unloaded' && e.last_error !== ''))
  const dot = disabled ? 'disabled' : failed ? 'failed' : e.state === 'ready' ? 'ready' : e.state === 'busy' ? 'busy' : e.state === 'loading' ? 'loading' : 'offline'
  const name = names[e.id] ?? e.id
  // Disabled or always loaded: the idle time does not apply, so it does not count against the default either.
  const noIdle = (r: Residency) => r === 'always' || r === 'disabled'
  return {
    id: e.id,
    name,
    kind: KIND_LABEL[e.kind] ?? e.kind,
    dot,
    state: disabled ? 'disabled' : failed ? 'failed' : (STATE_LABEL[e.state] ?? e.state),
    memory: gib(e.resident_bytes),
    loads: loadsIn(e.last_load_ms),
    residency: e.residency,
    choices: e.allowed_residency.map((value) => ({ value, label: RESIDENCY_LABEL[value] })),
    idle_ttl_s: e.idle_ttl_s,
    idle_shown: !noIdle(e.residency),
    disabled,
    warning: disabled && VOICE_KINDS.has(e.kind) ? VOICE_WARNING : '',
    is_default: e.residency === e.residency_default && (noIdle(e.residency) || e.idle_ttl_s === e.idle_ttl_s_default),
    default_text: `Default: ${RESIDENCY_LABEL[e.residency_default].toLowerCase()}${noIdle(e.residency_default) ? '' : `, unloads ${ttlText(e.idle_ttl_s_default)}`}`,
    error: e.last_error,
  }
}

export const NO_MANAGE = 'This computer’s pairing may read the models but not change them (that needs the manage scope). Pair again with a code made without --scopes to change them here.'

/**
 * The LocalRouter server as the card names it: its machine name from `/v1/capabilities` (`server.machine`) when it
 * gives one, else "this LocalRouter server". The card runs against any host (a Spark, a Mac Studio, a PC).
 */
export function serverName(machine: string | null | undefined): string {
  return machine?.trim() || 'this LocalRouter server'
}
export const oldBoxModels = (server: string) => `The LocalRouter on ${server} has no model settings yet: update it there to choose which models stay loaded.`
export const modelsIntro = (server: string) => `Which models stay loaded on ${server} and which it shuts down to free memory. The settings are kept there for every device, and a change applies at once.`

/** The Models card: the rows, the memory bar and whether they may be changed. */
export function modelsView(o: { tools: Tools | null; names: Record<string, string>; manage: boolean; server: string; problem?: string; at: number }): ModelsView {
  const intro = modelsIntro(o.server)
  if (!o.tools) return { available: false, intro, problem: o.problem || 'Not read yet', rows: [], unsupported: [], unsupported_text: '', budget: null, read_only: '', at: o.at }
  const supported = o.tools.residency || o.tools.entries.length === 0
  const away = o.tools.entries.filter((e) => !e.supported)
  return {
    available: true,
    intro,
    problem: o.problem ?? '',
    rows: o.tools.entries.filter((e) => e.supported).map((e) => modelRow(e, o.names)),
    unsupported: away.map((e) => ({ id: e.id, name: o.names[e.id] ?? e.id, kind: KIND_LABEL[e.kind] ?? e.kind, reason: e.unsupported_reason || `${o.server} cannot run it` })),
    unsupported_text: away.length ? `${away.length} ${away.length === 1 ? 'model' : 'models'} not available on ${o.server}` : '',
    budget: budgetView(o.tools.budget),
    read_only: !supported ? oldBoxModels(o.server) : o.manage ? '' : NO_MANAGE,
    at: o.at,
  }
}

export type SettingsChange = { residency?: Residency; idle_ttl_s?: number; reset?: true }

/** The longest idle time the field takes: a week. */
export const MAX_IDLE_S = 7 * 24 * 3600

/** What the page asked for, as the body for the box; a value the box would refuse is an Error saying why. */
export function settingsChange(input: unknown): SettingsChange {
  const i = (input && typeof input === 'object' ? input : {}) as Record<string, unknown>
  if (i.reset === true) return { reset: true }
  const out: SettingsChange = {}
  if (i.residency !== undefined) {
    if (!RESIDENCIES.includes(i.residency as Residency)) throw new Error('Choose how the model is kept')
    out.residency = i.residency as Residency
  }
  if (i.idle_ttl_s !== undefined) {
    const text = String(i.idle_ttl_s).trim()
    if (!/^\d+$/.test(text) || Number(text) > MAX_IDLE_S) throw new Error(`Enter whole seconds, from 0 (right after each use) to ${MAX_IDLE_S}`)
    out.idle_ttl_s = Number(text)
  }
  if (!Object.keys(out).length) throw new Error('Nothing to change')
  return out
}

/** Why the server did not take a change, for the row; `denied`: this token may not change models at all. */
export function settingsRefusal(error: unknown, server: string): { message: string; denied: boolean } {
  const message = error instanceof Error ? error.message : String(error)
  if (error instanceof SparkError) {
    if (error.status === 409) return { message, denied: false }
    if (error.status === 401 || error.status === 403) return { message: NO_MANAGE, denied: true }
    if (error.status === 404 || error.status === 405) return { message: oldBoxModels(server), denied: false }
    if (error.status === 400) return { message: `Refused by ${server}: ${message}`, denied: false }
  }
  return { message: `Not changed on ${server}: ${message}`, denied: false }
}

// ------------------------------------------------------------------ a call's voice models loading

export type CallLoading = { done: boolean; text: string; detail: string; progress: number | null; stopped?: true }

/**
 * The banner for a call whose models load when it starts: from `localrouter.session.loading` (`models`: each one's id,
 * state and time so far) and the server's last load time of each, a rough fraction done; `localrouter.session.ready`
 * ends it. A model in either that is "disabled" never loads, so the banner says so and stops (no progress, no spin).
 * Null for any other event.
 */
export function callLoading(ev: Record<string, unknown>, lastLoad: Record<string, number | null>, names: Record<string, string>, server = serverName(null)): CallLoading | null {
  if (ev.type !== 'localrouter.session.ready' && ev.type !== 'localrouter.session.loading') return null
  const models = (Array.isArray(ev.models) ? ev.models : []).filter((m): m is Record<string, unknown> => !!m && typeof m === 'object' && typeof (m as { id?: unknown }).id === 'string')
  const off = models.filter((m) => m.state === 'disabled').map((m) => names[m.id as string] ?? (m.id as string))
  if (off.length) {
    const list = off.length === 1 ? off[0] : `${off.slice(0, -1).join(', ')} and ${off[off.length - 1]}`
    return { done: false, stopped: true, text: 'Voice calls can’t start', detail: `${list} ${off.length === 1 ? 'is' : 'are'} disabled on ${server}. Enable ${off.length === 1 ? 'it' : 'them'} in Settings › Models to use voice calls.`, progress: null }
  }
  if (ev.type === 'localrouter.session.ready') {
    const ms = num(ev.loaded_ms, -1)
    return { done: true, text: 'Voice models ready', detail: ms >= 0 ? `loaded in ${(ms / 1000).toFixed(1)} s` : '', progress: 1 }
  }
  let known = 0
  let sum = 0
  const parts = models.map((m) => {
    const id = m.id as string
    const name = names[id] ?? id
    const ready = m.state === 'ready' || m.state === 'busy'
    const elapsed = num(m.elapsed_ms)
    const expected = lastLoad[id] ?? null
    if (ready) {
      known++
      sum += 1
      return `${name} ready`
    }
    if (expected && expected > 0) {
      known++
      sum += Math.min(0.95, elapsed / expected)
      return `${name} ${Math.round(elapsed / 1000)} s of ~${Math.max(1, Math.round(expected / 1000))} s`
    }
    return `${name} ${Math.round(elapsed / 1000)} s`
  })
  // Only an estimate for every model makes a bar; otherwise the banner just counts.
  const progress = models.length && known === models.length ? Math.round((sum / models.length) * 100) / 100 : null
  return { done: false, text: 'Loading voice models…', detail: parts.join(' · '), progress }
}
