import { execFile } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { createServer } from 'node:net'
import type { AddressInfo } from 'node:net'
import { hostname, tmpdir } from 'node:os'
import { dirname, extname, join, normalize, sep } from 'node:path'
import { app, BrowserWindow, dialog, globalShortcut, ipcMain, Menu, nativeImage, Notification, powerSaveBlocker, protocol, safeStorage, session, shell, systemPreferences, Tray } from 'electron'
import type { IpcMainEvent, IpcMainInvokeEvent, MenuItemConstructorOptions, WebContents } from 'electron'
import { AddressError, classify, CodeError, connectionOf, connectView, normalizeCode, parsePairLink, peerBases, serverOf, supportsApp, tailnetPeers, tooOld } from './connect.ts'
import type { ConnectError, Connection, Draft } from './connect.ts'
import { grantStates, signingId } from './grants.ts'
import type { Permission } from './grants.ts'
import { createLogger } from '../../../desktop-helper/src/log.ts'
import type { Card, PresenceGate } from '../executor/broker.ts'
import { LegacyBridge } from '../executor/legacy.ts'
import { dataDir, executorSocket } from '../executor/paths.ts'
import { ExecutorClient } from '../executor/server.ts'
import type { PageItem } from '../harness/journal.ts'
import { prepareVideo, restoreVoice } from '../harness/models.ts'
import { pageView } from '../harness/realtime.ts'
import type { ClientToolsSession } from '../harness/realtime.ts'
import { startRuntime } from '../harness/runtime.ts'
import { createPlatform } from '../platform/index.ts'
import { stopShortcut } from '../platform/shortcut.ts'
import type { PlatformPresence } from '../platform/types.ts'
import type { Runtime } from '../harness/runtime.ts'
import { normalizeBase, pair, Spark } from '../harness/spark.ts'
import type { Fetch } from '../harness/spark.ts'

/**
 * LocalRouter for the desktop: the reference harness on the user's computer (docs/HARNESS.md). The models stay on the
 * Spark; this app runs the harness (tools, history, realtime in client tool mode) and the executor (policy, approvals,
 * budgets, audit) that is the final authority over this computer.
 *
 * The UI is the repo's `/voice` page, bundled (never loaded from the box), with a small shim: its HTTP calls go to
 * `lr://app/...`, which this process answers. `/v1/computer/*` is served by the local executor (so the page's approval
 * and task cards are this computer's), everything else is passed to the Spark with the device token, which the page
 * never sees. Its realtime socket is bridged to the harness's client-tools session. Approving needs Touch ID where the
 * Mac has it, asked by this process, so a click in the page alone never approves.
 */
const VERSION = app.getVersion()
const log = createLogger()
const UI_DIR = join(__dirname, 'ui')
const PRELOAD = join(__dirname, 'preload.cjs')
const ORIGIN = 'lr://app'

/** `grants`: the build each macOS permission was last seen working in (to tell a stale grant from a revoked one). */
type Settings = { base: string; device: { id: string; name: string } | null; name: string; shared: string[]; legacy: boolean; grants?: Partial<Record<Permission, string>> }
type Secrets = { token: string; legacy_token: string }

protocol.registerSchemesAsPrivileged([{ scheme: 'lr', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true, codeCache: true } }])

// ------------------------------------------------------------------ settings and secrets

let dir = dataDir()
const settingsPath = () => join(dir, 'settings.json')
const secretsPath = () => join(dir, 'secrets.bin')
let memorySecrets: Secrets = { token: '', legacy_token: '' }

function loadSettings(): Settings {
  const dflt: Settings = { base: '', device: null, name: computerName(), shared: [], legacy: false }
  try {
    return { ...dflt, ...(JSON.parse(readFileSync(settingsPath(), 'utf8')) as Partial<Settings>) }
  } catch {
    return dflt
  }
}

function saveSettings(s: Settings) {
  mkdirSync(dir, { recursive: true, mode: 0o700 })
  writeFileSync(settingsPath(), `${JSON.stringify(s, null, 1)}\n`, { mode: 0o600 })
}

/** Tokens live in the OS keychain's protection (safeStorage); without it, only for this run. */
function loadSecrets(): Secrets {
  if (!safeStorage.isEncryptionAvailable() || !existsSync(secretsPath())) return memorySecrets
  try {
    return { ...memorySecrets, ...(JSON.parse(safeStorage.decryptString(readFileSync(secretsPath()))) as Partial<Secrets>) }
  } catch {
    return memorySecrets
  }
}

function saveSecrets(s: Secrets) {
  memorySecrets = s
  if (!safeStorage.isEncryptionAvailable()) return log.warn('secrets.session_only', { why: 'no OS encryption available' })
  mkdirSync(dir, { recursive: true, mode: 0o700 })
  writeFileSync(secretsPath(), safeStorage.encryptString(JSON.stringify(s)), { mode: 0o600 })
}

function computerName() {
  return hostname().replace(/\.local$/i, '') || 'This computer'
}

// ------------------------------------------------------------------ the presence gate

/**
 * The person's presence for an approval: Touch ID where the Mac has it, the platform's own check where it has one
 * (Windows Hello), otherwise a native confirmation dialog (weaker: documented in the README).
 */
class AppPresenceGate implements PresenceGate {
  private os: PlatformPresence | null = null
  private osReady = false

  /** The running platform's check, once the executor has started; its availability is read once. */
  async use(presence: PlatformPresence | null) {
    this.os = presence
    this.osReady = presence ? await presence.available().catch(() => false) : false
  }

  get method() {
    if (process.platform === 'darwin' && systemPreferences.canPromptTouchID()) return 'Touch ID'
    if (this.os && this.osReady) return this.os.method
    return 'a confirmation on this computer'
  }

  async verify(card: Card): Promise<boolean> {
    if (process.platform === 'darwin' && systemPreferences.canPromptTouchID()) {
      try {
        await systemPreferences.promptTouchID(`let Jarvis ${card.action}`.slice(0, 200))
        return true
      } catch {
        return false
      }
    }
    if (this.os && this.osReady) return this.os.verify(`Let Jarvis ${card.action}`.slice(0, 200)).catch(() => false)
    const where = [card.app, card.title, card.url].filter(Boolean).join(' · ')
    const options = {
      type: 'warning' as const,
      buttons: ['Deny', 'Approve'],
      defaultId: 0,
      cancelId: 0,
      noLink: true,
      message: `Let Jarvis ${card.action}?`,
      detail: `${where}\n${card.kind === 'type' ? `Text: ${card.text}\n` : ''}Why it asks: ${card.why}\nFor: ${card.goal}\nAsked by: ${card.requester}`,
    }
    const r = mainWindow ? await dialog.showMessageBox(mainWindow, options) : await dialog.showMessageBox(options)
    return r.response === 1
  }
}

const gate = new AppPresenceGate()
const STOP = stopShortcut(process.platform)
/** The tray's icon off macOS (a menu bar title needs none): an empty image is invisible in a Windows or Linux tray. */
const TRAY_ICON = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAg0lEQVR42u2XSw4AERBE3cfVHMLNzc5iMj41tCqiEuv3QvTHud3iQ4q1swKaGmeuDAD+FGGAx0UmwnEJAzgmYQTPEkx4XcLw6vueYhG8fAuowDtDAgi8lV8S2wj05gqcK3DuL6AXIqVSzGtG9HYsMZBIjGT0oVRiLJdYTGRWM4nl1DIPCdZ3uuZOWEUAAAAASUVORK5CYII='

// ------------------------------------------------------------------ the runtime

let runtime: Runtime | null = null
let legacy: LegacyBridge | null = null
let settings: Settings
let mainWindow: BrowserWindow | null = null
let setupWindow: BrowserWindow | null = null
let tray: Tray | null = null
const sessions = new Map<string, { session: ClientToolsSession; owner: WebContents }>()

function machineKind(): 'native' | 'fake' {
  return process.env.LOCALROUTER_COMPANION_FAKE === '1' ? 'fake' : 'native'
}

async function startRuntimeFromSettings(): Promise<void> {
  await stopRuntime()
  const secrets = loadSecrets()
  if (!settings.base || !secrets.token) return
  runtime = await startRuntime({ base: settings.base, token: secrets.token, dir, name: settings.name, machine: machineKind(), gate, shared: () => settings.shared, listen: true, ownApps: ['LocalRouter', 'Electron'], log })
  await gate.use(runtime.executor.built.platform.presence)
  runtime.executor.broker.on((e) => {
    if (e.type === 'approval') {
      showMain()
      if (Notification.isSupported()) new Notification({ title: 'Jarvis asks for your OK', body: `${e.card.action}. Approve it in LocalRouter.` }).show()
    }
    // A card nobody answered: say so, since the task ended while the person may have been away from the window.
    if (e.type === 'approval.end' && e.outcome === 'expired' && Notification.isSupported()) new Notification({ title: 'Jarvis stopped', body: 'Nobody approved the step in time, so nothing was done.' }).show()
    if (e.type === 'task' || e.type === 'approval' || e.type === 'approval.end') refreshTray()
    if (e.type === 'task' && e.task.status !== 'running' && /Accessibility|Screen Recording/i.test(e.task.reason)) void checkStale()
  })
  if (settings.legacy && secrets.legacy_token) {
    legacy = new LegacyBridge({ url: settings.base, token: secrets.legacy_token, name: settings.name, version: VERSION, broker: runtime.executor.broker, access: () => runtime!.executor.built.platform.access(), log })
    legacy.start()
  }
  refreshTray()
}

/** A computer task failed for lack of access: if macOS kept an older build's grant, say so and show the fix. */
async function checkStale() {
  const p = await permissions().catch(() => null)
  if (!p || (!p.stale.accessibility && !p.stale.screen && !p.stale.microphone)) return
  if (Notification.isSupported()) new Notification({ title: 'LocalRouter needs its permission again', body: STALE_GRANT }).show()
  showSetup()
}

async function stopRuntime() {
  for (const { session: s } of sessions.values()) s.close()
  sessions.clear()
  legacy?.close()
  legacy = null
  await runtime?.close()
  runtime = null
}

// ------------------------------------------------------------------ lr://app: the UI, the local executor, the Spark

const TYPES: Record<string, string> = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png', '.json': 'application/json' }
// The page may talk only to this app; media it shows comes through it too.
const CSP = "default-src 'self'; script-src 'self' 'unsafe-inline' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; media-src 'self' blob: data:; connect-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
const failure = (status: number, message: string) => json({ error: { message, type: 'companion_error' } }, status)

async function staticFile(path: string): Promise<Response> {
  const rel = path === '/' || path === '/voice' ? 'voice.html' : path.slice(1)
  const file = normalize(join(UI_DIR, rel))
  if (!file.startsWith(UI_DIR + sep)) return new Response('not found', { status: 404 })
  try {
    const data = await readFile(file)
    const type = TYPES[extname(file)] ?? 'application/octet-stream'
    return new Response(data, { headers: { 'content-type': type, ...(type.startsWith('text/html') ? { 'content-security-policy': CSP } : {}) } })
  } catch {
    return new Response('not found', { status: 404 })
  }
}

async function computerRoute(req: Request, path: string, url: URL): Promise<Response> {
  if (!runtime) return failure(503, 'LocalRouter is not paired with a Spark yet')
  const broker = runtime.executor.broker
  if (req.method === 'GET' && path === '/v1/computer/status') return json(broker.status())
  if (req.method === 'GET' && path === '/v1/computer/approvals') return json({ data: broker.approvals() })
  if (req.method === 'GET' && path === '/v1/computer/audit') return json({ data: broker.audit(Number(url.searchParams.get('limit') ?? 100)) })
  const body = req.method === 'POST' ? ((await req.json().catch(() => ({}))) as Record<string, unknown>) : {}
  if (req.method === 'POST' && path === '/v1/computer/stop') return json({ stopped: broker.stop(typeof body.task_id === 'string' ? body.task_id : undefined, 'the LocalRouter window') })
  const m = /^\/v1\/computer\/approvals\/([\w-]+)$/.exec(path)
  if (req.method === 'POST' && m) {
    if (body.decision !== 'approve' && body.decision !== 'deny') return failure(400, 'decision is approve or deny')
    const r = await broker.answer(m[1], body.decision, 'the LocalRouter window')
    return r.ok ? json({ ok: true }) : failure(r.status, r.error)
  }
  return failure(404, 'not a computer route of this app')
}

const PASS_REQUEST = ['content-type', 'accept', 'range']
const PASS_RESPONSE = ['content-type', 'content-length', 'content-range', 'accept-ranges', 'cache-control', 'x-sample-rate', 'x-channels', 'x-sample-format', 'last-modified', 'etag']

async function proxy(req: Request, pathAndQuery: string): Promise<Response> {
  if (!runtime) return failure(503, 'LocalRouter is not paired with a Spark yet: open Settings')
  const headers: Record<string, string> = {}
  for (const h of PASS_REQUEST) {
    const v = req.headers.get(h)
    if (v) headers[h] = v
  }
  const body = req.method === 'GET' || req.method === 'HEAD' ? null : await req.arrayBuffer()
  try {
    const r = await runtime.spark.request(pathAndQuery, { method: req.method, headers, body, signal: req.signal })
    const out = new Headers()
    for (const h of PASS_RESPONSE) {
      const v = r.headers.get(h)
      if (v) out.set(h, v)
    }
    return new Response(r.body, { status: r.status, headers: out })
  } catch (error) {
    return failure(502, `the Spark did not answer: ${error instanceof Error ? error.message : String(error)}`)
  }
}

function registerProtocol() {
  protocol.handle('lr', async (req) => {
    const url = new URL(req.url)
    if (url.host !== 'app') return new Response('not found', { status: 404 })
    const path = decodeURIComponent(url.pathname)
    if (path.startsWith('/v1/computer/')) return computerRoute(req, path, url)
    if (path.startsWith('/v1/') || path === '/mcp' || path === '/health') return proxy(req, url.pathname + url.search)
    return staticFile(path)
  })
}

// ------------------------------------------------------------------ IPC (checked: only this app's own pages)

const fromPage = (e: IpcMainEvent | IpcMainInvokeEvent, page: string) => {
  const u = e.senderFrame?.url ?? ''
  return u === `${ORIGIN}/${page}` || u.startsWith(`${ORIGIN}/${page}?`) || u.startsWith(`${ORIGIN}/${page}#`) || (page === 'voice.html' && (u === `${ORIGIN}/` || u.startsWith(`${ORIGIN}/voice`)))
}

let seq = 0

function registerIpc() {
  // The voice page's realtime socket, bridged to a client-tools session.
  ipcMain.handle('rt:open', async (e) => {
    if (!fromPage(e, 'voice.html')) throw new Error('not allowed')
    if (!runtime) throw new Error('LocalRouter is not paired with a Spark yet: open Settings')
    if (process.platform === 'darwin') await systemPreferences.askForMediaAccess('microphone').catch(() => false)
    const id = `rt${++seq}`
    const owner = e.sender
    // The page keeps the one conversation (typed and spoken) and mirrors it into the journal (`host:save`).
    const s = runtime.session({
      onEvent: (ev) => {
        if (owner.isDestroyed() || !pageView(ev)) return
        // When this process had each piece of speech, so the page can tell a slow box from a slow hop to it.
        if (ev.type === 'response.audio.delta') ev.lr_main_ms = performance.timeOrigin + performance.now()
        owner.send('rt:event', id, JSON.stringify(ev))
      },
      onClose: (reason) => {
        sessions.delete(id)
        callEnded()
        if (!owner.isDestroyed()) owner.send('rt:close', id, 1006, reason)
      },
    })
    sessions.set(id, { session: s, owner })
    callStarted()
    s.connect().then(
      () => !owner.isDestroyed() && owner.send('rt:open', id),
      (error: unknown) => {
        sessions.delete(id)
        callEnded()
        if (!owner.isDestroyed()) owner.send('rt:close', id, 1000, error instanceof Error ? error.message : String(error))
      },
    )
    return id
  })
  ipcMain.on('rt:send', (e, id: unknown, text: unknown) => {
    const s = typeof id === 'string' ? sessions.get(id) : undefined
    if (!s || s.owner !== e.sender || typeof text !== 'string' || text.length > 4 << 20) return
    try {
      s.session.send(JSON.parse(text) as Record<string, unknown>)
    } catch {
      // Not JSON: dropped, as the box would refuse it.
    }
  })
  ipcMain.on('rt:close', (e, id: unknown) => {
    const s = typeof id === 'string' ? sessions.get(id) : undefined
    if (!s || s.owner !== e.sender) return
    sessions.delete(id as string)
    callEnded()
    s.session.close()
  })

  // Typed chat on the page: this computer's tools, run here like a call's, under the same executor.
  const host = (channel: string, fn: (...args: unknown[]) => unknown) =>
    ipcMain.handle(channel, async (e, ...args: unknown[]) => {
      if (!fromPage(e, 'voice.html')) throw new Error('not allowed')
      return fn(...args)
    })
  const running = new Map<string, AbortController>()
  host('host:tools', () => runtime?.chat.chat() ?? [])
  host('host:run', async (callId, name, args, utterance) => {
    if (!runtime) return { ok: false, output: JSON.stringify({ error: 'LocalRouter is not paired with a Spark yet' }) }
    const id = String(callId)
    const abort = new AbortController()
    running.set(id, abort)
    try {
      const r = await runtime.chat.run(String(name), String(args ?? '{}'), { callId: id, utterance: String(utterance ?? ''), utteranceAt: Date.now(), signal: abort.signal })
      return { ok: r.ok, output: r.output }
    } finally {
      running.delete(id)
    }
  })
  host('host:cancel', (callId) => {
    running.get(String(callId))?.abort('stopped in the chat')
    return true
  })
  host('host:save', (conversation, items) => {
    if (!runtime || typeof conversation !== 'string' || !Array.isArray(items)) return 0
    return runtime.journal.mirror(conversation, items as PageItem[])
  })
  host('host:prepareVideo', () => (runtime ? prepareVideo(runtime.spark) : { ok: false, text: 'not paired' }))
  host('host:restoreVoice', () => (runtime ? restoreVoice(runtime.spark) : { ok: false, text: 'not paired' }))

  // Setup and settings.
  const setup = (channel: string, fn: (...args: unknown[]) => unknown) =>
    ipcMain.handle(channel, async (e, ...args: unknown[]) => {
      if (!fromPage(e, 'setup.html')) throw new Error('not allowed')
      return fn(...args)
    })
  setup('setup:state', () => state())
  // Connect: the address (checked before any code is spent), the code, the connection, re-pair and disconnect.
  setup('setup:checkAddress', async (input) => {
    errors = {}
    let linkCode: string | null = null
    try {
      const text = String(input ?? '').trim()
      const link = parsePairLink(text)
      linkCode = link?.code ?? null
      const b = sparkBase(link ? link.url : text)
      const caps = await new Spark(b, '', timed(CHECK_S)).capabilities()
      if (typeof caps?.contract !== 'string') throw new SyntaxError('no contract in /v1/capabilities')
      if (!supportsApp(caps)) errors.address = tooOld()
      else draft = { base: b, server: serverOf(caps), repair: Boolean(runtime) && b === settings.base }
    } catch (error) {
      errors.address = classify(error, 'address', CHECK_S)
    }
    return { ...(await state()), link_code: errors.address ? null : linkCode }
  })
  setup('setup:pair', async (code, name) => {
    if (!draft) throw new Error('Check the Spark address first')
    delete errors.code
    try {
      const c = normalizeCode(String(code ?? ''))
      if (!c) throw new CodeError('not a code')
      const nm = String(name || settings.name).trim() || settings.name
      const p = await pair(draft.base, c, nm, timed(PAIR_S))
      settings = { ...settings, base: draft.base, device: p.device, name: nm }
      saveSettings(settings)
      saveSecrets({ ...loadSecrets(), token: p.token })
      draft = null
      errors = {}
      await startRuntimeFromSettings()
      await checkConnection()
    } catch (error) {
      errors.code = classify(error, 'code', PAIR_S)
    }
    return state()
  })
  setup('setup:check', async () => {
    await checkConnection()
    return state()
  })
  // Pairs again with the same Spark; the working token stays until a new code replaces it.
  setup('setup:repair', async () => {
    if (settings.base) draft = { base: settings.base, server: connection?.ok ? connection.server : null, repair: true }
    errors = {}
    return state()
  })
  setup('setup:cancel', async () => {
    draft = null
    errors = {}
    return state()
  })
  // Forgets the Spark here: its token, the older clients' token and the address. The box keeps the entry until revoked there.
  setup('setup:disconnect', async () => {
    await stopRuntime()
    settings = { ...settings, base: '', device: null, legacy: false }
    saveSettings(settings)
    saveSecrets({ token: '', legacy_token: '' })
    draft = null
    connection = null
    errors = {}
    return state()
  })
  setup('setup:discover', () => discoverSparks())
  setup('setup:resetGrant', async (kind) => {
    if (process.platform !== 'darwin') return { ok: false, text: 'not a Mac', state: await state() }
    const which = kind === 'screen' ? 'screen' : kind === 'microphone' ? 'microphone' : 'accessibility'
    const r = await resetGrant(which === 'screen' ? 'ScreenCapture' : which === 'microphone' ? 'Microphone' : 'Accessibility')
    if (r.ok) {
      // Asks again: the microphone with macOS's own prompt; Accessibility lists LocalRouter, off, and the pane opens.
      if (which === 'microphone') await systemPreferences.askForMediaAccess('microphone').catch(() => false)
      else {
        if (which === 'accessibility') systemPreferences.isTrustedAccessibilityClient(true)
        await shell.openExternal(PANES[which])
      }
      settings = { ...settings, grants: { ...(settings.grants ?? {}), [which]: undefined } }
      saveSettings(settings)
    }
    return { ...r, state: await state() }
  })
  setup('setup:relaunch', () => {
    app.relaunch()
    app.exit(0)
  })
  setup('setup:permission', async (kind) => {
    if (process.platform !== 'darwin') return state()
    if (kind === 'microphone') await systemPreferences.askForMediaAccess('microphone')
    if (kind === 'accessibility') {
      systemPreferences.isTrustedAccessibilityClient(true)
      await shell.openExternal('x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility')
    }
    if (kind === 'screen') await shell.openExternal('x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture')
    if (kind === 'automation') await shell.openExternal('x-apple.systempreferences:com.apple.preference.security?Privacy_Automation')
    return state()
  })
  setup('setup:addFolder', async () => {
    const r = await dialog.showOpenDialog({ title: 'Share a folder with Jarvis', properties: ['openDirectory'] })
    if (!r.canceled && r.filePaths[0] && !settings.shared.includes(r.filePaths[0])) {
      settings = { ...settings, shared: [...settings.shared, r.filePaths[0]] }
      saveSettings(settings)
    }
    return state()
  })
  setup('setup:removeFolder', (path) => {
    settings = { ...settings, shared: settings.shared.filter((p) => p !== path) }
    saveSettings(settings)
    return state()
  })
  setup('setup:enroll', async (name) => {
    if (!runtime) throw new Error('Pair with your Spark first')
    const { caller, token } = runtime.executor.enrollment.add(String(name))
    return { caller: { id: caller.id, name: caller.name }, token, socket: executorSocket(dir), command: join(process.resourcesPath ?? '', 'localrouter-executor-mcp'), state: await state() }
  })
  setup('setup:revoke', (id) => {
    runtime?.executor.enrollment.revoke(String(id))
    return state()
  })
  setup('setup:legacy', async (enabled, code) => {
    if (enabled) {
      if (!settings.base) throw new Error('Pair with your Spark first')
      const p = await pair(settings.base, String(code), `${settings.name} (computer control for older clients)`)
      saveSecrets({ ...loadSecrets(), legacy_token: p.token })
    } else saveSecrets({ ...loadSecrets(), legacy_token: '' })
    settings = { ...settings, legacy: Boolean(enabled) }
    saveSettings(settings)
    await startRuntimeFromSettings()
    return state()
  })
  setup('setup:control', (on) => {
    runtime?.executor.broker.setDisabled(!on)
    refreshTray()
    return state()
  })
  setup('setup:export', async () => {
    if (!runtime) return null
    const r = await dialog.showSaveDialog({ title: 'Export conversations', defaultPath: `localrouter-conversations-${new Date().toISOString().slice(0, 10)}.json` })
    if (r.canceled || !r.filePath) return null
    const j = runtime.journal
    writeFileSync(r.filePath, `${JSON.stringify({ format: 'localrouter-conversations/1', conversations: j.list().map((id) => j.export(id)) }, null, 1)}\n`, { mode: 0o600 })
    return r.filePath
  })
  setup('setup:openVoice', () => {
    showMain()
    setupWindow?.close()
  })
}

// While a call is on, macOS must not nap the app (App Nap delays timers and the hop that brings the speech to the page).
let napBlock: number | null = null
function callStarted() {
  if (napBlock === null) napBlock = powerSaveBlocker.start('prevent-app-suspension')
}
function callEnded() {
  if (sessions.size === 0 && napBlock !== null) {
    powerSaveBlocker.stop(napBlock)
    napBlock = null
  }
}

// ------------------------------------------------------------------ macOS permissions, and grants macOS kept from an older build

export const BUNDLE_ID = 'com.localrouter.companion'
export const STALE_GRANT = 'macOS kept a permission from a LocalRouter build signed differently, and it does nothing for this one. Reset it here (macOS asks again), or in System Settings remove LocalRouter and add it again.'

let identity: Promise<string | null> | null = null

/**
 * This build as far as a grant is concerned: its designated requirement (the bundle id and the signing certificate),
 * which every build signed with the stable identity shares, so an update keeps its grants. Read once; null when it
 * cannot be read (unsigned in development), and then no grant is called stale from history.
 */
function signingIdentity(): Promise<string | null> {
  identity ??= new Promise((resolve) => {
    // …/LocalRouter.app/Contents/MacOS/LocalRouter → …/LocalRouter.app
    const bundle = dirname(dirname(dirname(app.getPath('exe'))))
    execFile('/usr/bin/codesign', ['-d', '-r-', bundle], { timeout: 10_000 }, (error, stdout, stderr) => {
      const requirement = /designated => (.+)/.exec(`${stdout}${stderr}`)?.[1]
      resolve(error || !requirement ? null : signingId(requirement))
    })
  })
  return identity
}

/**
 * Each permission as macOS answers this process (the truth: Accessibility's trust check, the in-process TCC status for
 * Screen Recording and the Microphone), the Accessibility probe (an AX read through the executor) only to catch a
 * "yes" that does not work, and whether a grant is stale: missing now, and seen working under another signing
 * identity (the ad-hoc builds before the stable certificate). See grants.ts.
 */
async function permissions() {
  if (process.platform !== 'darwin') {
    // Linux and Windows have no TCC grants to go stale: the platform says what it needs and how to get it.
    const report = runtime ? await runtime.executor.built.platform.access().catch(() => null) : null
    return { microphone: 'n/a', accessibility: 'n/a', screen: 'n/a', stale: { accessibility: false, screen: false, microphone: false }, backend: report?.backend ?? '', requirements: report?.requirements ?? [] }
  }
  const microphone = systemPreferences.getMediaAccessStatus('microphone')
  const said = { accessibility: systemPreferences.isTrustedAccessibilityClient(false), screen: systemPreferences.getMediaAccessStatus('screen') === 'granted', microphone: microphone === 'granted' }
  const works = runtime ? await runtime.executor.built.platform.access().then((a) => ({ accessibility: a.accessibility }), () => null) : null
  const { state: grant, grants } = grantStates(said, works, settings.grants ?? {}, await signingIdentity())
  if (JSON.stringify(grants) !== JSON.stringify(settings.grants ?? {})) {
    settings = { ...settings, grants }
    saveSettings(settings)
  }
  const stale = { accessibility: grant.accessibility === 'stale', screen: grant.screen === 'stale', microphone: grant.microphone === 'stale' }
  return { microphone: stale.microphone ? 'stale' : microphone, accessibility: grant.accessibility, screen: grant.screen, stale }
}

/** `tccutil reset` for this app only: runs as the user, no admin. What it said, and whether it worked. */
function resetGrant(service: 'Accessibility' | 'ScreenCapture' | 'Microphone'): Promise<{ ok: boolean; text: string }> {
  return new Promise((resolve) => {
    execFile('/usr/bin/tccutil', ['reset', service, BUNDLE_ID], { timeout: 10_000 }, (error, stdout, stderr) => {
      const text = `${stdout}${stderr}`.trim()
      resolve({ ok: !error, text: text || (error ? error.message : 'done') })
    })
  })
}

const PANES: Record<string, string> = {
  accessibility: 'x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility',
  screen: 'x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture',
}

// ------------------------------------------------------------------ connect: the address, the code, the connection

const CHECK_S = 8
const PAIR_S = 15
let draft: Draft | null = null
let connection: Connection | null = null
let errors: Partial<Record<'address' | 'code', ConnectError>> = {}

/** fetch with a time limit, so a host that never answers is an error the page can name, not a spinner. */
const timed = (s: number): Fetch => (input, init) => {
  const limit = AbortSignal.timeout(s * 1000)
  return fetch(input, { ...init, signal: init?.signal ? AbortSignal.any([init.signal, limit]) : limit })
}

/** The Spark's base URL from what was typed, or an AddressError saying why not. */
function sparkBase(text: string): string {
  if (!text) throw new AddressError('Enter the address your Spark answers on')
  try {
    return normalizeBase(text)
  } catch (error) {
    throw new AddressError(error instanceof Error && !/invalid url/i.test(error.message) ? error.message : 'That does not read as an address')
  }
}

/** The connection step: the box's discovery document and what it says about this token, both open reads. */
async function checkConnection() {
  if (!runtime) {
    connection = null
    return
  }
  try {
    const spark = new Spark(settings.base, loadSecrets().token, timed(CHECK_S))
    const [caps, who] = await Promise.all([spark.capabilities(), spark.whoami()])
    connection = connectionOf(caps, who, Date.now())
  } catch (error) {
    connection = { ok: false, at: Date.now(), error: classify(error, 'connection', CHECK_S) }
  }
  log.info('connect.check', { ok: connection.ok, ...(connection.ok ? {} : { kind: connection.error.kind, detail: connection.error.detail }) })
}

const TAILSCALE = ['/Applications/Tailscale.app/Contents/MacOS/Tailscale', '/opt/homebrew/bin/tailscale', '/usr/local/bin/tailscale', '/usr/bin/tailscale']

function tailscaleStatus(): Promise<unknown> {
  const tryAt = (i: number): Promise<unknown> =>
    i >= TAILSCALE.length
      ? Promise.resolve(null)
      : new Promise((resolve) => {
          if (!existsSync(TAILSCALE[i])) return resolve(tryAt(i + 1))
          execFile(TAILSCALE[i], ['status', '--json'], { timeout: 4000, maxBuffer: 8 << 20 }, (error, stdout) => {
            if (error && !stdout) return resolve(tryAt(i + 1))
            try {
              resolve(JSON.parse(stdout))
            } catch {
              resolve(tryAt(i + 1))
            }
          })
        })
  return tryAt(0)
}

/**
 * LocalRouter boxes on this Mac's tailnet: each online peer's MagicDNS name, asked for `/v1/capabilities` (open, no
 * secrets) at `tailscale serve`'s ports. `tailnet`: whether Tailscale runs here, so the page can say so.
 */
async function discoverSparks(): Promise<{ tailnet: 'running' | 'off'; found: Array<{ base: string; host: string; version: string; machine: string }> }> {
  const peers = tailnetPeers(await tailscaleStatus())
  if (!peers) return { tailnet: 'off', found: [] }
  const probes = peers.slice(0, 16).map(async (p) => {
    for (const base of peerBases(p)) {
      const caps = await new Spark(base, '', timed(3)).capabilities().catch(() => null)
      if (caps && typeof caps.contract === 'string') return { base, host: p.host, version: caps.server?.version ?? '', machine: caps.server?.machine ?? '' }
    }
    return null
  })
  const found = (await Promise.all(probes)).filter((f) => f !== null)
  log.info('connect.discover', { peers: peers.length, found: found.length })
  return { tailnet: 'running', found }
}

async function state() {
  const permissionState = await permissions()
  const callers = runtime?.executor.enrollment.list().map((c) => ({ id: c.id, name: c.name, created: c.created })) ?? []
  return {
    version: VERSION,
    paired: Boolean(runtime),
    base: settings.base,
    device: settings.device,
    name: settings.name,
    shared: settings.shared,
    legacy: { enabled: settings.legacy, connected: legacy?.status.connected ?? false },
    presence: gate.method,
    secrets: safeStorage.isEncryptionAvailable() ? 'keychain' : 'this run only',
    control: runtime ? !runtime.executor.broker.disabled : false,
    machine: machineKind(),
    permissions: permissionState,
    stop_words: runtime?.executor.built.platform.pointer ? `${STOP.words}, the tray menu, or move the pointer to the top-left corner` : `${STOP.words} or the tray menu`,
    connect: { ...connectView({ paired: Boolean(runtime), base: settings.base, draft, connection, errors, permissions: permissionState }), draft, connection, errors },
    stale_message: STALE_GRANT,
    callers,
    mcp: { command: join(process.resourcesPath ?? '', 'localrouter-executor-mcp'), socket: executorSocket(dir) },
  }
}

// ------------------------------------------------------------------ windows, tray, menu

const webPreferences = { preload: PRELOAD, contextIsolation: true, sandbox: true, nodeIntegration: false, webSecurity: true, spellcheck: false, backgroundThrottling: false }

function harden(win: BrowserWindow) {
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) void shell.openExternal(url)
    return { action: 'deny' }
  })
  win.webContents.on('will-navigate', (e, url) => {
    if (!url.startsWith(`${ORIGIN}/`)) e.preventDefault()
  })
}

function showMain() {
  if (!runtime) return showSetup()
  if (!mainWindow || mainWindow.isDestroyed()) {
    mainWindow = new BrowserWindow({ width: 1180, height: 820, minWidth: 420, minHeight: 520, title: 'LocalRouter', show: false, webPreferences })
    harden(mainWindow)
    mainWindow.once('ready-to-show', () => mainWindow?.show())
    mainWindow.on('closed', () => {
      mainWindow = null
    })
    void mainWindow.loadURL(`${ORIGIN}/voice.html`)
  } else {
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.show()
    mainWindow.focus()
  }
}

function showSetup() {
  if (setupWindow && !setupWindow.isDestroyed()) return setupWindow.focus()
  setupWindow = new BrowserWindow({ width: 720, height: 820, title: 'LocalRouter settings', show: false, webPreferences })
  harden(setupWindow)
  setupWindow.once('ready-to-show', () => setupWindow?.show())
  setupWindow.on('closed', () => {
    setupWindow = null
  })
  void setupWindow.loadURL(`${ORIGIN}/setup.html`)
}

function stopAll(by: string) {
  const n = runtime?.executor.broker.stop(undefined, by, 'You stopped it.') ?? 0
  if (n && Notification.isSupported()) new Notification({ title: 'LocalRouter', body: 'Stopped what Jarvis was doing on this computer.' }).show()
}

function refreshTray() {
  if (!tray) return
  const broker = runtime?.executor.broker
  const busy = broker?.status().tasks.find((t) => t.status === 'running')
  tray.setTitle(busy ? 'LR ●' : 'LR')
  const items: MenuItemConstructorOptions[] = [
    { label: 'Open LocalRouter', click: () => showMain() },
    { label: 'Settings…', click: () => showSetup() },
    { type: 'separator' },
    { label: busy ? `Stop: ${busy.goal.slice(0, 40)}` : 'Nothing running on this computer', enabled: Boolean(busy), click: () => stopAll('the menu bar') },
    { label: 'Let Jarvis control this computer', type: 'checkbox', enabled: Boolean(broker), checked: broker ? !broker.disabled : false, click: (item) => { broker?.setDisabled(!item.checked); refreshTray() } },
    { type: 'separator' },
    { label: 'Quit LocalRouter', click: () => app.quit() },
  ]
  tray.setContextMenu(Menu.buildFromTemplate(items))
}

function buildMenu() {
  const template: MenuItemConstructorOptions[] = [
    { label: 'LocalRouter', submenu: [{ role: 'about' }, { type: 'separator' }, { label: 'Settings…', accelerator: 'CommandOrControl+,', click: () => showSetup() }, { label: 'Stop what Jarvis is doing', accelerator: STOP.accelerator, click: () => stopAll('the keyboard') }, { type: 'separator' }, { role: 'hide' }, { role: 'quit' }] },
    { role: 'editMenu' },
    { role: 'viewMenu' },
    { role: 'windowMenu' },
  ]
  Menu.setApplicationMenu(Menu.buildFromTemplate(template))
}

function permissionHandlers() {
  const allowed = (origin: string, permission: string, mediaTypes?: string[]) =>
    origin.startsWith(ORIGIN) && ((permission === 'media' && (mediaTypes ?? ['audio']).every((t) => t === 'audio')) || permission === 'clipboard-sanitized-write')
  session.defaultSession.setPermissionRequestHandler((wc, permission, done, details) => {
    const d = details as { requestingUrl?: string; mediaTypes?: string[] }
    done(allowed(d.requestingUrl ?? wc.getURL(), permission, d.mediaTypes))
  })
  session.defaultSession.setPermissionCheckHandler((_wc, permission, origin) => allowed(origin, permission))
}

// ------------------------------------------------------------------ smoke (CI): starts headless, checks, exits

async function smoke() {
  const result: Record<string, unknown> = { version: VERSION, platform: process.platform, arch: process.arch }
  let ok = false
  try {
    dir = mkdtempSync(join(tmpdir(), 'lr-smoke-'))
    app.setPath('userData', dir)
    await app.whenReady()
    registerProtocol()
    registerIpc()
    settings = loadSettings()
    runtime = await startRuntime({ base: 'http://127.0.0.1:9', token: '', dir, name: 'smoke', machine: 'fake', gate: { method: 'none (smoke)', verify: async () => false }, shared: () => [], listen: true })
    const win = new BrowserWindow({ show: false, webPreferences })
    await win.loadURL(`${ORIGIN}/voice.html`)
    result.page = await win.webContents.executeJavaScript('({ shim: window.__companionShim === true, bridge: typeof (window.companion && window.companion.realtime && window.companion.realtime.open) === "function", title: document.title, socket: String(WebSocket.name) })')
    result.status = await win.webContents.executeJavaScript('fetch("/v1/computer/status").then(r => r.json()).then(s => ({ computers: s.computers.length, placement: s.computers[0] && s.computers[0].placement }))')
    result.proxy = await win.webContents.executeJavaScript('fetch("/v1/models").then(r => r.status)')
    const { token } = runtime.executor.enrollment.add('smoke')
    const client = new ExecutorClient(executorSocket(dir), token)
    const caps = await client.request('capabilities')
    client.close()
    result.executor = caps.ok ? caps.result : caps.error
    result.secrets = safeStorage.isEncryptionAvailable() ? 'keychain' : 'this run only'
    result.presence = gate.method
    // The settings window: its page draws the steps, and a closed port is named as refused through the IPC.
    const setupWin = new BrowserWindow({ show: false, webPreferences })
    await setupWin.loadURL(`${ORIGIN}/setup.html`)
    const closed = await new Promise<number>((resolve) => {
      const srv = createServer()
      srv.listen(0, '127.0.0.1', () => {
        const port = (srv.address() as AddressInfo).port
        srv.close(() => resolve(port))
      })
    })
    result.setup = await setupWin.webContents.executeJavaScript(`new Promise((r) => { const t = () => (document.getElementById("version").textContent ? r() : setTimeout(t, 50)); t() }).then(() => window.companion.setup.checkAddress("http://127.0.0.1:${closed}")).then(s => ({ version: document.getElementById("version").textContent, steps: s.connect.steps.map(x => x.id + ":" + x.status).join(","), error: s.connect.errors.address && s.connect.errors.address.kind }))`)
    const page = result.page as { shim: boolean; bridge: boolean }
    const status = result.status as { computers: number; placement: string }
    const setupPage = result.setup as { version: string; error: string }
    ok = page.shim && page.bridge && status.computers === 1 && status.placement === 'device_executor' && caps.ok && result.proxy === 502 && setupPage.version === VERSION && setupPage.error === 'refused'
    // This OS's own platform, read only (what it detects and needs; nothing is clicked or typed): the CI job shows it.
    const native = await createPlatform('native', { dataDir: dir })
    result.native = await native.platform.access().then(
      (a) => ({ os: native.platform.os, backend: a.backend, accessibility: a.accessibility, screen_recording: a.screen_recording, input: a.input, missing: a.requirements.filter((r) => r.granted === false).map((r) => r.id) }),
      (error: unknown) => ({ os: native.platform.os, error: error instanceof Error ? error.message : String(error) }),
    )
    native.platform.close?.()
    setupWin.destroy()
    win.destroy()
  } catch (error) {
    result.error = error instanceof Error ? error.stack ?? error.message : String(error)
  }
  process.stdout.write(`COMPANION_SMOKE ${JSON.stringify({ ok, ...result })}\n`)
  await runtime?.close().catch(() => undefined)
  app.exit(ok ? 0 : 1)
}

// ------------------------------------------------------------------ start

if (process.argv.includes('--version')) {
  process.stdout.write(`LocalRouter ${VERSION}\n`)
  process.exit(0)
} else if (process.argv.includes('--smoke')) {
  void smoke()
} else if (!app.requestSingleInstanceLock()) {
  app.quit()
} else {
  app.setPath('userData', dir)
  app.on('second-instance', () => showMain())
  app.on('window-all-closed', () => {
    // Stays in the menu bar: Quit there ends it.
  })
  app.on('activate', () => showMain())
  app.on('will-quit', () => {
    globalShortcut.unregisterAll()
    void stopRuntime()
  })
  void app.whenReady().then(async () => {
    registerProtocol()
    registerIpc()
    permissionHandlers()
    buildMenu()
    settings = loadSettings()
    tray = new Tray(process.platform === 'darwin' ? nativeImage.createEmpty() : nativeImage.createFromDataURL(TRAY_ICON))
    tray.setToolTip('LocalRouter')
    if (!globalShortcut.register(STOP.accelerator, () => stopAll('the keyboard'))) log.warn('stop_shortcut.unavailable', { accelerator: STOP.accelerator })
    try {
      await startRuntimeFromSettings()
    } catch (error) {
      log.error('runtime.start_failed', { error: error instanceof Error ? error.message : String(error) })
    }
    refreshTray()
    if (runtime) showMain()
    else showSetup()
  })
}
