import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { hostname, tmpdir } from 'node:os'
import { extname, join, normalize, sep } from 'node:path'
import { app, BrowserWindow, dialog, globalShortcut, ipcMain, Menu, nativeImage, Notification, protocol, safeStorage, session, shell, systemPreferences, Tray } from 'electron'
import type { IpcMainEvent, IpcMainInvokeEvent, MenuItemConstructorOptions, WebContents } from 'electron'
import { createLogger } from '../../../desktop-helper/src/log.ts'
import type { Card, PresenceGate } from '../executor/broker.ts'
import { LegacyBridge } from '../executor/legacy.ts'
import { dataDir, executorSocket } from '../executor/paths.ts'
import { ExecutorClient } from '../executor/server.ts'
import { pageView } from '../harness/realtime.ts'
import type { ClientToolsSession } from '../harness/realtime.ts'
import { startRuntime } from '../harness/runtime.ts'
import type { Runtime } from '../harness/runtime.ts'
import { normalizeBase, pair, Spark } from '../harness/spark.ts'

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

type Settings = { base: string; device: { id: string; name: string } | null; name: string; shared: string[]; legacy: boolean }
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

/** Touch ID where this Mac has it; otherwise a native confirmation (weaker: documented in the README). */
class MacPresenceGate implements PresenceGate {
  get method() {
    return process.platform === 'darwin' && systemPreferences.canPromptTouchID() ? 'Touch ID' : 'a confirmation on this computer'
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

const gate = new MacPresenceGate()

// ------------------------------------------------------------------ the runtime

let runtime: Runtime | null = null
let legacy: LegacyBridge | null = null
let settings: Settings
let mainWindow: BrowserWindow | null = null
let setupWindow: BrowserWindow | null = null
let tray: Tray | null = null
const sessions = new Map<string, { session: ClientToolsSession; owner: WebContents }>()

function machineKind(): 'mac' | 'fake' {
  return process.platform === 'darwin' && process.env.LOCALROUTER_COMPANION_FAKE !== '1' ? 'mac' : 'fake'
}

async function startRuntimeFromSettings(): Promise<void> {
  await stopRuntime()
  const secrets = loadSecrets()
  if (!settings.base || !secrets.token) return
  runtime = await startRuntime({ base: settings.base, token: secrets.token, dir, name: settings.name, machine: machineKind(), gate, shared: () => settings.shared, listen: true, ownApps: ['LocalRouter', 'Electron'], log })
  runtime.executor.broker.on((e) => {
    if (e.type === 'approval') {
      showMain()
      if (Notification.isSupported()) new Notification({ title: 'Jarvis asks for your OK', body: `${e.card.action}. Approve it in LocalRouter.` }).show()
    }
    if (e.type === 'task' || e.type === 'approval' || e.type === 'approval.end') refreshTray()
  })
  if (settings.legacy && secrets.legacy_token) {
    legacy = new LegacyBridge({ url: settings.base, token: secrets.legacy_token, name: settings.name, version: VERSION, broker: runtime.executor.broker, access: runtime.executor.built.access, log })
    legacy.start()
  }
  refreshTray()
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
    const conversation = runtime.journal.create()
    const s = runtime.session({
      conversation,
      onEvent: (ev) => {
        if (!owner.isDestroyed() && pageView(ev)) owner.send('rt:event', id, JSON.stringify(ev))
      },
      onClose: (reason) => {
        sessions.delete(id)
        if (!owner.isDestroyed()) owner.send('rt:close', id, 1006, reason)
      },
    })
    sessions.set(id, { session: s, owner })
    s.connect().then(
      () => !owner.isDestroyed() && owner.send('rt:open', id),
      (error: unknown) => {
        sessions.delete(id)
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
    s.session.close()
  })

  // Setup and settings.
  const setup = (channel: string, fn: (...args: unknown[]) => unknown) =>
    ipcMain.handle(channel, async (e, ...args: unknown[]) => {
      if (!fromPage(e, 'setup.html')) throw new Error('not allowed')
      return fn(...args)
    })
  setup('setup:state', () => state())
  setup('setup:pair', async (base, code, name) => {
    const b = normalizeBase(String(base))
    const caps = await new Spark(b, '').capabilities().catch((error: unknown) => {
      throw new Error(`That address does not answer as a LocalRouter box: ${error instanceof Error ? error.message : String(error)}`)
    })
    if (!caps.realtime?.tool_execution?.includes('client')) throw new Error('That box is too old for this app: it needs client tool execution (contract 2026-10-23). Update it first.')
    const p = await pair(b, String(code), String(name || settings.name))
    settings = { ...settings, base: b, device: p.device, name: String(name || settings.name) }
    saveSettings(settings)
    saveSecrets({ ...loadSecrets(), token: p.token })
    await startRuntimeFromSettings()
    return state()
  })
  setup('setup:unpair', async () => {
    await stopRuntime()
    settings = { ...settings, device: null, legacy: false }
    saveSettings(settings)
    saveSecrets({ token: '', legacy_token: '' })
    return state()
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
  setup('setup:enroll', (name) => {
    if (!runtime) throw new Error('Pair with your Spark first')
    const { caller, token } = runtime.executor.enrollment.add(String(name))
    return { caller: { id: caller.id, name: caller.name }, token, socket: executorSocket(dir), command: join(process.resourcesPath ?? '', 'localrouter-executor-mcp'), state: state() }
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

function permissions() {
  if (process.platform !== 'darwin') return { microphone: 'n/a', accessibility: 'n/a', screen: 'n/a' }
  return { microphone: systemPreferences.getMediaAccessStatus('microphone'), accessibility: systemPreferences.isTrustedAccessibilityClient(false) ? 'granted' : 'not-granted', screen: systemPreferences.getMediaAccessStatus('screen') }
}

function state() {
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
    permissions: permissions(),
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
    { label: 'LocalRouter', submenu: [{ role: 'about' }, { type: 'separator' }, { label: 'Settings…', accelerator: 'CommandOrControl+,', click: () => showSetup() }, { label: 'Stop what Jarvis is doing', accelerator: 'CommandOrControl+Shift+Escape', click: () => stopAll('the keyboard') }, { type: 'separator' }, { role: 'hide' }, { role: 'quit' }] },
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
    const page = result.page as { shim: boolean; bridge: boolean }
    const status = result.status as { computers: number; placement: string }
    ok = page.shim && page.bridge && status.computers === 1 && status.placement === 'device_executor' && caps.ok && result.proxy === 502
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
    tray = new Tray(nativeImage.createEmpty())
    tray.setToolTip('LocalRouter')
    globalShortcut.register('CommandOrControl+Shift+Escape', () => stopAll('the keyboard'))
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
