import { createHash } from 'node:crypto'
import { execFile } from 'node:child_process'
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { createServer as createHttpServer } from 'node:http'
import { createServer } from 'node:net'
import type { AddressInfo } from 'node:net'
import { homedir, hostname, tmpdir } from 'node:os'
import { dirname, extname, join, normalize, sep } from 'node:path'
import { app, BrowserWindow, dialog, globalShortcut, ipcMain, Menu, nativeImage, Notification, powerSaveBlocker, protocol, safeStorage, screen, session, shell, systemPreferences, Tray } from 'electron'
import type { IpcMainEvent, IpcMainInvokeEvent, MenuItemConstructorOptions, WebContents } from 'electron'
import { AddressError, classify, CodeError, connectionOf, connectView, mcpLauncher, normalizeCode, parsePairLink, peerBases, peerNameFor, serverOf, supportsApp, tailnetPeers, tailscaleCandidates, tooOld } from './connect.ts'
import type { ConnectError, Connection, Draft } from './connect.ts'
import { controlBox, controlTicked, everPaired, migrationDecision, StoredStanding } from './standing.ts'
import { PRESENCE_HELPER, TouchIdHelper } from './touchId.ts'
import { EMPTY, HELPER, KeychainHelper, run as runCommand, loadSecrets as loadStoredSecrets, MIGRATION_TIMEOUT_MS, migrateLegacy } from './secrets.ts'
import type { Migration, Secrets } from './secrets.ts'
import { grantStates, micAskAtStart, micResetBeforeAsk, signingId } from './grants.ts'
import type { Permission } from './grants.ts'
import { failure, json, mediaAllowed, ORIGIN, proxy } from './passthrough.ts'
import { createLogger } from '../../../desktop-helper/src/log.ts'
import { CONFIRMATION } from '../executor/broker.ts'
import type { Card, PresenceGate } from '../executor/broker.ts'
import { LegacyBridge } from '../executor/legacy.ts'
import { dataDir, executorSocket } from '../executor/paths.ts'
import { startExecutor } from '../executor/executor.ts'
import type { Executor } from '../executor/executor.ts'
import { ExecutorClient } from '../executor/server.ts'
import type { PageItem } from '../harness/journal.ts'
import { prepareVideo, restoreVoice } from '../harness/models.ts'
import { pageView } from '../harness/realtime.ts'
import type { ClientToolsSession } from '../harness/realtime.ts'
import { startRuntime, UNPAIRED } from '../harness/runtime.ts'
import { overlayHtml, VirtualCursor } from './overlay.ts'
import type { Surface } from './overlay.ts'
import { createPlatform } from '../platform/index.ts'
import { stopShortcut } from '../platform/shortcut.ts'
import type { PlatformPresence } from '../platform/types.ts'
import type { Runtime } from '../harness/runtime.ts'
import { normalizeBase, pair, PlainHttpError, Spark } from '../harness/spark.ts'
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

/**
 * The log, in a file as well as on stderr (an app opened from the Finder has no stderr anyone sees): on macOS
 * ~/Library/Logs/LocalRouter/LocalRouter.log, elsewhere <data>/logs. Kept under 5 MiB (the older half dropped).
 */
export const LOG_FILE = process.platform === 'darwin' ? join(homedir(), 'Library', 'Logs', 'LocalRouter', 'LocalRouter.log') : join(dataDir(), 'logs', 'LocalRouter.log')
function logLine(line: string) {
  process.stderr.write(`${line}\n`)
  try {
    mkdirSync(dirname(LOG_FILE), { recursive: true, mode: 0o700 })
    appendFileSync(LOG_FILE, `${line}\n`, { mode: 0o600 })
  } catch {
    // A log that cannot be written never stops the app.
  }
}
try {
  if (statSync(LOG_FILE).size > 5 << 20) writeFileSync(LOG_FILE, readFileSync(LOG_FILE, 'utf8').slice(-(2 << 20)), { mode: 0o600 })
} catch {
  // No log yet.
}
const log = createLogger(logLine)

/**
 * On macOS the app's tokens are in the keychain helper's item (src/app/secrets.ts). Chromium would still read its own
 * "LocalRouter Safe Storage" keychain item at start (safeStorage, cookie encryption): an item that belongs to the
 * build that made it, so every update made macOS ask again, and the read blocks the main thread until someone
 * answers. The app therefore runs on Chromium's mock keychain and never touches the real one; only the one-time move
 * of an older build's token (`--secrets-migrate`, a child process) and its tests use the real keychain.
 */
const REAL_KEYCHAIN_OPS = ['write-legacy', 'load-legacy']
const selftestOp = process.argv[process.argv.indexOf('--secrets-selftest') + 1] ?? ''
const keychainHelperShipped = process.platform === 'darwin' && existsSync(join(process.resourcesPath ?? '', HELPER))
const needsRealKeychain = process.argv.includes('--secrets-migrate') || (process.argv.includes('--secrets-selftest') && REAL_KEYCHAIN_OPS.includes(selftestOp))
if (keychainHelperShipped && !needsRealKeychain) app.commandLine.appendSwitch('use-mock-keychain')
const UI_DIR = join(__dirname, 'ui')
const PRELOAD = join(__dirname, 'preload.cjs')

/** `grants`: the build each macOS permission was last seen working in (to tell a stale grant from a revoked one). */
type Settings = { base: string; device: { id: string; name: string } | null; name: string; shared: string[]; legacy: boolean; askEachAction?: boolean; automationAsked?: string; pairedAt?: string; micAskedFor?: string; grants?: Partial<Record<Permission, string>>; secretsMigration?: Migration }

protocol.registerSchemesAsPrivileged([{ scheme: 'lr', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true, codeCache: true } }])

// ------------------------------------------------------------------ settings and secrets

let dir = dataDir()
const settingsPath = () => join(dir, 'settings.json')
const secretsPath = () => join(dir, 'secrets.bin')
let memorySecrets: Secrets = { ...EMPTY }
/** Where the tokens are kept: the keychain helper (macOS), Electron safeStorage (elsewhere), or only in memory. */
let secretsStore: 'keychain' | 'legacy' | 'none' = 'none'
let keychain: KeychainHelper | null = null

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

/**
 * Tokens: on macOS in the login keychain through the keychain helper (src/app/secrets.ts: no prompt after an update),
 * elsewhere Electron safeStorage; without either, only for this run. Read once at start; saved on each change.
 */
async function initSecrets() {
  keychain = process.platform === 'darwin' ? new KeychainHelper({ bundled: join(process.resourcesPath ?? '', HELPER), dataDir: dir }) : null
  const helper = keychain && (await keychain.path()) ? keychain : null
  if (keychain && !helper) log.warn('secrets.no_keychain_helper', { why: keychain.problem })
  try {
    if (helper) {
      // An older build's safeStorage file is moved later, off the main thread (moveLegacySecrets).
      memorySecrets = (await helper.get()) ?? { ...EMPTY }
      secretsStore = 'keychain'
    } else {
      const r = await loadStoredSecrets({ helper: null, legacyPath: secretsPath(), legacyDecrypt })
      memorySecrets = r.secrets
      secretsStore = r.store
    }
  } catch (error) {
    log.error('secrets.load_failed', { error: error instanceof Error ? error.message : String(error) })
  }
}

/** An older build's (or a non-Mac's) safeStorage file; only read when the file exists. */
function legacyDecrypt(data: Buffer): string | null {
  try {
    return safeStorage.isEncryptionAvailable() ? safeStorage.decryptString(data) : null
  } catch {
    return null
  }
}

function loadSecrets(): Secrets {
  return memorySecrets
}

function saveSecrets(s: Secrets) {
  memorySecrets = s
  void persistSecrets(s).catch((error: unknown) => log.error('secrets.save_failed', { error: error instanceof Error ? error.message : String(error) }))
}

async function persistSecrets(s: Secrets) {
  if (keychain && (await keychain.path())) {
    await keychain.set(s)
    secretsStore = 'keychain'
    return
  }
  if (!safeStorage.isEncryptionAvailable()) return log.warn('secrets.session_only', { why: 'no OS encryption available' })
  mkdirSync(dir, { recursive: true, mode: 0o700 })
  writeFileSync(secretsPath(), safeStorage.encryptString(JSON.stringify(s)), { mode: 0o600 })
  secretsStore = 'legacy'
}

/** For the state: never asks safeStorage on macOS (that alone can bring the keychain prompt). */
function secretsWhere() {
  return secretsStore === 'keychain' ? 'keychain' : secretsStore === 'legacy' ? 'keychain (safeStorage)' : 'this run only'
}

function computerName() {
  return hostname().replace(/\.local$/i, '') || 'This computer'
}

// ------------------------------------------------------------------ the presence gate

/**
 * The person's presence for an approval: Touch ID where the Mac has it, the platform's own check where it has one
 * (Windows Hello), otherwise a native confirmation dialog (weaker: documented in the README).
 */
/** A window's native handle as a number (an HWND on Windows), for an OS prompt it should own; undefined without one. */
function windowHandle(win: BrowserWindow | null): number | undefined {
  if (!win || win.isDestroyed()) return undefined
  const handle = win.getNativeWindowHandle()
  return handle.length >= 8 ? Number(handle.readBigUInt64LE(0)) : handle.length >= 4 ? handle.readUInt32LE(0) : undefined
}

class AppPresenceGate implements PresenceGate {
  private os: PlatformPresence | null = null
  private osReady = false
  /** Touch ID through localrouter-presence when the app ships it (dismissible, and it says why when unavailable). */
  private readonly touch = process.platform === 'darwin' && existsSync(join(process.resourcesPath ?? '', PRESENCE_HELPER)) ? new TouchIdHelper({ file: join(process.resourcesPath ?? '', PRESENCE_HELPER) }) : null
  private touchWhy = ''

  /** The running platform's check, once the executor has started; its availability is read once. */
  async use(presence: PlatformPresence | null) {
    this.os = presence
    this.osReady = presence ? await presence.available().catch(() => false) : false
    await this.probe()
  }

  /** Why Touch ID cannot be used now (LAError's reason), read again before each check: a lid can close meanwhile. */
  private async probe() {
    if (!this.touch) return
    const p = await this.touch.check()
    this.touchWhy = p.available ? '' : p.why
    log.info('presence.touch_id', { available: p.available, why: p.why })
  }

  private get touchId() {
    return process.platform === 'darwin' && systemPreferences.canPromptTouchID()
  }

  get method() {
    if (this.touchId) return 'Touch ID'
    if (this.os && this.osReady) return this.os.method
    return CONFIRMATION
  }

  /** On a fallback, why the OS check is not used, so the card and the audit name it. */
  get why() {
    if (this.method !== CONFIRMATION) return ''
    if (process.platform === 'darwin') return this.touchWhy || 'Touch ID is not available on this Mac right now'
    if (this.os) return this.os.why ?? `${this.os.method} is not available`
    return ''
  }

  async verify(card: Card, waiting?: (text: string) => void, signal?: AbortSignal): Promise<boolean> {
    const prompt = `let Jarvis ${card.action}`.slice(0, 200)
    if (this.touchId) {
      // The helper's prompt is dismissed when the approval ends another way; Electron's (a dev build) cannot be.
      if (this.touch) return this.touch.verify(prompt, signal)
      return systemPreferences.promptTouchID(prompt).then(() => !signal?.aborted, () => false)
    }
    if (this.os && this.osReady) {
      // The app's window owns the OS prompt (Windows Hello opens over it, not behind it).
      const window = windowHandle(mainWindow)
      return this.os.verify(`Let Jarvis ${card.action}`.slice(0, 200), { ...(window !== undefined ? { window } : {}), ...(waiting ? { waiting } : {}), ...(signal ? { signal } : {}) }).catch(() => false)
    }
    void this.probe()
    const where = [card.app, card.title, card.url].filter(Boolean).join(' · ')
    const options = {
      type: 'warning' as const,
      buttons: ['Deny', 'Approve'],
      defaultId: 0,
      cancelId: 0,
      noLink: true,
      message: `Let Jarvis ${card.action}?`,
      detail: `${where}\n${card.kind === 'type' ? `Text: ${card.text}\n` : ''}Why it asks: ${card.why}\nFor: ${card.goal}\nAsked by: ${card.requester}`,
      // A stop, or the approval ending another way, closes the sheet (it must have a parent window to be closable).
      ...(signal ? { signal } : {}),
    }
    const parent = mainWindow && !mainWindow.isDestroyed() ? mainWindow : null
    const r = parent ? await dialog.showMessageBox(parent, options) : await dialog.showMessageBox(options)
    return r.response === 1 && !signal?.aborted
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
let virtualCursor: VirtualCursor | null = null

/**
 * The overlay cursor, built on first use over the whole virtual desktop: transparent, always on top, click-through and
 * never focused, so it shows where the agent acts without touching anything. Rebuilt after `end()` closes its window.
 */
function cursor(): VirtualCursor {
  if (virtualCursor) return virtualCursor
  let win: BrowserWindow | null = null
  // The page loads after the window is built: the first ring of a task waits for it instead of being dropped (Windows
  // run, 2026-10-26: no ring 241 ms after a background click).
  let loaded = false
  let shownOnce = false
  let pending: string | null = null
  const build = (): BrowserWindow => {
    loaded = false
    const area = screen.getDisplayNearestPoint(screen.getCursorScreenPoint()).bounds
    const bounds = screen.getAllDisplays().reduce((b, d) => ({ x: Math.min(b.x, d.bounds.x), y: Math.min(b.y, d.bounds.y), right: Math.max(b.right, d.bounds.x + d.bounds.width), bottom: Math.max(b.bottom, d.bounds.y + d.bounds.height) }), { x: area.x, y: area.y, right: area.x + area.width, bottom: area.y + area.height })
    const w = new BrowserWindow({ x: bounds.x, y: bounds.y, width: bounds.right - bounds.x, height: bounds.bottom - bounds.y, frame: false, transparent: true, resizable: false, movable: false, focusable: false, skipTaskbar: true, hasShadow: false, show: false, webPreferences: { contextIsolation: true } })
    w.setIgnoreMouseEvents(true)
    w.setAlwaysOnTop(true, 'screen-saver')
    if (process.platform === 'darwin') w.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true })
    w.webContents.once('did-finish-load', () => {
      loaded = true
      if (pending) run(pending)
      pending = null
    })
    void w.loadURL(`data:text/html,${encodeURIComponent(overlayHtml())}`)
    return w
  }
  const run = (js: string) => {
    if (!win || win.isDestroyed()) return
    if (!loaded) {
      pending = js
      return
    }
    void win.webContents.executeJavaScript(js).catch(() => undefined)
  }
  const surface: Surface = {
    warm: () => {
      if (!win || win.isDestroyed()) win = build()
    },
    showAt: (x, y, mode) => {
      if (!win || win.isDestroyed()) win = build()
      const b = win.getBounds()
      win.showInactive()
      if (!shownOnce) log.info('overlay.shown', { at: { x, y }, mode, bounds: b, loaded, visible: win.isVisible() })
      shownOnce = true
      run(`window.lrCursor(${x - b.x}, ${y - b.y}, ${JSON.stringify(mode)})`)
    },
    hide: () => {
      run('window.lrHide && window.lrHide()')
      if (win && !win.isDestroyed()) win.hide()
    },
    close: () => {
      if (win && !win.isDestroyed()) win.destroy()
      win = null
    },
  }
  virtualCursor = new VirtualCursor(surface)
  return virtualCursor
}
let tray: Tray | null = null
const sessions = new Map<string, { session: ClientToolsSession; owner: WebContents }>()

function machineKind(): 'native' | 'fake' {
  return process.env.LOCALROUTER_COMPANION_FAKE === '1' ? 'fake' : 'native'
}

/**
 * This computer's executor, started at launch whether or not LocalRouter is paired: its lease, policy, approvals,
 * audit, stop and caller enrollment are all local. Pairing only gives it the Spark's decisions (computer use);
 * before that, other harnesses enrolled in Settings › Harnesses can run fixed controls and read the screen, and a step
 * that needs an approval is refused (the window that shows approval cards opens once paired).
 */
let executor: Executor | null = null

export const NO_APPROVAL_WINDOW = 'Approving needs LocalRouter’s window, which opens once LocalRouter is paired with a Spark: pair it on the Connect page.'

/** The person's grant for computer control, kept with the tokens; "Ask before each action" is a setting. */
const standing = new StoredStanding({ load: () => loadSecrets(), save: (s) => saveSecrets(s) }, () => settings?.askEachAction === true)

/**
 * An install already set up (paired, with the OS grants computer control needs) that never decided gets the standing
 * grant at start with no card: that setup was the consent (JJ, 2026-10-26). Audited as `source: migrated`.
 */
async function migrateStanding(when: 'launch' | 'pairing' | 'permissions') {
  if (!executor || standing.decision !== 'undecided') return
  const access = await executor.built.platform.access().catch(() => null)
  const app = process.platform === 'darwin' ? { accessibility: systemPreferences.isTrustedAccessibilityClient(false), screen_recording: systemPreferences.getMediaAccessStatus('screen') === 'granted' } : null
  const d = migrationDecision({ grant: standing.decision, everPaired: everPaired({ base: settings.base, token: loadSecrets().token, pairedAt: settings.pairedAt }), os: process.platform, access, app })
  // Logged when it changes (state() polls this on the setup page), always at launch and on a pairing.
  const key = `${d.eligible}:${d.reason}`
  if (when !== 'permissions' || key !== lastMigrate) log.info('consent.migrate', { ...d, when, executor: access ? { accessibility: access.accessibility, screen_recording: access.screen_recording } : null, app })
  lastMigrate = key
  if (!d.eligible) return
  executor.broker.grantStanding('migrated')
  log.info('consent.migrated', { os: process.platform, when })
}
let lastMigrate = ''

/**
 * macOS: the Automation grants (System Events, and Chrome when it runs), asked once, from setup, where the person
 * pressed the button: the one moment a "LocalRouter wants to control…" prompt may come. During tasks nothing asks.
 */
async function askAutomationOnce() {
  const automation = executor?.built.platform.automation
  if (!automation || settings.automationAsked) return
  const answers = await automation.ask().catch(() => ({}))
  settings = { ...settings, automationAsked: new Date().toISOString() }
  saveSettings(settings)
  log.info('automation.asked', answers)
}

async function startLocalExecutor(): Promise<void> {
  if (executor) return
  executor = await startExecutor({ dir, name: settings.name, machine: machineKind(), decider: null, gate, standing, listen: true, ownApps: ['LocalRouter', 'Electron'], approvalSurface: () => (runtime ? null : NO_APPROVAL_WINDOW), log })
  executor.broker.setDecider(null, UNPAIRED)
  await gate.use(executor.built.platform.presence)
  await migrateStanding('launch')
  executor.broker.on((e) => {
    // Refused for want of the grant: a notification only, never a window over the person's work.
    if (e.type === 'consent.needed' && Notification.isSupported()) new Notification({ title: 'Computer control is off', body: `Jarvis asked to ${e.goal.slice(0, 80)}. Turn computer control on in LocalRouter → Settings.` }).show()
    if (e.type === 'approval') {
      showMain()
      if (Notification.isSupported()) new Notification({ title: 'Jarvis asks for your OK', body: `${e.card.action}. Approve it in LocalRouter.` }).show()
    }
    // A card nobody answered: say so, since the task ended while the person may have been away from the window.
    if (e.type === 'approval.end' && e.outcome === 'expired' && Notification.isSupported()) new Notification({ title: 'Jarvis stopped', body: 'Nobody approved the step in time, so nothing was done.' }).show()
    // The OS check is open behind other windows: say where it is, in a notification as well as on the card.
    if (e.type === 'approval.update' && e.card.waiting && Notification.isSupported()) new Notification({ title: 'Windows Hello is waiting', body: e.card.waiting }).show()
    // The visible virtual cursor: show where the agent acted; let go of the overlay when the task ends.
    if (e.type === 'cursor') {
      // Windows and Linux report screen pixels; the overlay is placed in Electron's DIPs (a 150 % display differs).
      const at = process.platform === 'darwin' ? e.at : screen.screenToDipPoint(e.at)
      cursor().show({ x: at.x, y: at.y, mode: e.mode })
    }
    if (e.type === 'task' && e.task.status === 'running') cursor().ready()
    if (e.type === 'task' && e.task.status !== 'running') cursor().end()
    if (e.type === 'task' || e.type === 'approval' || e.type === 'approval.end') refreshTray()
    if (e.type === 'task' && e.task.status !== 'running' && /Accessibility|Screen Recording/i.test(e.task.reason)) void checkStale()
  })
}

async function startRuntimeFromSettings(): Promise<void> {
  await stopRuntime()
  await startLocalExecutor()
  const secrets = loadSecrets()
  if (!settings.base || !secrets.token || !executor) return
  runtime = await startRuntime({ base: settings.base, token: secrets.token, dir, name: settings.name, machine: machineKind(), gate, shared: () => settings.shared, executor, log })
  if (settings.legacy && secrets.legacy_token) {
    const local = executor
    legacy = new LegacyBridge({ url: settings.base, token: secrets.legacy_token, name: settings.name, version: VERSION, broker: local.broker, access: () => local.built.platform.access(), log })
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
  if (!executor) return failure(503, 'LocalRouter’s executor is starting')
  const broker = executor.broker
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

function registerProtocol() {
  protocol.handle('lr', async (req) => {
    const url = new URL(req.url)
    if (url.host !== 'app') return new Response('not found', { status: 404 })
    const path = decodeURIComponent(url.pathname)
    if (path.startsWith('/v1/computer/')) return computerRoute(req, path, url)
    if (path.startsWith('/v1/') || path === '/mcp' || path === '/health') return proxy(runtime?.spark ?? null, req, url.pathname + url.search)
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
      // A LAN address over plain http://: name the same Spark's tailnet address, when Tailscale knows it.
      if (error instanceof PlainHttpError) error.suggestion = peerNameFor(await tailscaleStatus().catch(() => null), error.host)
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
      settings = { ...settings, base: draft.base, device: p.device, name: nm, pairedAt: new Date().toISOString() }
      saveSettings(settings)
      saveSecrets({ ...loadSecrets(), token: p.token })
      // A pairing (first, or again after the box revoked the token) reruns the migration: never a step left at "not allowed".
      await migrateStanding('pairing')
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
    // The person's grant for computer control is theirs, not the Spark's: it stays until revoked in Settings.
    saveSecrets({ ...loadSecrets(), token: '', legacy_token: '' })
    draft = null
    connection = null
    errors = {}
    return state()
  })
  setup('setup:discover', () => discoverSparks())
  setup('setup:resetGrant', async (kind) => {
    if (process.platform !== 'darwin') return { ok: false, text: 'only macOS keeps these grants', state: await state() }
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
    if (kind === 'microphone') {
      // Not granted now (denied, or an older build's entry that no longer matches this signature): macOS never asks
      // again by itself, so this app's entry is reset first and macOS shows its prompt, once (grants.ts).
      if (micResetBeforeAsk(systemPreferences.getMediaAccessStatus('microphone'))) await resetGrant('Microphone')
      await systemPreferences.askForMediaAccess('microphone')
    }
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
    if (!executor) throw new Error('LocalRouter’s executor is starting: try again in a moment')
    const { caller, token } = executor.enrollment.add(String(name))
    return { caller: { id: caller.id, name: caller.name }, token, socket: executorSocket(dir), command: mcpLauncher(process.resourcesPath ?? '', process.platform), state: await state() }
  })
  setup('setup:revoke', (id) => {
    executor?.enrollment.revoke(String(id))
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
  // The box is ticked only when control is on and allowed (controlBox); ticking it once also grants.
  setup('setup:control', async (on) => {
    const t = controlTicked(Boolean(on), standing.granted())
    executor?.broker.setDisabled(!t.enable)
    if (t.grant && executor) {
      executor.broker.grantStanding('settings')
      await askAutomationOnce()
    }
    refreshTray()
    return state()
  })
  // "Ask before each action" (off by default): a card per consequential step instead of the standing grant.
  setup('setup:askEach', (on) => {
    settings = { ...settings, askEachAction: Boolean(on) }
    saveSettings(settings)
    return state()
  })
  // The one consent, in setup (the onboarding step, or Settings): never a card over a task. On a Mac it also asks the
  // Automation grants once, here, so no "wants to control" prompt can come up during a task.
  setup('setup:grantControl', async (from) => {
    if (!executor) return state()
    executor.broker.grantStanding(from === 'settings' ? 'settings' : 'onboarding')
    await askAutomationOnce()
    return state()
  })
  // "Revoke computer control": the standing grant is cleared; tasks are refused until it is given again here.
  setup('setup:revokeControl', () => {
    if (executor) executor.broker.revokeStanding('settings')
    else standing.set(false)
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
    const report = executor ? await executor.built.platform.access().catch(() => null) : null
    // Windows asks about the microphone too (Settings › Privacy & security › Microphone); Linux has no such switch.
    const mic = process.platform === 'win32' ? systemPreferences.getMediaAccessStatus('microphone') : 'n/a'
    const microphone = mic === 'n/a' ? [] : [{ id: 'microphone', label: 'Microphone', granted: mic === 'granted' ? true : mic === 'denied' || mic === 'restricted' ? false : null, for: 'calls', fix: 'Settings › Privacy & security › Microphone: turn on “Let desktop apps access your microphone”.' }]
    return { microphone: 'n/a', accessibility: 'n/a', screen: 'n/a', stale: { accessibility: false, screen: false, microphone: false }, backend: report?.backend ?? '', requirements: [...microphone, ...(report?.requirements ?? [])] }
  }
  const microphone = systemPreferences.getMediaAccessStatus('microphone')
  const said = { accessibility: systemPreferences.isTrustedAccessibilityClient(false), screen: systemPreferences.getMediaAccessStatus('screen') === 'granted', microphone: microphone === 'granted' }
  const works = executor ? await executor.built.platform.access().then((a) => ({ accessibility: a.accessibility }), () => null) : null
  const { state: grant, grants } = grantStates(said, works, settings.grants ?? {}, await signingIdentity())
  if (JSON.stringify(grants) !== JSON.stringify(settings.grants ?? {})) {
    settings = { ...settings, grants }
    saveSettings(settings)
  }
  const stale = { accessibility: grant.accessibility === 'stale', screen: grant.screen === 'stale', microphone: grant.microphone === 'stale' }
  return { microphone: stale.microphone ? 'stale' : microphone, accessibility: grant.accessibility, screen: grant.screen, stale }
}

/** `tccutil reset` for this app only: runs as the user, no admin. What it said, and whether it worked. */
/**
 * macOS: an older build's Microphone entry that no longer matches this signature never prompts again by itself, so
 * once per signing identity, when the Microphone is not granted, the app's own entry is reset (`tccutil reset
 * Microphone com.localrouter.companion`) and macOS asks, at start, never during a task.
 */
async function askMicOnce() {
  if (process.platform !== 'darwin') return
  const identity = await signingIdentity()
  const status = systemPreferences.getMediaAccessStatus('microphone')
  if (!micAskAtStart({ status, identity, askedFor: settings.micAskedFor })) return
  settings = { ...settings, micAskedFor: identity ?? undefined }
  saveSettings(settings)
  const reset = await resetGrant('Microphone')
  const granted = await systemPreferences.askForMediaAccess('microphone')
  log.info('microphone.asked', { before: status, reset: reset.ok, granted })
}

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

/** The Spark's base URL from what was typed, or an AddressError (a PlainHttpError for http:// to a LAN host) saying why not. */
function sparkBase(text: string): string {
  if (!text) throw new AddressError('Enter the address your Spark answers on')
  try {
    return normalizeBase(text)
  } catch (error) {
    if (error instanceof PlainHttpError) throw error
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

const TAILSCALE = tailscaleCandidates(process.platform, process.env)

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
 * LocalRouter boxes on this computer's tailnet: each online peer's MagicDNS name, asked for `/v1/capabilities` (open, no
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
  // Permissions granted after launch (the setup page polls this): the migration is decided again.
  await migrateStanding('permissions')
  const callers = executor?.enrollment.list().map((c) => ({ id: c.id, name: c.name, created: c.created })) ?? []
  return {
    version: VERSION,
    paired: Boolean(runtime),
    base: settings.base,
    device: settings.device,
    name: settings.name,
    shared: settings.shared,
    legacy: { enabled: settings.legacy, connected: legacy?.status.connected ?? false },
    presence: gate.method,
    presence_why: gate.why,
    secrets: secretsWhere(),
    control: executor ? controlBox({ enabled: !executor.broker.disabled, granted: standing.granted() }) : false,
    approvals: executor?.broker.approvalMode() ?? 'consent',
    control_granted: standing.granted(),
    ask_each: settings.askEachAction === true,
    granted_since: standing.since,
    machine: machineKind(),
    permissions: permissionState,
    stop_words: executor?.built.platform.pointer ? `${STOP.words}, the tray menu, or move the pointer to the top-left corner` : `${STOP.words} or the tray menu`,
    connect: { ...connectView({ paired: Boolean(runtime), base: settings.base, draft, connection, errors, permissions: permissionState, control: { granted: standing.granted() } }), draft, connection, errors },
    stale_message: STALE_GRANT,
    callers,
    mcp: { command: mcpLauncher(process.resourcesPath ?? '', process.platform), socket: executorSocket(dir) },
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
  const n = executor?.broker.stop(undefined, by, 'You stopped it.') ?? 0
  if (n && Notification.isSupported()) new Notification({ title: 'LocalRouter', body: 'Stopped what Jarvis was doing on this computer.' }).show()
}

function refreshTray() {
  if (!tray) return
  const broker = executor?.broker
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
  session.defaultSession.setPermissionRequestHandler((wc, permission, done, details) => {
    const d = details as { requestingUrl?: string; mediaTypes?: string[] }
    const ok = mediaAllowed(d.requestingUrl ?? wc.getURL(), permission, d.mediaTypes)
    // The page's own microphone (recording a voice to clone, not only a call's rt:open): macOS asks the person once,
    // and its answer is the answer.
    if (ok && permission === 'media' && process.platform === 'darwin') void systemPreferences.askForMediaAccess('microphone').then(done, () => done(false))
    else done(ok)
  })
  session.defaultSession.setPermissionCheckHandler((_wc, permission, origin) => mediaAllowed(origin, permission))
}

// ------------------------------------------------------------------ smoke (CI): starts headless, checks, exits

async function smoke() {
  const result: Record<string, unknown> = { version: VERSION, platform: process.platform, arch: process.arch }
  let ok = false
  // A stand-in Spark that says what reached it: the page's calls must arrive whole through lr://app (a voice clone's
  // multipart clips from a Blob, PATCH, DELETE, a query).
  const echo = createHttpServer((req, res) => {
    const parts: Buffer[] = []
    req.on('data', (c: Buffer) => parts.push(c))
    req.on('end', () => {
      const body = Buffer.concat(parts)
      res.setHeader('content-type', 'application/json')
      res.end(JSON.stringify({ method: req.method, url: req.url, type: req.headers['content-type'] ?? '', bytes: body.length, clip: body.includes('RIFFclip'), auth: req.headers.authorization ?? '' }))
    })
  })
  try {
    await new Promise<void>((resolve) => echo.listen(0, '127.0.0.1', resolve))
    dir = mkdtempSync(join(tmpdir(), 'lr-smoke-'))
    app.setPath('userData', dir)
    await app.whenReady()
    registerProtocol()
    registerIpc()
    settings = loadSettings()
    // The app's own path: the executor starts unpaired (the fake backend), answers an enrolled harness on its socket
    // with the reason computer use waits, and then pairing gives it the Spark's decisions.
    process.env.LOCALROUTER_COMPANION_FAKE = '1'
    await startLocalExecutor()
    const local = executor!
    const { token } = local.enrollment.add('smoke')
    const client = new ExecutorClient(executorSocket(dir), token)
    const before = await client.request('capabilities')
    const unpairedCaps = (before.ok ? before.result : {}) as { computer_use?: boolean; computer_use_unavailable?: string }
    result.unpaired = { answered: before.ok, computer_use: unpairedCaps.computer_use ?? null, says: unpairedCaps.computer_use_unavailable === UNPAIRED ? 'not paired' : (unpairedCaps.computer_use_unavailable ?? (before.ok ? '' : before.error)) }
    runtime = await startRuntime({ base: `http://127.0.0.1:${(echo.address() as AddressInfo).port}`, token: 'smoke-token', dir, name: 'smoke', machine: 'fake', gate: { method: 'none (smoke)', verify: async () => false }, shared: () => [], executor: local })
    const win = new BrowserWindow({ show: false, webPreferences })
    await win.loadURL(`${ORIGIN}/voice.html`)
    result.page = await win.webContents.executeJavaScript('({ shim: window.__companionShim === true, bridge: typeof (window.companion && window.companion.realtime && window.companion.realtime.open) === "function", title: document.title, socket: String(WebSocket.name) })')
    result.status = await win.webContents.executeJavaScript('fetch("/v1/computer/status").then(r => r.json()).then(s => ({ computers: s.computers.length, placement: s.computers[0] && s.computers[0].placement }))')
    result.proxy = await win.webContents.executeJavaScript('fetch("/v1/models").then(r => r.status)')
    result.voices = await win.webContents.executeJavaScript(`(async () => {
      const fd = new FormData();
      fd.append('file', new Blob([new TextEncoder().encode('RIFFclip'), new Uint8Array(4096)], { type: 'audio/wav' }), 'take.wav');
      fd.append('consent', 'true');
      const go = (path, init) => fetch(path, init).then(r => r.json());
      return { post: await go('/v1/voices', { method: 'POST', body: fd }),
        patch: await go('/v1/voices/draft-1', { method: 'PATCH', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ name: 'JJ' }) }),
        del: await go('/v1/voices/JJ', { method: 'DELETE' }), list: await go('/v1/voices?model=m', { headers: { authorization: 'Bearer page' } }) };
    })()`)
    const caps = await client.request('capabilities')
    client.close()
    result.executor = caps.ok ? caps.result : caps.error
    result.secrets = process.platform === 'darwin' ? 'keychain helper (see --secrets-selftest)' : safeStorage.isEncryptionAvailable() ? 'safeStorage' : 'this run only'
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
    type Echo = { method: string; url: string; type: string; bytes: number; clip: boolean; auth: string }
    const v = result.voices as { post: Echo; patch: Echo; del: Echo; list: Echo }
    const voices = v.post.method === 'POST' && v.post.url === '/v1/voices' && v.post.type.startsWith('multipart/form-data; boundary=') && v.post.clip && v.post.bytes > 4104 &&
      v.patch.method === 'PATCH' && v.patch.bytes === 13 && v.del.method === 'DELETE' && v.del.bytes === 0 && v.list.url === '/v1/voices?model=m' && v.list.auth === 'Bearer smoke-token'
    const unpaired = result.unpaired as { answered: boolean; computer_use: boolean | null; says: string }
    ok = unpaired.answered && unpaired.computer_use === false && unpaired.says === 'not paired' && caps.ok && (caps.result as { computer_use?: boolean }).computer_use === true && page.shim && page.bridge && status.computers === 1 && status.placement === 'device_executor' && caps.ok && result.proxy === 200 && voices && setupPage.version === VERSION && setupPage.error === 'refused'
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
  echo.close()
  await runtime?.close().catch(() => undefined)
  await executor?.close().catch(() => undefined)
  app.exit(ok ? 0 : 1)
}

// ------------------------------------------------------------------ moving an older build's token, once

/** The move itself, in a child process: the read may wait on a keychain prompt, and never blocks this one. */
function runMigrationChild(timeoutMs: number): Promise<{ ok: boolean; reason: string }> {
  // `run` ends on the child's result line or its pipes closing, never only on an exit event (which Electron's main
  // process can miss); the child's own Chromium helpers may keep the pipe open a moment after its line.
  const env = { ...process.env, LOCALROUTER_COMPANION_HOME: dir }
  return runCommand(process.execPath, ['--secrets-migrate'], undefined, { timeoutMs, env, done: (out) => /^COMPANION_MIGRATE .+\n/m.test(out) }).then((r) => {
    const line = /^COMPANION_MIGRATE (.+)$/m.exec(r.stdout)?.[1]
    let j: { ok?: boolean; reason?: string } = {}
    try {
      j = line ? (JSON.parse(line) as typeof j) : {}
    } catch {
      // Not JSON: a failure below.
    }
    if (j.ok) return { ok: true, reason: '' }
    if (r.code === 137 && !line) return { ok: false, reason: `no answer to the keychain prompt within ${Math.round(timeoutMs / 1000)} s` }
    return { ok: false, reason: j.reason ?? (r.stderr.trim().split('\n').pop() || 'the move did not answer') }
  })
}

/**
 * After the window is up: an older build's safeStorage file is moved into the keychain helper, once. macOS asks once;
 * Deny, Allow-once-then-fail or no answer in time is recorded, the file is dropped, and the person pairs again.
 */
async function moveLegacySecrets(o: { interactive: boolean; timeoutMs?: number } = { interactive: true }) {
  if (!keychain || !(await keychain.path()) || !existsSync(secretsPath())) return 'none'
  const previous = settings.secretsMigration
  if (o.interactive && !memorySecrets.token && previous?.state !== 'failed') {
    void dialog.showMessageBox({ type: 'info', message: 'macOS will ask once to move your Spark login to the new storage', detail: 'Choose Always Allow (or Allow). It is the last time: LocalRouter keeps your login in its own keychain item from now on, and updates do not ask again.', buttons: ['OK'] })
  }
  const r = await migrateLegacy({ legacyPath: secretsPath(), previous, haveToken: Boolean(memorySecrets.token), run: runMigrationChild, timeoutMs: o.timeoutMs, log: (event, fields) => log.info(event, fields) })
  if (r.migration) {
    settings = { ...settings, secretsMigration: r.migration }
    saveSettings(settings)
  }
  if (r.outcome === 'migrated') {
    memorySecrets = (await keychain.get()) ?? { ...EMPTY }
    await migrateStanding('launch')
    if (o.interactive) {
      await startRuntimeFromSettings().catch((error: unknown) => log.error('runtime.start_failed', { error: error instanceof Error ? error.message : String(error) }))
      refreshTray()
      if (runtime) {
        showMain()
        setupWindow?.close()
      }
    }
  } else if (r.outcome === 'failed' && o.interactive) {
    showSetup()
    void dialog.showMessageBox({ type: 'warning', message: 'Pair LocalRouter with your Spark again', detail: `Your Spark login from the older LocalRouter could not be moved (${r.migration?.reason ?? 'refused'}). Make a new pairing code on the Spark and enter it on the Connect page. LocalRouter will not ask the keychain about the old one again.`, buttons: ['OK'] })
  }
  return r.outcome
}

/**
 * `--secrets-migrate`: the child that moves the token. It reads the older build's file with the real safeStorage
 * (macOS may ask), stores it through the keychain helper (this process is LocalRouter, so the helper answers it),
 * then deletes the file and, if it can without asking, the old "LocalRouter Safe Storage" item.
 */
async function secretsMigrate() {
  const out: Record<string, unknown> = {}
  try {
    app.setPath('userData', mkdtempSync(join(tmpdir(), 'lr-migrate-')))
    await app.whenReady()
    const text = legacyDecrypt(readFileSync(secretsPath()))
    if (text === null) throw new Error('the keychain refused (Deny), or the old login could not be read')
    const helper = new KeychainHelper({ bundled: join(process.resourcesPath ?? '', HELPER), dataDir: dir })
    await helper.set({ ...EMPTY, ...(JSON.parse(text) as Partial<Secrets>) })
    rmSync(secretsPath(), { force: true })
    out.ok = true
    const del = await runCommand('/bin/sh', ['-c', '/usr/bin/security delete-generic-password -s "$1" > /dev/null; echo "LR_STATUS $?" >&2', 'sh', `${app.getName()} Safe Storage`], undefined, { timeoutMs: 10_000 })
    out.legacy_item = del.code === 0 ? 'deleted' : del.code === 137 ? 'kept (it would have asked)' : `kept (${del.stderr.trim().split('\n')[0] || `exit ${del.code}`})`
  } catch (error) {
    out.ok = false
    out.reason = error instanceof Error ? error.message : String(error)
  }
  process.stdout.write(`COMPANION_MIGRATE ${JSON.stringify(out)}\n`)
  app.exit(out.ok ? 0 : 1)
}

// ------------------------------------------------------------------ secrets self-test (CI: install over, read back)

/**
 * `--secrets-selftest OP [VALUE]` with LOCALROUTER_COMPANION_HOME: the app's own secret store, headless, for the
 * update test (build A stores, build B installed over it reads). Prints `COMPANION_SECRETS {json}`; the token only as
 * a SHA-256 prefix. OPs: `write-legacy V` (an older build's safeStorage file), `migrate` (the one-time move, as the
 * app does after its window is up; LR_MIGRATE_TIMEOUT_MS shortens its limit), `load` (as the app starts), `save V`,
 * `load-legacy` (safeStorage directly: the control, which an update makes ask again). Each stays up LR_SELFTEST_HOLD_MS
 * (default 4 s) after its work, as a running app would, so anything that reads the keychain at start is caught.
 */
async function secretsSelftest() {
  const at = process.argv.indexOf('--secrets-selftest')
  const op = process.argv[at + 1] ?? ''
  const value = process.argv[at + 2] ?? ''
  const digest = (t: string) => (t ? createHash('sha256').update(t).digest('hex').slice(0, 12) : '')
  const out: Record<string, unknown> = { op }
  try {
    await app.whenReady()
    settings = loadSettings()
    if (op === 'migrate') {
      await initSecrets()
      out.outcome = await moveLegacySecrets({ interactive: false, timeoutMs: Number(process.env.LR_MIGRATE_TIMEOUT_MS ?? MIGRATION_TIMEOUT_MS) })
      out.ok = true
      out.migration = settings.secretsMigration ?? null
      out.token = digest(memorySecrets.token)
      out.legacy_file = existsSync(secretsPath())
    } else if (op === 'write-legacy') {
      mkdirSync(dir, { recursive: true, mode: 0o700 })
      writeFileSync(secretsPath(), safeStorage.encryptString(JSON.stringify({ token: value, legacy_token: '' })), { mode: 0o600 })
      out.ok = true
    } else if (op === 'load-legacy') {
      const text = legacyDecrypt(readFileSync(secretsPath()))
      out.ok = text !== null
      out.token = text ? digest((JSON.parse(text) as Secrets).token) : ''
    } else if (op === 'load' || op === 'save') {
      await initSecrets()
      if (op === 'save') {
        await persistSecrets({ ...EMPTY, token: value })
        await initSecrets()
      }
      out.ok = true
      out.store = secretsStore
      out.token = digest(memorySecrets.token)
      out.helper = keychain ? await keychain.path() : null
      out.problem = keychain?.problem || undefined
      out.legacy_file = existsSync(secretsPath())
      out.migration = settings.secretsMigration ?? null
    } else throw new Error(`unknown op ${op}`)
    await new Promise((resolve) => setTimeout(resolve, Number(process.env.LR_SELFTEST_HOLD_MS ?? 4000)))
  } catch (error) {
    out.ok = false
    out.error = error instanceof Error ? error.message : String(error)
  }
  process.stdout.write(`COMPANION_SECRETS ${JSON.stringify(out)}\n`)
  app.exit(out.ok ? 0 : 1)
}

// ------------------------------------------------------------------ start

if (process.argv.includes('--version')) {
  process.stdout.write(`LocalRouter ${VERSION}\n`)
  process.exit(0)
} else if (process.argv.includes('--smoke')) {
  void smoke()
} else if (process.argv.includes('--secrets-selftest')) {
  void secretsSelftest()
} else if (process.argv.includes('--secrets-migrate')) {
  void secretsMigrate()
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
    void stopRuntime().then(() => executor?.close())
  })
  void app.whenReady().then(async () => {
    registerProtocol()
    registerIpc()
    permissionHandlers()
    buildMenu()
    settings = loadSettings()
    await initSecrets()
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
    void askMicOnce().catch((error: unknown) => log.warn('microphone.ask_failed', { error: error instanceof Error ? error.message : String(error) }))
    log.info('app.started', { version: VERSION, secrets: secretsStore, keychain: keychainHelperShipped ? 'helper (Chromium on its mock keychain)' : 'safeStorage', log: LOG_FILE })
    void moveLegacySecrets().catch((error: unknown) => log.error('secrets.migration.failed', { error: error instanceof Error ? error.message : String(error) }))
  })
}
