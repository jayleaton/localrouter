import { readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { ControlError } from '../../../../desktop-helper/src/computer/macControls.ts'
import type { CommandRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import type { BrowserTab, BrowserTabs } from '../../../../desktop-helper/src/computer/types.ts'
import type { Logger } from '../../../../desktop-helper/src/log.ts'
import { array, boolean, integerIn, number, object, string } from '../../../../desktop-helper/src/validate.ts'
import type { AccessReport, Platform, Requirement } from '../types.ts'
import { WindowsControls } from './controls.ts'
import type { DirEntry } from './controls.ts'
import { WindowsHello } from './presence.ts'
import { ask, WindowsScreen } from './screen.ts'
import { PowerShellWorker } from './worker.ts'
import type { WindowsCall, WorkerSpawn } from './worker.ts'

export const WINDOWS_BACKEND = 'Windows (UI Automation, SendInput)'

/** Integrity levels (the mandatory label's RID): an elevated process is at High or above. */
const HIGH_INTEGRITY = 0x3000

export type AccessFacts = { uia: boolean; secure: boolean; session: number; integrity: number; dpi: string; interactive: boolean }

export function accessShape(value: unknown): AccessFacts {
  const a = object(value, 'access')
  return { uia: boolean(a.uia, 'access uia'), secure: boolean(a.secure, 'access secure'), session: number(a.session, 'access session'), integrity: number(a.integrity, 'access integrity'), dpi: string(a.dpi, 'access dpi'), interactive: boolean(a.interactive, 'access interactive') }
}

/** What Settings shows: each thing computer control needs on Windows, whether it holds now, and how to fix it. */
export function accessReport(facts: AccessFacts | null, error?: string): AccessReport {
  if (!facts) {
    return {
      accessibility: false,
      screen_recording: false,
      input: false,
      backend: WINDOWS_BACKEND,
      requirements: [{ id: 'powershell', label: 'Windows PowerShell 5.1', granted: false, for: 'reading and driving the screen (its helper compiles and runs there)', fix: `Windows PowerShell must be able to start and run its helper${error ? ` (${error})` : ''}. It ships with Windows 10 and 11; check that it is not blocked by policy.` }],
    }
  }
  const desktop = facts.session !== 0 && facts.interactive
  const elevated = facts.integrity >= HIGH_INTEGRITY
  const requirements: Requirement[] = [
    { id: 'uia', label: 'UI Automation', granted: facts.uia, for: 'reading the window in front and pressing its buttons without the pointer', ...(facts.uia ? {} : { fix: 'Run LocalRouter in your own signed-in Windows session (not as a service, over SSH, or in a disconnected remote session).' }) },
    { id: 'desktop', label: 'Your desktop session', granted: desktop, for: 'clicks, typing and screenshots', ...(desktop ? {} : { fix: 'Run LocalRouter as the signed-in user from the Start menu, not as a Windows service.' }) },
    { id: 'secure_desktop', label: 'Unlocked, no prompt open', granted: !facts.secure, for: 'anything on screen: no app can read or use a UAC prompt or the lock screen', ...(facts.secure ? { fix: 'Unlock the computer or answer the User Account Control prompt.' } : {}) },
    { id: 'not_elevated', label: 'Runs as you, not as administrator', granted: !elevated, for: 'keeping LocalRouter at the same rights as the apps it uses', ...(elevated ? { fix: 'Run LocalRouter as the signed-in user, not as administrator, unless the apps you control run as administrator.' } : {}) },
    { id: 'dpi', label: 'Per-monitor DPI awareness', granted: facts.dpi === 'per_monitor_v2' || facts.dpi === 'per_monitor', for: 'clicks landing on their control on scaled and mixed-scale displays', ...(facts.dpi.startsWith('per_monitor') ? {} : { fix: 'Windows 10 version 1703 or later is needed; on older builds clicks may miss on displays scaled differently.' }) },
  ]
  return { accessibility: facts.uia && !facts.secure, screen_recording: desktop && !facts.secure, input: desktop && !facts.secure, backend: WINDOWS_BACKEND, requirements }
}

/** A Chrome or Edge tab as the helper reads the tab strip. */
function tabsShape(value: unknown): { app: string; tabs: BrowserTab[] } {
  const raw = object(value, 'tabs')
  return {
    app: string(raw.app, 'tabs app'),
    tabs: array(raw.tabs, 'tabs', (entry, what) => {
      const t = object(entry, what)
      return { window: 0, index: integerIn(t.index, 0, 10_000, `${what} index`), title: string(t.title, `${what} title`), url: '', active: boolean(t.active, `${what} active`) }
    }),
  }
}

/**
 * Chrome's or Edge's tabs from the tab strip through UI Automation (the window in front when it is one, else the
 * frontmost one). The strip gives titles, not addresses: `url` is empty. Switching selects the tab with
 * SelectionItem (no pointer), then brings the browser forward.
 */
export class WindowsTabs implements BrowserTabs {
  private readonly worker: WindowsCall
  private name = 'Google Chrome'

  constructor(worker: WindowsCall) {
    this.worker = worker
  }

  /** The browser last read ("Google Chrome", "Microsoft Edge"): the app name the controller brings forward. */
  get browser(): string {
    return this.name
  }

  async list(): Promise<BrowserTab[]> {
    const read = tabsShape(await ask(this.worker, 'tabs', {}, { timeoutMs: 10_000 }))
    if (read.app) this.name = read.app
    return read.tabs
  }

  async activate(tab: BrowserTab): Promise<void> {
    await ask(this.worker, 'tab_select', { index: tab.index, title: tab.title }, { timeoutMs: 10_000 })
  }
}

export type WindowsPlatformOptions = {
  /** Starts a PowerShell child (tests pass a fake). */
  spawn?: WorkerSpawn
  /** Runs explorer.exe for opening apps. */
  runner?: CommandRunner
  env?: NodeJS.ProcessEnv
  listDir?: (path: string) => Promise<DirEntry[]>
  log?: Logger
  /** Ready-made workers instead of PowerShell (tests): `screen` for the window and pointer, `system` for the rest. */
  workers?: { screen: WindowsCall; system: WindowsCall }
}

const listDir = async (path: string): Promise<DirEntry[]> => (await readdir(path, { withFileTypes: true })).map((entry) => ({ name: entry.name, directory: entry.isDirectory() }))

/**
 * The Windows platform. Two PowerShell workers, each started on first use: `screen` (observe, act, screenshots,
 * the pointer and displays, so the local stop's pointer reads only ever wait behind a short action) and `system`
 * (Start menu, windows, controls, tabs, Windows Hello, whose prompt may wait for a minute). `close()` ends both.
 */
export function createWindowsPlatform(options: WindowsPlatformOptions = {}): Platform & { close(): void } {
  const env = options.env ?? process.env
  const made = options.workers ? null : {
    screen: new PowerShellWorker({ spawn: options.spawn, log: options.log, env }),
    system: new PowerShellWorker({ spawn: options.spawn, log: options.log, env }),
  }
  const workers = options.workers ?? made!
  const screen = new WindowsScreen(workers.screen)
  const controls = new WindowsControls({ worker: workers.system, runner: options.runner, env, listDir: options.listDir })
  const list = options.listDir ?? listDir

  /** Brings a running app's front window forward; one not running is opened from the Start menu. */
  const activate = async (name: string): Promise<void> => {
    const found = await controls.running(name)
    if (!found) {
      await controls.open(name)
      return
    }
    const front = found.windows[0]
    if (!front) throw new ControlError('not_found', `${name} has no window to bring forward.`)
    // Restore + SetForegroundWindow, within Windows' foreground lock rules: LocalRouter may take the foreground
    // while it (or a process it started, such as this worker) is in front or got the last input. When Windows
    // refuses, this fails with a sentence; it never forces the foreground (no AttachThreadInput, no key tricks).
    await ask(workers.system, 'foreground', { hwnd: front.hwnd })
  }

  return {
    os: 'win32',
    backend: WINDOWS_BACKEND,
    screen,
    tabs: new WindowsTabs(workers.system),
    launcher: {
      /** The taskbar's pinned apps, by their shortcut names, in folder order. */
      async list() {
        const folder = env.APPDATA ? join(env.APPDATA, 'Microsoft', 'Internet Explorer', 'Quick Launch', 'User Pinned', 'TaskBar') : null
        if (!folder) return []
        const entries = await list(folder).catch((): DirEntry[] => [])
        return entries.filter((e) => !e.directory && /\.lnk$/i.test(e.name)).map((e) => e.name.slice(0, -4).trim()).filter(Boolean)
      },
      activate,
    },
    apps: {
      async running() {
        return [...new Set((await controls.windows()).map((w) => w.app).filter(Boolean))]
      },
      activate,
    },
    controls,
    pointer: () => screen.pointer(),
    displays: () => screen.displays(),
    presence: new WindowsHello(workers.system),
    async access() {
      try {
        return accessReport(accessShape(await ask(workers.system, 'access', {}, { timeoutMs: 60_000 })))
      } catch (error) {
        return accessReport(null, error instanceof Error ? error.message : String(error))
      }
    },
    close() {
      made?.screen.close()
      made?.system.close()
    },
  }
}
