import { readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { ControlError, resolveInstalled } from '../../../../desktop-helper/src/computer/macControls.ts'
import type { ComputerControl, InstalledApp } from '../../../../desktop-helper/src/computer/macControls.ts'
import { execRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import type { CommandRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import { array, boolean, integerIn, number, object, string } from '../../../../desktop-helper/src/validate.ts'
import type { PlatformControls } from '../types.ts'
import { ask } from './screen.ts'
import type { WindowsCall } from './worker.ts'

/**
 * What people call apps on Windows, mapped to the Start menu names that answer to it, in order of preference: the
 * first one installed wins.
 */
export const WINDOWS_ALIASES: Record<string, string[]> = {
  chrome: ['Google Chrome', 'Google Chrome Beta', 'Google Chrome Dev', 'Google Chrome Canary', 'Chromium'],
  'google chrome': ['Google Chrome'],
  edge: ['Microsoft Edge', 'Microsoft Edge Beta', 'Microsoft Edge Dev', 'Microsoft Edge Canary'],
  'microsoft edge': ['Microsoft Edge'],
  firefox: ['Firefox', 'Mozilla Firefox', 'Firefox Developer Edition', 'Firefox Nightly'],
  brave: ['Brave', 'Brave Browser'],
  'vs code': ['Visual Studio Code', 'Visual Studio Code - Insiders'],
  vscode: ['Visual Studio Code', 'Visual Studio Code - Insiders'],
  code: ['Visual Studio Code', 'Visual Studio Code - Insiders'],
  word: ['Word', 'Microsoft Word'],
  'microsoft word': ['Word', 'Microsoft Word'],
  excel: ['Excel', 'Microsoft Excel'],
  'microsoft excel': ['Excel', 'Microsoft Excel'],
  powerpoint: ['PowerPoint', 'Microsoft PowerPoint'],
  'microsoft powerpoint': ['PowerPoint', 'Microsoft PowerPoint'],
  outlook: ['Outlook', 'Outlook (new)', 'Outlook (classic)', 'Microsoft Outlook'],
  'microsoft outlook': ['Outlook', 'Outlook (new)', 'Outlook (classic)', 'Microsoft Outlook'],
  mail: ['Outlook (new)', 'Mail', 'Outlook'],
  email: ['Outlook (new)', 'Mail', 'Outlook'],
  teams: ['Microsoft Teams', 'Microsoft Teams (work or school)', 'Teams', 'Microsoft Teams classic'],
  'microsoft teams': ['Microsoft Teams', 'Microsoft Teams (work or school)', 'Teams', 'Microsoft Teams classic'],
  onenote: ['OneNote', 'Microsoft OneNote'],
  settings: ['Settings'],
  'system settings': ['Settings'],
  preferences: ['Settings'],
  'control panel': ['Control Panel'],
  files: ['File Explorer'],
  explorer: ['File Explorer'],
  'windows explorer': ['File Explorer'],
  finder: ['File Explorer'],
  terminal: ['Windows Terminal', 'Terminal'],
  'windows terminal': ['Windows Terminal', 'Terminal'],
  powershell: ['Windows PowerShell', 'PowerShell'],
  cmd: ['Command Prompt'],
  'command prompt': ['Command Prompt'],
  store: ['Microsoft Store'],
  'app store': ['Microsoft Store'],
  'microsoft store': ['Microsoft Store'],
  photos: ['Photos', 'Microsoft Photos'],
  calculator: ['Calculator'],
  notepad: ['Notepad'],
  paint: ['Paint'],
  'task manager': ['Task Manager'],
  'snipping tool': ['Snipping Tool'],
  zoom: ['Zoom', 'Zoom Workplace'],
  whatsapp: ['WhatsApp'],
  spotify: ['Spotify'],
  'media player': ['Media Player', 'Windows Media Player'],
}

/** How an app from the Start menu is opened: its AppUserModelID through the shell's AppsFolder, or its shortcut. */
export type StartApp = { name: string; launch: { kind: 'app'; id: string } | { kind: 'lnk'; path: string } }

/** A top-level window the person can switch to, front to back. */
export type WindowInfo = { hwnd: number; pid: number; app: string; process: string; title: string; minimized: boolean }

export function windowsShape(value: unknown): WindowInfo[] {
  return array(value, 'windows', (raw, what) => {
    const w = object(raw, what)
    return { hwnd: number(w.hwnd, `${what} hwnd`), pid: number(w.pid, `${what} pid`), app: string(w.app, `${what} app`), process: string(w.process, `${what} process`), title: string(w.title, `${what} title`), minimized: boolean(w.minimized, `${what} minimized`) }
  })
}

/** Uninstallers sit next to their apps in the Start menu; "open Foo" must never find "Uninstall Foo". */
const UNWANTED = /\buninstall\b/i

/**
 * AppUserModelIDs as Get-StartApps lists them: a package family and app id (`Microsoft.WindowsCalculator_8wekyb3d8bbwe!App`),
 * a known-folder path (`{6D809377-…}\notepad.exe`) or a plain id (`Chrome`, `Microsoft.Windows.Explorer`). No quotes,
 * no line breaks, nothing that starts like a flag.
 */
const APP_ID = /^[^\s"<>|-][^"<>|\r\n]{0,259}$/

/** Get-StartApps' reply, checked: names a person reads and ids that are only ever an argument to explorer.exe. */
export function startAppsShape(value: unknown): StartApp[] {
  const out: StartApp[] = []
  for (const app of array(value, 'start apps', (raw, what) => {
    const a = object(raw, what)
    return { name: string(a.name, `${what} name`).trim(), id: string(a.id, `${what} id`).trim() }
  })) {
    if (!app.name || app.name.length > 120 || UNWANTED.test(app.name) || !APP_ID.test(app.id)) continue
    out.push({ name: app.name, launch: { kind: 'app', id: app.id } })
  }
  return out
}

export type DirEntry = { name: string; directory: boolean }

const listDir = async (path: string): Promise<DirEntry[]> => (await readdir(path, { withFileTypes: true })).map((entry) => ({ name: entry.name, directory: entry.isDirectory() }))

/** `.lnk` names under a Start menu folder, two folders deep, as StartApps opened through their shortcut. */
export async function shortcuts(root: string, list: (path: string) => Promise<DirEntry[]> = listDir, depth = 2): Promise<StartApp[]> {
  const out: StartApp[] = []
  const entries = await list(root).catch((): DirEntry[] => [])
  for (const entry of entries) {
    const path = join(root, entry.name)
    if (entry.directory) {
      if (depth > 0) out.push(...(await shortcuts(path, list, depth - 1)))
    } else if (/\.lnk$/i.test(entry.name)) {
      const name = entry.name.slice(0, -4).trim()
      if (name && !name.startsWith('-') && !UNWANTED.test(name)) out.push({ name, launch: { kind: 'lnk', path } })
    }
  }
  return out
}

const clamp = (level: number) => Math.min(100, Math.max(0, Math.round(level)))
const VOLUME_STEP = 20
const APP_CACHE_MS = 5 * 60_000

export const MUSIC_VOLUME = 'Windows has no Music app whose volume LocalRouter can set. Ask for the system volume, or Spotify’s.'

const MEDIA_DONE = { play: 'Playing', pause: 'Paused', next: 'Skipped to the next track on', previous: 'Went back a track on' } as const
const MEDIA_KEY_DONE = { play: 'Pressed the play/pause media key.', pause: 'Pressed the play/pause media key.', next: 'Pressed the next-track media key.', previous: 'Pressed the previous-track media key.' } as const

/** A media session's app, from its AppUserModelID ("Spotify.exe", "SpotifyAB.SpotifyMusic_…!Spotify", "Chrome"). */
export function sessionApp(id: string): string {
  if (/spotify/i.test(id)) return 'Spotify'
  if (/chrome/i.test(id)) return 'Google Chrome'
  if (/msedge|microsoftedge/i.test(id)) return 'Microsoft Edge'
  if (/firefox/i.test(id)) return 'Firefox'
  if (/zunemusic|mediaplayer/i.test(id)) return 'Media Player'
  return id.replace(/\.exe$/i, '').replace(/_[a-z0-9]{13}!.*$/i, '').split(/[.\\/]/).pop() || 'the player'
}

export type WindowsControlsOptions = {
  worker: WindowsCall
  /** Runs explorer.exe (argv only, never a shell). */
  runner?: CommandRunner
  env?: NodeJS.ProcessEnv
  listDir?: (path: string) => Promise<DirEntry[]>
  now?: () => number
}

/**
 * The fixed controls on Windows. App names are resolved against the Start menu before anything starts: Get-StartApps
 * (every Win32 and Store app, opened as `explorer.exe shell:AppsFolder\<AppUserModelID>`) and the Start menu's
 * shortcuts. No user text reaches a command line except as one argv entry naming something from those lists.
 */
export class WindowsControls implements PlatformControls {
  private readonly worker: WindowsCall
  private readonly runner: CommandRunner
  private readonly env: NodeJS.ProcessEnv
  private readonly listDir: (path: string) => Promise<DirEntry[]>
  private readonly now: () => number
  private cache: { at: number; apps: StartApp[] } | null = null

  constructor(options: WindowsControlsOptions) {
    this.worker = options.worker
    this.runner = options.runner ?? execRunner({ timeoutMs: 15_000, slow: 'Windows' })
    this.env = options.env ?? process.env
    this.listDir = options.listDir ?? listDir
    this.now = options.now ?? Date.now
  }

  private get explorer(): string {
    return join(this.env.SystemRoot ?? this.env.windir ?? 'C:\\Windows', 'explorer.exe')
  }

  /** Start menu apps: Get-StartApps first, then shortcuts it did not list; cached for five minutes. */
  async startApps(): Promise<StartApp[]> {
    if (this.cache && this.now() - this.cache.at < APP_CACHE_MS) return this.cache.apps
    const listed = await ask(this.worker, 'start_apps', {}, { timeoutMs: 30_000 }).then(startAppsShape, (): StartApp[] => [])
    const folders = [
      this.env.ProgramData && join(this.env.ProgramData, 'Microsoft', 'Windows', 'Start Menu', 'Programs'),
      this.env.APPDATA && join(this.env.APPDATA, 'Microsoft', 'Windows', 'Start Menu', 'Programs'),
    ].filter((folder): folder is string => Boolean(folder))
    const seen = new Set(listed.map((app) => app.name.toLowerCase()))
    const apps = [...listed]
    for (const folder of folders) {
      for (const app of await shortcuts(folder, this.listDir)) {
        if (seen.has(app.name.toLowerCase())) continue
        seen.add(app.name.toLowerCase())
        apps.push(app)
      }
    }
    if (apps.length) this.cache = { at: this.now(), apps }
    return apps
  }

  /** The Start menu app a spoken or typed name means; equally good matches are a question (ControlError 'ambiguous'). */
  async resolve(requested: string): Promise<StartApp> {
    const apps = await this.startApps()
    const name = resolveInstalled(requested, apps.map((app, i) => ({ id: String(i), name: app.name })), WINDOWS_ALIASES)
    return apps.find((app) => app.name === name)!
  }

  /** Opens an app from the Start menu. */
  async open(requested: string): Promise<string> {
    let app: StartApp
    try {
      app = await this.resolve(requested)
    } catch (error) {
      // Settings is always there, even when the Start menu could not be read.
      if (error instanceof ControlError && error.code === 'not_found' && /^\s*(windows\s+)?settings\s*$/i.test(requested)) {
        await this.launch('ms-settings:')
        return 'Settings'
      }
      throw error
    }
    await this.launch(app.launch.kind === 'app' ? `shell:AppsFolder\\${app.launch.id}` : app.launch.path)
    return app.name
  }

  /** explorer.exe exits with 1 even when it opened the app, so only a failure to start it at all is an error. */
  private async launch(argument: string): Promise<void> {
    try {
      await this.runner.run(this.explorer, [argument])
    } catch (error) {
      if (error instanceof Error && /ENOENT|EACCES|could not be started/i.test(error.message)) throw new ControlError('unavailable', 'Windows Explorer could not be started to open the app.')
    }
  }

  async windows(): Promise<WindowInfo[]> {
    return windowsShape(await ask(this.worker, 'windows', {}, { timeoutMs: 10_000 }))
  }

  /** The running app a name means, among the apps with a window, and its windows (front first). */
  async running(requested: string): Promise<{ app: string; windows: WindowInfo[] } | null> {
    const windows = await this.windows()
    const apps = new Map<string, InstalledApp>()
    for (const w of windows) {
      const known = apps.get(w.app)
      if (!known) apps.set(w.app, { id: String(apps.size), name: w.app, aliases: w.process && w.process.toLowerCase() !== w.app.toLowerCase() ? [w.process] : [] })
    }
    let app: string
    try {
      app = resolveInstalled(requested, [...apps.values()], WINDOWS_ALIASES)
    } catch (error) {
      if (error instanceof ControlError && error.code === 'not_found') return null
      throw error
    }
    return { app, windows: windows.filter((w) => w.app === app) }
  }

  async run(control: ComputerControl): Promise<{ done: string }> {
    switch (control.kind) {
      case 'open_app':
        return { done: `Opened ${await this.open(control.app)}.` }
      case 'quit_app': {
        const found = await this.running(control.app)
        if (!found) {
          const installed = await this.resolve(control.app)
          return { done: `${installed.name} wasn't open.` }
        }
        // WM_CLOSE to each window, what closing it by hand does: the app may still ask about unsaved work.
        for (const w of found.windows) await ask(this.worker, 'close', { hwnd: w.hwnd })
        return { done: `Quit ${found.app}.` }
      }
      case 'volume':
        return this.volume(control)
      case 'media':
        return this.media(control)
      case 'dark_mode': {
        const result = object(await ask(this.worker, 'dark_mode', { action: control.action }), 'dark mode')
        return { done: `Dark mode is ${boolean(result.dark, 'dark mode dark') ? 'on' : 'off'}.` }
      }
      case 'lock_screen':
        await ask(this.worker, 'lock')
        return { done: 'Locked the screen.' }
    }
  }

  private async volume(control: Extract<ComputerControl, { kind: 'volume' }>): Promise<{ done: string }> {
    if (control.target === 'Music') throw new ControlError('unavailable', MUSIC_VOLUME)
    const target = control.target
    const who = target === 'system' ? 'Volume' : `${target} volume`
    const read = (value: unknown) => {
      const v = object(value, 'volume')
      return { level: integerIn(v.level, 0, 100, 'volume level'), muted: boolean(v.muted, 'volume muted') }
    }
    try {
      if (control.action === 'mute' || control.action === 'unmute') {
        await ask(this.worker, 'volume_write', { target, muted: control.action === 'mute' })
        return { done: control.action === 'mute' ? (target === 'system' ? 'Muted.' : `Muted ${target}.`) : (target === 'system' ? 'Unmuted.' : `Unmuted ${target}.`) }
      }
      let level: number
      if (control.action === 'set') {
        if (control.level === undefined || !Number.isFinite(control.level)) throw new ControlError('invalid', 'A volume level from 0 to 100 is required')
        level = clamp(control.level)
      } else {
        const now = read(await ask(this.worker, 'volume_read', { target }))
        level = clamp(now.level + (control.action === 'up' ? VOLUME_STEP : -VOLUME_STEP))
      }
      // Changing the level unmutes, as the volume keys do.
      const after = read(await ask(this.worker, 'volume_write', { target, level, muted: false }))
      return { done: `${who} is at ${after.level}%.` }
    } catch (error) {
      if (error instanceof Error && /LR_NOT_PLAYING\|/.test(error.message)) return { done: `${target} isn't playing.` }
      throw error
    }
  }

  /**
   * The system media session (what the media flyout controls): Spotify's own session when Spotify is asked for, else
   * whatever is playing. When the session cannot be reached (an older Windows), the media keys do the same.
   */
  private async media(control: Extract<ComputerControl, { kind: 'media' }>): Promise<{ done: string }> {
    let session: { session: boolean; done: boolean; app: string } | null = null
    try {
      const raw = object(await ask(this.worker, 'media', { action: control.action, player: control.player }, { timeoutMs: 25_000 }), 'media')
      session = { session: boolean(raw.session, 'media session'), done: boolean(raw.done, 'media done'), app: string(raw.app, 'media app') }
    } catch {
      session = null
    }
    if (session?.done) return { done: `${MEDIA_DONE[control.action]} ${control.player === 'Spotify' ? 'Spotify' : sessionApp(session.app)}.` }
    if (session && !session.session) {
      if (control.player === 'Spotify') return { done: "Spotify isn't playing." }
      if (control.action !== 'play') return { done: 'Nothing is playing.' }
    }
    await ask(this.worker, 'media_key', { action: control.action })
    return { done: MEDIA_KEY_DONE[control.action] }
  }
}
