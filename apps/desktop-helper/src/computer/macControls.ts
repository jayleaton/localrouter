// Ported from SideBud (github.com/jayleaton/sidebud @70da540, apps/companion/src/computer/macControls.ts), reused in LocalRouter with the owner's permission.
import { execFile } from 'node:child_process'
import { readdir, readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { jxa } from './desktop.ts'
import { integerIn, isObject, oneOf, string } from '../validate.ts'

/** `name`: the bundle's file name (what `open -a` takes); `aliases`: its bundle and display names, when they differ. */
export type InstalledApp = { id: string; name: string; aliases?: string[] }

export const PLAYERS = ['Spotify', 'Music'] as const
export type Player = (typeof PLAYERS)[number]

/** The closed set of controls the box may ask for, in the protocol's order. */
export const CONTROL_KINDS = ['open_app', 'quit_app', 'volume', 'media', 'dark_mode', 'lock_screen'] as const

/** The closed set of computer actions the box may request; nothing here reaches a shell. */
export type ComputerControl =
  | { kind: 'open_app'; app: string }
  | { kind: 'quit_app'; app: string }
  | { kind: 'volume'; action: 'up' | 'down' | 'mute' | 'unmute' | 'set'; level?: number; target: 'system' | Player }
  | { kind: 'media'; action: 'play' | 'pause' | 'next' | 'previous'; player: Player }
  | { kind: 'dark_mode'; action: 'on' | 'off' | 'toggle' }
  | { kind: 'lock_screen' }

/** A control that could not run, with a code the box can sort on and a message for the user. */
export class ControlError extends Error {
  readonly code: 'unavailable' | 'not_found' | 'forbidden' | 'invalid' | 'ambiguous'

  constructor(code: ControlError['code'], message: string) {
    super(message)
    this.code = code
  }
}

const appName = (value: unknown, what: string) => {
  const name = string(value, what)
  if (name.length < 1 || name.length > 120) throw new ControlError('invalid', `${what} should be 1 to 120 characters`)
  return name
}

/** Checks a `control` from the box against the closed set; throws a ControlError on anything else. */
export function parseControl(value: unknown): ComputerControl {
  try {
    return readControl(value)
  } catch (error) {
    if (error instanceof ControlError) throw error
    throw new ControlError('invalid', error instanceof Error ? error.message : String(error))
  }
}

function readControl(value: unknown): ComputerControl {
  if (!isObject(value)) throw new ControlError('invalid', 'control should be an object')
  const kind = oneOf(value.kind, CONTROL_KINDS, 'control.kind')
  switch (kind) {
    case 'open_app':
      return { kind: 'open_app', app: appName(value.app, 'control.app') }
    case 'quit_app':
      return { kind: 'quit_app', app: appName(value.app, 'control.app') }
    case 'volume': {
      const action = oneOf(value.action, ['up', 'down', 'mute', 'unmute', 'set'] as const, 'control.action')
      const target = oneOf(value.target, ['system', ...PLAYERS] as const, 'control.target')
      const level = value.level === undefined || value.level === null ? undefined : integerIn(value.level, 0, 100, 'control.level')
      return level === undefined ? { kind: 'volume', action, target } : { kind: 'volume', action, level, target }
    }
    case 'media':
      return { kind: 'media', action: oneOf(value.action, ['play', 'pause', 'next', 'previous'] as const, 'control.action'), player: oneOf(value.player, PLAYERS, 'control.player') }
    case 'dark_mode':
      return { kind: 'dark_mode', action: oneOf(value.action, ['on', 'off', 'toggle'] as const, 'control.action') }
    case 'lock_screen':
      return { kind: 'lock_screen' }
  }
}

import type { CommandRunner } from './runner.ts'
export type { CommandRunner } from './runner.ts'

export type MacControlsOptions = {
  runner?: CommandRunner
  platform?: NodeJS.Platform
  appDirectories?: string[]
  readDir?: (path: string) => Promise<string[]>
  /** An app's Info.plist as text (its bundle and display names, when it is XML). */
  readText?: (path: string) => Promise<string>
  now?: () => number
}

const execFileAsync = promisify(execFile)
const execRunner: CommandRunner = {
  async run(file, args) {
    const { stdout } = await execFileAsync(file, args, { timeout: 10_000 })
    return stdout
  },
}

const APP_CACHE_MS = 5 * 60_000
const VOLUME_STEP = 20
const PERMISSION_ERROR = /not allowed|assistive access|accessibility|\(1002\)|-1743|-25211/i

// Player scripts name the app literally because `next track` and `sound volume` only compile
// against a known dictionary; the literal comes from the closed Player union, never from input.
const MEDIA_VERBS = { play: 'play', pause: 'pause', next: 'next track', previous: 'previous track' }
const MEDIA_DONE = { play: 'Playing', pause: 'Paused', next: 'Skipped to the next track on', previous: 'Went back a track on' }

/**
 * A Notification Center banner, title and text through argv. Shown by osascript, so macOS lists it
 * under Script Editor in System Settings › Notifications.
 */
export const NOTIFY_SCRIPT = `function run(argv) {
  var app = Application.currentApplication();
  app.includeStandardAdditions = true;
  app.displayNotification(argv[1], { withTitle: argv[0], soundName: 'Glass' });
  return 'ok';
}`

/** Scripts take dynamic values only through argv so names are never parsed as AppleScript. */
function script(lines: string[], args: string[] = []): string[] {
  return [...['on run argv', ...lines, 'end run'].flatMap((line) => ['-e', line]), ...args]
}

function slugify(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'app'
}

const clamp = (level: number) => Math.min(100, Math.max(0, Math.round(level)))

/**
 * Deterministic macOS controls the helper runs directly when the box sends a `control`.
 * Every app name is resolved against the installed-app scan before any command runs.
 */
export class MacControls {
  private readonly runner: CommandRunner
  private readonly platform: NodeJS.Platform
  private readonly appDirectories: string[]
  private readonly readDir: (path: string) => Promise<string[]>
  private readonly readText: (path: string) => Promise<string>
  private readonly now: () => number
  private cache: { at: number; apps: InstalledApp[] } | null = null

  constructor(options: MacControlsOptions = {}) {
    this.runner = options.runner ?? execRunner
    this.platform = options.platform ?? process.platform
    this.appDirectories = options.appDirectories ?? [
      '/Applications',
      '/System/Applications',
      '/System/Applications/Utilities',
      // Finder lives here; the folder is read one level deep like the others.
      '/System/Library/CoreServices',
      join(homedir(), 'Applications'),
    ]
    this.readDir = options.readDir ?? ((path) => readdir(path))
    this.readText = options.readText ?? ((path) => readFile(path, 'utf8'))
    this.now = options.now ?? Date.now
  }

  get available(): boolean {
    return this.platform === 'darwin'
  }

  /** Installed .app bundles, top level and one folder deep, sorted and cached for five minutes. */
  async apps(): Promise<InstalledApp[]> {
    if (this.cache && this.now() - this.cache.at < APP_CACHE_MS) return this.cache.apps
    const names = new Map<string, string>()
    const paths = new Map<string, string>()
    const add = (directory: string) => (entry: string) => {
      const name = entry.slice(0, -'.app'.length)
      // A leading dash could be read as an osascript/open flag when passed as argv.
      if (name && !name.startsWith('-') && !names.has(name.toLowerCase())) {
        names.set(name.toLowerCase(), name)
        paths.set(name, join(directory, entry))
      }
    }
    const list = (path: string) => this.readDir(path).catch((): string[] => [])
    for (const directory of this.appDirectories) {
      for (const entry of await list(directory)) {
        if (entry.endsWith('.app')) add(directory)(entry)
        else if (!entry.startsWith('.')) (await list(join(directory, entry))).filter((inner) => inner.endsWith('.app')).forEach(add(join(directory, entry)))
      }
    }
    const aliases = new Map<string, string[]>()
    await Promise.all([...paths].map(async ([name, path]) => {
      const plist = await this.readText(join(path, 'Contents', 'Info.plist')).catch(() => '')
      const found = plistNames(plist).filter((alias) => alias.toLowerCase() !== name.toLowerCase())
      if (found.length) aliases.set(name, found)
    }))
    const used = new Set<string>()
    const apps = [...names.values()]
      .sort((a, b) => a.localeCompare(b))
      .map((name) => {
        const base = slugify(name)
        let id = base
        for (let n = 2; used.has(id); n += 1) id = `${base}_${n}`
        used.add(id)
        const also = aliases.get(name)
        return also ? { id, name, aliases: also } : { id, name }
      })
    this.cache = { at: this.now(), apps }
    return apps
  }

  /**
   * Whether the user seems to be at this Mac: the screen is unlocked and it had keyboard or mouse
   * input in the last `idleSeconds`. False when unknown.
   */
  async atComputer(idleSeconds = 300): Promise<boolean> {
    if (!this.available) return false
    try {
      const [idle, root] = await Promise.all([
        this.runner.run('ioreg', ['-c', 'IOHIDSystem', '-d', '4', '-r', '-k', 'HIDIdleTime']),
        this.runner.run('ioreg', ['-n', 'Root', '-d', '1']),
      ])
      const nanoseconds = Number(idle.match(/"HIDIdleTime"\s*=\s*(\d+)/)?.[1] ?? NaN)
      const locked = /"CGSSessionScreenIsLocked"\s*=\s*Yes/.test(root)
      return !locked && Number.isFinite(nanoseconds) && nanoseconds / 1e9 < idleSeconds
    } catch {
      return false
    }
  }

  /** Names of running foreground apps; empty when System Events is unavailable. */
  async running(): Promise<string[]> {
    if (!this.available) return []
    try {
      const out = await this.osascript([
        'tell application "System Events" to set names to name of every application process whose background only is false',
        'set AppleScript\'s text item delimiters to linefeed',
        'return names as text',
      ])
      return out.split('\n').map((line) => line.trim()).filter(Boolean)
    } catch {
      return []
    }
  }

  /** Brings a running or installed app to the front by name (no Dock click, so the pointer does not move). */
  async activate(name: string): Promise<void> {
    if (!this.available) throw new ControlError('unavailable', 'Computer controls work on macOS only')
    if (!name || name.startsWith('-')) throw new ControlError('invalid', `${name} is not an app name`)
    await this.runner.run('open', ['-a', name])
  }

  /** A Notification Center banner on this Mac (the task-start notice); failures are ignored. */
  async notify(title: string, text: string): Promise<void> {
    if (!this.available) return
    await this.runner.run(...jxa(NOTIFY_SCRIPT, [title, text].map((value) => value.replace(/^-+\s*/, '')))).catch(() => undefined)
  }

  describe(control: ComputerControl): string {
    switch (control.kind) {
      case 'open_app':
        return `open ${control.app}`
      case 'quit_app':
        return `quit ${control.app}`
      case 'volume': {
        const target = control.target === 'system' ? 'system volume' : `${control.target} volume`
        return control.action === 'set' ? `set ${target} to ${control.level ?? '?'}%` : `${control.action} ${target}`
      }
      case 'media':
        return `${MEDIA_VERBS[control.action]} ${control.player}`
      case 'dark_mode':
        return `dark mode ${control.action}`
      case 'lock_screen':
        return 'lock the screen'
    }
  }

  /** Runs one control and returns a short sentence suitable for speaking back. */
  async run(control: ComputerControl): Promise<{ done: string }> {
    if (!this.available) throw new ControlError('unavailable', 'Computer controls work on macOS only')
    switch (control.kind) {
      case 'open_app': {
        const app = await this.resolveApp(control.app)
        await this.runner.run('open', ['-a', app])
        return { done: `Opened ${app}.` }
      }
      case 'quit_app': {
        const app = await this.resolveApp(control.app)
        if (!(await this.isRunning(app))) return { done: `${app} wasn't open.` }
        await this.osascript(['tell application (item 1 of argv) to quit'], [app])
        return { done: `Quit ${app}.` }
      }
      case 'volume':
        return control.target === 'system' ? this.systemVolume(control) : this.playerVolume(control, control.target)
      case 'media': {
        const player = await this.resolveApp(control.player)
        if (control.action !== 'play' && !(await this.isRunning(player))) return { done: `${player} isn't playing.` }
        await this.osascript([`tell application "${control.player}" to ${MEDIA_VERBS[control.action]}`])
        return { done: `${MEDIA_DONE[control.action]} ${player}.` }
      }
      case 'dark_mode': {
        const next = { on: 'true', off: 'false', toggle: 'not dark mode' }[control.action]
        const out = await this.osascript([
          'tell application "System Events" to tell appearance preferences',
          `set dark mode to ${next}`,
          'return dark mode',
          'end tell',
        ])
        return { done: `Dark mode is ${out.trim() === 'true' ? 'on' : 'off'}.` }
      }
      case 'lock_screen':
        try {
          await this.osascript(['tell application "System Events" to keystroke "q" using {control down, command down}'])
        } catch (error) {
          if (error instanceof Error && PERMISSION_ERROR.test(error.message)) {
            throw new ControlError(
              'forbidden',
              'Locking needs Accessibility access. Allow the node binary that runs LocalRouter Desktop under System Settings → Privacy & Security → Accessibility.'
            )
          }
          throw error
        }
        return { done: 'Locked the screen.' }
    }
  }

  private async systemVolume(control: Extract<ComputerControl, { kind: 'volume' }>): Promise<{ done: string }> {
    if (control.action === 'mute' || control.action === 'unmute') {
      await this.osascript([`set volume output muted ${control.action === 'mute'}`])
      return { done: control.action === 'mute' ? 'Muted.' : 'Unmuted.' }
    }
    const level = await this.nextLevel(control, () => this.osascript(['return output volume of (get volume settings)']))
    await this.osascript(['set volume output volume ((item 1 of argv) as integer)'], [String(level)])
    return { done: `Volume is at ${level}%.` }
  }

  private async playerVolume(
    control: Extract<ComputerControl, { kind: 'volume' }>,
    target: Player
  ): Promise<{ done: string }> {
    const player = await this.resolveApp(target)
    if (!(await this.isRunning(player))) return { done: `${player} isn't playing.` }
    const level =
      control.action === 'mute' ? 0 : control.action === 'unmute' ? 50 : await this.nextLevel(control, () =>
        this.osascript([`return sound volume of application "${target}"`])
      )
    await this.osascript([`tell application "${target}" to set sound volume to ((item 1 of argv) as integer)`], [String(level)])
    return { done: `${player} volume is at ${level}%.` }
  }

  private async nextLevel(
    control: Extract<ComputerControl, { kind: 'volume' }>,
    read: () => Promise<string>
  ): Promise<number> {
    if (control.action === 'set') {
      if (control.level === undefined || !Number.isFinite(control.level)) {
        throw new ControlError('invalid', 'A volume level from 0 to 100 is required')
      }
      return clamp(control.level)
    }
    const current = Number((await read()).trim())
    if (!Number.isFinite(current)) throw new ControlError('unavailable', 'This output device has no adjustable volume')
    return clamp(current + (control.action === 'up' ? VOLUME_STEP : -VOLUME_STEP))
  }

  private async resolveApp(requested: string): Promise<string> {
    return resolveInstalled(requested, await this.apps())
  }

  // `is running` checks without launching, unlike a `tell` block.
  private async isRunning(app: string): Promise<boolean> {
    return (await this.osascript(['return application (item 1 of argv) is running'], [app])).trim() === 'true'
  }

  private osascript(lines: string[], args: string[] = []): Promise<string> {
    return this.runner.run('osascript', script(lines, args))
  }
}

/** CFBundleDisplayName and CFBundleName from an XML Info.plist (a binary one gives none: the file name is used). */
export function plistNames(plist: string): string[] {
  if (!plist.trimStart().startsWith('<?xml') && !plist.includes('<plist')) return []
  const out: string[] = []
  for (const key of ['CFBundleDisplayName', 'CFBundleName']) {
    const m = new RegExp(`<key>${key}</key>\\s*<string>([^<]{1,80})</string>`).exec(plist)
    const v = m?.[1]?.trim()
    if (v && !v.startsWith('-') && !v.includes('$(') && !out.includes(v)) out.push(v)
  }
  return out
}

/**
 * What people call apps, mapped to the bundles that answer to it, in order of preference: the first one installed
 * wins ("Chrome" is Google Chrome even with Chrome Canary there too).
 */
export const APP_ALIASES: Record<string, string[]> = {
  chrome: ['Google Chrome', 'Google Chrome Beta', 'Google Chrome Canary', 'Chromium'],
  'google chrome': ['Google Chrome'],
  'vs code': ['Visual Studio Code', 'Visual Studio Code - Insiders'],
  vscode: ['Visual Studio Code', 'Visual Studio Code - Insiders'],
  code: ['Visual Studio Code', 'Visual Studio Code - Insiders'],
  'visual studio': ['Visual Studio Code'],
  word: ['Microsoft Word'],
  excel: ['Microsoft Excel'],
  powerpoint: ['Microsoft PowerPoint'],
  outlook: ['Microsoft Outlook'],
  teams: ['Microsoft Teams', 'Microsoft Teams (work or school)', 'Microsoft Teams classic'],
  edge: ['Microsoft Edge'],
  onenote: ['Microsoft OneNote'],
  settings: ['System Settings', 'System Preferences'],
  'system preferences': ['System Settings', 'System Preferences'],
  preferences: ['System Settings', 'System Preferences'],
  mail: ['Mail'],
  'apple mail': ['Mail'],
  email: ['Mail'],
  itunes: ['Music'],
  'apple music': ['Music'],
  imessage: ['Messages'],
  texts: ['Messages'],
  ical: ['Calendar'],
  iterm: ['iTerm', 'iTerm2'],
  zoom: ['zoom.us'],
  firefox: ['Firefox', 'Firefox Developer Edition', 'Firefox Nightly'],
  brave: ['Brave Browser'],
  arc: ['Arc'],
  photos: ['Photos'],
  'app store': ['App Store'],
  files: ['Finder'],
  whatsapp: ['WhatsApp'],
}

const fold = (s: string) => s.toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
const squash = (s: string) => fold(s).replace(/[^a-z0-9]/g, '')
const words = (s: string) => fold(s).split(/[^a-z0-9]+/).filter(Boolean)

/**
 * The installed app a spoken or typed name means: its exact name (or bundle/display name), a common alias, the same
 * letters without spaces or punctuation ("googlechrome", "zoom us"), then a case-insensitive word-prefix match
 * ("photoshop" is "Adobe Photoshop 2026"). Several equally good matches are a question, never a guess.
 */
export function resolveInstalled(requested: string, apps: InstalledApp[]): string {
  const asked = requested.trim()
  const names = (app: InstalledApp) => [app.name, ...(app.aliases ?? [])]
  const unique = (found: InstalledApp[]) => [...new Map(found.map((app) => [app.name, app])).values()]
  const pick = (found: InstalledApp[]): string | null => {
    const u = unique(found)
    if (u.length === 1) return u[0].name
    if (u.length > 1) throw new ControlError('ambiguous', `Which one do you mean: ${u.map((app) => app.name).sort((a, b) => a.localeCompare(b)).slice(0, 5).join(', ')}?`)
    return null
  }
  const exact = pick(apps.filter((app) => names(app).some((n) => fold(n) === fold(asked))))
  if (exact) return exact
  for (const preferred of APP_ALIASES[fold(asked).replace(/\s+/g, ' ')] ?? []) {
    const hit = apps.find((app) => names(app).some((n) => fold(n) === fold(preferred)))
    if (hit) return hit.name
  }
  const flat = squash(asked)
  if (flat) {
    const same = pick(apps.filter((app) => names(app).some((n) => squash(n) === flat)))
    if (same) return same
    const want = words(asked)
    const prefixed = apps.filter((app) => names(app).some((n) => {
      const have = words(n)
      return squash(n).startsWith(flat) || want.every((w) => have.some((h) => h.startsWith(w)))
    }))
    const found = pick(prefixed)
    if (found) return found
  }
  throw new ControlError('not_found', `${requested} isn't installed on this computer`)
}
