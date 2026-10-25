import { readdir, readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { ControlError } from '../../../../desktop-helper/src/computer/macControls.ts'
import type { ComputerControl, Player } from '../../../../desktop-helper/src/computer/macControls.ts'
import type { CommandRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import type { ControlSupport, PlatformControls } from '../types.ts'
import { clientWindows } from './compositor.ts'
import type { Session } from './session.ts'

/**
 * The fixed controls on Linux. Apps are the installed .desktop entries (the menu a person sees), launched with
 * gtk-launch or gio launch by their id or path: never a shell, and never their Exec= line run by us. Everything
 * else is one well-known tool per job (wpctl/pactl/amixer, playerctl, gsettings or plasma-apply-colorscheme,
 * loginctl), each run with argv only.
 */

/** One application entry. `id` is the desktop file id ("org.gnome.Nautilus.desktop"), unique across directories. */
export type DesktopEntry = { id: string; path: string; name: string; generic?: string; keywords: string[]; wmClass?: string; command?: string }

/** The [Desktop Entry] group of a .desktop file, or null when it is not a visible application. */
export function parseDesktopEntry(text: string, id: string, path: string): DesktopEntry | null {
  const fields: Record<string, string> = {}
  let inMain = false
  for (const line of text.split('\n')) {
    const trimmed = line.trim()
    if (trimmed.startsWith('[')) {
      inMain = trimmed === '[Desktop Entry]'
      continue
    }
    if (!inMain || trimmed.startsWith('#')) continue
    const eq = trimmed.indexOf('=')
    // Localised keys (Name[de]) are skipped: names are matched as the person types them in English or as installed.
    if (eq > 0) fields[trimmed.slice(0, eq).trim()] ??= trimmed.slice(eq + 1).trim()
  }
  if (fields.Type !== 'Application' || fields.NoDisplay === 'true' || fields.Hidden === 'true' || !fields.Name) return null
  const command = (fields.Exec ?? '').split(/\s+/).filter((token) => token && token !== 'env' && !token.includes('='))[0]?.replace(/^["']|["']$/g, '')
  const entry: DesktopEntry = { id, path, name: fields.Name, keywords: (fields.Keywords ?? '').split(';').map((k) => k.trim()).filter(Boolean) }
  if (fields.GenericName) entry.generic = fields.GenericName
  if (fields.StartupWMClass) entry.wmClass = fields.StartupWMClass
  if (command) entry.command = command.split('/').pop()
  return entry
}

/** Where .desktop files live, in precedence order (the first file with an id wins), flatpak's exports included. */
export function applicationDirs(env: NodeJS.ProcessEnv, home: string): string[] {
  const dataHome = env.XDG_DATA_HOME || join(home, '.local/share')
  const dataDirs = (env.XDG_DATA_DIRS || '/usr/local/share:/usr/share').split(':').filter(Boolean)
  const dirs = [dataHome, ...dataDirs, join(home, '.local/share/flatpak/exports/share'), '/var/lib/flatpak/exports/share'].map((dir) => join(dir, 'applications'))
  return [...new Set(dirs)]
}

/**
 * What people call apps, mapped to the desktop ids that answer to it, in order of preference: the first one
 * installed wins ("Chrome" is Google Chrome even with Chromium there too).
 */
export const LINUX_ALIASES: Record<string, string[]> = {
  chrome: ['google-chrome', 'com.google.Chrome', 'google-chrome-beta', 'google-chrome-unstable', 'chromium', 'chromium-browser', 'org.chromium.Chromium'],
  'google chrome': ['google-chrome', 'com.google.Chrome'],
  chromium: ['chromium', 'chromium-browser', 'org.chromium.Chromium'],
  firefox: ['firefox', 'org.mozilla.firefox', 'firefox-esr'],
  'vs code': ['code', 'visual-studio-code', 'com.visualstudio.code', 'code-oss', 'codium', 'com.vscodium.codium'],
  vscode: ['code', 'visual-studio-code', 'com.visualstudio.code', 'code-oss', 'codium', 'com.vscodium.codium'],
  code: ['code', 'visual-studio-code', 'com.visualstudio.code', 'code-oss', 'codium'],
  'visual studio code': ['code', 'visual-studio-code', 'com.visualstudio.code'],
  settings: ['org.gnome.Settings', 'gnome-control-center', 'systemsettings', 'kdesystemsettings', 'xfce-settings-manager'],
  'system settings': ['org.gnome.Settings', 'gnome-control-center', 'systemsettings', 'kdesystemsettings'],
  preferences: ['org.gnome.Settings', 'gnome-control-center', 'systemsettings'],
  files: ['org.gnome.Nautilus', 'nautilus', 'org.kde.dolphin', 'thunar', 'nemo', 'pcmanfm'],
  'file manager': ['org.gnome.Nautilus', 'nautilus', 'org.kde.dolphin', 'thunar', 'nemo', 'pcmanfm'],
  finder: ['org.gnome.Nautilus', 'nautilus', 'org.kde.dolphin', 'thunar', 'nemo'],
  terminal: ['org.gnome.Console', 'org.gnome.Terminal', 'org.kde.konsole', 'com.mitchellh.ghostty', 'Alacritty', 'kitty', 'foot', 'xfce4-terminal'],
  'text editor': ['org.gnome.TextEditor', 'org.gnome.gedit', 'org.kde.kate', 'org.kde.kwrite', 'mousepad'],
  calculator: ['org.gnome.Calculator', 'org.kde.kcalc', 'galculator'],
  mail: ['thunderbird', 'org.mozilla.Thunderbird', 'net.thunderbird.Thunderbird', 'org.gnome.Evolution', 'org.gnome.Geary', 'org.kde.kmail2'],
  email: ['thunderbird', 'org.mozilla.Thunderbird', 'net.thunderbird.Thunderbird', 'org.gnome.Evolution', 'org.gnome.Geary', 'org.kde.kmail2'],
  spotify: ['spotify', 'com.spotify.Client'],
  brave: ['brave-browser', 'com.brave.Browser'],
  edge: ['microsoft-edge', 'com.microsoft.Edge'],
  zoom: ['Zoom', 'us.zoom.Zoom'],
  slack: ['slack', 'com.slack.Slack'],
  discord: ['discord', 'com.discordapp.Discord'],
}

const fold = (s: string) => s.toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').trim()
const squash = (s: string) => fold(s).replace(/[^a-z0-9]/g, '')
const words = (s: string) => fold(s).split(/[^a-z0-9]+/).filter(Boolean)
const bare = (id: string) => id.replace(/\.desktop$/, '')
/** The short name in a reverse-DNS id: "org.gnome.Nautilus" is "Nautilus". */
const short = (id: string) => bare(id).split('.').pop() ?? bare(id)

/**
 * The entry a spoken or typed name means: its Name or file id exactly, a common alias, the same letters without
 * spaces or punctuation, its GenericName or a keyword ("browser"), then a word-prefix of its Name. Several equally
 * good matches are a question, never a guess.
 */
export function resolveEntry(requested: string, entries: DesktopEntry[]): DesktopEntry {
  const asked = fold(requested)
  const pick = (found: DesktopEntry[]): DesktopEntry | null => {
    const unique = [...new Map(found.map((entry) => [entry.id, entry])).values()]
    if (unique.length === 1) return unique[0]
    if (unique.length > 1) {
      const names = unique.map((entry) => entry.name)
      const label = (entry: DesktopEntry) => (names.filter((n) => n === entry.name).length > 1 ? `${entry.name} (${bare(entry.id)})` : entry.name)
      throw new ControlError('ambiguous', `Which one do you mean: ${unique.map(label).sort((a, b) => a.localeCompare(b)).slice(0, 5).join(', ')}?`)
    }
    return null
  }
  const ids = (entry: DesktopEntry) => [bare(entry.id), short(entry.id)]
  const exact = pick(entries.filter((entry) => fold(entry.name) === asked || ids(entry).some((id) => fold(id) === asked)))
  if (exact) return exact
  for (const preferred of LINUX_ALIASES[asked.replace(/\s+/g, ' ')] ?? []) {
    const hit = entries.find((entry) => fold(bare(entry.id)) === fold(preferred))
    if (hit) return hit
  }
  const flat = squash(requested)
  if (flat) {
    const same = pick(entries.filter((entry) => [entry.name, ...ids(entry)].some((n) => squash(n) === flat)))
    if (same) return same
    const generic = pick(entries.filter((entry) => [entry.generic ?? '', ...entry.keywords].some((n) => n && fold(n) === asked)))
    if (generic) return generic
    const want = words(requested)
    const prefixed = pick(entries.filter((entry) => {
      const have = words(entry.name)
      return squash(entry.name).startsWith(flat) || want.every((w) => have.some((h) => h.startsWith(w)))
    }))
    if (prefixed) return prefixed
  }
  throw new ControlError('not_found', `${requested} isn't installed on this computer`)
}

/**
 * Whether a window's app (an X11 WM_CLASS "instance.Class", a Wayland app_id or Hyprland class) belongs to `entry`:
 * equal to its StartupWMClass, id, short id, command or Name, or ending or starting with one of them at a dot.
 */
export function ownsWindow(entry: DesktopEntry, app: string): boolean {
  const have = fold(app)
  if (!have) return false
  const keys = [entry.wmClass, bare(entry.id), short(entry.id), entry.command, entry.name].filter((k): k is string => Boolean(k)).map(fold)
  return keys.some((key) => have === key || have.startsWith(`${key}.`) || have.endsWith(`.${key}`))
}

export type LinuxControlsOptions = {
  session: Session
  runner: CommandRunner
  home?: string
  /** A directory's entries; readdir with file types by default. */
  list?: (path: string) => Promise<{ name: string; dir: boolean }[]>
  readText?: (path: string) => Promise<string>
  now?: () => number
}

const APP_CACHE_MS = 5 * 60_000
const VOLUME_STEP = 20
const clamp = (level: number) => Math.min(100, Math.max(0, Math.round(level)))
const MEDIA_DONE = { play: 'Playing', pause: 'Paused', next: 'Skipped to the next track on', previous: 'Went back a track on' }
const playerName = (player: Player) => (player === 'Spotify' ? 'Spotify' : 'the music player')
/** Spotify by its MPRIS name; Music is whichever player playerctl picks (the most recently active one). */
const playerArgs = (player: Player) => (player === 'Spotify' ? ['-p', 'spotify'] : [])
const NO_PLAYER = /no players? (found|could handle)/i

export class LinuxControls implements PlatformControls {
  readonly unsupported: ControlSupport
  private readonly session: Session
  private readonly runner: CommandRunner
  private readonly home: string
  private readonly list: (path: string) => Promise<{ name: string; dir: boolean }[]>
  private readonly readText: (path: string) => Promise<string>
  private readonly now: () => number
  private cache: { at: number; entries: DesktopEntry[] } | null = null

  constructor(options: LinuxControlsOptions) {
    this.session = options.session
    this.runner = options.runner
    this.home = options.home ?? homedir()
    this.list = options.list ?? (async (path) => (await readdir(path, { withFileTypes: true })).map((d) => ({ name: d.name, dir: d.isDirectory() })))
    this.readText = options.readText ?? ((path) => readFile(path, 'utf8'))
    this.now = options.now ?? Date.now
    this.unsupported = unsupportedControls(options.session)
  }

  /** Installed application entries, cached for five minutes. Subdirectories make ids with dashes, as the spec says. */
  async entries(): Promise<DesktopEntry[]> {
    if (this.cache && this.now() - this.cache.at < APP_CACHE_MS) return this.cache.entries
    const byId = new Map<string, DesktopEntry | null>()
    const walk = async (dir: string, prefix: string, depth: number): Promise<void> => {
      const items = await this.list(dir).catch(() => [])
      for (const item of items.sort((a, b) => a.name.localeCompare(b.name))) {
        if (item.dir && depth < 3) await walk(join(dir, item.name), `${prefix}${item.name}-`, depth + 1)
        else if (!item.dir && item.name.endsWith('.desktop')) {
          const id = `${prefix}${item.name}`
          // A user's own file with the same id hides the system one, even when it hides the app (NoDisplay).
          if (byId.has(id)) continue
          const path = join(dir, item.name)
          byId.set(id, parseDesktopEntry(await this.readText(path).catch(() => ''), id, path))
        }
      }
    }
    for (const dir of applicationDirs(this.session.env, this.home)) await walk(dir, '', 0)
    const entries = [...byId.values()].filter((entry): entry is DesktopEntry => entry !== null && !entry.id.startsWith('-'))
    this.cache = { at: this.now(), entries }
    return entries
  }

  async resolve(name: string): Promise<DesktopEntry> {
    return resolveEntry(name, await this.entries())
  }

  /** Launches an entry; for a single-instance app that is already open this brings its window forward. */
  async launch(entry: DesktopEntry): Promise<void> {
    const t = this.session.tools
    if (t['gtk-launch']) await this.runner.run(t['gtk-launch'], [bare(entry.id)])
    else if (t.gio) await this.runner.run(t.gio, ['launch', entry.path])
    else throw new ControlError('unavailable', this.unsupported.open_app ?? 'Opening apps needs gtk-launch or gio.')
  }

  async notify(title: string, text: string): Promise<void> {
    const send = this.session.tools['notify-send']
    if (send) await this.runner.run(send, ['--app-name=LocalRouter', '--', title, text]).catch(() => undefined)
  }

  async run(control: ComputerControl): Promise<{ done: string }> {
    // A player's own volume needs playerctl, not a system mixer.
    const refused = control.kind === 'volume' && control.target !== 'system' ? this.unsupported.media : this.unsupported[control.kind]
    if (refused) throw new ControlError('unavailable', refused)
    switch (control.kind) {
      case 'open_app': {
        const entry = await this.resolve(control.app)
        await this.launch(entry)
        return { done: `Opened ${entry.name}.` }
      }
      case 'quit_app':
        return this.quit(await this.resolve(control.app))
      case 'volume':
        return control.target === 'system' ? this.systemVolume(control) : this.playerVolume(control, control.target)
      case 'media':
        return this.media(control)
      case 'dark_mode':
        return this.darkMode(control.action)
      case 'lock_screen':
        return this.lock()
    }
  }

  /** Asks each of the app's windows to close, as its close button would; never a signal to the process. */
  private async quit(entry: DesktopEntry): Promise<{ done: string }> {
    const t = this.session.tools
    const windows = (await clientWindows(this.session, this.runner)) ?? []
    const mine = windows.filter((w) => ownsWindow(entry, w.app))
    if (!mine.length) return { done: `${entry.name} wasn't open.` }
    for (const win of mine) {
      if (this.session.server === 'x11' && t.wmctrl) await this.runner.run(t.wmctrl, ['-i', '-c', win.id])
      else if (this.session.compositor === 'hyprland' && t.hyprctl) await this.runner.run(t.hyprctl, ['dispatch', 'closewindow', `address:${win.id}`])
      else if (this.session.compositor === 'sway' && t.swaymsg) await this.runner.run(t.swaymsg, [`[con_id=${win.id}]`, 'kill'])
    }
    return { done: `Quit ${entry.name}.` }
  }

  private async systemVolume(control: Extract<ComputerControl, { kind: 'volume' }>): Promise<{ done: string }> {
    const t = this.session.tools
    const mute = control.action === 'mute' || control.action === 'unmute' ? control.action === 'mute' : null
    if (t.wpctl) {
      const sink = '@DEFAULT_AUDIO_SINK@'
      if (mute !== null) {
        await this.runner.run(t.wpctl, ['set-mute', sink, mute ? '1' : '0'])
        return { done: mute ? 'Muted.' : 'Unmuted.' }
      }
      const wpctl = t.wpctl
      const level = await this.nextLevel(control, async () => {
        const m = /Volume:\s*([\d.]+)/.exec(await this.runner.run(wpctl, ['get-volume', sink]))
        return m ? Number(m[1]) * 100 : NaN
      })
      await this.runner.run(wpctl, ['set-volume', '-l', '1.0', sink, (level / 100).toFixed(2)])
      return { done: `Volume is at ${level}%.` }
    }
    if (t.pactl) {
      const sink = '@DEFAULT_SINK@'
      if (mute !== null) {
        await this.runner.run(t.pactl, ['set-sink-mute', sink, mute ? '1' : '0'])
        return { done: mute ? 'Muted.' : 'Unmuted.' }
      }
      const pactl = t.pactl
      const level = await this.nextLevel(control, async () => Number(/(\d+)%/.exec(await this.runner.run(pactl, ['get-sink-volume', sink]))?.[1] ?? NaN))
      await this.runner.run(pactl, ['set-sink-volume', sink, `${level}%`])
      return { done: `Volume is at ${level}%.` }
    }
    if (t.amixer) {
      if (mute !== null) {
        await this.runner.run(t.amixer, ['-q', 'set', 'Master', mute ? 'mute' : 'unmute'])
        return { done: mute ? 'Muted.' : 'Unmuted.' }
      }
      const amixer = t.amixer
      const level = await this.nextLevel(control, async () => Number(/\[(\d+)%\]/.exec(await this.runner.run(amixer, ['-M', 'get', 'Master']))?.[1] ?? NaN))
      await this.runner.run(amixer, ['-M', '-q', 'set', 'Master', `${level}%`])
      return { done: `Volume is at ${level}%.` }
    }
    throw new ControlError('unavailable', this.unsupported.volume ?? 'No volume control is installed.')
  }

  /** A player's own volume over MPRIS (0.0–1.0). */
  private async playerVolume(control: Extract<ComputerControl, { kind: 'volume' }>, player: Player): Promise<{ done: string }> {
    const playerctl = this.playerctl()
    const args = playerArgs(player)
    let current: string
    try {
      current = await this.runner.run(playerctl, [...args, 'volume'])
    } catch (error) {
      if (error instanceof Error && NO_PLAYER.test(error.message)) return { done: `${capital(playerName(player))} isn't playing.` }
      throw error
    }
    const level = control.action === 'mute' ? 0 : control.action === 'unmute' ? 50 : await this.nextLevel(control, async () => Number(current.trim()) * 100)
    await this.runner.run(playerctl, [...args, 'volume', (level / 100).toFixed(2)])
    return { done: `${capital(playerName(player))} volume is at ${level}%.` }
  }

  private async media(control: Extract<ComputerControl, { kind: 'media' }>): Promise<{ done: string }> {
    const playerctl = this.playerctl()
    try {
      await this.runner.run(playerctl, [...playerArgs(control.player), control.action])
    } catch (error) {
      if (!(error instanceof Error && NO_PLAYER.test(error.message))) throw error
      if (control.action !== 'play') return { done: `${capital(playerName(control.player))} isn't playing.` }
      if (control.player !== 'Spotify') throw new ControlError('unavailable', 'No music player is open.')
      // Spotify answers MPRIS only once it is running: open it, as asking a Mac app to play would.
      const entry = await this.resolve('Spotify')
      await this.launch(entry)
      return { done: `Opened ${entry.name}; press play once it has loaded.` }
    }
    return { done: `${MEDIA_DONE[control.action]} ${playerName(control.player)}.` }
  }

  private playerctl(): string {
    const playerctl = this.session.tools.playerctl
    if (!playerctl) throw new ControlError('unavailable', this.unsupported.media ?? 'Media controls need playerctl.')
    return playerctl
  }

  private async nextLevel(control: Extract<ComputerControl, { kind: 'volume' }>, read: () => Promise<number>): Promise<number> {
    if (control.action === 'set') {
      if (control.level === undefined || !Number.isFinite(control.level)) throw new ControlError('invalid', 'A volume level from 0 to 100 is required')
      return clamp(control.level)
    }
    const current = await read()
    if (!Number.isFinite(current)) throw new ControlError('unavailable', 'This output device has no adjustable volume')
    return clamp(current + (control.action === 'up' ? VOLUME_STEP : -VOLUME_STEP))
  }

  /**
   * KDE: its colour scheme. Elsewhere GNOME's color-scheme key, which libadwaita and GTK apps and the settings portal
   * follow (also on Hyprland and sway when gsettings is installed); the classic Adwaita theme is switched with it.
   */
  private async darkMode(action: 'on' | 'off' | 'toggle'): Promise<{ done: string }> {
    const t = this.session.tools
    if (this.session.compositor === 'kde' && t['plasma-apply-colorscheme']) {
      const reader = t.kreadconfig6 ?? t.kreadconfig5
      const dark = action === 'toggle' ? !(reader && /dark/i.test(await this.runner.run(reader, ['--file', 'kdeglobals', '--group', 'General', '--key', 'ColorScheme']).catch(() => ''))) : action === 'on'
      await this.runner.run(t['plasma-apply-colorscheme'], [dark ? 'BreezeDark' : 'BreezeLight'])
      return { done: `Dark mode is ${dark ? 'on' : 'off'}.` }
    }
    const gsettings = t.gsettings
    if (!gsettings) throw new ControlError('unavailable', this.unsupported.dark_mode ?? 'Dark mode needs gsettings.')
    const schema = 'org.gnome.desktop.interface'
    const dark = action === 'toggle' ? !(await this.runner.run(gsettings, ['get', schema, 'color-scheme'])).includes('prefer-dark') : action === 'on'
    await this.runner.run(gsettings, ['set', schema, 'color-scheme', dark ? 'prefer-dark' : 'default'])
    const theme = (await this.runner.run(gsettings, ['get', schema, 'gtk-theme']).catch(() => '')).trim().replace(/^'|'$/g, '')
    if (theme === 'Adwaita' || theme === 'Adwaita-dark') await this.runner.run(gsettings, ['set', schema, 'gtk-theme', dark ? 'Adwaita-dark' : 'Adwaita'])
    return { done: `Dark mode is ${dark ? 'on' : 'off'}.` }
  }

  /** logind's lock (the desktop's locker answers it), else the XDG screensaver. */
  private async lock(): Promise<{ done: string }> {
    const t = this.session.tools
    const session = this.session.env.XDG_SESSION_ID
    if (t.loginctl) {
      try {
        await this.runner.run(t.loginctl, ['lock-session', ...(session ? [session] : [])])
        return { done: 'Locked the screen.' }
      } catch (error) {
        if (!t['xdg-screensaver']) throw error
      }
    }
    if (t['xdg-screensaver']) {
      await this.runner.run(t['xdg-screensaver'], ['lock'])
      return { done: 'Locked the screen.' }
    }
    throw new ControlError('unavailable', this.unsupported.lock_screen ?? 'Locking needs loginctl or xdg-screensaver.')
  }
}

const capital = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)

/** The controls this session cannot run, with why. */
export function unsupportedControls(session: Session): ControlSupport {
  const t = session.tools
  const out: ControlSupport = {}
  if (!t['gtk-launch'] && !t.gio) out.open_app = 'Opening apps needs gtk-launch (GTK) or gio (GLib) on this computer.'
  const canClose = session.server === 'x11' ? Boolean(t.wmctrl) : (session.compositor === 'hyprland' && Boolean(t.hyprctl)) || (session.compositor === 'sway' && Boolean(t.swaymsg))
  if (!canClose) {
    out.quit_app = session.server === 'x11'
      ? 'Quitting apps needs wmctrl (it asks the windows to close).'
      : `${session.compositor === 'gnome' ? 'GNOME' : session.compositor === 'kde' ? 'KDE Plasma' : 'This desktop'} on Wayland does not let other apps close windows; quit it from the app itself.`
  }
  if (!t.wpctl && !t.pactl && !t.amixer) out.volume = 'Volume needs wpctl (PipeWire), pactl (PulseAudio) or amixer (ALSA).'
  if (!t.playerctl) out.media = 'Media controls need playerctl.'
  if (!(session.compositor === 'kde' && t['plasma-apply-colorscheme']) && !t.gsettings) out.dark_mode = 'Dark mode works on GNOME and GTK desktops (gsettings) and KDE Plasma; this desktop has neither.'
  if (!t.loginctl && !t['xdg-screensaver']) out.lock_screen = 'Locking needs loginctl (systemd-logind) or xdg-screensaver.'
  return out
}
