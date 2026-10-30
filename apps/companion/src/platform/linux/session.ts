import { constants } from 'node:fs'
import { access, readFile } from 'node:fs/promises'
import { delimiter, join } from 'node:path/posix'
import type { CommandRunner } from '../../../../desktop-helper/src/computer/runner.ts'

/**
 * The Linux session the companion runs in: X11 or Wayland, which compositor, and which of the tools the executor
 * drives are installed. Everything else in platform/linux/ chooses its path from this, so it is detected once and
 * passed around rather than re-read from the environment in each module.
 */
export type DisplayServer = 'x11' | 'wayland'
export type Compositor = 'gnome' | 'kde' | 'hyprland' | 'sway' | 'other'
/** The package manager family, for the install line a missing piece names. */
export type Distro = 'debian' | 'fedora' | 'arch' | 'other'

/** The executables looked up on PATH. */
export const TOOLS = [
  'xdotool', 'wmctrl', 'xrandr', 'import', 'xwd', 'convert', 'magick',
  'ydotool', 'wtype', 'grim', 'hyprctl', 'swaymsg',
  'playerctl', 'wpctl', 'pactl', 'amixer',
  'gsettings', 'plasma-apply-colorscheme', 'kreadconfig6', 'kreadconfig5',
  'loginctl', 'xdg-screensaver', 'notify-send', 'gtk-launch', 'gio', 'python3',
] as const
export type Tool = (typeof TOOLS)[number]

export type Session = {
  server: DisplayServer
  compositor: Compositor
  distro: Distro
  /** Each tool's absolute path, absent when it is not installed. */
  tools: Partial<Record<Tool, string>>
  /** ydotoold's socket exists and is writable by this user (ydotool does nothing without it). */
  ydotoold: boolean
  /** python3 can import gi with Atspi 2.0 (the tree reader). */
  atspi: boolean
  /** python3 gi has Gio (the portals) and GdkPixbuf (cropping and scaling a portal screenshot). */
  gio: boolean
  pixbuf: boolean
  /** XDG_CURRENT_DESKTOP as set, for messages. */
  desktop: string
  env: NodeJS.ProcessEnv
}

export type DetectOptions = {
  env?: NodeJS.ProcessEnv
  runner: CommandRunner
  /** Whether `path` exists with `mode` (fs.constants.X_OK, W_OK); fs.access by default. */
  can?: (path: string, mode: number) => Promise<boolean>
  readText?: (path: string) => Promise<string>
  uid?: number
}

const defaultCan = (path: string, mode: number) => access(path, mode).then(() => true, () => false)

/** Wayland or X11 from the session variables; a Wayland session with only DISPLAY set (XWayland) is still Wayland. */
export function displayServer(env: NodeJS.ProcessEnv): DisplayServer {
  const type = (env.XDG_SESSION_TYPE ?? '').toLowerCase()
  if (type === 'wayland' || type === 'x11') return type
  return env.WAYLAND_DISPLAY ? 'wayland' : 'x11'
}

/** The compositor (or desktop) from its own variables first, which are exact, then XDG_CURRENT_DESKTOP. */
export function compositor(env: NodeJS.ProcessEnv): Compositor {
  if (env.HYPRLAND_INSTANCE_SIGNATURE) return 'hyprland'
  if (env.SWAYSOCK) return 'sway'
  const desktop = `${env.XDG_CURRENT_DESKTOP ?? ''}:${env.XDG_SESSION_DESKTOP ?? ''}:${env.DESKTOP_SESSION ?? ''}`.toLowerCase()
  if (/hyprland/.test(desktop)) return 'hyprland'
  if (/sway|wlroots|river|labwc|wayfire/.test(desktop)) return 'sway'
  if (/kde|plasma/.test(desktop)) return 'kde'
  if (/gnome|unity|ubuntu|pop|budgie/.test(desktop)) return 'gnome'
  return 'other'
}

/** The distro family from os-release's ID and ID_LIKE. */
export function distro(osRelease: string): Distro {
  const ids = [...osRelease.matchAll(/^(?:ID|ID_LIKE)=["']?([^"'\n]*)/gm)].flatMap((m) => m[1].toLowerCase().split(/\s+/))
  if (ids.some((id) => id === 'debian' || id === 'ubuntu')) return 'debian'
  if (ids.some((id) => id === 'fedora' || id === 'rhel' || id === 'centos')) return 'fedora'
  if (ids.some((id) => id === 'arch' || id === 'archlinux')) return 'arch'
  return 'other'
}

/** Where ydotoold listens: YDOTOOL_SOCKET, else the user's runtime directory, else /tmp (its older default). */
export function ydotoolSockets(env: NodeJS.ProcessEnv, uid: number): string[] {
  if (env.YDOTOOL_SOCKET) return [env.YDOTOOL_SOCKET]
  return [join(env.XDG_RUNTIME_DIR || `/run/user/${uid}`, '.ydotool_socket'), '/tmp/.ydotool_socket']
}

/** One python3 run answers which gi namespaces load, so detection costs one process, not three. */
export const GI_PROBE = String.raw`
import json
out = {}
try:
    import gi
except Exception:
    print(json.dumps(out))
    raise SystemExit(0)
for name, version in (('Atspi', '2.0'), ('Gio', '2.0'), ('GdkPixbuf', '2.0')):
    try:
        gi.require_version(name, version)
        __import__('gi.repository.' + name)
        out[name] = True
    except Exception:
        out[name] = False
print(json.dumps(out))
`

/**
 * Detects the session from the environment, PATH and one python3 probe. It never talks to the compositor or the
 * accessibility bus: that is the backends' job, and detection must be cheap and side-effect free.
 */
export async function detectSession(options: DetectOptions): Promise<Session> {
  const env = options.env ?? process.env
  const can = options.can ?? defaultCan
  const readText = options.readText ?? ((path: string) => readFile(path, 'utf8'))
  const dirs = (env.PATH ?? '').split(delimiter).filter(Boolean)
  const tools: Partial<Record<Tool, string>> = {}
  await Promise.all(TOOLS.map(async (tool) => {
    for (const dir of dirs) {
      const path = join(dir, tool)
      if (await can(path, constants.X_OK)) {
        tools[tool] = path
        return
      }
    }
  }))
  const uid = options.uid ?? process.getuid?.() ?? 1000
  let ydotoold = false
  if (tools.ydotool) for (const socket of ydotoolSockets(env, uid)) if (await can(socket, constants.W_OK)) ydotoold = true
  let gi: Record<string, unknown> = {}
  if (tools.python3) {
    try {
      const parsed: unknown = JSON.parse(await options.runner.run(tools.python3, ['-I', '-c', GI_PROBE], { timeoutMs: 10_000 }))
      if (parsed && typeof parsed === 'object') gi = parsed as Record<string, unknown>
    } catch {
      gi = {}
    }
  }
  const release = await readText('/etc/os-release').catch(() => '')
  return {
    server: displayServer(env),
    compositor: compositor(env),
    distro: distro(release),
    tools,
    ydotoold,
    atspi: gi.Atspi === true,
    gio: gi.Gio === true,
    pixbuf: gi.GdkPixbuf === true,
    desktop: env.XDG_CURRENT_DESKTOP ?? '',
    env,
  }
}

/** wlroots-style compositors that speak the virtual-keyboard protocol wtype uses (GNOME and KDE do not). */
export const wlroots = (session: Session) => session.server === 'wayland' && (session.compositor === 'hyprland' || session.compositor === 'sway' || session.compositor === 'other')

/** Words for the session, for backend names and messages: "X11", "Wayland/Hyprland", "Wayland/GNOME". */
export function sessionName(session: Session): string {
  if (session.server === 'x11') return 'X11'
  const names: Record<Compositor, string> = { gnome: 'GNOME', kde: 'KDE Plasma', hyprland: 'Hyprland', sway: 'sway', other: session.desktop || 'wlroots' }
  return `Wayland/${names[session.compositor]}`
}

/** The command that installs `packages` (per family names) on this distro. */
export function installLine(session: Session, packages: Record<Distro, string>): string {
  const commands: Record<Distro, string> = { debian: 'sudo apt install', fedora: 'sudo dnf install', arch: 'sudo pacman -S', other: 'Install' }
  return `${commands[session.distro]} ${packages[session.distro]}`
}
