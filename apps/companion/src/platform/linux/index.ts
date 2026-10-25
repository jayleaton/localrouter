import { join } from 'node:path'
import { execRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import type { CommandRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import type { BrowserTab, BrowserTabs } from '../../../../desktop-helper/src/computer/types.ts'
import { array, boolean, integerIn, json, object, string } from '../../../../desktop-helper/src/validate.ts'
import type { AccessReport, Platform, Requirement } from '../types.ts'
import { A11Y_FIX, ATSPI_PACKAGES, LinuxScreen, actShape, atspiCall, explainAtspi, windowOffset } from './atspi.ts'
import { captureName, screenCapture } from './capture.ts'
import { activeWindow, clientWindows, displays, pointerReader } from './compositor.ts'
import { LinuxControls } from './controls.ts'
import type { LinuxControlsOptions } from './controls.ts'
import { chooseInjector, fileTokenStore } from './input.ts'
import type { Injector } from './input.ts'
import { detectSession, installLine, sessionName } from './session.ts'
import type { DetectOptions, Session } from './session.ts'

/**
 * The Linux platform: AT-SPI for the tree and semantic actions on every session, the session's injector (input.ts)
 * for keys and the pointer fallback, the session's capture (capture.ts), and .desktop entries for the fixed controls.
 * See docs/dev/COMPUTER.md › Linux for what each compositor allows.
 */
export type LinuxPlatformOptions = {
  runner?: CommandRunner
  env?: NodeJS.ProcessEnv
  /** The companion's data directory: the RemoteDesktop portal's restore token is kept here. */
  dataDir?: string
  /** An already detected session (tests; a caller that detected it for other reasons). */
  session?: Session
  /** File checks for detection (tests). */
  detect?: Omit<DetectOptions, 'runner' | 'env'>
  /** Where the .desktop entries are read from (tests). */
  controls?: Omit<LinuxControlsOptions, 'session' | 'runner'>
}

const fold = (s: string) => s.toLowerCase().trim()

export async function createLinuxPlatform(options: LinuxPlatformOptions = {}): Promise<Platform> {
  const runner = options.runner ?? execRunner({ timeoutMs: 15_000, slow: 'A desktop tool' })
  const env = options.env ?? process.env
  const session = options.session ?? (await detectSession({ env, runner, ...options.detect }))
  const t = session.tools
  const tokens = options.dataDir ? fileTokenStore(join(options.dataDir, 'linux-remote-desktop-token')) : undefined
  const injector = chooseInjector(session, runner, { tokens })
  const controls = new LinuxControls({ ...options.controls, session, runner })
  const atspi = async (op: string, payload: object = {}): Promise<unknown> => {
    try {
      return json((await runner.run(...atspiCall(session, op), { input: JSON.stringify(payload), timeoutMs: 12_000 })).trim(), op)
    } catch (error) {
      throw explainAtspi(session, error instanceof Error ? error.message : String(error)) ?? error
    }
  }

  const canDisplays = (session.server === 'x11' && t.xrandr) || (session.compositor === 'hyprland' && t.hyprctl) || (session.compositor === 'sway' && t.swaymsg) || t.python3
  const displaysOf = canDisplays ? () => displays(session, runner) : null
  const screen = session.atspi ? new LinuxScreen({ session, runner, injector }) : null
  const capture = screenCapture({ session, runner, displays: displaysOf ?? (async () => []) })
  if (screen && capture) screen.screenshot = capture

  /** Running apps with a window: AT-SPI's applications, else the compositor's window list. */
  const running = async (): Promise<{ name: string; pid: number }[]> => {
    if (session.atspi) {
      return array(await atspi('apps'), 'apps', (raw, what) => {
        const app = object(raw, what)
        return { name: string(app.name, `${what} name`), pid: integerIn(app.pid, 0, 2 ** 31, `${what} pid`) }
      })
    }
    return ((await clientWindows(session, runner)) ?? []).map((w) => ({ name: w.app, pid: w.pid ?? 0 }))
  }

  /**
   * Brings an app's window forward by its process: X11 activates it, Hyprland and sway focus it over their IPC.
   * GNOME and KDE on Wayland let no other client raise a window: AT-SPI's grab_focus is tried, then the app is launched
   * again, which single-instance apps (most GTK and KDE apps, browsers) answer by raising their window.
   */
  const activate = async (name: string): Promise<void> => {
    const found = (await running()).find((app) => fold(app.name) === fold(name))
    if (!found) {
      await controls.launch(await controls.resolve(name))
      return
    }
    if (session.server === 'x11' && t.xdotool && found.pid) {
      const ids = (await runner.run(t.xdotool, ['search', '--onlyvisible', '--pid', String(found.pid)]).catch(() => '')).split('\n').map((s) => s.trim()).filter((s) => /^\d+$/.test(s))
      if (ids.length) {
        await runner.run(t.xdotool, ['windowactivate', '--sync', ids[ids.length - 1]])
        return
      }
    }
    if (session.server === 'wayland' && session.compositor === 'hyprland' && t.hyprctl && found.pid) {
      await runner.run(t.hyprctl, ['dispatch', 'focuswindow', `pid:${found.pid}`])
      return
    }
    if (session.server === 'wayland' && session.compositor === 'sway' && t.swaymsg && found.pid) {
      await runner.run(t.swaymsg, [`[pid=${found.pid}]`, 'focus'])
      return
    }
    if (session.atspi) {
      const focus = object(await atspi('focus', { name: found.name }), 'focus')
      if (boolean(focus.focused, 'focused')) return
    }
    await controls.launch(await controls.resolve(name))
  }

  /** GNOME's dash favourites, as names; other desktops keep their docks in their own formats. */
  const gsettings = t.gsettings
  const launcher = session.compositor === 'gnome' && gsettings
    ? {
        list: async () => {
          const ids = [...(await runner.run(gsettings, ['get', 'org.gnome.shell', 'favorite-apps'])).matchAll(/'([^']+)'/g)].map((m) => m[1])
          const entries = await controls.entries()
          return ids.flatMap((id) => entries.filter((entry) => entry.id === id).map((entry) => entry.name))
        },
        activate,
      }
    : null

  const tabs: BrowserTabs | null = session.atspi ? linuxTabs(session, runner, injector, atspi, activate) : null

  return {
    os: 'linux',
    backend: `Linux ${sessionName(session)} (AT-SPI, ${injector.name})`,
    screen,
    tabs,
    launcher,
    apps: { running: async () => [...new Set((await running()).map((app) => app.name).filter(Boolean))], activate },
    controls,
    pointer: pointerReader(session, runner),
    displays: displaysOf,
    presence: null,
    access: () => linuxAccess(session, injector, Boolean(capture), session.atspi ? () => atspi('probe') : null),
  }
}

/**
 * A browser's tabs from its tab strip (AT-SPI page tabs), Chromium and Firefox alike. The tab strip has no URLs, so
 * `url` is empty. Switching selects the tab through AT-SPI (no pointer), then brings the browser forward.
 */
function linuxTabs(session: Session, runner: CommandRunner, injector: Injector, atspi: (op: string, payload?: object) => Promise<unknown>, activate: (name: string) => Promise<void>): BrowserTabs {
  return {
    browser: 'browser',
    async list(): Promise<BrowserTab[]> {
      return array(await atspi('tabs'), 'tabs', (raw, what) => {
        const tab = object(raw, what)
        return { window: integerIn(tab.window, 0, 10_000, `${what} window`), index: integerIn(tab.index, 0, 10_000, `${what} index`), title: string(tab.title, `${what} title`), url: '', active: boolean(tab.active, `${what} active`) }
      })
    },
    async activate(tab: BrowserTab): Promise<void> {
      const raw = await atspi('tab', { window: tab.window, index: tab.index, title: tab.title })
      const app = string(object(raw, 'tab').app ?? '', 'tab app')
      const result = actShape(raw)
      if (app) await activate(app)
      if (result.done) return
      // No semantic switch: click the tab, which needs the window's global position (see atspi.ts).
      const offset = windowOffset(session, result.frame, result.pid, await activeWindow(session, runner))
      if (!result.point || !offset || injector.noPointer) throw new Error(injector.noPointer ?? 'Could not switch to that tab here: the browser did not offer a way without the pointer, and this desktop does not say where the tab is.')
      await injector.click({ x: result.point.x + offset.dx, y: result.point.y + offset.dy })
    },
  }
}

const INPUT_PACKAGES = {
  x11: { debian: 'xdotool wmctrl', fedora: 'xdotool wmctrl', arch: 'xdotool wmctrl', other: 'xdotool and wmctrl' },
  wlroots: { debian: 'wtype ydotool', fedora: 'wtype ydotool', arch: 'wtype ydotool', other: 'wtype and ydotool' },
  ydotool: { debian: 'ydotool', fedora: 'ydotool', arch: 'ydotool', other: 'ydotool' },
}
const CAPTURE_PACKAGES = {
  x11: { debian: 'imagemagick xdotool', fedora: 'ImageMagick xdotool', arch: 'imagemagick xdotool', other: 'ImageMagick and xdotool' },
  wlroots: { debian: 'grim', fedora: 'grim', arch: 'grim', other: 'grim' },
  portal: { debian: 'python3-gi gir1.2-gdkpixbuf-2.0 xdg-desktop-portal', fedora: 'python3-gobject gdk-pixbuf2 xdg-desktop-portal', arch: 'python-gobject gdk-pixbuf2 xdg-desktop-portal', other: 'python3-gi, GdkPixbuf and xdg-desktop-portal' },
}
const YDOTOOLD = 'ydotool needs its daemon: run ydotoold with access to /dev/uinput (Arch: `systemctl --user enable --now ydotool`; elsewhere add yourself to the input group with a udev rule for uinput, or run ydotoold as a service).'

/** What this computer can do and what each missing piece needs, as Settings shows it. */
export async function linuxAccess(session: Session, injector: Injector, captures: boolean, probe: (() => Promise<unknown>) | null): Promise<AccessReport> {
  const requirements: Requirement[] = []
  requirements.push({ id: 'atspi', label: 'AT-SPI reader (python3-gi, Atspi 2.0)', granted: session.atspi, for: 'Reading the window in front: its controls, text and address bar, and acting on them without the pointer', ...(session.atspi ? {} : { fix: installLine(session, ATSPI_PACKAGES) }) })
  let reachable = false
  if (probe) {
    try {
      reachable = integerIn(object(await probe(), 'probe').apps, 0, 100_000, 'apps') > 0
    } catch {
      reachable = false
    }
    requirements.push({ id: 'a11y', label: 'Apps on the accessibility bus', granted: reachable, for: 'Apps show their windows to AT-SPI only with accessibility on', ...(reachable ? {} : { fix: A11Y_FIX }) })
  }
  const wl = session.server === 'wayland'
  const wlroots = wl && (session.compositor === 'hyprland' || session.compositor === 'sway' || session.compositor === 'other')
  const portal = injector.name.includes('portal')
  const keysFix = session.server === 'x11' ? installLine(session, INPUT_PACKAGES.x11) : wlroots ? `${installLine(session, INPUT_PACKAGES.wlroots)} (wtype for keys)` : `${installLine(session, INPUT_PACKAGES.ydotool)}. ${YDOTOOLD} Without it, the RemoteDesktop portal is used and asks once.`
  requirements.push({
    id: 'keys',
    label: `Keys (${injector.noKeys ? 'none' : portal ? 'RemoteDesktop portal' : injector.name})`,
    // The portal cannot be checked without asking the person.
    granted: injector.noKeys ? false : portal ? null : true,
    for: 'Pressing Enter, Tab, Escape and Space, and typing where a text box cannot be set directly',
    ...(injector.noKeys ? { fix: keysFix } : portal ? { fix: 'The portal asks once whether LocalRouter may control the keyboard; the answer is kept.' } : {}),
  })
  const pointerFix = session.server === 'x11'
    ? installLine(session, INPUT_PACKAGES.x11)
    : session.compositor === 'hyprland'
      ? `${installLine(session, INPUT_PACKAGES.ydotool)}. ${YDOTOOLD}`
      : session.compositor === 'sway'
        ? 'Install sway\'s swaymsg.'
        : 'Not possible here: GNOME and KDE on Wayland do not let other apps place the pointer or say where a window is. Controls with an accessibility action (most buttons, links, tabs and text boxes) still work.'
  requirements.push({ id: 'pointer', label: 'Pointer fallback', granted: !injector.noPointer, for: 'Clicking or scrolling a control that has no accessibility action', ...(injector.noPointer ? { fix: pointerFix } : {}) })
  const capture = captureName(session)
  const captureFix = session.server === 'x11' ? installLine(session, CAPTURE_PACKAGES.x11) : wlroots && session.compositor !== 'other' ? installLine(session, CAPTURE_PACKAGES.wlroots) : installLine(session, CAPTURE_PACKAGES.portal)
  requirements.push({ id: 'capture', label: `Screen capture (${capture ?? 'none'})`, granted: capture === 'Screenshot portal' ? null : captures, for: 'Screenshots of the window in front', ...(captures ? (capture === 'Screenshot portal' ? { fix: 'The portal may ask once whether LocalRouter may take screenshots.' } : {}) : { fix: captureFix }) })
  return {
    accessibility: session.atspi && reachable,
    screen_recording: captures,
    input: !injector.noKeys || !injector.noPointer,
    backend: `Linux ${sessionName(session)} (AT-SPI, ${injector.name})`,
    requirements,
  }
}

