import { describe, expect, it } from 'vitest'
import { createLinuxPlatform } from './index.ts'
import type { Session } from './session.ts'
import { atspiOp, fakeRunner, fakeSession, tool } from './testing.ts'
import type { Call } from './testing.ts'

/** .desktop entries the platform sees: Files and Firefox, nothing read from this machine. */
const APPS = {
  home: '/home/jj',
  list: async (path: string) => (path === '/usr/share/applications' ? [{ name: 'org.gnome.Nautilus.desktop', dir: false }, { name: 'firefox.desktop', dir: false }] : []),
  readText: async (path: string) => `[Desktop Entry]\nType=Application\nName=${path.includes('Nautilus') ? 'Files' : 'Firefox'}\n`,
}

/** A platform over a fake runner: `atspi` answers each script op, `other` every other tool. */
async function platform(session: Session, atspi: (op: string, payload: Record<string, unknown>) => unknown = () => ({}), other: (call: Call) => string | Error = () => '') {
  const fake = fakeRunner((call) => {
    const op = atspiOp(call)
    return op ? JSON.stringify(atspi(op, JSON.parse(call.input ?? '{}'))) : other(call)
  })
  return { ...fake, platform: await createLinuxPlatform({ session: { ...session, env: { XDG_DATA_DIRS: '/usr/share' } }, runner: fake.runner, controls: APPS }) }
}

const X11 = fakeSession({ server: 'x11', tools: ['xdotool', 'xrandr', 'import', 'wmctrl', 'python3', 'gtk-launch', 'wpctl', 'playerctl', 'gsettings', 'loginctl'] })
const GNOME = fakeSession({ server: 'wayland', compositor: 'gnome', tools: ['python3', 'gtk-launch', 'gsettings', 'loginctl'] })
const HYPRLAND = fakeSession({ server: 'wayland', compositor: 'hyprland', ydotoold: true, tools: ['hyprctl', 'ydotool', 'wtype', 'grim', 'python3', 'gtk-launch'] })

describe('createLinuxPlatform', () => {
  it('assembles X11: AT-SPI with xdotool, a capture, the pointer and displays, and no OS presence check', async () => {
    const { platform: x11 } = await platform(X11)
    expect(x11).toMatchObject({ os: 'linux', backend: 'Linux X11 (AT-SPI, xdotool)', presence: null, launcher: null })
    expect(x11.screen?.screenshot).toBeTypeOf('function')
    expect(x11.pointer).toBeTypeOf('function')
    expect(x11.displays).toBeTypeOf('function')
  })

  it('on GNOME Wayland has no pointer for the stop, and lists the dash favourites by name', async () => {
    const { platform: gnome } = await platform(GNOME, () => ({}), () => "['firefox.desktop', 'org.gnome.Nautilus.desktop', 'gone.desktop']\n")
    expect(gnome.pointer).toBeNull()
    expect(await gnome.launcher?.list()).toEqual(['Firefox', 'Files'])
  })

  it('has no screen without the AT-SPI reader, but still its controls', async () => {
    const { platform: bare } = await platform({ ...GNOME, atspi: false })
    expect(bare.screen).toBeNull()
    expect(bare.tabs).toBeNull()
    expect(bare.controls.unsupported?.quit_app).toMatch(/GNOME on Wayland/)
  })
})

describe('bringing an app forward', () => {
  const RUNNING = [{ name: 'Firefox', pid: 50 }, { name: 'gedit', pid: 60 }]

  it('focuses by process over Hyprland’s IPC', async () => {
    const { platform: hypr, lines } = await platform(HYPRLAND, (op) => (op === 'apps' ? RUNNING : {}))
    expect(await hypr.apps?.running()).toEqual(['Firefox', 'gedit'])
    await hypr.apps?.activate('firefox')
    expect(lines().slice(-1)).toEqual(['hyprctl dispatch focuswindow pid:50'])
  })

  it('activates the process’s visible X11 window', async () => {
    const { platform: x11, lines } = await platform(X11, (op) => (op === 'apps' ? RUNNING : {}), (call) => (call.args[0] === 'search' ? '1234\n5678\n' : ''))
    await x11.apps?.activate('gedit')
    expect(lines().slice(-2)).toEqual(['xdotool search --onlyvisible --pid 60', 'xdotool windowactivate --sync 5678'])
  })

  it('on GNOME Wayland tries AT-SPI focus, then launches the app again (single-instance apps raise their window)', async () => {
    const { platform: gnome, lines } = await platform(GNOME, (op) => (op === 'apps' ? RUNNING : op === 'focus' ? { pid: 50, focused: false } : {}))
    await gnome.apps?.activate('Firefox')
    expect(lines()).toEqual(['atspi apps', 'atspi focus', 'gtk-launch firefox'])
  })

  it('opens an app that is not running', async () => {
    const { platform: hypr, lines } = await platform(HYPRLAND, (op) => (op === 'apps' ? [] : {}))
    await hypr.apps?.activate('Files')
    expect(lines()).toEqual(['atspi apps', 'gtk-launch org.gnome.Nautilus'])
  })
})

describe('browser tabs', () => {
  it('lists tab-strip tabs and switches through AT-SPI, then brings the browser forward', async () => {
    const tabs = [{ window: 0, index: 0, title: 'Inbox', active: true, app: 'Firefox' }, { window: 0, index: 1, title: 'News', active: false, app: 'Firefox' }]
    const { platform: hypr, lines } = await platform(HYPRLAND, (op) => (op === 'tabs' ? tabs : op === 'tab' ? { done: true, app: 'Firefox' } : op === 'apps' ? [{ name: 'Firefox', pid: 50 }] : {}))
    const listed = await hypr.tabs!.list()
    expect(listed[1]).toEqual({ window: 0, index: 1, title: 'News', url: '', active: false })
    await hypr.tabs!.activate(listed[1]!)
    expect(lines()).toEqual(['atspi tabs', 'atspi tab', 'atspi apps', 'hyprctl dispatch focuswindow pid:50'])
  })
})

describe('access report', () => {
  it('on a complete X11 desktop grants everything', async () => {
    const { platform: x11 } = await platform(X11, (op) => (op === 'probe' ? { apps: 4 } : {}))
    const access = await x11.access()
    expect(access).toMatchObject({ accessibility: true, screen_recording: true, input: true, backend: 'Linux X11 (AT-SPI, xdotool)' })
    expect(access.requirements.map((r) => [r.id, r.granted])).toEqual([['atspi', true], ['a11y', true], ['keys', true], ['pointer', true], ['capture', true]])
    expect(access.requirements.every((r) => r.fix === undefined)).toBe(true)
  })

  it('names each missing piece with the distro’s install line, and accessibility turned off', async () => {
    const debian = await platform({ ...fakeSession({ server: 'x11', distro: 'debian', atspi: false, tools: [] }) })
    const report = await debian.platform.access()
    expect(report).toMatchObject({ accessibility: false, screen_recording: false, input: false })
    const fix = Object.fromEntries(report.requirements.map((r) => [r.id, r.fix]))
    expect(fix.atspi).toBe('sudo apt install python3-gi gir1.2-atspi-2.0 at-spi2-core')
    expect(fix.keys).toBe('sudo apt install xdotool wmctrl')
    expect(fix.capture).toBe('sudo apt install imagemagick xdotool')
    const off = await platform(fakeSession({ server: 'x11', distro: 'arch', tools: ['python3', 'xdotool'] }), (op) => (op === 'probe' ? { apps: 0 } : {}))
    const a11y = (await off.platform.access()).requirements.find((r) => r.id === 'a11y')
    expect(a11y).toMatchObject({ granted: false })
    expect(a11y?.fix).toMatch(/toolkit-accessibility true.*QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1.*--force-renderer-accessibility/)
  })

  it('on GNOME Wayland says the pointer is impossible, and that the portals ask (unknown until asked)', async () => {
    const { platform: gnome } = await platform(GNOME, (op) => (op === 'probe' ? { apps: 2 } : {}))
    const byId = Object.fromEntries((await gnome.access()).requirements.map((r) => [r.id, r]))
    expect(byId.pointer).toMatchObject({ granted: false })
    expect(byId.pointer?.fix).toMatch(/^Not possible here: GNOME and KDE on Wayland/)
    expect(byId.capture).toMatchObject({ label: 'Screen capture (Screenshot portal)', granted: null })
    // No token store was given to this platform, so no portal for keys: ydotool is the fix.
    expect(byId.keys?.fix).toMatch(/sudo apt install ydotool\. ydotool needs its daemon/)
  })

  it('reports portal keys as needing the person’s answer when a data directory keeps the token', async () => {
    const fake = fakeRunner()
    const gnome = await createLinuxPlatform({ session: GNOME, runner: fake.runner, dataDir: '/nonexistent/localrouter', controls: APPS })
    expect(gnome.backend).toBe('Linux Wayland/GNOME (AT-SPI, RemoteDesktop portal)')
    const keys = (await gnome.access()).requirements.find((r) => r.id === 'keys')
    expect(keys).toMatchObject({ label: 'Keys (RemoteDesktop portal)', granted: null })
    expect(fake.calls.filter((call) => tool(call) !== 'python3')).toHaveLength(0)
  })
})
