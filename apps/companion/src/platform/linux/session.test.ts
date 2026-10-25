import { constants } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { compositor, detectSession, displayServer, distro } from './session.ts'
import { fakeRunner } from './testing.ts'

describe('session detection', () => {
  it('tells X11 from Wayland, and keeps XWayland (DISPLAY set too) a Wayland session', () => {
    expect(displayServer({ XDG_SESSION_TYPE: 'x11', DISPLAY: ':0' })).toBe('x11')
    expect(displayServer({ XDG_SESSION_TYPE: 'wayland', DISPLAY: ':0', WAYLAND_DISPLAY: 'wayland-1' })).toBe('wayland')
    expect(displayServer({ WAYLAND_DISPLAY: 'wayland-0', DISPLAY: ':1' })).toBe('wayland')
    expect(displayServer({ DISPLAY: ':0' })).toBe('x11')
  })

  it('names the compositor from its own socket variables before XDG_CURRENT_DESKTOP', () => {
    expect(compositor({ HYPRLAND_INSTANCE_SIGNATURE: 'abc', XDG_CURRENT_DESKTOP: 'GNOME' })).toBe('hyprland')
    expect(compositor({ SWAYSOCK: '/run/user/1000/sway-ipc.sock' })).toBe('sway')
    expect(compositor({ XDG_CURRENT_DESKTOP: 'ubuntu:GNOME' })).toBe('gnome')
    expect(compositor({ XDG_CURRENT_DESKTOP: 'KDE' })).toBe('kde')
    expect(compositor({ XDG_CURRENT_DESKTOP: 'Hyprland' })).toBe('hyprland')
    expect(compositor({ XDG_CURRENT_DESKTOP: 'labwc:wlroots' })).toBe('sway')
    expect(compositor({ XDG_CURRENT_DESKTOP: 'XFCE' })).toBe('other')
  })

  it('reads the distro family from os-release ID and ID_LIKE', () => {
    expect(distro('NAME="Ubuntu"\nID=ubuntu\nID_LIKE=debian\n')).toBe('debian')
    expect(distro('ID="rocky"\nID_LIKE="rhel centos fedora"\n')).toBe('fedora')
    expect(distro('ID=arch\n')).toBe('arch')
    expect(distro('ID=omarchy\nID_LIKE=arch\n')).toBe('arch')
    expect(distro('ID=nixos\n')).toBe('other')
  })

  it('finds tools on PATH, a writable ydotoold socket and which gi namespaces load, with one python3 run', async () => {
    const executables = new Set(['/usr/bin/python3', '/usr/local/bin/ydotool', '/usr/bin/hyprctl', '/usr/bin/wtype'])
    const sockets = new Set(['/run/user/1000/.ydotool_socket'])
    const { runner, calls } = fakeRunner(() => JSON.stringify({ Atspi: true, Gio: true, GdkPixbuf: false }))
    const session = await detectSession({
      env: { PATH: '/usr/local/bin:/usr/bin', XDG_SESSION_TYPE: 'wayland', HYPRLAND_INSTANCE_SIGNATURE: 'x', XDG_RUNTIME_DIR: '/run/user/1000' },
      runner,
      uid: 1000,
      can: async (path, mode) => (mode === constants.X_OK ? executables.has(path) : sockets.has(path)),
      readText: async () => 'ID=arch\n',
    })
    expect(session).toMatchObject({ server: 'wayland', compositor: 'hyprland', distro: 'arch', ydotoold: true, atspi: true, gio: true, pixbuf: false })
    expect(session.tools).toEqual({ python3: '/usr/bin/python3', ydotool: '/usr/local/bin/ydotool', hyprctl: '/usr/bin/hyprctl', wtype: '/usr/bin/wtype' })
    expect(calls).toHaveLength(1)
    expect(calls[0]?.args.slice(0, 2)).toEqual(['-I', '-c'])
  })

  it('reports no AT-SPI when python3 is missing or its probe fails, without throwing', async () => {
    const failing = fakeRunner(() => new Error('ModuleNotFoundError'))
    const session = await detectSession({ env: { PATH: '/usr/bin' }, runner: failing.runner, can: async (path) => path === '/usr/bin/python3', readText: async () => '' })
    expect(session).toMatchObject({ atspi: false, gio: false, ydotoold: false, server: 'x11' })
    const none = fakeRunner()
    expect((await detectSession({ env: { PATH: '/usr/bin' }, runner: none.runner, can: async () => false, readText: async () => '' })).atspi).toBe(false)
    expect(none.calls).toHaveLength(0)
  })
})
