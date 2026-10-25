import { describe, expect, it } from 'vitest'
import { activeWindow, displays, hyprCursor, pointerReader, swayWindow, union, xrandrMonitors } from './compositor.ts'
import { fakeRunner, fakeSession, tool } from './testing.ts'

const HYPR_WINDOW = { address: '0x55d0c3a0', at: [1930, 52], size: [1260, 800], pid: 4242, class: 'org.gnome.TextEditor', title: 'notes.txt' }

/** A sway tree: a focused tiled window inside an output and workspace, with a 2 px border and a title bar. */
const SWAY_TREE = {
  type: 'root', id: 1, nodes: [{
    type: 'output', id: 3, rect: { x: 0, y: 0, width: 2560, height: 1440 }, nodes: [{
      type: 'workspace', id: 4, nodes: [
        { type: 'con', id: 7, pid: 900, app_id: 'foot', name: 'shell', focused: false, rect: { x: 0, y: 0, width: 1280, height: 1440 }, window_rect: { x: 2, y: 2, width: 1276, height: 1436 } },
        { type: 'con', id: 8, pid: 901, app_id: 'firefox', name: 'Mozilla Firefox', focused: true, rect: { x: 1280, y: 0, width: 1280, height: 1440 }, window_rect: { x: 2, y: 24, width: 1276, height: 1414 } },
      ],
    }],
  }],
}

describe('the focused window from the compositor', () => {
  it('reads Hyprland’s activewindow (global, logical position and size)', async () => {
    const { runner, lines } = fakeRunner(() => JSON.stringify(HYPR_WINDOW))
    const win = await activeWindow(fakeSession({ server: 'wayland', compositor: 'hyprland', tools: ['hyprctl'] }), runner)
    expect(win).toEqual({ x: 1930, y: 52, w: 1260, h: 800, id: '0x55d0c3a0', pid: 4242, app: 'org.gnome.TextEditor', title: 'notes.txt' })
    expect(lines()).toEqual(['hyprctl activewindow -j'])
  })

  it('adds sway’s window_rect to its container rect, so borders and the title bar are not counted as the window', () => {
    expect(swayWindow(SWAY_TREE)).toEqual({ x: 1282, y: 24, w: 1276, h: 1414, id: '8', pid: 901, app: 'firefox', title: 'Mozilla Firefox' })
  })

  it('reads the X11 active window through xdotool', async () => {
    const answers: Record<string, string> = { getactivewindow: '73400323\n', getwindowgeometry: 'WINDOW=73400323\nX=40\nY=60\nWIDTH=800\nHEIGHT=600\nSCREEN=0\n', getwindowname: 'Inbox\n', getwindowpid: '321\n', getwindowclassname: 'Thunderbird\n' }
    const { runner } = fakeRunner((call) => answers[call.args[0] ?? ''] ?? '')
    expect(await activeWindow(fakeSession({ server: 'x11', tools: ['xdotool'] }), runner)).toEqual({ x: 40, y: 60, w: 800, h: 600, id: '73400323', title: 'Inbox', pid: 321, app: 'Thunderbird' })
  })

  it('has no answer on GNOME or KDE Wayland, which do not tell other clients where windows are', async () => {
    const { runner, calls } = fakeRunner()
    expect(await activeWindow(fakeSession({ server: 'wayland', compositor: 'gnome', tools: ['hyprctl', 'swaymsg', 'xdotool'] }), runner)).toBeNull()
    expect(calls).toHaveLength(0)
  })
})

describe('displays', () => {
  it('parses xrandr’s active monitors, including negative offsets', () => {
    const text = 'Monitors: 2\n 0: +*eDP-1 1920/344x1080/194+0+0  eDP-1\n 1: +HDMI-1 2560/597x1440/336-2560+0  HDMI-1\n'
    expect(xrandrMonitors(text)).toEqual([{ x: 0, y: 0, w: 1920, h: 1080 }, { x: -2560, y: 0, w: 2560, h: 1440 }])
  })

  it('gives Hyprland monitors their logical size: pixels over scale, turned for a quarter transform', async () => {
    const monitors = [
      { name: 'eDP-1', x: 0, y: 0, width: 2880, height: 1800, scale: 2, transform: 0 },
      { name: 'DP-1', x: 1440, y: 0, width: 2560, height: 1440, scale: 1.25, transform: 1 },
    ]
    const { runner } = fakeRunner(() => JSON.stringify(monitors))
    expect(await displays(fakeSession({ server: 'wayland', compositor: 'hyprland', tools: ['hyprctl'] }), runner)).toEqual([{ x: 0, y: 0, w: 1440, h: 900 }, { x: 1440, y: 0, w: 1152, h: 2048 }])
  })

  it('takes sway’s output rects as they are and skips inactive outputs', async () => {
    const outputs = [{ name: 'eDP-1', active: true, rect: { x: 0, y: 0, width: 1280, height: 800 } }, { name: 'HDMI-A-1', active: false, rect: { x: 0, y: 0, width: 0, height: 0 } }]
    const { runner } = fakeRunner(() => JSON.stringify(outputs))
    expect(await displays(fakeSession({ server: 'wayland', compositor: 'sway', tools: ['swaymsg'] }), runner)).toEqual([{ x: 0, y: 0, w: 1280, h: 800 }])
  })

  it('asks Gdk through python3 on GNOME and KDE', async () => {
    const { runner, calls } = fakeRunner(() => '[{"x":0,"y":0,"w":1920,"h":1080}]')
    expect(await displays(fakeSession({ server: 'wayland', compositor: 'gnome', tools: ['python3'] }), runner)).toEqual([{ x: 0, y: 0, w: 1920, h: 1080 }])
    expect(tool(calls[0]!)).toBe('python3')
  })

  it('bounds every display in one rectangle', () => {
    expect(union([{ x: -2560, y: 0, w: 2560, h: 1440 }, { x: 0, y: 200, w: 1920, h: 1080 }])).toEqual({ x: -2560, y: 0, w: 4480, h: 1440 })
    expect(union([])).toBeNull()
  })
})

describe('the pointer for the local stop', () => {
  it('reads it on X11 and Hyprland, and is null on GNOME/KDE Wayland and sway', async () => {
    const x11 = fakeRunner(() => 'X=812\nY=44\nSCREEN=0\nWINDOW=1\n')
    expect(await pointerReader(fakeSession({ server: 'x11', tools: ['xdotool'] }), x11.runner)?.()).toEqual({ x: 812, y: 44 })
    const hypr = fakeRunner(() => '{"x": 1500, "y": 20}')
    expect(await pointerReader(fakeSession({ server: 'wayland', compositor: 'hyprland', tools: ['hyprctl'] }), hypr.runner)?.()).toEqual({ x: 1500, y: 20 })
    expect(hyprCursor('12, 34')).toEqual({ x: 12, y: 34 })
    expect(pointerReader(fakeSession({ server: 'wayland', compositor: 'gnome', tools: ['xdotool', 'hyprctl'] }), x11.runner)).toBeNull()
    expect(pointerReader(fakeSession({ server: 'wayland', compositor: 'sway', tools: ['swaymsg'] }), x11.runner)).toBeNull()
  })
})
