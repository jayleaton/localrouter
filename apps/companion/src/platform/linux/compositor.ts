import type { CommandRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import type { PointerReading, Rect } from '../../../../desktop-helper/src/computer/types.ts'
import { array, isObject, json, number, object, optional, string } from '../../../../desktop-helper/src/validate.ts'
import type { Fields } from '../../../../desktop-helper/src/validate.ts'
import type { Session } from './session.ts'

/**
 * What the compositor (or the X server) says about windows, outputs and the pointer. AT-SPI knows a window's
 * contents but, on Wayland, not where the window is: a Wayland client is never told its global position, so its
 * accessible extents are relative to its own surface. Hyprland and sway answer over their IPC; GNOME and KDE have no
 * such interface for other clients, and there these functions return null.
 */

/** The focused window in global coordinates. `id`: the X window id, the Hyprland address or the sway con_id. */
export type ActiveWindow = Rect & { id: string; pid?: number; title: string; app: string }

/** A top-level window as the compositor lists it (for closing an app's windows). */
export type ClientWindow = { id: string; pid?: number; app: string; title: string }

const QUICK = { timeoutMs: 5_000 }

/** `xdotool ... --shell` output (`X=10\nY=20\n…`) as numbers. */
export function shellVars(text: string): Record<string, number> {
  const out: Record<string, number> = {}
  for (const line of text.split('\n')) {
    const m = /^([A-Z]+)=(-?\d+)$/.exec(line.trim())
    if (m) out[m[1]] = Number(m[2])
  }
  return out
}

/** Hyprland's `activewindow -j`: `at` and `size` are the window's global, logical position and size. */
export function hyprWindow(value: unknown): ActiveWindow | null {
  if (!isObject(value) || !Array.isArray(value.at)) return null
  const at = array(value.at, 'window at', number)
  const size = array(value.size, 'window size', number)
  if (at.length !== 2 || size.length !== 2) return null
  const pid = typeof value.pid === 'number' && value.pid > 0 ? value.pid : undefined
  return { x: at[0], y: at[1], w: size[0], h: size[1], id: string(value.address, 'window address'), title: optional(value.title, 'window title', string) ?? '', app: optional(value.class, 'window class', string) ?? '', ...(pid ? { pid } : {}) }
}

/** Every window in a sway tree (floating ones too), depth first. */
function swayWindows(node: Fields, out: Fields[] = []): Fields[] {
  if ((node.type === 'con' || node.type === 'floating_con') && (node.pid !== undefined || node.app_id !== undefined || isObject(node.window_properties))) out.push(node)
  for (const key of ['nodes', 'floating_nodes']) {
    const kids = node[key]
    if (Array.isArray(kids)) for (const kid of kids) if (isObject(kid)) swayWindows(kid, out)
  }
  return out
}

const swayApp = (node: Fields) => (typeof node.app_id === 'string' && node.app_id ? node.app_id : isObject(node.window_properties) && typeof node.window_properties.class === 'string' ? node.window_properties.class : '')

/**
 * The focused window of `swaymsg -t get_tree`. `rect` is the container with its borders and title bar;
 * `window_rect` is the client's area inside it, so the client's global origin is their sum.
 */
export function swayWindow(tree: unknown): ActiveWindow | null {
  const node = swayWindows(object(tree, 'sway tree')).find((n) => n.focused === true)
  if (!node) return null
  const rect = object(node.rect, 'sway rect')
  const inner = isObject(node.window_rect) ? node.window_rect : { x: 0, y: 0, width: rect.width, height: rect.height }
  const pid = typeof node.pid === 'number' && node.pid > 0 ? node.pid : undefined
  return {
    x: number(rect.x, 'rect x') + number(inner.x, 'window_rect x'),
    y: number(rect.y, 'rect y') + number(inner.y, 'window_rect y'),
    w: number(inner.width, 'window_rect width'),
    h: number(inner.height, 'window_rect height'),
    id: String(number(node.id, 'sway id')),
    title: typeof node.name === 'string' ? node.name : '',
    app: swayApp(node),
    ...(pid ? { pid } : {}),
  }
}

/** The focused window, or null where the compositor does not say (GNOME and KDE on Wayland) or nothing is focused. */
export async function activeWindow(session: Session, runner: CommandRunner): Promise<ActiveWindow | null> {
  const t = session.tools
  try {
    if (session.server === 'x11' && t.xdotool) {
      const id = (await runner.run(t.xdotool, ['getactivewindow'], QUICK)).trim()
      if (!/^\d+$/.test(id)) return null
      const geometry = shellVars(await runner.run(t.xdotool, ['getwindowgeometry', '--shell', id], QUICK))
      const title = (await runner.run(t.xdotool, ['getwindowname', id], QUICK).catch(() => '')).trim()
      const pid = Number((await runner.run(t.xdotool, ['getwindowpid', id], QUICK).catch(() => '')).trim())
      // getwindowclassname is in xdotool 3.2021 and later; older ones leave the app unnamed.
      const app = (await runner.run(t.xdotool, ['getwindowclassname', id], QUICK).catch(() => '')).trim()
      if (![geometry.X, geometry.Y, geometry.WIDTH, geometry.HEIGHT].every(Number.isFinite)) return null
      return { x: geometry.X, y: geometry.Y, w: geometry.WIDTH, h: geometry.HEIGHT, id, title, app, ...(pid > 0 ? { pid } : {}) }
    }
    if (session.server === 'wayland' && session.compositor === 'hyprland' && t.hyprctl) return hyprWindow(json(await runner.run(t.hyprctl, ['activewindow', '-j'], QUICK), 'active window'))
    if (session.server === 'wayland' && session.compositor === 'sway' && t.swaymsg) return swayWindow(json(await runner.run(t.swaymsg, ['-t', 'get_tree'], QUICK), 'sway tree'))
  } catch {
    return null
  }
  return null
}

/** Top-level windows from the compositor, for closing an app's windows; null where it cannot list them. */
export async function clientWindows(session: Session, runner: CommandRunner): Promise<ClientWindow[] | null> {
  const t = session.tools
  if (session.server === 'x11' && t.wmctrl) {
    // wmctrl -lpx: id, desktop, pid, WM_CLASS (instance.Class), host, title.
    return (await runner.run(t.wmctrl, ['-lpx'], QUICK)).split('\n').flatMap((line) => {
      const m = /^(0x[0-9a-f]+)\s+-?\d+\s+(\d+)\s+(\S+)\s+\S+\s?(.*)$/i.exec(line.trim())
      if (!m) return []
      const pid = Number(m[2])
      return [{ id: m[1], app: m[3].split('.').pop() ?? m[3], title: m[4], ...(pid > 0 ? { pid } : {}) }]
    })
  }
  if (session.server === 'wayland' && session.compositor === 'hyprland' && t.hyprctl) {
    return array(json(await runner.run(t.hyprctl, ['clients', '-j'], QUICK), 'clients'), 'clients', (raw, what) => {
      const c = object(raw, what)
      const pid = typeof c.pid === 'number' && c.pid > 0 ? c.pid : undefined
      return { id: string(c.address, `${what} address`), app: optional(c.class, `${what} class`, string) ?? '', title: optional(c.title, `${what} title`, string) ?? '', ...(pid ? { pid } : {}) }
    })
  }
  if (session.server === 'wayland' && session.compositor === 'sway' && t.swaymsg) {
    return swayWindows(object(json(await runner.run(t.swaymsg, ['-t', 'get_tree'], QUICK), 'sway tree'), 'sway tree')).map((node) => {
      const pid = typeof node.pid === 'number' && node.pid > 0 ? node.pid : undefined
      return { id: String(number(node.id, 'sway id')), app: swayApp(node), title: typeof node.name === 'string' ? node.name : '', ...(pid ? { pid } : {}) }
    })
  }
  return null
}

/** `xrandr --listactivemonitors`: ` 0: +*eDP-1 1920/344x1080/194+0+0  eDP-1`. */
export function xrandrMonitors(text: string): Rect[] {
  return [...text.matchAll(/(\d+)\/\d+x(\d+)\/\d+([+-]\d+)([+-]\d+)/g)].map((m) => ({ x: Number(m[3]), y: Number(m[4]), w: Number(m[1]), h: Number(m[2]) }))
}

/**
 * Hyprland's `monitors -j`: `width`/`height` are the mode's pixels, `x`/`y` the logical origin; the logical size is
 * the pixels over `scale`, swapped when `transform` turns the output a quarter (1, 3, 5, 7).
 */
export function hyprMonitors(value: unknown): Rect[] {
  return array(value, 'monitors', (raw, what) => {
    const m = object(raw, what)
    const scale = typeof m.scale === 'number' && m.scale > 0 ? m.scale : 1
    const turned = typeof m.transform === 'number' && m.transform % 2 === 1
    const w = number(m.width, `${what} width`) / scale
    const h = number(m.height, `${what} height`) / scale
    return { x: number(m.x, `${what} x`), y: number(m.y, `${what} y`), w: Math.round(turned ? h : w), h: Math.round(turned ? w : h) }
  })
}

/** sway's `get_outputs`: `rect` is already logical and global; inactive outputs are skipped. */
export function swayOutputs(value: unknown): Rect[] {
  return array(value, 'outputs', (raw, what) => object(raw, what)).filter((o) => o.active !== false && isObject(o.rect)).map((o) => {
    const r = object(o.rect, 'output rect')
    return { x: number(r.x, 'rect x'), y: number(r.y, 'rect y'), w: number(r.width, 'rect width'), h: number(r.height, 'rect height') }
  })
}

/** Gdk's monitors (logical, global), for GNOME and KDE where the compositor has no IPC for other clients. */
export const GDK_MONITORS = String.raw`
import json, gi
gi.require_version('Gdk', '3.0')
from gi.repository import Gdk
Gdk.init([])
display = Gdk.Display.get_default()
out = []
if display is not None:
    for i in range(display.get_n_monitors()):
        g = display.get_monitor(i).get_geometry()
        out.append({'x': g.x, 'y': g.y, 'w': g.width, 'h': g.height})
print(json.dumps(out))
`

const rectShape = (raw: unknown, what: string): Rect => {
  const r = object(raw, what)
  return { x: number(r.x, `${what} x`), y: number(r.y, `${what} y`), w: number(r.w, `${what} w`), h: number(r.h, `${what} h`) }
}

/** Every display's bounds in the global space, from whatever this session can ask. */
export async function displays(session: Session, runner: CommandRunner): Promise<Rect[]> {
  const t = session.tools
  if (session.server === 'x11' && t.xrandr) return xrandrMonitors(await runner.run(t.xrandr, ['--listactivemonitors'], QUICK))
  if (session.compositor === 'hyprland' && t.hyprctl) return hyprMonitors(json(await runner.run(t.hyprctl, ['monitors', '-j'], QUICK), 'monitors'))
  if (session.compositor === 'sway' && t.swaymsg) return swayOutputs(json(await runner.run(t.swaymsg, ['-t', 'get_outputs'], QUICK), 'outputs'))
  if (t.python3) return array(json(await runner.run(t.python3, ['-I', '-c', GDK_MONITORS], QUICK), 'monitors'), 'monitors', rectShape)
  throw new Error('The displays of this session cannot be read (no xrandr, compositor IPC or python3).')
}

/** The smallest rectangle holding every display. */
export function union(rects: Rect[]): Rect | null {
  if (!rects.length) return null
  const x = Math.min(...rects.map((r) => r.x))
  const y = Math.min(...rects.map((r) => r.y))
  return { x, y, w: Math.max(...rects.map((r) => r.x + r.w)) - x, h: Math.max(...rects.map((r) => r.y + r.h)) - y }
}

/** Hyprland's `cursorpos -j` ({"x":1,"y":2}), or its plain "1, 2". */
export function hyprCursor(text: string): PointerReading {
  const trimmed = text.trim()
  if (trimmed.startsWith('{')) {
    const p = object(json(trimmed, 'cursor'), 'cursor')
    return { x: number(p.x, 'cursor x'), y: number(p.y, 'cursor y') }
  }
  const m = /^(-?[\d.]+),\s*(-?[\d.]+)$/.exec(trimmed)
  if (!m) throw new Error('Hyprland did not say where the pointer is.')
  return { x: Number(m[1]), y: Number(m[2]) }
}

/**
 * The pointer reader for the local stop, or null where no client may read the pointer: GNOME and KDE on Wayland (the
 * stop there is the keyboard shortcut and the tray), and sway, whose IPC does not report the cursor.
 */
export function pointerReader(session: Session, runner: CommandRunner): (() => Promise<PointerReading>) | null {
  const t = session.tools
  if (session.server === 'x11' && t.xdotool) {
    const xdotool = t.xdotool
    return async () => {
      const v = shellVars(await runner.run(xdotool, ['getmouselocation', '--shell'], QUICK))
      if (!Number.isFinite(v.X) || !Number.isFinite(v.Y)) throw new Error('xdotool did not say where the pointer is.')
      return { x: v.X, y: v.Y }
    }
  }
  if (session.server === 'wayland' && session.compositor === 'hyprland' && t.hyprctl) {
    const hyprctl = t.hyprctl
    return async () => hyprCursor(await runner.run(hyprctl, ['cursorpos', '-j'], QUICK))
  }
  return null
}
