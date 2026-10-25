import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { CommandRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import { SCREENSHOT_MAX_EDGE, fitScale } from '../../../../desktop-helper/src/computer/types.ts'
import type { Rect, Screenshot, ScreenshotOptions } from '../../../../desktop-helper/src/computer/types.ts'
import { integerIn, json, object } from '../../../../desktop-helper/src/validate.ts'
import { activeWindow, union } from './compositor.ts'
import type { Session } from './session.ts'

/**
 * Screenshots of the front window. The capture is written to a private temporary directory and read back, because
 * the runner only carries text; the directory is removed whatever happens.
 *
 *   X11              ImageMagick `import -window <id>` (else `xwd` then `convert`), scaled in the same command.
 *   Hyprland, sway   `grim -g "x,y wxh" -s <scale>` of the window's global, logical rectangle.
 *   GNOME, KDE       org.freedesktop.portal.Screenshot (interactive false), scaled with GdkPixbuf. These compositors
 *                    do not tell other apps where a window is, so the image is the whole desktop, not the window.
 */

/** A PNG's pixel size from its IHDR chunk; throws when the bytes are not a PNG. */
export function pngSize(png: Uint8Array): { width: number; height: number } {
  const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
  if (png.length < 24 || signature.some((byte, i) => png[i] !== byte)) throw new Error('The screenshot is not a PNG image.')
  const view = new DataView(png.buffer, png.byteOffset, png.byteLength)
  return { width: view.getUint32(16), height: view.getUint32(20) }
}

/** The image size and `-s`/`-resize` scale for a `w`×`h` region at most `maxEdge` on its longer side. */
export function scaled(region: Rect, maxEdge: number): { scale: number; width: number; height: number } {
  const scale = fitScale(region.w, region.h, maxEdge)
  return { scale, width: Math.max(1, Math.round(region.w * scale)), height: Math.max(1, Math.round(region.h * scale)) }
}

/**
 * Takes a screenshot through the Screenshot portal, deletes the file the portal saved (GNOME writes it into the
 * person's Pictures folder), scales it to at most `max_edge` with GdkPixbuf and writes a PNG to `out`.
 * stdin: {"out": path, "max_edge": n}; stdout: {"width", "height", "source_width", "source_height"}.
 */
export const PORTAL_SCREENSHOT = String.raw`
import json, os, sys
try:
    import gi
    gi.require_version('Gio', '2.0')
    gi.require_version('GdkPixbuf', '2.0')
    from gi.repository import Gio, GLib, GdkPixbuf
except Exception:
    sys.stderr.write('LOCALROUTER_NO_GIO')
    raise SystemExit(3)

req = json.loads(sys.stdin.read() or '{}')
bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
sender = bus.get_unique_name()[1:].replace('.', '_')
token = 'localrouter_shot'
path = '/org/freedesktop/portal/desktop/request/%s/%s' % (sender, token)
loop = GLib.MainLoop()
got = {}
def on_response(_c, _s, _o, _i, _n, params):
    code, results = params.unpack()
    got['code'] = code
    got['results'] = results
    loop.quit()
def on_timeout():
    got['code'] = 9
    loop.quit()
    return False
bus.signal_subscribe('org.freedesktop.portal.Desktop', 'org.freedesktop.portal.Request', 'Response', path, None, Gio.DBusSignalFlags.NONE, on_response)
GLib.timeout_add_seconds(50, on_timeout)
try:
    bus.call_sync('org.freedesktop.portal.Desktop', '/org/freedesktop/portal/desktop', 'org.freedesktop.portal.Screenshot', 'Screenshot',
                  GLib.Variant('(sa{sv})', ('', {'handle_token': GLib.Variant('s', token), 'interactive': GLib.Variant('b', False)})),
                  None, Gio.DBusCallFlags.NONE, -1, None)
except GLib.Error as error:
    sys.stderr.write('LOCALROUTER_NO_PORTAL:' + str(error.message))
    raise SystemExit(4)
loop.run()
if got.get('code') != 0:
    sys.stderr.write('LOCALROUTER_PORTAL_DENIED:%s' % got.get('code'))
    raise SystemExit(5)
source = Gio.File.new_for_uri(got['results']['uri']).get_path()
try:
    image = GdkPixbuf.Pixbuf.new_from_file(source)
finally:
    os.remove(source)
w, h = image.get_width(), image.get_height()
scale = min(1.0, float(req['max_edge']) / max(w, h))
out_w, out_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
if (out_w, out_h) != (w, h):
    image = image.scale_simple(out_w, out_h, GdkPixbuf.InterpType.BILINEAR)
image.savev(req['out'], 'png', [], [])
print(json.dumps({'width': out_w, 'height': out_h, 'source_width': w, 'source_height': h}))
`

export type CaptureOptions = {
  session: Session
  runner: CommandRunner
  /** Every display's bounds (the portal's image covers all of them). */
  displays: () => Promise<Rect[]>
  /** A new private directory; mkdtemp under the system temp directory by default. */
  makeDir?: () => Promise<string>
  read?: (path: string) => Promise<Uint8Array>
  remove?: (path: string) => Promise<void>
}

/** The front-window screenshot for this session, or null when it has no way to capture. */
export function screenCapture(options: CaptureOptions): ((o?: ScreenshotOptions) => Promise<Screenshot>) | null {
  const { session, runner } = options
  const t = session.tools
  const makeDir = options.makeDir ?? (() => mkdtemp(join(tmpdir(), 'localrouter-shot-')))
  const read = options.read ?? ((path: string) => readFile(path))
  const remove = options.remove ?? ((path: string) => rm(path, { recursive: true, force: true }))
  const resizer = t.magick ?? t.convert

  /** Runs `capture(dir)` for the file it writes, reads it and always removes the directory. */
  const take = async (capture: (dir: string) => Promise<string>): Promise<Uint8Array> => {
    const dir = await makeDir()
    try {
      return await read(await capture(dir))
    } finally {
      await remove(dir)
    }
  }
  const front = async () => {
    const win = await activeWindow(session, runner)
    if (!win || win.w <= 0 || win.h <= 0) throw new Error('No window is in front to capture.')
    return win
  }
  const shot = (png: Uint8Array, region: Rect, app: string, title: string): Screenshot => {
    const size = pngSize(png)
    return { png, width: size.width, height: size.height, region, scale: size.width / region.w, app, title }
  }

  if (session.server === 'x11' && t.xdotool && (t.import || (t.xwd && resizer))) {
    const { import: importer, xwd } = t
    return async (o = {}) => {
      const win = await front()
      const size = scaled(win, o.maxEdge ?? SCREENSHOT_MAX_EDGE)
      const resize = size.scale < 1 ? ['-resize', `${size.width}x${size.height}!`] : []
      const png = await take(async (dir) => {
        const out = join(dir, 'window.png')
        if (importer) await runner.run(importer, ['-window', win.id, ...resize, `png:${out}`], { signal: o.signal })
        else if (xwd && resizer) {
          const raw = join(dir, 'window.xwd')
          await runner.run(xwd, ['-id', win.id, '-silent', '-out', raw], { signal: o.signal })
          await runner.run(resizer, [`xwd:${raw}`, ...resize, `png:${out}`], { signal: o.signal })
        }
        return out
      })
      return shot(png, { x: win.x, y: win.y, w: win.w, h: win.h }, win.app, win.title)
    }
  }
  if (session.server === 'wayland' && (session.compositor === 'hyprland' || session.compositor === 'sway') && t.grim) {
    const grim = t.grim
    return async (o = {}) => {
      const win = await front()
      const size = scaled(win, o.maxEdge ?? SCREENSHOT_MAX_EDGE)
      const png = await take(async (dir) => {
        const out = join(dir, 'window.png')
        await runner.run(grim, ['-g', `${Math.round(win.x)},${Math.round(win.y)} ${Math.round(win.w)}x${Math.round(win.h)}`, '-s', String(size.scale), '-t', 'png', out], { signal: o.signal })
        return out
      })
      return shot(png, { x: win.x, y: win.y, w: win.w, h: win.h }, win.app, win.title)
    }
  }
  if (session.server === 'wayland' && t.python3 && session.gio && session.pixbuf) {
    const python3 = t.python3
    return async (o = {}) => {
      const region = union(await options.displays())
      if (!region) throw new Error('The displays of this session cannot be read, so a screenshot cannot be placed on them.')
      const maxEdge = o.maxEdge ?? SCREENSHOT_MAX_EDGE
      const png = await take(async (dir) => {
        const out = join(dir, 'screen.png')
        const answer = object(json((await runner.run(python3, ['-I', '-c', PORTAL_SCREENSHOT], { input: JSON.stringify({ out, max_edge: maxEdge }), signal: o.signal, timeoutMs: 60_000 })).trim(), 'screenshot'), 'screenshot')
        integerIn(answer.width, 1, 100_000, 'screenshot width')
        return out
      })
      return shot(png, region, '', '')
    }
  }
  return null
}

/** What a session's capture path is, for the access report: the tool, and what to install when it has none. */
export function captureName(session: Session): string | null {
  const t = session.tools
  if (session.server === 'x11') return t.xdotool && (t.import || (t.xwd && (t.magick || t.convert))) ? (t.import ? 'ImageMagick import' : 'xwd') : null
  if ((session.compositor === 'hyprland' || session.compositor === 'sway') && t.grim) return 'grim'
  return t.python3 && session.gio && session.pixbuf ? 'Screenshot portal' : null
}
