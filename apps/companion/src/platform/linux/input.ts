import { chmod, mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path/posix'
import type { CommandRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import type { ActOptions, Point } from '../../../../desktop-helper/src/computer/types.ts'
import { json, object, optional, string } from '../../../../desktop-helper/src/validate.ts'
import { wlroots } from './session.ts'
import type { Session } from './session.ts'

/**
 * Synthetic input on Linux, chosen by session. AT-SPI actions come first (atspi.ts); this is the fallback for a
 * control with no action, and the only way to press keys on Wayland. What each session allows:
 *
 *   X11          xdotool (XTest): exact pointer, keys, Unicode text.
 *   Hyprland     `hyprctl dispatch movecursor` places the pointer exactly; ydotool clicks and scrolls there; wtype types.
 *   sway         `swaymsg seat - cursor set|press|release` for the pointer (exact, no setup); wtype types.
 *   GNOME, KDE   no pointer: no client may place it, and these compositors do not say where a window is, so a
 *                position could not be trusted anyway (ydotool's "absolute" move is a relative move from the corner,
 *                bent by pointer acceleration). Keys: ydotool (uinput, needs ydotoold), else the RemoteDesktop portal
 *                (asks the person once; a restore token keeps the grant).
 */

/** The keys an action may press; `select_all` is Ctrl+A before replacing a field's text through the keyboard. */
export type Key = 'enter' | 'escape' | 'tab' | 'space' | 'select_all'
export type ScrollDirection = 'up' | 'down' | 'top' | 'bottom'

export interface Injector {
  /** "xdotool", "hyprctl + ydotool", "wtype", "RemoteDesktop portal", for the backend's name and access report. */
  readonly name: string
  /** Why the pointer cannot be placed, when it cannot (the refusal a pointer fallback gives). */
  readonly noPointer?: string
  /** Why keys cannot be sent, when they cannot. */
  readonly noKeys?: string
  click(point: Point, options?: ActOptions): Promise<void>
  scroll(point: Point, direction: ScrollDirection, options?: ActOptions): Promise<void>
  key(key: Key, options?: ActOptions): Promise<void>
  /** Types plain text (no line breaks), in chunks, stopping between chunks when `signal` aborts. */
  type(text: string, options?: ActOptions): Promise<void>
}

/** At most this many characters a call: a long text stops soon after a stop, and no argv grows large. */
export const CHUNK = 200
const XDOTOOL_DELAY_MS = '12'

/** Wheel notches for a scroll: a few for up/down, many for top/bottom. */
export const notches = (direction: ScrollDirection) => (direction === 'top' || direction === 'bottom' ? 25 : 5)
const upward = (direction: ScrollDirection) => direction === 'up' || direction === 'top'

/** Text in chunks of at most CHUNK characters, never splitting a surrogate pair. */
export function chunks(text: string): string[] {
  const chars = [...text]
  const out: string[] = []
  for (let i = 0; i < chars.length; i += CHUNK) out.push(chars.slice(i, i + CHUNK).join(''))
  return out
}

export const MULTILINE = 'Multi-line text needs a text box that accepts it directly; typing it as keys would press Enter at each line break.'

/** Throws for text the keyboard path must not type: a line break would be Enter (sending a message), not a new line. */
export function plainText(text: string): string {
  if (/[\r\n]/.test(text)) throw new Error(MULTILINE)
  return text
}

const aborted = (signal?: AbortSignal) => {
  if (signal?.aborted) throw new Error(typeof signal.reason === 'string' && signal.reason ? signal.reason : 'It was stopped.')
}

const at = (p: Point) => [String(Math.round(p.x)), String(Math.round(p.y))]

type PointerDriver = { name: string; click(p: Point, o?: ActOptions): Promise<void>; scroll(p: Point, d: ScrollDirection, o?: ActOptions): Promise<void> }
type KeyDriver = { name: string; key(k: Key, o?: ActOptions): Promise<void>; chunk(text: string, o?: ActOptions): Promise<void> }

const XDOTOOL_KEYS: Record<Key, string> = { enter: 'Return', escape: 'Escape', tab: 'Tab', space: 'space', select_all: 'ctrl+a' }

function xdotoolPointer(runner: CommandRunner, xdotool: string): PointerDriver {
  return {
    name: 'xdotool',
    async click(p, o) {
      await runner.run(xdotool, ['mousemove', '--sync', ...at(p), 'click', '1'], { signal: o?.signal })
    },
    async scroll(p, d, o) {
      await runner.run(xdotool, ['mousemove', '--sync', ...at(p), 'click', '--repeat', String(notches(d)), '--delay', '20', upward(d) ? '4' : '5'], { signal: o?.signal })
    },
  }
}

function xdotoolKeys(runner: CommandRunner, xdotool: string): KeyDriver {
  return {
    name: 'xdotool',
    async key(k, o) {
      await runner.run(xdotool, ['key', '--clearmodifiers', XDOTOOL_KEYS[k]], { signal: o?.signal })
    },
    async chunk(text, o) {
      await runner.run(xdotool, ['type', '--clearmodifiers', '--delay', XDOTOOL_DELAY_MS, '--', text], { signal: o?.signal })
    },
  }
}

const WTYPE_KEYS: Record<Key, string[]> = { enter: ['-k', 'Return'], escape: ['-k', 'Escape'], tab: ['-k', 'Tab'], space: ['-k', 'space'], select_all: ['-M', 'ctrl', '-k', 'a', '-m', 'ctrl'] }

function wtypeKeys(runner: CommandRunner, wtype: string): KeyDriver {
  return {
    name: 'wtype',
    async key(k, o) {
      await runner.run(wtype, WTYPE_KEYS[k], { signal: o?.signal })
    },
    async chunk(text, o) {
      await runner.run(wtype, ['-d', XDOTOOL_DELAY_MS, '--', text], { signal: o?.signal })
    },
  }
}

/** Linux input event codes (KEY_ENTER 28, KEY_ESC 1, KEY_TAB 15, KEY_SPACE 57, KEY_LEFTCTRL 29, KEY_A 30) as ydotool takes them. */
const YDOTOOL_KEYS: Record<Key, string[]> = { enter: ['28:1', '28:0'], escape: ['1:1', '1:0'], tab: ['15:1', '15:0'], space: ['57:1', '57:0'], select_all: ['29:1', '30:1', '30:0', '29:0'] }
/** ydotool click: 0x40 down + 0x80 up of button 0 (left). */
const YDOTOOL_LEFT_CLICK = '0xC0'

function ydotoolKeys(runner: CommandRunner, ydotool: string): KeyDriver {
  return {
    name: 'ydotool',
    async key(k, o) {
      await runner.run(ydotool, ['key', ...YDOTOOL_KEYS[k]], { signal: o?.signal })
    },
    async chunk(text, o) {
      await runner.run(ydotool, ['type', '--key-delay', XDOTOOL_DELAY_MS, '--', text], { signal: o?.signal })
    },
  }
}

/** ydotool's wheel: REL_WHEEL, positive away from the person (up). */
const ydotoolWheel = (runner: CommandRunner, ydotool: string, d: ScrollDirection, o?: ActOptions) =>
  runner.run(ydotool, ['mousemove', '--wheel', '-x', '0', '-y', String(upward(d) ? notches(d) : -notches(d))], { signal: o?.signal })

function hyprPointer(runner: CommandRunner, hyprctl: string, ydotool: string): PointerDriver {
  const move = (p: Point, o?: ActOptions) => runner.run(hyprctl, ['dispatch', 'movecursor', ...at(p)], { signal: o?.signal })
  return {
    name: 'hyprctl + ydotool',
    async click(p, o) {
      await move(p, o)
      await runner.run(ydotool, ['click', YDOTOOL_LEFT_CLICK], { signal: o?.signal })
    },
    async scroll(p, d, o) {
      await move(p, o)
      await ydotoolWheel(runner, ydotool, d, o)
    },
  }
}

/** sway's own seat commands: exact, and need no uinput. Buttons 4 and 5 are the wheel. */
function swayPointer(runner: CommandRunner, swaymsg: string): PointerDriver {
  const seat = (args: string[], o?: ActOptions) => runner.run(swaymsg, ['seat', '-', 'cursor', ...args], { signal: o?.signal })
  return {
    name: 'swaymsg',
    async click(p, o) {
      await seat(['set', ...at(p)], o)
      await seat(['press', 'button1'], o)
      await seat(['release', 'button1'], o)
    },
    async scroll(p, d, o) {
      await seat(['set', ...at(p)], o)
      const button = upward(d) ? 'button4' : 'button5'
      for (let i = 0; i < notches(d); i += 1) {
        aborted(o?.signal)
        await seat(['press', button], o)
        await seat(['release', button], o)
      }
    },
  }
}

/** Where the RemoteDesktop portal's restore token is kept, so the person is asked once, not every session. */
export interface TokenStore {
  read(): Promise<string | null>
  write(token: string): Promise<void>
}

/** A token file readable only by this user. */
export function fileTokenStore(path: string): TokenStore {
  return {
    read: () => readFile(path, 'utf8').then((text) => text.trim() || null, () => null),
    async write(token) {
      await mkdir(dirname(path), { recursive: true, mode: 0o700 })
      await writeFile(path, `${token}\n`, { mode: 0o600 })
      await chmod(path, 0o600)
    },
  }
}

/**
 * Keys through org.freedesktop.portal.RemoteDesktop: CreateSession, SelectDevices (keyboard, persist_mode 2 with the
 * kept restore_token), Start, then NotifyKeyboardKeysym for each event, and Close. Without a valid token Start shows
 * the portal's dialog and waits for the person. Only keys: a button press would land wherever the person's pointer
 * is, and absolute motion needs a ScreenCast stream this backend does not open.
 * stdin: {"token": str|null, "events": [[keysym, pressed 1|0], ...]}; stdout: {"token": str|null}.
 */
export const PORTAL_KEYS = String.raw`
import json, sys
try:
    import gi
    gi.require_version('Gio', '2.0')
    from gi.repository import Gio, GLib
except Exception:
    sys.stderr.write('LOCALROUTER_NO_GIO')
    raise SystemExit(3)

NAME = 'org.freedesktop.portal.Desktop'
PATH = '/org/freedesktop/portal/desktop'
REMOTE = 'org.freedesktop.portal.RemoteDesktop'
req = json.loads(sys.stdin.read() or '{}')
bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
sender = bus.get_unique_name()[1:].replace('.', '_')
loop = GLib.MainLoop()
count = [0]

def call(method, args):
    return bus.call_sync(NAME, PATH, REMOTE, method, args, None, Gio.DBusCallFlags.NONE, -1, None)

def request(method, build):
    count[0] += 1
    token = 'localrouter%d' % count[0]
    path = '/org/freedesktop/portal/desktop/request/%s/%s' % (sender, token)
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
    sub = bus.signal_subscribe(NAME, 'org.freedesktop.portal.Request', 'Response', path, None, Gio.DBusSignalFlags.NONE, on_response)
    timer = GLib.timeout_add_seconds(110, on_timeout)
    try:
        call(method, build(token))
    except GLib.Error as error:
        bus.signal_unsubscribe(sub)
        sys.stderr.write('LOCALROUTER_NO_PORTAL:' + str(error.message))
        raise SystemExit(4)
    loop.run()
    if got.get('code') != 9:
        GLib.source_remove(timer)
    bus.signal_unsubscribe(sub)
    if got.get('code') != 0:
        sys.stderr.write('LOCALROUTER_PORTAL_DENIED:%s' % got.get('code'))
        raise SystemExit(5)
    return got['results']

created = request('CreateSession', lambda t: GLib.Variant('(a{sv})', ({'handle_token': GLib.Variant('s', t), 'session_handle_token': GLib.Variant('s', 'localrouter')},)))
session = created['session_handle']
select = {'types': GLib.Variant('u', 1), 'persist_mode': GLib.Variant('u', 2)}
if req.get('token'):
    select['restore_token'] = GLib.Variant('s', req['token'])
def selecting(t):
    select['handle_token'] = GLib.Variant('s', t)
    return GLib.Variant('(oa{sv})', (session, select))
request('SelectDevices', selecting)
started = request('Start', lambda t: GLib.Variant('(osa{sv})', (session, '', {'handle_token': GLib.Variant('s', t)})))
if not (started.get('devices', 0) & 1):
    sys.stderr.write('LOCALROUTER_PORTAL_DENIED:keyboard')
    raise SystemExit(5)
for keysym, pressed in req.get('events', []):
    call('NotifyKeyboardKeysym', GLib.Variant('(oa{sv}iu)', (session, {}, int(keysym), int(pressed))))
bus.call_sync(NAME, session, 'org.freedesktop.portal.Session', 'Close', None, None, Gio.DBusCallFlags.NONE, -1, None)
print(json.dumps({'token': started.get('restore_token') or req.get('token')}))
`

/** X keysyms: Latin-1 printable characters are their own code; anything else is 0x01000000 + the code point. */
export function keysym(char: string): number {
  const code = char.codePointAt(0) ?? 0
  return (code >= 0x20 && code <= 0x7e) || (code >= 0xa0 && code <= 0xff) ? code : 0x01000000 + code
}

/** The keys a `press` action names. */
export type PressKey = Exclude<Key, 'select_all'>
/** Their X keysyms, for the portal and for AT-SPI's own key events on X11. */
export const PRESS_KEYSYMS: Record<PressKey, number> = { enter: 0xff0d, escape: 0xff1b, tab: 0xff09, space: 0x20 }
const CONTROL_L = 0xffe3

/** Press and release events for a key, as the portal script takes them. */
export function keyEvents(key: Key): number[][] {
  if (key === 'select_all') return [[CONTROL_L, 1], [0x61, 1], [0x61, 0], [CONTROL_L, 0]]
  return [[PRESS_KEYSYMS[key], 1], [PRESS_KEYSYMS[key], 0]]
}

function portalKeys(runner: CommandRunner, python3: string, tokens: TokenStore): KeyDriver {
  const send = async (events: number[][], o?: ActOptions) => {
    const token = await tokens.read()
    // The first run waits for the person to answer the portal's dialog.
    const out = await runner.run(python3, ['-I', '-c', PORTAL_KEYS], { input: JSON.stringify({ token, events }), signal: o?.signal, timeoutMs: token ? 15_000 : 120_000 })
    const next = optional(object(json(out.trim(), 'portal answer'), 'portal answer').token, 'portal token', (v, w) => (v === null ? null : string(v, w)))
    if (next && next !== token) await tokens.write(next)
  }
  return {
    name: 'RemoteDesktop portal',
    key: (k, o) => send(keyEvents(k), o),
    chunk: (text, o) => send([...text].flatMap((c) => [[keysym(c), 1], [keysym(c), 0]]), o),
  }
}

export type InjectorOptions = {
  /** Where the portal's restore token is kept; without a store the portal is not used (it would ask on every key). */
  tokens?: TokenStore
}

const NO_POINTER: Record<string, string> = {
  x11: 'Clicking where a control has no accessibility action needs xdotool.',
  hyprland: 'Clicking where a control has no accessibility action needs ydotool with ydotoold running (Hyprland places the pointer; ydotool clicks).',
  sway: 'Clicking where a control has no accessibility action needs swaymsg.',
  desktop: 'This desktop does not let another app place the pointer, and this control has no accessibility action to use instead.',
}

/** Chooses the pointer and key paths for this session (see the top of this file). */
export function chooseInjector(session: Session, runner: CommandRunner, options: InjectorOptions = {}): Injector {
  const t = session.tools
  const ydotool = t.ydotool && session.ydotoold ? t.ydotool : undefined
  let pointer: PointerDriver | null = null
  let keys: KeyDriver | null = null
  let noPointer: string = NO_POINTER.desktop
  if (session.server === 'x11') {
    noPointer = NO_POINTER.x11
    if (t.xdotool) {
      pointer = xdotoolPointer(runner, t.xdotool)
      keys = xdotoolKeys(runner, t.xdotool)
    }
  } else {
    if (session.compositor === 'hyprland') {
      noPointer = NO_POINTER.hyprland
      if (t.hyprctl && ydotool) pointer = hyprPointer(runner, t.hyprctl, ydotool)
    } else if (session.compositor === 'sway') {
      noPointer = NO_POINTER.sway
      if (t.swaymsg) pointer = swayPointer(runner, t.swaymsg)
    }
    if (wlroots(session) && t.wtype) keys = wtypeKeys(runner, t.wtype)
    else if (ydotool) keys = ydotoolKeys(runner, ydotool)
    else if (t.python3 && session.gio && options.tokens && (session.compositor === 'gnome' || session.compositor === 'kde')) keys = portalKeys(runner, t.python3, options.tokens)
  }
  const names = [...new Set([pointer?.name, keys?.name].filter((n): n is string => Boolean(n)))]
  const noKeys = keys ? undefined : session.server === 'x11' ? 'Pressing keys needs xdotool.' : 'Pressing keys on this desktop needs wtype (wlroots compositors), ydotool with ydotoold, or the RemoteDesktop portal.'
  return {
    name: names.join(' + ') || 'none',
    ...(pointer ? {} : { noPointer }),
    ...(noKeys ? { noKeys } : {}),
    async click(p, o) {
      if (!pointer) throw new Error(noPointer)
      aborted(o?.signal)
      await pointer.click(p, o)
    },
    async scroll(p, d, o) {
      if (!pointer) throw new Error(noPointer)
      aborted(o?.signal)
      await pointer.scroll(p, d, o)
    },
    async key(k, o) {
      if (!keys) throw new Error(noKeys)
      aborted(o?.signal)
      await keys.key(k, o)
    },
    async type(text, o) {
      if (!keys) throw new Error(noKeys)
      for (const piece of chunks(plainText(text))) {
        aborted(o?.signal)
        await keys.chunk(piece, o)
      }
    },
  }
}
