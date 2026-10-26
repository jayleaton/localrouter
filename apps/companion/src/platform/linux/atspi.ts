import type { CommandRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import { CONTROL_ROLES, NeedsPointer } from '../../../../desktop-helper/src/computer/types.ts'
import type { ActOptions, ActResult, Observation, ObserveOptions, Point, Rect, ScreenAction, ScreenBackend, ScreenControl, Screenshot, ScreenshotOptions } from '../../../../desktop-helper/src/computer/types.ts'
import { array, boolean, integerIn, json, nullable, number, object, oneOf, optional, string } from '../../../../desktop-helper/src/validate.ts'
import { activeWindow } from './compositor.ts'
import type { ActiveWindow } from './compositor.ts'
import { MULTILINE, PRESS_KEYSYMS, plainText } from './input.ts'
import type { Injector, PressKey } from './input.ts'
import { installLine } from './session.ts'
import type { Session } from './session.ts'

/**
 * The AT-SPI reader and actor, one python3 script (python3-gi, Atspi 2.0) run per call with its op as argv and its
 * payload on stdin, so typed text never shows in a process list. Like the macOS backend, a ref is the child-index
 * path from the front window, re-resolved when the action runs; the walk is breadth first and bounded (2,500 nodes,
 * 150 controls, 3 s). Actions go through AT-SPI first (Action, Selection, EditableText, Value): those never touch the
 * person's pointer or keyboard, and work the same on X11 and every Wayland compositor. Only when a control has no
 * such action does the script hand back its centre for the injector (input.ts).
 *
 * Coordinates: on X11, `get_extents(SCREEN)` is global. On Wayland a client does not know where its window is, so
 * the extents are relative to its own surface (XWayland apps still report X coordinates). Either way an element's
 * offset inside its window is right, so its global position is the compositor's window origin (Hyprland, sway) plus
 * that offset. Where the compositor does not say (GNOME, KDE), bounds are left out and a pointer fallback is refused
 * rather than clicking a guess.
 *
 * Python 3.8+; the script must not contain a backtick or a dollar-brace (it is a TypeScript template).
 */
export const ATSPI_SCRIPT = String.raw`
import json, os, re, sys, time
try:
    import gi
    gi.require_version('Atspi', '2.0')
    from gi.repository import Atspi, GLib
except Exception:
    sys.stderr.write('LOCALROUTER_NO_ATSPI')
    raise SystemExit(3)

class Fail(Exception):
    pass

S = Atspi.StateType
R = Atspi.Role

def roles(*names):
    return set(r for r in (getattr(R, n, None) for n in names) if r is not None)

WINDOWS = roles('FRAME', 'WINDOW', 'DIALOG', 'ALERT', 'FILE_CHOOSER')
KINDS = {}
for kind, names in (
    ('button', ('PUSH_BUTTON', 'TOGGLE_BUTTON', 'PUSH_BUTTON_MENU')),
    ('link', ('LINK',)),
    ('textbox', ('ENTRY', 'PASSWORD_TEXT', 'SPIN_BUTTON')),
    ('tab', ('PAGE_TAB',)),
    ('row', ('LIST_ITEM', 'TABLE_ROW', 'TREE_ITEM', 'TABLE_CELL')),
    ('menuitem', ('MENU_ITEM', 'CHECK_MENU_ITEM', 'RADIO_MENU_ITEM')),
    ('checkbox', ('CHECK_BOX', 'RADIO_BUTTON', 'SWITCH')),
):
    for r in roles(*names):
        KINDS[r] = kind
# Editable only when the EDITABLE state says so: a read-only text view is text, a contenteditable section a textbox.
EDITABLE_ROLES = roles('TEXT', 'DOCUMENT_TEXT', 'PARAGRAPH', 'SECTION', 'TERMINAL', 'COMBO_BOX')
PASSWORD = roles('PASSWORD_TEXT')
OPTION_PARENTS = roles('LIST_BOX', 'COMBO_BOX', 'MENU')
COMBOS = roles('COMBO_BOX')
FRAMES = roles('FRAME')
PAGE_TABS = roles('PAGE_TAB')
MENUS = roles('MENU')
MENU_BARS = roles('MENU_BAR')
TEXT_ROLES = roles('LABEL', 'STATIC', 'HEADING', 'PARAGRAPH', 'TEXT', 'CAPTION', 'DESCRIPTION_VALUE', 'BLOCK_QUOTE')
LEAVES = roles('LABEL', 'STATIC', 'SCROLL_BAR', 'PROGRESS_BAR', 'SLIDER', 'IMAGE', 'ICON', 'SEPARATOR')
SCROLL_BARS = roles('SCROLL_BAR')
TAB_LISTS = roles('PAGE_TAB_LIST')
WEB = roles('DOCUMENT_WEB', 'DOCUMENT_FRAME')
# Chrome/Chromium "Address and search bar", Firefox "Search or enter address", Epiphany/others "Address" / "Location".
ADDRESS = re.compile(r'address and search bar|search or enter address|search with .* or enter address|^address( bar)?$|^location$', re.I)
SELECTABLE = ('tab', 'row', 'option')
PREFERRED = ('click', 'press', 'activate', 'jump', 'toggle', 'switch', 'select', 'open')

MAX_NODES, MAX_CONTROLS, MAX_TEXT, BUDGET_S = 2500, 150, 1500, 3.0

DEBUG = os.environ.get('LOCALROUTER_ATSPI_DEBUG', '')

def safe(fn, default=None):
    try:
        value = fn()
        return default if value is None else value
    except Exception as error:
        # LOCALROUTER_ATSPI_DEBUG=<file> (CI): every swallowed failure appended there, for a backend not run elsewhere.
        if DEBUG:
            with open(DEBUG, 'a') as log:
                log.write('%s: %r\n' % (getattr(fn, '__name__', 'call'), error))
        return default

def clip(text, limit):
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    return text[:limit - 1] + '…' if len(text) > limit else text

def has(states, state):
    return safe(lambda: states.contains(state), False)

def extents(acc):
    r = safe(lambda: acc.get_extents(Atspi.CoordType.SCREEN))
    return [r.x, r.y, r.width, r.height] if r is not None and r.width > 0 and r.height > 0 else None

def center(box):
    return [box[0] + box[2] / 2.0, box[1] + box[3] / 2.0] if box else None

def text_of(acc, limit):
    t = safe(acc.get_text_iface)
    if t is None:
        return ''
    n = safe(t.get_character_count, 0)
    # Atspi.Text.get_text, not t.get_text: on an Accessible, get_text() is the one-argument alias for the interface.
    return safe(lambda: Atspi.Text.get_text(t, 0, min(n, limit)), '') or ''

def children(acc):
    n = safe(acc.get_child_count, 0)
    for i in range(min(n, 2000)):
        kid = safe(lambda: acc.get_child_at_index(i))
        if kid is not None:
            yield i, kid

def windows():
    desktop = Atspi.get_desktop(0)
    for _, app in children(desktop):
        for _, win in children(app):
            if safe(win.get_role) in WINDOWS:
                yield app, win

def front(hint):
    found = list(windows())
    active = lambda pair: has(safe(pair[1].get_state_set), S.ACTIVE)
    pid = hint.get('pid')
    if pid:
        # The compositor named the focused process: its window or nothing, never another app's stale ACTIVE one.
        mine = [p for p in found if safe(p[0].get_process_id) == pid]
        title = hint.get('title') or ''
        for group in ([p for p in mine if active(p)], [p for p in mine if title and safe(p[1].get_name, '') == title],
                      [p for p in mine if has(safe(p[1].get_state_set), S.SHOWING)], mine):
            if group:
                return group[0]
        raise Fail('LOCALROUTER_UNREACHABLE')
    for pair in found:
        if active(pair):
            return pair
    raise Fail('LOCALROUTER_NO_WINDOW')

def kind_of(acc, role, states, parent_role):
    kind = KINDS.get(role)
    if role in EDITABLE_ROLES and has(states, S.EDITABLE) and safe(acc.get_editable_text_iface) is not None:
        return 'textbox'
    if role in COMBOS:
        return 'button'
    if role in MENUS and parent_role in MENU_BARS:
        return 'menuitem'
    if kind == 'row' and parent_role in OPTION_PARENTS:
        return 'option'
    return kind

def inner_text(acc, depth=0):
    out = ''
    for _, kid in children(acc):
        if len(out) >= 100:
            break
        words = safe(kid.get_name, '') or (text_of(kid, 100) if not safe(kid.get_child_count, 0) else '')
        if not words and depth < 3:
            words = inner_text(kid, depth + 1)
        if words:
            out += (' ' if out else '') + words
    return out

def overlaps(box, frame):
    return not (box and frame) or (box[0] < frame[0] + frame[2] and box[1] < frame[1] + frame[3] and box[0] + box[2] > frame[0] and box[1] + box[3] > frame[1])

def observe(req):
    app, win = front(req.get('hint') or {})
    frame = extents(win)
    deadline = time.monotonic() + BUDGET_S
    queue, visited, controls, text, url, truncated = [(win, [], 0)], 0, [], '', None, False
    while queue:
        node, path, depth = queue.pop(0)
        node_role = safe(node.get_role)
        for i, acc in children(node):
            if visited >= MAX_NODES or len(controls) >= MAX_CONTROLS or time.monotonic() > deadline:
                truncated = True
                break
            visited += 1
            states = safe(acc.get_state_set)
            if not has(states, S.SHOWING):
                continue
            role = safe(acc.get_role)
            box = extents(acc)
            if not overlaps(box, frame):
                continue
            kind = kind_of(acc, role, states, node_role)
            here = path + [i]
            if kind:
                secret = role in PASSWORD
                name = safe(acc.get_name, '')
                label = name or safe(acc.get_description, '')
                if not label and kind not in ('textbox',):
                    label = text_of(acc, 100) or (inner_text(acc) if kind in ('row', 'option', 'button', 'link') else '')
                control = {'ref': '.'.join(str(n) for n in here), 'role': kind, 'label': clip(label, 100)}
                if box:
                    control['bounds'] = box
                if kind == 'textbox':
                    if not secret:
                        control['value'] = clip(text_of(acc, 400), 200)
                        if url is None and ADDRESS.search(name or ''):
                            url = text_of(acc, 2000).strip() or None
                    if has(states, S.FOCUSED):
                        control['focused'] = True
                if kind != 'row' or control['label']:
                    controls.append(control)
            elif role in TEXT_ROLES and len(text) < MAX_TEXT:
                words = safe(acc.get_name, '') or (text_of(acc, MAX_TEXT) if not safe(acc.get_child_count, 0) else '')
                if words:
                    text += (' ' if text else '') + clip(words, MAX_TEXT)
            if kind in ('textbox', 'button', 'checkbox') or role in LEAVES or depth + 1 >= 20:
                continue
            queue.append((acc, here, depth + 1))
        if truncated:
            break
    order = {id(c): n for n, c in enumerate(controls)}
    controls.sort(key=lambda c: (round(c['bounds'][1] / 8.0), c['bounds'][0]) if 'bounds' in c else (1e9, order[id(c)]))
    out = {'app': safe(app.get_name, ''), 'title': safe(win.get_name, ''), 'pid': safe(app.get_process_id, 0),
           'frame': frame, 'controls': controls, 'text': text[:MAX_TEXT], 'truncated': truncated}
    if url:
        out['url'] = url
    return out

def resolve(win, ref):
    el = win
    for part in str(ref).split('.'):
        if not part.isdigit() or int(part) >= safe(el.get_child_count, 0):
            raise Fail('LOCALROUTER_GONE')
        el = safe(lambda: el.get_child_at_index(int(part)))
        if el is None:
            raise Fail('LOCALROUTER_GONE')
    if not has(safe(el.get_state_set), S.SHOWING):
        raise Fail('LOCALROUTER_GONE')
    return el

def named_action(el):
    action = safe(el.get_action_iface)
    if action is None:
        return False
    names = [(safe(lambda: action.get_action_name(i), '') or '').lower() for i in range(safe(action.get_n_actions, 0))]
    for want in PREFERRED:
        if want in names:
            return bool(safe(lambda: action.do_action(names.index(want)), False))
    return False

def select_in_parent(el):
    parent = safe(el.get_parent)
    selection = safe(parent.get_selection_iface) if parent is not None else None
    index = safe(el.get_index_in_parent, -1)
    return selection is not None and index >= 0 and bool(safe(lambda: selection.select_child(index), False))

def fallback(el, win, app, **extra):
    out = {'done': False, 'point': center(extents(el)), 'frame': extents(win), 'pid': safe(app.get_process_id, 0)}
    out.update(extra)
    return out

def click(el, win, app):
    parent = safe(el.get_parent)
    kind = kind_of(el, safe(el.get_role), safe(el.get_state_set), safe(parent.get_role) if parent is not None else None)
    tries = (select_in_parent, named_action) if kind in SELECTABLE else (named_action, select_in_parent)
    for attempt in tries:
        if attempt(el):
            return {'done': True}
    return fallback(el, win, app)

def landed(el, text):
    for wait in (0, 0.15, 0.4):
        time.sleep(wait)
        if text_of(el, len(text) + 64) == text:
            return True
    return False

def type_text(el, win, app, text):
    states = safe(el.get_state_set)
    secret = safe(el.get_role) in PASSWORD
    if '\n' in text and has(states, S.SINGLE_LINE):
        raise Fail('LOCALROUTER_MULTILINE')
    editable = safe(el.get_editable_text_iface) if has(states, S.EDITABLE) else None
    if editable is not None:
        # Focus first, as a click would: an Enter after this goes to the field, not to whatever had the focus.
        safe(el.grab_focus)
        if safe(lambda: editable.set_text_contents(text), False) and (secret or landed(el, text)):
            return {'done': True}
        safe(el.grab_focus)
        count = safe(lambda: el.get_text_iface().get_character_count(), 0)
        safe(lambda: editable.delete_text(0, count))
        if safe(lambda: editable.insert_text(0, text, len(text.encode('utf-8'))), False) and (secret or landed(el, text)):
            return {'done': True}
    return fallback(el, win, app, secret=secret)

def scroll_bar(target, limit=400):
    queue, seen = [target], 0
    while queue and seen < limit:
        node = queue.pop(0)
        for _, kid in children(node):
            seen += 1
            if safe(kid.get_role) in SCROLL_BARS:
                states = safe(kid.get_state_set)
                if has(states, S.SHOWING) and not has(states, S.HORIZONTAL) and safe(kid.get_value_iface) is not None:
                    return kid
            elif safe(kid.get_role) not in LEAVES:
                queue.append(kid)
    return None

def scroll(el, win, app, direction):
    bar, node = None, el or win
    # The area's own scroll bar, else one beside it in an enclosing scrolled pane (a few levels up).
    for _ in range(6):
        if node is None:
            break
        bar = scroll_bar(node)
        if bar is not None:
            break
        node = safe(node.get_parent) if el is not None else None
    if bar is not None:
        value = bar.get_value_iface()
        low, high, now = safe(value.get_minimum_value, 0.0), safe(value.get_maximum_value, 0.0), safe(value.get_current_value, 0.0)
        step = max((high - low) / 10.0, safe(value.get_minimum_increment, 0.0) * 3)
        goal = {'top': low, 'bottom': high, 'up': max(low, now - step), 'down': min(high, now + step)}[direction]
        if high > low and safe(lambda: value.set_current_value(goal), False) and abs(safe(value.get_current_value, now) - goal) < max(step, 1) / 2:
            return {'done': True}
        if high > low and goal == now:
            return {'done': True}
    return fallback(el or win, win, app)

def act(req):
    app, win = front(req.get('hint') or {})
    op = req.get('kind')
    if op == 'scroll':
        return scroll(resolve(win, req['ref']) if req.get('ref') else None, win, app, req.get('direction', 'down'))
    el = resolve(win, req.get('ref'))
    if op == 'click':
        return click(el, win, app)
    if op == 'type':
        return type_text(el, win, app, str(req.get('text', '')))
    raise Fail('LOCALROUTER_UNKNOWN_OP')

def read(req):
    app, win = front(req.get('hint') or {})
    el = resolve(win, req.get('ref'))
    return {'text': '' if safe(el.get_role) in PASSWORD else text_of(el, 100000)}

def key(req):
    return {'done': bool(Atspi.generate_keyboard_event(int(req['keysym']), None, Atspi.KeySynthType.SYM))}

def apps(req):
    out = []
    for _, app in children(Atspi.get_desktop(0)):
        if any(safe(w.get_role) in WINDOWS for _, w in children(app)):
            name = safe(app.get_name, '')
            if name:
                out.append({'name': name, 'pid': safe(app.get_process_id, 0)})
    return out

def focus(req):
    wanted = str(req.get('name', '')).lower()
    for app, win in windows():
        if safe(app.get_name, '').lower() == wanted:
            return {'pid': safe(app.get_process_id, 0), 'focused': bool(safe(win.grab_focus, False))}
    raise Fail('LOCALROUTER_NOT_RUNNING')

BROWSERS = re.compile(r'chrom|firefox|brave|edge|vivaldi|opera|librewolf|zen|epiphany|web', re.I)

def tab_strip(win, limit=1500):
    queue, seen = [win], 0
    while queue and seen < limit:
        node = queue.pop(0)
        for _, kid in children(node):
            seen += 1
            role = safe(kid.get_role)
            if role in TAB_LISTS:
                return kid
            if role not in WEB and role not in LEAVES:
                queue.append(kid)
    return None

def browser_tabs():
    found, window = [], 0
    for app, win in windows():
        if not BROWSERS.search(safe(app.get_name, '')) or safe(win.get_role) not in FRAMES:
            continue
        strip = tab_strip(win)
        if strip is not None:
            index = 0
            for _, tab in children(strip):
                if safe(tab.get_role) in PAGE_TABS:
                    found.append((window, index, tab, app, win))
                    index += 1
        window += 1
    return found

def tabs(req):
    return [{'window': w, 'index': i, 'title': clip(safe(t.get_name, ''), 200), 'active': has(safe(t.get_state_set), S.SELECTED), 'app': safe(a.get_name, '')}
            for w, i, t, a, _ in browser_tabs()]

def tab(req):
    found = browser_tabs()
    title = str(req.get('title', ''))
    same = lambda t: clip(safe(t.get_name, ''), 200) == title
    hit = [f for f in found if f[0] == req.get('window') and f[1] == req.get('index') and same(f[2])] or [f for f in found if same(f[2])]
    if not hit:
        raise Fail('LOCALROUTER_NO_TAB')
    _, _, el, app, win = hit[0]
    for attempt in (select_in_parent, named_action):
        if attempt(el):
            return {'done': True, 'app': safe(app.get_name, '')}
    return fallback(el, win, app, app=safe(app.get_name, ''))

def probe(req):
    return {'apps': len(apps(req))}

OPS = {'observe': observe, 'act': act, 'read': read, 'key': key, 'apps': apps, 'focus': focus, 'tabs': tabs, 'tab': tab, 'probe': probe}

try:
    request = json.loads(sys.stdin.read() or '{}')
    print(json.dumps(OPS[sys.argv[1]](request)))
except Fail as error:
    sys.stderr.write(str(error))
    raise SystemExit(2)
except GLib.Error as error:
    sys.stderr.write('LOCALROUTER_NO_BUS:' + str(error.message))
    raise SystemExit(2)
`

/** One op of the script: argv carries the op only, the payload goes on stdin. */
export function atspiCall(session: Session, op: string): [string, string[]] {
  return [session.tools.python3 ?? 'python3', ['-I', '-c', ATSPI_SCRIPT, op]]
}

/** The x/y offset from an element's raw extents to the global space, or null when it is unknown (see the top). */
export type Offset = { dx: number; dy: number }

/**
 * X11: none. Wayland: the compositor's window origin less the window's own reported origin, when the compositor's
 * focused window is the process AT-SPI read (a mismatch means they are looking at different windows).
 */
export function windowOffset(session: Session, frame: Rect | null, pid: number, win: ActiveWindow | null): Offset | null {
  if (session.server === 'x11') return { dx: 0, dy: 0 }
  if (!win || !frame) return null
  if (win.pid && pid && win.pid !== pid) return null
  return { dx: win.x - frame.x, dy: win.y - frame.y }
}

const shift = (r: Rect, o: Offset): Rect => ({ x: r.x + o.dx, y: r.y + o.dy, w: r.w, h: r.h })

const boxShape = (value: unknown, what: string): Rect => {
  const [x, y, w, h] = array(value, what, number)
  if (h === undefined) throw new Error(`${what} should be [x, y, w, h]`)
  return { x, y, w, h }
}

const pointShape = (value: unknown, what: string): Point => {
  const [x, y] = array(value, what, number)
  if (y === undefined) throw new Error(`${what} should be [x, y]`)
  return { x, y }
}

type RawControl = ScreenControl & { bounds?: Rect }
export type RawObservation = { app: string; title: string; pid: number; frame: Rect | null; controls: RawControl[]; text: string; url?: string; truncated: boolean }

function controlShape(value: unknown, what: string): RawControl {
  const raw = object(value, what)
  const control: RawControl = { ref: string(raw.ref, `${what} ref`), role: oneOf(raw.role, CONTROL_ROLES, `${what} role`), label: string(raw.label, `${what} label`) }
  const text = optional(raw.value, `${what} value`, string)
  const focused = optional(raw.focused, `${what} focused`, boolean)
  const bounds = optional(raw.bounds, `${what} bounds`, boxShape)
  if (text !== undefined) control.value = text
  if (focused !== undefined) control.focused = focused
  if (bounds) control.bounds = bounds
  return control
}

export function observationShape(value: unknown): RawObservation {
  const raw = object(value, 'observation')
  const url = optional(raw.url, 'observation url', string)
  return {
    app: string(raw.app, 'observation app'),
    title: string(raw.title, 'observation title'),
    pid: integerIn(raw.pid, 0, 2 ** 31, 'observation pid'),
    frame: nullable(raw.frame, 'observation frame', boxShape),
    controls: array(raw.controls, 'observation controls', controlShape),
    text: string(raw.text, 'observation text'),
    truncated: boolean(raw.truncated, 'observation truncated'),
    ...(url ? { url } : {}),
  }
}

/** What an act op answered: done through AT-SPI, or the element's raw centre for the pointer. */
export type RawAct = { done: true } | { done: false; point: Point | null; frame: Rect | null; pid: number; secret: boolean }

export function actShape(value: unknown): RawAct {
  const raw = object(value, 'action result')
  if (boolean(raw.done, 'action done')) return { done: true }
  return {
    done: false,
    point: nullable(raw.point, 'action point', pointShape),
    frame: nullable(raw.frame, 'action frame', boxShape),
    pid: integerIn(raw.pid, 0, 2 ** 31, 'action pid'),
    secret: optional(raw.secret, 'action secret', boolean) ?? false,
  }
}

/**
 * An address bar's text as the observation URL, exactly as shown: Chromium hides the scheme of both http and https,
 * so none is guessed (an approval card shows what the person saw). Text being typed there (words, spaces) is not a URL.
 */
export function addressUrl(text: string): string | undefined {
  const value = text.trim()
  if (!value || /\s/.test(value)) return undefined
  // "localhost:3000" has a colon but no scheme; a real one has "//" or is one of the browsers' own.
  if (/^[a-z][a-z0-9+.-]*:\/\//i.test(value) || /^(about|chrome|edge|brave|file|data|view-source|moz-extension):/i.test(value)) return value
  return /^[^/]+\.[a-z]{2,}(?::\d+)?(\/|$)/i.test(value) || /^localhost(:\d+)?(\/|$)/i.test(value) ? value : undefined
}

export type LinuxScreenOptions = { session: Session; runner: CommandRunner; injector: Injector }

/** Refusal for a pointer fallback where the window's global position is unknown. */
export const NO_ORIGIN = 'This desktop does not tell other apps where its windows are, so LocalRouter will not click by position here (it could hit the wrong place), and this control has no accessibility action to use instead.'

/** The front window read through AT-SPI and acted on semantically first, with the session's injector as fallback. */
export class LinuxScreen implements ScreenBackend {
  readonly kind = 'desktop'
  private readonly session: Session
  private readonly runner: CommandRunner
  private readonly injector: Injector
  /** Set by index.ts when the session can capture the front window. */
  screenshot?: (options?: ScreenshotOptions) => Promise<Screenshot>

  constructor(options: LinuxScreenOptions) {
    this.session = options.session
    this.runner = options.runner
    this.injector = options.injector
  }

  async observe(options: ObserveOptions = {}): Promise<Observation> {
    const win = await this.compositorWindow()
    const raw = observationShape(json((await this.call('observe', { hint: hint(win) }, options.signal, win)).trim(), 'observation'))
    const offset = windowOffset(this.session, raw.frame, raw.pid, win)
    const controls = raw.controls.map(({ bounds, ...control }) => (bounds && offset ? { ...control, bounds: shift(bounds, offset) } : control))
    const url = raw.url !== undefined ? addressUrl(raw.url) : undefined
    const window = raw.frame && offset ? shift(raw.frame, offset) : undefined
    return {
      backend: 'desktop',
      app: raw.app,
      title: raw.title,
      controls,
      text: raw.text,
      ...(url ? { url } : {}),
      ...(window ? { window } : {}),
      ...(raw.truncated ? { truncated: true } : {}),
    }
  }

  async act(action: ScreenAction, options: ActOptions = {}): Promise<ActResult> {
    if (action.kind === 'press') return this.press(action.key, options)
    const win = await this.compositorWindow()
    // AT-SPI first (an action or EditableText): the person's pointer and keyboard are never touched.
    const result = actShape(json((await this.call('act', { ...action, hint: hint(win) }, options.signal, win)).trim(), 'action result'))
    if (result.done) return { mode: 'background' }
    // The background could not do it. Without the person's leave for the real pointer, say why and stop here.
    if (!options.allowPointer) throw new NeedsPointer(this.pointerReason(result, win, action))
    if (action.kind === 'type') plainText(action.text) // a line break would be Enter: refused before the keyboard
    const point = this.globalPoint(result, win)
    if (action.kind === 'click') {
      await this.injector.click(point, options)
      return { mode: 'pointer', pointer: point, at: point }
    }
    if (action.kind === 'scroll') {
      await this.injector.scroll(point, action.direction, options)
      return { mode: 'pointer', pointer: point, at: point }
    }
    await this.injector.click(point, options)
    await this.injector.key('select_all', options)
    await this.injector.type(action.text, options)
    if (!result.secret) {
      const read = object(json((await this.call('read', { ref: action.ref, hint: hint(win) }, options.signal, win)).trim(), 'text read back'), 'text read back')
      if (string(read.text, 'text read back') !== action.text) throw new Error('The text did not land in that box as typed; look again before trying another way.')
    }
    return { mode: 'pointer', pointer: point, at: point }
  }

  /** Why a step needs the real pointer: the compositor cannot inject one, or the control has no accessible action. */
  private pointerReason(result: Extract<RawAct, { done: false }>, win: ActiveWindow | null, action: ScreenAction): string {
    if (this.injector.noPointer) return this.injector.noPointer
    if (action.kind !== 'scroll' && !result.point) return 'that control has no accessible action and is not on screen'
    if (action.kind !== 'scroll' && !windowOffset(this.session, result.frame, result.pid, win)) return NO_ORIGIN
    return 'that control has no accessible action'
  }

  /** Keys are global events either way (not background): the injector, or AT-SPI's own on X11 when xdotool is missing. */
  /**
   * A key press. On Linux every route (XTest, ydotool, AT-SPI's own generate_keyboard_event) sends the key to the
   * focused window globally: there is no per-window background key. So a press needs the real keyboard, which the
   * person allows for the step ("needs your mouse for a moment"); until then it is refused.
   */
  private async press(key: PressKey, options: ActOptions): Promise<ActResult> {
    if (!options.allowPointer) throw new NeedsPointer('a key press reaches the focused window on Linux, so it needs the real keyboard')
    if (this.injector.noKeys) {
      if (this.session.server === 'x11') {
        const out = object(json((await this.call('key', { keysym: PRESS_KEYSYMS[key] }, options.signal, null)).trim(), 'key'), 'key')
        if (boolean(out.done, 'key done')) return { mode: 'pointer' }
      }
      throw new NeedsPointer(this.injector.noKeys)
    }
    await this.injector.key(key, options)
    return { mode: 'pointer' }
  }

  /** The element's centre in the global space, or the refusal when it cannot be known. */
  private globalPoint(result: Extract<RawAct, { done: false }>, win: ActiveWindow | null): Point {
    if (this.injector.noPointer) throw new Error(this.injector.noPointer)
    if (!result.point) throw new Error('That control is not on screen; scroll to it or look again.')
    const offset = windowOffset(this.session, result.frame, result.pid, win)
    if (!offset) throw new Error(NO_ORIGIN)
    return { x: result.point.x + offset.dx, y: result.point.y + offset.dy }
  }

  /** On Wayland, the compositor's focused window: the hint that picks the AT-SPI window, and the origin of its extents. */
  private async compositorWindow(): Promise<ActiveWindow | null> {
    return this.session.server === 'wayland' ? activeWindow(this.session, this.runner) : null
  }

  private async call(op: string, payload: object, signal: AbortSignal | undefined, win: ActiveWindow | null): Promise<string> {
    try {
      return await this.runner.run(...atspiCall(this.session, op), { input: JSON.stringify(payload), signal, timeoutMs: 12_000 })
    } catch (error) {
      throw explainAtspi(this.session, error instanceof Error ? error.message : String(error), win?.app) ?? error
    }
  }
}

const hint = (win: ActiveWindow | null) => (win ? { pid: win.pid ?? 0, title: win.title } : {})

export const ATSPI_PACKAGES = { debian: 'python3-gi gir1.2-atspi-2.0 at-spi2-core', fedora: 'python3-gobject at-spi2-core', arch: 'python-gobject at-spi2-core', other: 'python3 GObject introspection (gi) with the Atspi 2.0 typelib and at-spi2-core' }

/** How to make apps show their windows to AT-SPI, for the message and the access report. */
export const A11Y_FIX = 'Turn accessibility on: `gsettings set org.gnome.desktop.interface toolkit-accessibility true`, and for Qt apps set ACCESSIBILITY_ENABLED=1 and QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1 in the session environment; Electron apps need --force-renderer-accessibility. Then reopen the app.'

/** A script failure in words a person can act on, or null to pass it through. */
export function explainAtspi(session: Session, message: string, app?: string): Error | null {
  if (message.includes('LOCALROUTER_GONE')) return new Error('That control is gone; look again.')
  if (message.includes('LOCALROUTER_NO_WINDOW')) return new Error('No window is in front, or the app in front is not showing it to accessibility.')
  if (message.includes('LOCALROUTER_UNREACHABLE')) return new Error(`${app || 'The app in front'} is not showing its window to accessibility. ${A11Y_FIX}`)
  if (message.includes('LOCALROUTER_MULTILINE')) return new Error(MULTILINE)
  if (message.includes('LOCALROUTER_NO_TAB')) return new Error("Could not find that tab in the browser's tab strip.")
  if (message.includes('LOCALROUTER_NOT_RUNNING')) return new Error('That app has no open window.')
  if (message.includes('LOCALROUTER_NO_ATSPI')) return new Error(`Reading the screen needs python3 with GObject introspection and AT-SPI: ${installLine(session, ATSPI_PACKAGES)}.`)
  if (message.includes('LOCALROUTER_NO_BUS')) return new Error('The accessibility bus (at-spi2-core) is not running in this session; install at-spi2-core and log in again.')
  return null
}
