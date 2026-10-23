// Ported from SideBud (github.com/jayleaton/sidebud @70da540, apps/companion/src/computer/use/desktop.ts), reused in LocalRouter with the owner's permission.
import { execFile } from 'node:child_process'
import { rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { array, boolean, json, nullable, number, object, oneOf, optional, string } from '../validate.ts'
import type { CommandRunner } from './macControls.ts'
import { CONTROL_ROLES } from './types.ts'
import type { ActResult, Observation, Point, ScreenAction, ScreenBackend, ScreenControl } from './types.ts'

const execFileAsync = promisify(execFile)

/**
 * Default runner for the computer-use backends. osascript prints the script's return value on
 * stdout; on failure only stderr is kept, because execFile's own message repeats the whole script
 * and the error patterns below would otherwise match the script source instead of the real error.
 */
export const jxaRunner: CommandRunner = {
  async run(file, args) {
    try {
      const { stdout } = await execFileAsync(file, args, { timeout: 15_000, maxBuffer: 4 * 1024 * 1024 })
      return stdout
    } catch (error) {
      const stderr = typeof error === 'object' && error !== null && 'stderr' in error ? String(error.stderr).trim() : ''
      const killed = typeof error === 'object' && error !== null && 'killed' in error && error.killed === true
      throw new Error(killed ? 'The Mac took too long to answer (osascript timed out)' : stderr || (error instanceof Error ? error.message : String(error)))
    }
  },
}

/**
 * osascript argv for a JXA script whose `run(argv)` receives `argv`. The `--` stops osascript from
 * reading a user value that starts with a dash as one of its own flags.
 */
export function jxa(script: string, argv: string[] = []): [string, string[]] {
  return ['osascript', ['-l', 'JavaScript', '-e', script, '--', ...argv]]
}

/**
 * JXA helpers shared by the desktop scripts and Chrome tab switching. Reads use the AXUIElement C
 * API through the ObjC bridge rather than System Events: System Events costs one Apple Event
 * (~25 ms) per property, so a deep Electron window took ~10 s for 100 elements, while direct AX reads
 * walk ~700 in ~0.3 s. Bridging notes: out-params use `Ref()`, and the result must go through
 * `ObjC.castRefToObject` before it can be passed back into another AX call. AXValue structs
 * (position, size) cannot be read with AXValueGetValue from JXA, so their description
 * "{x:1 y:2}" / "{w:3 h:4}" is parsed. AX positions are global, top-left-origin points, the same
 * space CGEvent locations use, so an element's center can be clicked directly.
 * Every action is a real pointer or keyboard event; `dryrun` builds the events without posting
 * them, which is how the bridging is checked without touching the screen.
 */
export const JXA_HELPERS = `
ObjC.import('ApplicationServices');
ObjC.import('AppKit');
ObjC.import('CoreGraphics');
function attr(el, name) {
  var ref = Ref();
  if ($.AXUIElementCopyAttributeValue(el, $(name), ref) !== 0) return null;
  return ObjC.castRefToObject(ref[0]);
}
function plain(el, name) {
  var value = attr(el, name);
  if (value == null) return null;
  try { var unwrapped = ObjC.unwrap(value); return typeof unwrapped === 'object' ? null : unwrapped; } catch (error) { return null; }
}
function pair(el, name) {
  var value = attr(el, name);
  if (value == null) return null;
  var match = ObjC.unwrap(value.description).match(/(?:x|w):(-?[\\d.]+) (?:y|h):(-?[\\d.]+)/);
  return match ? [Number(match[1]), Number(match[2])] : null;
}
function children(el) {
  var list = attr(el, 'AXChildren'), out = [];
  if (!list) return out;
  for (var i = 0; i < list.count; i++) out.push(list.objectAtIndex(i));
  return out;
}
function center(el) {
  var p = pair(el, 'AXPosition'), s = pair(el, 'AXSize');
  return p && s && s[0] > 0 && s[1] > 0 ? [p[0] + s[0] / 2, p[1] + s[1] / 2] : null;
}
// The app macOS asks about for Accessibility: the one that launched the helper (Terminal, iTerm),
// which marks its child processes with its bundle ID. Under the LaunchAgent nothing marks it, and
// macOS asks about the node binary itself (empty here).
function hostApp() {
  var env = $.NSProcessInfo.processInfo.environment, id = env.objectForKey('__CFBundleIdentifier');
  var url = id && !id.isNil() ? $.NSWorkspace.sharedWorkspace.URLForApplicationWithBundleIdentifier(id) : null;
  if (url && !url.isNil()) return ObjC.unwrap(url.URLByDeletingPathExtension.lastPathComponent);
  var term = env.objectForKey('TERM_PROGRAM');
  return term && !term.isNil() ? ObjC.unwrap(term).replace(/^Apple_/, '').replace(/\.app$/, '') : '';
}
function trusted() {
  if (!$.AXIsProcessTrusted()) throw new Error('LOCALROUTER_NO_ACCESSIBILITY:' + hostApp() + ':');
}
// The AX root of one app process. With Accessibility granted, an app can still refuse every request
// (kAXErrorAPIDisabled / CannotComplete): the app asks macOS whether the caller is trusted, and macOS
// refuses when it cannot identify the app itself, as when the program file it runs from was deleted
// while it was open (on 2026-09-26 a disk cleanup removed the owner's Chrome's code-sign clone). That
// is reported as the app's refusal, not as a missing control or a missing permission.
function axApp(pid) {
  var app = $.AXUIElementCreateApplication(pid), ref = Ref(), code = $.AXUIElementCopyAttributeValue(app, $('AXRole'), ref);
  if (code === -25211 || code === -25204) {
    var running = $.NSRunningApplication.runningApplicationWithProcessIdentifier(pid);
    throw new Error('LOCALROUTER_UNREACHABLE:' + (running && !running.isNil() ? ObjC.unwrap(running.localizedName) : 'The app') + ':');
  }
  return app;
}
function post(event, dryrun) { if (!dryrun) $.CGEventPost($.kCGHIDEventTap, event); }
function mouse(type, point, dryrun) {
  var event = $.CGEventCreateMouseEvent($(), type, $.CGPointMake(point[0], point[1]), $.kCGMouseButtonLeft);
  $.CGEventSetIntegerValueField(event, $.kCGMouseEventClickState, 1);
  post(event, dryrun);
}
// Moves the real pointer onto the point first so hover-dependent controls are ready for the click.
function clickAt(point, dryrun) {
  mouse($.kCGEventMouseMoved, point, dryrun);
  if (!dryrun) delay(0.08);
  mouse($.kCGEventLeftMouseDown, point, dryrun);
  mouse($.kCGEventLeftMouseUp, point, dryrun);
}
`

// The main window is used rather than "window 1" because apps often keep small floating panels
// (input-method badges, palettes) ahead of the real window in the accessibility order.
const FRONT = `
function front() {
  trusted();
  var running = $.NSWorkspace.sharedWorkspace.frontmostApplication;
  if (!running || running.isNil()) throw new Error('LOCALROUTER_NO_APP');
  var app = axApp(running.processIdentifier);
  var win = attr(app, 'AXMainWindow') || attr(app, 'AXFocusedWindow');
  if (!win) { var windows = attr(app, 'AXWindows'); win = windows && windows.count ? windows.objectAtIndex(0) : null; }
  return { name: ObjC.unwrap(running.localizedName) || '', win: win };
}
`

const OBSERVE_SCRIPT = `${JXA_HELPERS}${FRONT}
var ROLES = {
  AXButton: 'button', AXLink: 'link', AXTextField: 'textbox', AXTextArea: 'textbox', AXComboBox: 'textbox',
  AXSearchField: 'textbox', AXRadioButton: 'tab', AXTab: 'tab', AXRow: 'row', AXCell: 'row', AXMenuItem: 'menuitem',
  AXMenuButton: 'button', AXPopUpButton: 'button', AXCheckBox: 'checkbox'
};
var LEAVES = { AXStaticText: 1, AXImage: 1, AXTextField: 1, AXTextArea: 1, AXSearchField: 1, AXButton: 1, AXMenuButton: 1,
  AXPopUpButton: 1, AXCheckBox: 1, AXRadioButton: 1, AXScrollBar: 1, AXValueIndicator: 1, AXComboBox: 1 };
function clip(text, max) {
  text = String(text == null ? '' : text).replace(/\\s+/g, ' ').trim();
  return text.length > max ? text.slice(0, max - 1) + '\\u2026' : text;
}
// Table rows rarely carry their own title; their label is the text of their cells.
function innerText(el, depth) {
  var out = '', kids = children(el);
  for (var i = 0; i < kids.length && out.length < 100; i++) {
    var text = plain(kids[i], 'AXRole') === 'AXStaticText' ? plain(kids[i], 'AXValue') : depth < 3 ? innerText(kids[i], depth + 1) : '';
    if (text) out += (out ? ' ' : '') + text;
  }
  return out;
}
function run() {
  var target = front();
  if (!target.win) return JSON.stringify({ app: target.name, title: '', controls: [], text: '' });
  var win = target.win;
  var wp = pair(win, 'AXPosition'), ws = pair(win, 'AXSize');
  var left = wp[0], top = wp[1], right = wp[0] + ws[0], bottom = wp[1] + ws[1];
  var queue = [{ el: win, path: [], depth: 0 }], visited = 0, controls = [], text = '';
  while (queue.length && visited < 2500 && controls.length < 150) {
    var node = queue.shift(), kids = children(node.el);
    for (var i = 0; i < kids.length && visited < 2500 && controls.length < 150; i++) {
      visited++;
      var el = kids[i], role = plain(el, 'AXRole'), p = pair(el, 'AXPosition'), s = pair(el, 'AXSize');
      var sized = p && s && s[0] > 0 && s[1] > 0;
      if (sized && (p[0] >= right || p[1] >= bottom || p[0] + s[0] <= left || p[1] + s[1] <= top)) continue;
      var path = node.path.concat([i]);
      if (role === 'AXStaticText') {
        var words = plain(el, 'AXValue');
        if (text.length < 1500 && words) text += (text ? ' ' : '') + clip(words, 1500);
      } else if (ROLES[role] && sized) {
        var kind = ROLES[role], value = plain(el, 'AXValue');
        var label = plain(el, 'AXTitle') || plain(el, 'AXDescription') || (kind === 'textbox' ? '' : value) ||
          (kind === 'row' ? innerText(el, 0) : '') ||
          (plain(el, 'AXSubrole') ? plain(el, 'AXRoleDescription') : '') || plain(el, 'AXHelp') || '';
        label = clip(label, 100);
        if (kind !== 'row' || label) {
          var control = { ref: path.join('.'), role: kind, label: label, x: p[0], y: p[1] };
          if (kind === 'textbox') {
            control.value = clip(value, 200);
            if (plain(el, 'AXFocused') === true) control.focused = true;
          }
          controls.push(control);
        }
      }
      if (!LEAVES[role] && node.depth + 1 < 20) queue.push({ el: el, path: path, depth: node.depth + 1 });
    }
  }
  controls.sort(function (a, b) { return Math.round(a.y / 8) - Math.round(b.y / 8) || a.x - b.x; });
  controls.forEach(function (control) { delete control.x; delete control.y; });
  return JSON.stringify({ app: target.name, title: plain(win, 'AXTitle') || '', controls: controls, text: text.slice(0, 1500) });
}`

/**
 * The frontmost app's front window from the window server's list, which needs no accessibility: its
 * id, bounds (global, top-left-origin points, the space clicks use), and title (the title needs Screen
 * Recording access, which is also what a screenshot of it needs).
 */
const WINDOW_LIST = `
function frontWindow() {
  var running = $.NSWorkspace.sharedWorkspace.frontmostApplication;
  if (!running || running.isNil()) throw new Error('LOCALROUTER_NO_APP');
  var pid = running.processIdentifier;
  var list = ObjC.deepUnwrap(ObjC.castRefToObject($.CGWindowListCopyWindowInfo($.kCGWindowListOptionOnScreenOnly | $.kCGWindowListExcludeDesktopElements, 0))) || [];
  // The frontmost real window, not a small helper window the app keeps in the same layer (Chrome has a
  // 66×20 one over its tab strip): a screenshot of that shows nothing to read.
  var win = list.filter(function (w) { return w.kCGWindowOwnerPID === pid && w.kCGWindowLayer === 0 && w.kCGWindowBounds.Width >= 200 && w.kCGWindowBounds.Height >= 150; })[0];
  var b = win && win.kCGWindowBounds;
  return { app: ObjC.unwrap(running.localizedName) || '', window: win ? { id: win.kCGWindowNumber, x: b.X, y: b.Y, w: b.Width, h: b.Height, title: String(win.kCGWindowName || '') } : null };
}
`

/**
 * One action on the frontmost main window, always through the real pointer and keyboard. The ref is
 * re-resolved here so the click lands where the control is now, not where observe() saw it. A ref
 * "v:x,y" is a spot read from a screenshot (the app refused accessibility), clicked where it was seen.
 */
const ACT_SCRIPT = `${JXA_HELPERS}${FRONT}${WINDOW_LIST}
function resolve(win, ref) {
  var el = win, parts = String(ref).split('.');
  for (var i = 0; i < parts.length; i++) {
    var kids = children(el), index = Number(parts[i]);
    if (!(index >= 0) || index >= kids.length) throw new Error('LOCALROUTER_GONE');
    el = kids[index];
  }
  return el;
}
// The window's middle, from accessibility when the app answers, else from the window list.
function middleOfWindow() {
  try {
    var target = front();
    if (!target.win) throw new Error('LOCALROUTER_NO_WINDOW');
    return center(target.win);
  } catch (error) {
    if (String(error.message).indexOf('LOCALROUTER_UNREACHABLE') < 0) throw error;
    var seen = frontWindow().window;
    if (!seen) throw new Error('LOCALROUTER_NO_WINDOW');
    return [seen.x + seen.w / 2, seen.y + seen.h / 2];
  }
}
function run(argv) {
  var op = argv[0], payload = JSON.parse(argv[1] || '{}'), dryrun = payload.dryrun === true;
  var events = Application('System Events');
  trusted();
  if (op === 'press') {
    if (!dryrun) events.keyCode(payload.code);
    return 'ok';
  }
  if (op === 'scroll') {
    var middle = middleOfWindow();
    if (!middle) throw new Error('LOCALROUTER_NO_WINDOW');
    var up = payload.direction === 'up' || payload.direction === 'top';
    var steps = payload.direction === 'top' || payload.direction === 'bottom' ? 20 : 3;
    // Scroll events go to the view under the pointer, so it is moved over the window first.
    mouse($.kCGEventMouseMoved, middle, dryrun);
    for (var s = 0; s < steps; s++) post($.CGEventCreateScrollWheelEvent($(), $.kCGScrollEventUnitLine, 1, up ? 15 : -15), dryrun);
    return JSON.stringify({ x: middle[0], y: middle[1] });
  }
  var spot = /^v:(-?[\\d.]+),(-?[\\d.]+)$/.exec(String(payload.ref || '')), point;
  if (spot) point = [Number(spot[1]), Number(spot[2])];
  else {
    var win = front().win;
    if (!win) throw new Error('LOCALROUTER_NO_WINDOW');
    point = center(resolve(win, payload.ref));
  }
  if (!point) throw new Error('LOCALROUTER_GONE');
  clickAt(point, dryrun);
  // Where the pointer was left, for the local stop: a later move away from it is the person's.
  var left = JSON.stringify({ x: point[0], y: point[1] });
  if (op === 'click') return left;
  if (op === 'type') {
    if (dryrun) return left;
    delay(0.1);
    // Key code 0 is "a": command-A selects the field's current text so typing replaces it.
    events.keyCode(0, { using: 'command down' });
    events.keystroke(payload.text);
    return left;
  }
  throw new Error('LOCALROUTER_UNKNOWN_OP');
}`

const FRONT_WINDOW_SCRIPT = `ObjC.import('AppKit');
ObjC.import('CoreGraphics');
${WINDOW_LIST}
function run() { return JSON.stringify(frontWindow()); }`

type Window = { id: number; x: number; y: number; w: number; h: number; title: string }
type FrontWindow = { app: string; window: Window | null }

function frontWindowShape(value: unknown): FrontWindow {
  const front = object(value, 'front window')
  return {
    app: string(front.app, 'front window app'),
    window: nullable(front.window, 'front window', (raw, what) => {
      const win = object(raw, what)
      return { id: number(win.id, `${what} id`), x: number(win.x, `${what} x`), y: number(win.y, `${what} y`), w: number(win.w, `${what} w`), h: number(win.h, `${what} h`), title: string(win.title, `${what} title`) }
    }),
  }
}

/**
 * On-device text recognition (Vision) of a screenshot: each line of text with its box, normalized to
 * the image with a bottom-left origin. argv: the image's path.
 */
const OCR_SCRIPT = `ObjC.import('Vision');
ObjC.import('Foundation');
function run(argv) {
  var request = $.VNRecognizeTextRequest.alloc.init;
  request.recognitionLevel = 0;
  request.usesLanguageCorrection = true;
  var handler = $.VNImageRequestHandler.alloc.initWithURLOptions($.NSURL.fileURLWithPath(argv[0]), $({}));
  if (!handler.performRequestsError($([request]), Ref())) throw new Error('LOCALROUTER_OCR_FAILED');
  var results = request.results, out = [];
  for (var i = 0; results && i < results.count; i++) {
    var line = results.objectAtIndex(i), box = line.boundingBox, best = line.topCandidates(1).firstObject;
    if (best && !best.isNil()) out.push({ text: ObjC.unwrap(best.string), x: box.origin.x, y: box.origin.y, w: box.size.width, h: box.size.height });
  }
  return JSON.stringify(out);
}`
export type OcrLine = { text: string; x: number; y: number; w: number; h: number }

function ocrShape(value: unknown): OcrLine[] {
  return array(value, 'text lines', (raw, what) => {
    const line = object(raw, what)
    return { text: string(line.text, `${what} text`), x: number(line.x, `${what} x`), y: number(line.y, `${what} y`), w: number(line.w, `${what} w`), h: number(line.h, `${what} h`) }
  })
}

/**
 * Text read from a screenshot of the window, as clickable spots ("v:x,y" at each line's middle, in
 * the global points clicks use), top to bottom, then left to right.
 */
export function visualControls(lines: OcrLine[], window: Window): ScreenControl[] {
  return lines
    .filter((line) => /[a-z0-9]/i.test(line.text) && line.text.trim().length > 1)
    .map((line) => ({ label: line.text.trim().slice(0, 100), x: Math.round(window.x + (line.x + line.w / 2) * window.w), y: Math.round(window.y + (1 - line.y - line.h / 2) * window.h) }))
    .sort((a, b) => Math.round(a.y / 8) - Math.round(b.y / 8) || a.x - b.x)
    .slice(0, 150)
    .map(({ label, x, y }) => ({ ref: `v:${x},${y}`, role: 'text' as const, label }))
}

/** macOS virtual key codes for the keys a step may press. */
const KEY_CODES = { enter: 36, escape: 53, tab: 48, space: 49 } as const

/** The JSON handed to the act script as argv: key codes for presses, otherwise the action's own fields. */
function actionPayload(action: ScreenAction): object {
  if (action.kind === 'press') return { code: KEY_CODES[action.key] }
  if (action.kind === 'scroll') return { direction: action.direction }
  if (action.kind === 'type') return { ref: action.ref, text: action.text }
  return { ref: action.ref }
}

// -1743 is the separate Automation prompt (permission to send Apple Events to System Events,
// used for key presses and typed text). A refusal from the one app being read (LOCALROUTER_UNREACHABLE)
// is not a missing permission: Accessibility is on, and that app has to be reopened.
const AUTOMATION = /-1743|Not authorized to send Apple events/i
const ACCESSIBILITY = /LOCALROUTER_NO_ACCESSIBILITY|-1719|assistive access|not allowed/i
export const UNREACHABLE = /LOCALROUTER_UNREACHABLE:([^:]+):/

/** The app macOS asks about in Privacy & Security: the one running the helper, named by the script when it can. */
function hostFrom(message: string) {
  return message.match(/LOCALROUTER_NO_ACCESSIBILITY:([^:]+):/)?.[1]?.trim() || 'the node binary that runs LocalRouter Desktop (for example /opt/homebrew/bin/node)'
}

/** Turns an access failure from a script into an error that says which app and which setting to change. */
export function macAccessError(message: string): Error | null {
  const unreachable = message.match(UNREACHABLE)
  if (unreachable) {
    const app = unreachable[1].trim()
    return new Error(`${app} is refusing to show its window to other apps: macOS can no longer verify the running ${app} (a file it runs from was changed or removed while it was open). Quitting and reopening ${app} fixes this.`)
  }
  if (AUTOMATION.test(message)) {
    return new Error('LocalRouter Desktop needs permission to control System Events. Allow the node binary that runs it under System Settings › Privacy & Security › Automation, then try again.')
  }
  if (ACCESSIBILITY.test(message)) {
    return new Error(`LocalRouter Desktop needs Accessibility access to see and use apps. Allow ${hostFrom(message)} in System Settings › Privacy & Security › Accessibility, then try again.`)
  }
  return null
}

// Screen Recording is only needed for the screenshot fallback (apps that refuse accessibility). Its check
// is guarded: should the bridge not expose it, Accessibility is still reported.
const ACCESS_SCRIPT = `${JXA_HELPERS}
function run() {
  var screen = false;
  try { screen = Boolean($.CGPreflightScreenCaptureAccess()); } catch (error) { screen = false; }
  return JSON.stringify({ accessibility: Boolean($.AXIsProcessTrusted()), screenRecording: screen, host: hostApp() });
}`

export type MacAccess = { accessibility: boolean; screenRecording: boolean; host: string }

/** Whether the process running the helper has Accessibility and Screen Recording access, and the app macOS attributes it to. */
export async function macAccess(runner: CommandRunner = jxaRunner): Promise<MacAccess> {
  const access = object(json((await runner.run(...jxa(ACCESS_SCRIPT))).trim(), 'access check'), 'access check')
  return { accessibility: boolean(access.accessibility, 'accessibility'), screenRecording: boolean(access.screenRecording, 'screenRecording'), host: string(access.host, 'host') }
}

/** Where the pointer is now (global, top-left-origin points), without moving it or needing any permission. */
const POINTER_SCRIPT = `ObjC.import('CoreGraphics');
function run() {
  var point = $.CGEventGetLocation($.CGEventCreate($()));
  return JSON.stringify({ x: point.x, y: point.y });
}`

/** A point printed by a script (`{"x":1,"y":2}`). */
export function pointShape(value: unknown, what = 'point'): Point {
  const point = object(value, what)
  return { x: number(point.x, `${what} x`), y: number(point.y, `${what} y`) }
}

function controlShape(value: unknown, what: string): ScreenControl {
  const raw = object(value, what)
  const control: ScreenControl = { ref: string(raw.ref, `${what} ref`), role: oneOf(raw.role, CONTROL_ROLES, `${what} role`), label: string(raw.label, `${what} label`) }
  const text = optional(raw.value, `${what} value`, string)
  const focused = optional(raw.focused, `${what} focused`, boolean)
  if (text !== undefined) control.value = text
  if (focused !== undefined) control.focused = focused
  return control
}

function observationShape(value: unknown): Omit<Observation, 'backend'> {
  const raw = object(value, 'observation')
  return { app: string(raw.app, 'observation app'), title: string(raw.title, 'observation title'), controls: array(raw.controls, 'observation controls', controlShape), text: string(raw.text, 'observation text') }
}

/** Reads the frontmost app's main window through accessibility and drives it with the real pointer and keyboard. */
export class DesktopBackend implements ScreenBackend {
  readonly kind = 'desktop'
  private readonly runner: CommandRunner

  constructor(runner: CommandRunner = jxaRunner) {
    this.runner = runner
  }

  async observe(): Promise<Observation> {
    let output: string
    try {
      output = await this.runner.run(...jxa(OBSERVE_SCRIPT))
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      if (UNREACHABLE.test(message)) return this.observeScreenshot(macAccessError(message) ?? new Error(message))
      throw explain(message) ?? error
    }
    return { backend: 'desktop', ...observationShape(json(output.trim(), 'observation')) }
  }

  /**
   * For an app that refuses accessibility: its front window's text, read on this Mac from a screenshot
   * (Vision), as spots to click. The screenshot is deleted at once. `refused` is thrown when nothing
   * could be read (no window, or no Screen Recording access, which leaves the screenshot blank).
   */
  private async observeScreenshot(refused: Error): Promise<Observation> {
    const front = frontWindowShape(json((await this.runner.run(...jxa(FRONT_WINDOW_SCRIPT))).trim(), 'front window'))
    if (!front.window) throw refused
    const window = front.window
    const file = join(tmpdir(), `localrouter-screen-${process.pid}-${Date.now()}.png`)
    let controls: ScreenControl[]
    try {
      await this.runner.run('screencapture', ['-x', '-o', `-l${window.id}`, file])
      controls = visualControls(ocrShape(json((await this.runner.run(...jxa(OCR_SCRIPT, [file]))).trim() || '[]', 'text lines')), window)
    } catch {
      throw refused
    } finally {
      await rm(file, { force: true })
    }
    if (!controls.length) throw refused
    return { backend: 'desktop', app: front.app, title: window.title, controls, text: controls.map((control) => control.label).join(' ').slice(0, 1_500), visual: true }
  }

  /** Acts, and reports where a click, typing, or scroll left the pointer. */
  async act(action: ScreenAction): Promise<ActResult> {
    const output = (await this.call(ACT_SCRIPT, [action.kind, JSON.stringify(actionPayload(action))])).trim()
    if (!output.startsWith('{')) return {}
    try {
      return { pointer: pointShape(JSON.parse(output)) }
    } catch {
      return {}
    }
  }

  async pointer(): Promise<Point> {
    return pointShape(json((await this.call(POINTER_SCRIPT, [])).trim(), 'pointer'), 'pointer')
  }

  private async call(script: string, argv: string[]): Promise<string> {
    try {
      return await this.runner.run(...jxa(script, argv))
    } catch (error) {
      throw explain(error instanceof Error ? error.message : String(error)) ?? error
    }
  }
}

/** A script failure in words a person can act on, or null to pass it through. */
function explain(message: string): Error | null {
  if (message.includes('LOCALROUTER_GONE')) return new Error('That control is gone; look again.')
  if (message.includes('LOCALROUTER_NO_APP')) return new Error('No app is in front on the Mac.')
  if (message.includes('LOCALROUTER_NO_WINDOW')) return new Error('The app in front has no open window.')
  return macAccessError(message)
}
