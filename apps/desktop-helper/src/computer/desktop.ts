// Ported from SideBud (github.com/jayleaton/sidebud @70da540, apps/companion/src/computer/use/desktop.ts), reused in LocalRouter with the owner's permission.
import { readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { array, boolean, json, nullable, number, object, oneOf, optional, string } from '../validate.ts'
import { execRunner } from './runner.ts'
import type { CommandRunner } from './runner.ts'
import { CONTROL_ROLES, fitScale, SCREENSHOT_MAX_EDGE } from './types.ts'
import type { ActOptions, ActResult, Observation, ObserveOptions, Point, PointerReading, Rect, ScreenAction, ScreenBackend, ScreenControl, Screenshot, ScreenshotOptions } from './types.ts'

/**
 * Default runner for the computer-use backends. osascript prints the script's return value on
 * stdout; on failure only stderr is kept, because execFile's own message repeats the whole script
 * and the error patterns below would otherwise match the script source instead of the real error.
 * A stop (`signal`) kills the script under way.
 */
export const jxaRunner: CommandRunner = execRunner({ timeoutMs: 15_000, maxBuffer: 16 * 1024 * 1024, slow: 'The Mac (osascript)' })

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
// A hung app answers every AX call only after the messaging timeout (6 s by default), so a walk of a few hundred
// elements would outlast the script's own limit: 1.5 s an answer, and the walk keeps its own time budget too.
function axApp(pid) {
  var app = $.AXUIElementCreateApplication(pid), ref = Ref();
  $.AXUIElementSetMessagingTimeout(app, 1.5);
  var code = $.AXUIElementCopyAttributeValue(app, $('AXRole'), ref);
  if (code === -25211 || code === -25204) {
    var running = $.NSRunningApplication.runningApplicationWithProcessIdentifier(pid);
    throw new Error('LOCALROUTER_UNREACHABLE:' + (running && !running.isNil() ? ObjC.unwrap(running.localizedName) : 'The app') + ':');
  }
  return app;
}
// Every display's bounds in global top-left-origin points. NSScreen frames have a bottom-left origin at the
// main display (the first screen, with the menu bar), so y is flipped against its height.
function displays() {
  var screens = $.NSScreen.screens, out = [];
  if (!screens.count) return out;
  var mainHeight = screens.objectAtIndex(0).frame.size.height;
  for (var i = 0; i < screens.count; i++) {
    var f = screens.objectAtIndex(i).frame;
    out.push({ x: f.origin.x, y: mainHeight - f.origin.y - f.size.height, w: f.size.width, h: f.size.height });
  }
  return out;
}
// A click outside every display would land on the nearest edge instead (CGEvent clamps it): never clicked.
function onScreen(point) {
  var list = displays();
  if (!list.length) return true;
  for (var i = 0; i < list.length; i++) {
    var d = list[i];
    if (point[0] >= d.x && point[0] < d.x + d.w && point[1] >= d.y && point[1] < d.y + d.h) return true;
  }
  return false;
}
function post(event, dryrun) { if (!dryrun) $.CGEventPost($.kCGHIDEventTap, event); }
function mouse(type, point, dryrun) {
  var event = $.CGEventCreateMouseEvent($(), type, $.CGPointMake(point[0], point[1]), $.kCGMouseButtonLeft);
  $.CGEventSetIntegerValueField(event, $.kCGMouseEventClickState, 1);
  post(event, dryrun);
}
// Moves the real pointer onto the point first so hover-dependent controls are ready for the click.
function clickAt(point, dryrun) {
  if (!onScreen(point)) throw new Error('LOCALROUTER_OFFSCREEN');
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
  return { name: ObjC.unwrap(running.localizedName) || '', win: win, app: app };
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
// The page a browser window shows: the URL of its web area (Safari, Chrome, Electron), else the window's document.
function urlOf(el) {
  var value = attr(el, 'AXURL');
  if (value == null) return '';
  try { return ObjC.unwrap(value.absoluteString) || ''; } catch (error) { return ''; }
}
var WALK_MS = 4000;
function run() {
  var target = front();
  if (!target.win) return JSON.stringify({ app: target.name, title: '', controls: [], text: '' });
  var win = target.win, started = Date.now(), truncated = false;
  var wp = pair(win, 'AXPosition'), ws = pair(win, 'AXSize');
  var left = wp[0], top = wp[1], right = wp[0] + ws[0], bottom = wp[1] + ws[1];
  var queue = [{ el: win, path: [], depth: 0 }], visited = 0, controls = [], text = '', url = '';
  while (queue.length) {
    if (visited >= 2500 || controls.length >= 150 || Date.now() - started > WALK_MS) { truncated = true; break; }
    var node = queue.shift(), kids = children(node.el);
    for (var i = 0; i < kids.length; i++) {
      if (visited >= 2500 || controls.length >= 150) { truncated = true; break; }
      visited++;
      var el = kids[i], role = plain(el, 'AXRole'), p = pair(el, 'AXPosition'), s = pair(el, 'AXSize');
      var sized = p && s && s[0] > 0 && s[1] > 0;
      if (sized && (p[0] >= right || p[1] >= bottom || p[0] + s[0] <= left || p[1] + s[1] <= top)) continue;
      var path = node.path.concat([i]);
      if (role === 'AXWebArea' && !url) url = urlOf(el);
      if (role === 'AXStaticText') {
        var words = plain(el, 'AXValue');
        if (text.length < 1500 && words) text += (text ? ' ' : '') + clip(words, 1500);
      } else if (ROLES[role] && sized) {
        // A password field (subrole AXSecureTextField) is listed so it can be typed into, never with its value.
        var kind = ROLES[role], value = plain(el, 'AXSubrole') === 'AXSecureTextField' ? '' : plain(el, 'AXValue');
        var label = plain(el, 'AXTitle') || plain(el, 'AXDescription') || (kind === 'textbox' ? '' : value) ||
          (kind === 'row' ? innerText(el, 0) : '') ||
          (plain(el, 'AXSubrole') ? plain(el, 'AXRoleDescription') : '') || plain(el, 'AXHelp') || '';
        label = clip(label, 100);
        if (kind !== 'row' || label) {
          var control = { ref: path.join('.'), role: kind, label: label, bounds: { x: p[0], y: p[1], w: s[0], h: s[1] } };
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
  if (!url) { var doc = plain(win, 'AXDocument'); if (typeof doc === 'string' && /^(https?|file):/.test(doc)) url = doc; }
  controls.sort(function (a, b) { return Math.round(a.bounds.y / 8) - Math.round(b.bounds.y / 8) || a.bounds.x - b.bounds.x; });
  var out = { app: target.name, title: plain(win, 'AXTitle') || '', controls: controls, text: text.slice(0, 1500), window: { x: left, y: top, w: ws[0], h: ws[1] } };
  if (url) out.url = url;
  if (truncated) out.truncated = true;
  return JSON.stringify(out);
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
var TEXT_ROLES = { AXTextField: 1, AXTextArea: 1, AXComboBox: 1, AXSearchField: 1 };
// The scroll view a person would scroll: the largest scroll or web area inside the window (not a sidebar), else null.
function scrollArea(win) {
  var queue = [{ el: win, depth: 0 }], seen = 0, best = null, bestArea = 0;
  while (queue.length && seen < 400) {
    var node = queue.shift(), kids = children(node.el);
    for (var i = 0; i < kids.length && seen < 400; i++) {
      seen++;
      var el = kids[i], role = plain(el, 'AXRole');
      if (role === 'AXScrollArea' || role === 'AXWebArea') {
        var s = pair(el, 'AXSize'), area = s ? s[0] * s[1] : 0;
        if (area > bestArea) { best = el; bestArea = area; }
      }
      if (node.depth + 1 < 10 && role !== 'AXWebArea') queue.push({ el: el, depth: node.depth + 1 });
    }
  }
  return best ? center(best) : null;
}
// Whether the text box took the typing focus after the click: the app's focused element is it, or sits inside it
// (a combo box's own field), and is a text element.
function hasFocus(app, el) {
  var focused = attr(app, 'AXFocusedUIElement');
  if (!focused) return false;
  if (!TEXT_ROLES[plain(focused, 'AXRole')] && plain(el, 'AXFocused') !== true) return false;
  var c = center(focused), p = pair(el, 'AXPosition'), s = pair(el, 'AXSize');
  return Boolean(c && p && s && c[0] >= p[0] - 2 && c[0] <= p[0] + s[0] + 2 && c[1] >= p[1] - 2 && c[1] <= p[1] + s[1] + 2);
}
// The app's own menu item for a Command shortcut (Edit › Select All is ⌘A, Paste ⌘V), pressed through
// accessibility. Unlike a key code, this does not depend on the keyboard layout: key code 0 with Command is ⌘A on a
// US layout but ⌘Q (quit) on AZERTY.
function menuCommand(app, letter) {
  var bar = attr(app, 'AXMenuBar');
  if (!bar) return false;
  var menus = children(bar);
  for (var i = 0; i < menus.length; i++) {
    var lists = children(menus[i]);
    for (var j = 0; j < lists.length; j++) {
      var items = children(lists[j]);
      for (var k = 0; k < items.length && k < 80; k++) {
        if (String(plain(items[k], 'AXMenuItemCmdChar') || '').toUpperCase() === letter && plain(items[k], 'AXMenuItemCmdModifiers') === 0 && plain(items[k], 'AXEnabled') !== false) {
          return $.AXUIElementPerformAction(items[k], $('AXPress')) === 0;
        }
      }
    }
  }
  return false;
}
// A Command shortcut by its character: the menu item, else System Events, which maps the character to the key that
// types it on the current layout.
function command(app, letter, events) {
  if (menuCommand(app, letter)) return;
  events.keystroke(letter.toLowerCase(), { using: 'command down' });
}
// Pastes text through the pasteboard, marked transient and concealed so clipboard managers do not keep it, then
// puts back what the pasteboard held. For text the keyboard cannot type as is (other scripts, emoji, line breaks).
function paste(app, text, events) {
  var pb = $.NSPasteboard.generalPasteboard, saved = [], items = pb.pasteboardItems;
  for (var i = 0; items && i < items.count; i++) {
    var item = items.objectAtIndex(i), types = item.types, entries = [];
    for (var j = 0; j < types.count; j++) {
      var type = types.objectAtIndex(j), data = item.dataForType(type);
      if (data && !data.isNil()) entries.push([type, data]);
    }
    saved.push(entries);
  }
  pb.clearContents;
  var mine = $.NSPasteboardItem.alloc.init;
  mine.setStringForType($(text), $.NSPasteboardTypeString);
  mine.setStringForType($(''), $('org.nspasteboard.TransientType'));
  mine.setStringForType($(''), $('org.nspasteboard.ConcealedType'));
  pb.writeObjects($([mine]));
  try {
    command(app, 'V', events);
    delay(0.5);
  } finally {
    pb.clearContents;
    var back = saved.map(function (entries) {
      var it = $.NSPasteboardItem.alloc.init;
      entries.forEach(function (entry) { it.setDataForType(entry[1], entry[0]); });
      return it;
    });
    if (back.length) pb.writeObjects($(back));
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
    var target = null;
    if (payload.ref) {
      var w = front().win;
      if (!w) throw new Error('LOCALROUTER_NO_WINDOW');
      target = center(resolve(w, payload.ref));
      if (!target) throw new Error('LOCALROUTER_GONE');
    } else {
      try { var fw = front().win; target = fw ? scrollArea(fw) : null; } catch (error) { target = null; }
      target = target || middleOfWindow();
    }
    if (!target) throw new Error('LOCALROUTER_NO_WINDOW');
    if (!onScreen(target)) throw new Error('LOCALROUTER_OFFSCREEN');
    var up = payload.direction === 'up' || payload.direction === 'top';
    var steps = payload.direction === 'top' || payload.direction === 'bottom' ? 20 : 3;
    // Scroll events go to the view under the pointer, so it is moved over the scroll area first.
    mouse($.kCGEventMouseMoved, target, dryrun);
    for (var n = 0; n < steps; n++) post($.CGEventCreateScrollWheelEvent($(), $.kCGScrollEventUnitLine, 1, up ? 15 : -15), dryrun);
    return JSON.stringify({ x: target[0], y: target[1] });
  }
  if (op === 'keys') {
    if (/[\\r\\n]/.test(payload.text)) throw new Error('LOCALROUTER_NEWLINE');
    if (!dryrun) events.keystroke(payload.text);
    return 'ok';
  }
  var spot = /^v:(-?[\\d.]+),(-?[\\d.]+)$/.exec(String(payload.ref || '')), point, el = null, ax = null;
  if (spot) point = [Number(spot[1]), Number(spot[2])];
  else {
    var here = front();
    if (!here.win) throw new Error('LOCALROUTER_NO_WINDOW');
    ax = here.app;
    el = resolve(here.win, payload.ref);
    point = center(el);
  }
  if (op === 'value') return JSON.stringify({ value: el && plain(el, 'AXSubrole') !== 'AXSecureTextField' ? plain(el, 'AXValue') : null });
  if (op === 'paste') {
    if (!dryrun) paste(ax || front().app, payload.text, events);
    return 'ok';
  }
  if (!point) throw new Error('LOCALROUTER_GONE');
  clickAt(point, dryrun);
  // Where the pointer was left, for the local stop: a later move away from it is the person's.
  var left = { x: point[0], y: point[1] };
  if (op === 'click') return JSON.stringify(left);
  if (op === 'focus') {
    if (dryrun || !el) return JSON.stringify(left);
    delay(0.15);
    if (!hasFocus(ax, el)) {
      $.AXUIElementSetAttributeValue(el, $('AXFocused'), $.NSNumber.numberWithBool(true));
      delay(0.1);
      if (!hasFocus(ax, el)) throw new Error('LOCALROUTER_NOT_FOCUSED');
    }
    // The text it holds is selected (to be replaced) only when there is any.
    var current = plain(el, 'AXValue');
    if (typeof current === 'string' && current.length) command(ax, 'A', events);
    return JSON.stringify(left);
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
  if (action.kind === 'scroll') return action.ref ? { direction: action.direction, ref: action.ref } : { direction: action.direction }
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

/**
 * Where the pointer is now (global, top-left-origin points) and how long ago the left button last went down, without
 * moving it or needing any permission.
 */
const POINTER_SCRIPT = `ObjC.import('CoreGraphics');
function run() {
  var point = $.CGEventGetLocation($.CGEventCreate($()));
  var click = $.CGEventSourceSecondsSinceLastEventType($.kCGEventSourceStateHIDSystemState, $.kCGEventLeftMouseDown);
  return JSON.stringify({ x: point.x, y: point.y, since_click_s: click });
}`

/** A point printed by a script (`{"x":1,"y":2}`). */
export function pointShape(value: unknown, what = 'point'): Point {
  const point = object(value, what)
  return { x: number(point.x, `${what} x`), y: number(point.y, `${what} y`) }
}

function rectShape(value: unknown, what: string): Rect {
  const raw = object(value, what)
  return { x: number(raw.x, `${what} x`), y: number(raw.y, `${what} y`), w: number(raw.w, `${what} w`), h: number(raw.h, `${what} h`) }
}

function controlShape(value: unknown, what: string): ScreenControl {
  const raw = object(value, what)
  const control: ScreenControl = { ref: string(raw.ref, `${what} ref`), role: oneOf(raw.role, CONTROL_ROLES, `${what} role`), label: string(raw.label, `${what} label`) }
  const text = optional(raw.value, `${what} value`, string)
  const focused = optional(raw.focused, `${what} focused`, boolean)
  const bounds = optional(raw.bounds, `${what} bounds`, rectShape)
  if (text !== undefined) control.value = text
  if (focused !== undefined) control.focused = focused
  if (bounds !== undefined) control.bounds = bounds
  return control
}

function observationShape(value: unknown): Omit<Observation, 'backend'> {
  const raw = object(value, 'observation')
  const seen: Omit<Observation, 'backend'> = { app: string(raw.app, 'observation app'), title: string(raw.title, 'observation title'), controls: array(raw.controls, 'observation controls', controlShape), text: string(raw.text, 'observation text') }
  const url = optional(raw.url, 'observation url', string)
  const window = optional(raw.window, 'observation window', rectShape)
  if (url) seen.url = url
  if (window) seen.window = window
  if (optional(raw.truncated, 'observation truncated', boolean)) seen.truncated = true
  return seen
}

/** Every display's bounds (global, top-left-origin points). */
const DISPLAYS_SCRIPT = `${JXA_HELPERS}
function run() { return JSON.stringify(displays()); }`

export function displaysShape(value: unknown): Rect[] {
  return array(value, 'displays', rectShape)
}

/**
 * Text the keyboard types as is: printable ASCII on one line. Anything else (other scripts, emoji, line breaks,
 * where a typed Return would send a message) is pasted through a transient pasteboard instead.
 */
const KEYBOARD_TEXT = /^[\x20-\x7e]{1,400}$/
/** Characters a keystroke script types at once: a stop takes effect between chunks. */
export const KEY_CHUNK = 50
/** Time for a field (a web page's especially) to report what was typed before it is read back. */
const READBACK_MS = 200

/** Splits text into chunks of at most `size` characters, never inside a surrogate pair. */
export function chunks(text: string, size: number): string[] {
  const out: string[] = []
  let at = 0
  while (at < text.length) {
    let end = Math.min(text.length, at + size)
    if (end < text.length && /[\ud800-\udbff]/.test(text[end - 1])) end -= 1
    out.push(text.slice(at, end))
    at = end
  }
  return out
}

const squash = (text: string) => text.replace(/\s+/g, '')

/**
 * Whether typed text is in the field now. A field that cannot be read back (null) is taken at its word; one that
 * reformats what was typed (spaces in a card number) still holds the same characters.
 */
export function landed(value: string | null, text: string): boolean {
  if (value === null) return true
  return squash(value).includes(squash(text))
}

export type DesktopOptions = { sleep?: (ms: number) => Promise<void>; tmp?: () => string }

const pause = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))
const stopped = (signal: AbortSignal) => new Error(typeof signal.reason === 'string' && signal.reason ? signal.reason : 'It was stopped.')

/** Reads the frontmost app's main window through accessibility and drives it with the real pointer and keyboard. */
export class DesktopBackend implements ScreenBackend {
  readonly kind = 'desktop'
  private readonly runner: CommandRunner
  private readonly sleep: (ms: number) => Promise<void>
  private readonly tmp: () => string

  constructor(runner: CommandRunner = jxaRunner, options: DesktopOptions = {}) {
    this.runner = runner
    this.sleep = options.sleep ?? pause
    this.tmp = options.tmp ?? (() => join(tmpdir(), `localrouter-screen-${process.pid}-${Date.now()}.png`))
  }

  async observe(options: ObserveOptions = {}): Promise<Observation> {
    let output: string
    try {
      output = await this.runner.run(...jxa(OBSERVE_SCRIPT), options.signal ? { signal: options.signal } : undefined)
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
    const file = this.tmp()
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
    return { backend: 'desktop', app: front.app, title: window.title, controls, text: controls.map((control) => control.label).join(' ').slice(0, 1_500), visual: true, window: { x: window.x, y: window.y, w: window.w, h: window.h } }
  }

  /** Acts, and reports where a click, typing, or scroll left the pointer. */
  async act(action: ScreenAction, options: ActOptions = {}): Promise<ActResult> {
    const signal = options.signal
    if (signal?.aborted) throw stopped(signal)
    if (action.kind === 'type') return this.type(action.ref, action.text, signal)
    return pointerOf(await this.call(ACT_SCRIPT, [action.kind, JSON.stringify(actionPayload(action))], signal))
  }

  /**
   * Typing: click the box, check it took the focus (else nothing is typed, so a select-all never lands on the page),
   * select what it holds through the app's Edit menu, type plain text in chunks a stop can come between (or paste
   * anything else), then read the box back: text that did not land is a failure, not a step done.
   */
  private async type(ref: string, text: string, signal: AbortSignal | undefined): Promise<ActResult> {
    const left = pointerOf(await this.call(ACT_SCRIPT, ['focus', JSON.stringify({ ref })], signal))
    if (KEYBOARD_TEXT.test(text)) {
      for (const chunk of chunks(text, KEY_CHUNK)) {
        if (signal?.aborted) throw stopped(signal)
        await this.call(ACT_SCRIPT, ['keys', JSON.stringify({ text: chunk })], signal)
      }
    } else {
      await this.call(ACT_SCRIPT, ['paste', JSON.stringify({ ref, text })], signal)
    }
    await this.sleep(READBACK_MS)
    const read = object(json((await this.call(ACT_SCRIPT, ['value', JSON.stringify({ ref })], signal)).trim(), 'typed value'), 'typed value')
    const value = read.value === null || read.value === undefined ? null : string(read.value, 'typed value')
    if (!landed(value, text)) throw new Error('The text did not land in the text box, so it stopped.')
    return left
  }

  async pointer(): Promise<PointerReading> {
    const value = json((await this.call(POINTER_SCRIPT, [])).trim(), 'pointer')
    const point = pointShape(value, 'pointer')
    const since = (value as { since_click_s?: unknown }).since_click_s
    return typeof since === 'number' && Number.isFinite(since) && since >= 0 ? { ...point, sinceClickS: since } : point
  }

  async displays(): Promise<Rect[]> {
    return displaysShape(json((await this.call(DISPLAYS_SCRIPT, [])).trim(), 'displays'))
  }

  /**
   * The front window as a PNG, scaled so its longer edge is at most `maxEdge` pixels (a Retina capture is twice the
   * window's points). Needs Screen Recording: without it macOS returns the desktop instead of the window, so it is
   * checked first. The file is deleted at once.
   */
  async screenshot(options: ScreenshotOptions = {}): Promise<Screenshot> {
    const access = await macAccess(this.runner)
    if (!access.screenRecording) throw new Error('A screenshot needs Screen Recording access. Allow LocalRouter in System Settings › Privacy & Security › Screen Recording, then try again.')
    const front = frontWindowShape(json((await this.call(FRONT_WINDOW_SCRIPT, [], options.signal)).trim(), 'front window'))
    if (!front.window) throw new Error('The app in front has no open window.')
    const window = front.window
    const file = this.tmp()
    try {
      await this.runner.run('screencapture', ['-x', '-o', '-t', 'png', `-l${window.id}`, file], options.signal ? { signal: options.signal } : undefined)
      let size = pixelSize(await this.runner.run('sips', ['-g', 'pixelWidth', '-g', 'pixelHeight', file]))
      const fit = fitScale(size.w, size.h, options.maxEdge ?? SCREENSHOT_MAX_EDGE)
      if (fit < 1) {
        await this.runner.run('sips', ['--resampleWidth', String(Math.max(1, Math.round(size.w * fit))), file])
        size = pixelSize(await this.runner.run('sips', ['-g', 'pixelWidth', '-g', 'pixelHeight', file]))
      }
      const png = new Uint8Array(await readFile(file))
      return { png, width: size.w, height: size.h, region: { x: window.x, y: window.y, w: window.w, h: window.h }, scale: window.w > 0 ? size.w / window.w : 1, app: front.app, title: window.title }
    } finally {
      await rm(file, { force: true })
    }
  }

  private async call(script: string, argv: string[], signal?: AbortSignal): Promise<string> {
    try {
      return await this.runner.run(...jxa(script, argv), signal ? { signal } : undefined)
    } catch (error) {
      if (signal?.aborted) throw stopped(signal)
      throw explain(error instanceof Error ? error.message : String(error)) ?? error
    }
  }
}

/** The pointer position an act script printed (`{"x":…,"y":…}`), or nothing when it did not move it. */
function pointerOf(output: string): ActResult {
  const text = output.trim()
  if (!text.startsWith('{')) return {}
  try {
    return { pointer: pointShape(JSON.parse(text)) }
  } catch {
    return {}
  }
}

/** `sips -g pixelWidth -g pixelHeight` output → the image's size. */
export function pixelSize(output: string): { w: number; h: number } {
  const w = Number(/pixelWidth:\s*(\d+)/.exec(output)?.[1])
  const h = Number(/pixelHeight:\s*(\d+)/.exec(output)?.[1])
  if (!(w > 0 && h > 0)) throw new Error('The screenshot could not be measured.')
  return { w, h }
}

/** Every script this backend runs, by name: the tests parse each one, as osascript would. */
export const SCRIPTS = { observe: OBSERVE_SCRIPT, act: ACT_SCRIPT, frontWindow: FRONT_WINDOW_SCRIPT, ocr: OCR_SCRIPT, access: ACCESS_SCRIPT, pointer: POINTER_SCRIPT, displays: DISPLAYS_SCRIPT }

/** A script failure in words a person can act on, or null to pass it through. */
function explain(message: string): Error | null {
  if (message.includes('LOCALROUTER_GONE')) return new Error('That control is gone; look again.')
  if (message.includes('LOCALROUTER_NO_APP')) return new Error('No app is in front on the Mac.')
  if (message.includes('LOCALROUTER_NO_WINDOW')) return new Error('The app in front has no open window.')
  if (message.includes('LOCALROUTER_OFFSCREEN')) return new Error('That control is off screen; scroll it into view first.')
  if (message.includes('LOCALROUTER_NOT_FOCUSED')) return new Error('The text box did not take the typing focus, so nothing was typed.')
  if (message.includes('LOCALROUTER_NEWLINE')) return new Error('Line breaks are never typed as keys (a Return could send a message).')
  return macAccessError(message)
}
