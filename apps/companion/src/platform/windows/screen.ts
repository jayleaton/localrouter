import { CONTROL_ROLES, fitScale, NeedsPointer, SCREENSHOT_MAX_EDGE } from '../../../../desktop-helper/src/computer/types.ts'
import type { ActOptions, ActResult, Observation, ObserveOptions, Point, PointerReading, Rect, ScreenAction, ScreenBackend, ScreenControl, Screenshot, ScreenshotOptions } from '../../../../desktop-helper/src/computer/types.ts'
import { array, boolean, integerIn, number, object, oneOf, optional, string } from '../../../../desktop-helper/src/validate.ts'
import type { WindowsCall } from './worker.ts'

/**
 * Turns a helper failure (`LR_CODE|detail|`, see helper.cs.ts) into a sentence a person can act on, or null to pass
 * the message through.
 */
export function explain(message: string): Error | null {
  const m = /LR_([A-Z_]+)\|([^|]*)\|/.exec(message)
  if (!m) return null
  const [, code, detail] = m
  const app = detail.trim() || 'That app'
  switch (code) {
    case 'GONE':
      return new Error('That control is gone; look again.')
    case 'NO_APP':
      return new Error('No app is in front on this computer.')
    case 'NO_WINDOW':
      return new Error('The app in front has no open window.')
    case 'ELEVATED':
      return new Error(`${app} runs as administrator, so LocalRouter cannot see, click or type in it. Run ${app} normally (not as administrator), or do this step yourself.`)
    case 'SECURE_DESKTOP':
      return new Error('Windows is showing a secure screen (a User Account Control prompt, or the lock or sign-in screen) that no app can read or use. Answer it yourself, then try again.')
    case 'NOT_FRONT':
      return new Error(`Windows did not let LocalRouter bring ${app} to the front. Click ${app} in the taskbar, then try again.`)
    case 'FRONT_CHANGED':
      return new Error('Another window came to the front, so it stopped before clicking or typing there. Look again.')
    case 'DID_NOT_TAKE':
      return new Error(`The text did not land in ${detail.trim() ? `“${detail.trim()}”` : 'the field'}: it shows something else now. Look again.`)
    case 'NEWLINE':
      return new Error(NEWLINE_REFUSED)
    case 'INPUT_REFUSED':
      return new Error('Windows refused the click or key press (another program may be blocking input). Try again.')
    case 'NO_POINTER':
      return new Error('Windows did not say where the pointer is.')
    case 'NO_TAB':
      return new Error("Could not find that tab in the browser's tab strip.")
    case 'NO_AUDIO':
      return new Error('This computer has no sound output to change.')
    default:
      return null
  }
}

export const NEWLINE_REFUSED = 'That text has more than one line, and this field only takes typed keys, where a new line would press Enter. Nothing was typed: type one line at a time.'

/** Runs a request and rewrites a helper failure into words. */
export async function ask(worker: WindowsCall, op: string, args: Record<string, unknown> = {}, options: { timeoutMs?: number; signal?: AbortSignal } = {}): Promise<unknown> {
  try {
    return await worker.call(op, args, options)
  } catch (error) {
    throw explain(error instanceof Error ? error.message : String(error)) ?? error
  }
}

/** Refs look like "<hwnd>:<path>#<runtime id>" (helper.cs.ts, Resolve); anything else is never sent. */
const REF = /^\d+:\d+(?:\.\d+)*(?:#-?\d+(?:\.-?\d+)*)?$/

function checkRef(ref: string): string {
  if (!REF.test(ref)) throw new Error('That control is gone; look again.')
  return ref
}

export function rectShape(value: unknown, what: string): Rect {
  const r = object(value, what)
  return { x: number(r.x, `${what} x`), y: number(r.y, `${what} y`), w: number(r.w, `${what} w`), h: number(r.h, `${what} h`) }
}

export function pointShape(value: unknown, what = 'point'): Point {
  const p = object(value, what)
  return { x: number(p.x, `${what} x`), y: number(p.y, `${what} y`) }
}

/** One control as the helper lists it. A password field's value is dropped here too, whatever the helper sent. */
export function controlShape(value: unknown, what: string): ScreenControl {
  const raw = object(value, what)
  const control: ScreenControl = { ref: string(raw.ref, `${what} ref`), role: oneOf(raw.role, CONTROL_ROLES, `${what} role`), label: string(raw.label, `${what} label`) }
  const password = optional(raw.password, `${what} password`, boolean) === true
  const text = optional(raw.value, `${what} value`, string)
  const focused = optional(raw.focused, `${what} focused`, boolean)
  const bounds = optional(raw.bounds, `${what} bounds`, rectShape)
  if (text !== undefined && !password) control.value = text
  if (focused !== undefined) control.focused = focused
  if (bounds) control.bounds = bounds
  return control
}

export function observationShape(value: unknown): Observation {
  const raw = object(value, 'observation')
  const observation: Observation = {
    backend: 'desktop',
    app: string(raw.app, 'observation app'),
    title: string(raw.title, 'observation title'),
    controls: array(raw.controls, 'observation controls', controlShape),
    text: string(raw.text, 'observation text'),
  }
  const url = optional(raw.url, 'observation url', string)
  const window = optional(raw.window, 'observation window', rectShape)
  if (url) observation.url = url
  if (window) observation.window = window
  if (optional(raw.truncated, 'observation truncated', boolean)) observation.truncated = true
  return observation
}

export function displaysShape(value: unknown): Rect[] {
  const list = array(value, 'displays', (raw, what) => ({ ...rectShape(raw, what), primary: optional(object(raw, what).primary, `${what} primary`, boolean) === true }))
  // The primary display first: its top-left corner is (0, 0) on Windows, the local stop's corner.
  return [...list.filter((d) => d.primary), ...list.filter((d) => !d.primary)].map(({ x, y, w, h }) => ({ x, y, w, h }))
}

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]

/**
 * A screenshot reply as the contract's Screenshot. The scale is fitScale's for the captured region, and the image must
 * have the size that scale gives (the helper rounds the same way), so a pixel maps back to the right global point.
 */
export function screenshotShape(value: unknown, maxEdge: number): Screenshot {
  const raw = object(value, 'screenshot')
  const region = rectShape(raw.region, 'screenshot region')
  const width = number(raw.width, 'screenshot width')
  const height = number(raw.height, 'screenshot height')
  const scale = fitScale(region.w, region.h, maxEdge)
  const expected = (edge: number) => Math.max(1, Math.round(edge * scale))
  if (Math.abs(width - expected(region.w)) > 1 || Math.abs(height - expected(region.h)) > 1) {
    throw new Error(`The screenshot is ${width}×${height}, not the ${expected(region.w)}×${expected(region.h)} its scale gives.`)
  }
  const png = new Uint8Array(Buffer.from(string(raw.png, 'screenshot png'), 'base64'))
  if (png.length < PNG_SIGNATURE.length || PNG_SIGNATURE.some((byte, i) => png[i] !== byte)) throw new Error('The screenshot is not a PNG image.')
  return { png, width, height, region, scale, app: string(raw.app, 'screenshot app'), title: string(raw.title, 'screenshot title') }
}

/** Virtual-key codes for the keys a step may press. */
export const VK = { enter: 0x0d, escape: 0x1b, tab: 0x09, space: 0x20 } as const
const VK_DELETE = 0x2e

/** Typed text goes in chunks of at most this many UTF-16 code units; a stop is checked between chunks. */
export const TYPE_CHUNK = 200

/** Splits text into chunks of at most `size` UTF-16 code units, never between the two halves of a surrogate pair. */
export function chunks(text: string, size: number = TYPE_CHUNK): string[] {
  const out: string[] = []
  for (let at = 0; at < text.length;) {
    let end = Math.min(text.length, at + size)
    const last = text.charCodeAt(end - 1)
    if (end < text.length && end - at > 1 && last >= 0xd800 && last <= 0xdbff) end -= 1
    out.push(text.slice(at, end))
    at = end
  }
  return out
}

/** Wheel notches for one scroll (positive is up): a few for a step, many for the top or bottom. */
export const NOTCHES = { up: 3, down: -3, top: 25, bottom: -25 } as const

const done = (value: unknown, what: string) => boolean(object(value, what).done, `${what} done`)

/** Where a semantic action acted (the element's centre, screen pixels), for the overlay ring; absent when unknown. */
function spot(value: unknown): { at: { x: number; y: number } } | Record<string, never> {
  const v = object(value, 'semantic')
  return typeof v.x === 'number' && typeof v.y === 'number' && Number.isFinite(v.x) && Number.isFinite(v.y) ? { at: { x: v.x, y: v.y } } : {}
}

function target(value: unknown): { hwnd: number; x: number; y: number } {
  const t = object(value, 'target')
  return { hwnd: number(t.hwnd, 'target hwnd'), x: number(t.x, 'target x'), y: number(t.y, 'target y') }
}

function throwIfAborted(signal?: AbortSignal) {
  if (signal?.aborted) throw new Error(typeof signal.reason === 'string' && signal.reason ? signal.reason : 'It was stopped.')
}

export type WindowsScreenOptions = {
  /** The observe walk's wall-time budget; past it the observation is `truncated`. */
  observeBudgetMs?: number
}

/**
 * The window in front, read with UI Automation and driven semantically first: a click is Invoke, Select, Toggle or
 * Expand; typing is ValuePattern.SetValue, read back to check it landed; scrolling is ScrollPattern. Those touch
 * neither the pointer nor the keyboard (`background: true`), so the person can keep working. Only a control without
 * the pattern gets real input through SendInput, after the window is in front and checked again before each event.
 */
export class WindowsScreen implements ScreenBackend {
  readonly kind = 'desktop'
  private readonly worker: WindowsCall
  private readonly observeBudgetMs: number

  constructor(worker: WindowsCall, options: WindowsScreenOptions = {}) {
    this.worker = worker
    this.observeBudgetMs = options.observeBudgetMs ?? 3_000
  }

  async observe(options: ObserveOptions = {}): Promise<Observation> {
    return observationShape(await ask(this.worker, 'observe', { budgetMs: this.observeBudgetMs }, { signal: options.signal, timeoutMs: this.observeBudgetMs + 12_000 }))
  }

  async act(action: ScreenAction, options: ActOptions = {}): Promise<ActResult> {
    const signal = options.signal
    const call = (op: string, args: Record<string, unknown>) => ask(this.worker, op, args, { signal })
    switch (action.kind) {
      case 'press': {
        // In the background the key is posted to the foreground window's focused control (no SendInput). When that is
        // refused — or the person allowed the real keyboard — a SendInput key event.
        if (!options.allowPointer) {
          await call('postkey', { vk: VK[action.key] })
          return { mode: 'background', how: 'key posted to the focused control' }
        }
        await call('key', { vk: VK[action.key] })
        return { mode: 'pointer', how: 'key event (SendInput)' }
      }
      case 'click': {
        const ref = checkRef(action.ref)
        // A UI Automation pattern (Invoke, Select, Toggle, Expand): no pointer, no focus change. A field, or anything else
        // that takes the keyboard focus, is focused (SetFocus): that is all a click there does.
        const semantic = await call('semantic', { kind: 'click', ref })
        if (done(semantic, 'click')) return { mode: 'background', ...spot(semantic), ...(object(semantic, 'click').how === 'focus' ? { how: 'focused through UI Automation' } : {}) }
        // No pattern: the real pointer, only with the person's leave ("needs your mouse for a moment").
        if (!options.allowPointer) throw new NeedsPointer('that control has no UI Automation pattern to click')
        const at = target(await call('point', { ref }))
        return { mode: 'pointer', pointer: pointShape(await call('click', at), 'pointer'), at: { x: at.x, y: at.y } }
      }
      case 'type': {
        const ref = checkRef(action.ref)
        // ValuePattern sets the field's text in the background: no focus change, the person's keyboard untouched.
        const semantic = object(await call('semantic', { kind: 'type', ref, text: action.text }), 'type')
        if (boolean(semantic.done, 'type done')) return { mode: 'background', ...spot(semantic), how: semantic.focused === true ? 'value set, field focused' : 'value set, field not focused' }
        if (!options.allowPointer) throw new NeedsPointer('that field has no value pattern and must be typed with the keyboard')
        // Keys cannot type a line break without pressing Enter, which could send or submit.
        if (/[\r\n]/.test(action.text)) throw new Error(NEWLINE_REFUSED)
        const at = target(await call('point', { ref }))
        const pointer = pointShape(await call('click', at), 'pointer')
        await call('select_all', { hwnd: at.hwnd })
        if (!action.text) await call('key', { hwnd: at.hwnd, vk: VK_DELETE })
        for (const chunk of chunks(action.text)) {
          throwIfAborted(signal)
          await call('type', { hwnd: at.hwnd, text: chunk })
        }
        return { mode: 'pointer', pointer, at: { x: at.x, y: at.y }, how: 'clicked into the field, selected its text, typed keys' }
      }
      case 'scroll': {
        const ref = action.ref === undefined ? undefined : checkRef(action.ref)
        const where = ref ? { ref } : {}
        const scrolled = await call('semantic', { kind: 'scroll', direction: action.direction, ...where })
        if (done(scrolled, 'scroll')) return { mode: 'background', ...spot(scrolled) }
        if (!options.allowPointer) throw new NeedsPointer('nothing here scrolls through UI Automation')
        const at = target(await call('point', where))
        return { mode: 'pointer', pointer: pointShape(await call('wheel', { ...at, notches: NOTCHES[action.direction] }), 'pointer'), at: { x: at.x, y: at.y } }
      }
    }
  }

  async pointer(): Promise<PointerReading> {
    // No sinceClickS: Windows has no cheap, reliable "time since the last left click" (GetAsyncKeyState misses clicks
    // between polls, and GetLastInputInfo counts every key and move). The local stop then works from movement alone.
    return pointShape(await ask(this.worker, 'pointer', {}, { timeoutMs: 5_000 }), 'pointer')
  }

  async displays(): Promise<Rect[]> {
    return displaysShape(await ask(this.worker, 'displays', {}, { timeoutMs: 5_000 }))
  }

  async screenshot(options: ScreenshotOptions = {}): Promise<Screenshot> {
    const maxEdge = integerIn(options.maxEdge ?? SCREENSHOT_MAX_EDGE, 16, 16_384, 'maxEdge')
    return screenshotShape(await ask(this.worker, 'screenshot', { maxEdge }, { signal: options.signal, timeoutMs: 20_000 }), maxEdge)
  }
}
