// Ported from SideBud (github.com/jayleaton/sidebud @70da540, apps/companion/src/computer/use/types.ts), reused in LocalRouter with the owner's permission.

/**
 * Computer use: the helper moves the real cursor and presses real keys on this Mac (a browser tab, an
 * app window) toward a goal, one small step at a time, with the box's decision provider choosing each
 * step from the controls the screen actually has, and the box granting each action before it happens.
 * Backends read the screen through accessibility and act with the cursor and keyboard; the controller
 * (controller.ts) asks the box and acts.
 */

/**
 * A point in global, top-left-origin screen coordinates: the space the platform's accessibility positions and
 * input events share (points on macOS, physical pixels on Windows with per-monitor DPI awareness, X11/compositor
 * pixels on Linux). Every Point and Rect a backend reports or takes is in this one space.
 */
export type Point = { x: number; y: number }

/** A rectangle in the same global space as Point. */
export type Rect = { x: number; y: number; w: number; h: number }

/** The pointer as the local stop reads it: where it is, and how long ago the left button last went down, if known. */
export type PointerReading = Point & { sinceClickS?: number }

/** One control on screen that can be acted on. */
export type ScreenControl = {
  /** The backend's handle for it, valid until the next observation. */
  ref: string
  /** `text` is a line of text read from a screenshot, when the app would not show its window through accessibility. */
  role: 'button' | 'link' | 'textbox' | 'tab' | 'row' | 'menuitem' | 'checkbox' | 'option' | 'text' | 'other'
  /** Its visible text or accessible name, clipped. */
  label: string
  /** A text box's current text, clipped. */
  value?: string
  focused?: boolean
  /** Where it is on screen, when the backend knows (for a caller that reads the tree next to a screenshot). */
  bounds?: Rect
}

export const CONTROL_ROLES = ['button', 'link', 'textbox', 'tab', 'row', 'menuitem', 'checkbox', 'option', 'text', 'other'] as const

/** What is on screen now, for one decision. */
export type Observation = {
  backend: 'browser' | 'desktop'
  app: string
  /** The window or tab title. */
  title: string
  url?: string
  /** Visible, actionable controls, in reading order (at most about 150). */
  controls: ScreenControl[]
  /** A short excerpt of the visible text, for context (at most about 1,500 characters). */
  text?: string
  /** Read from a screenshot: only text is listed, not icons such as a play button. */
  visual?: boolean
  /** The window observed, in global coordinates, when the backend knows. */
  window?: Rect
  /** The walk stopped early (its node, control or time budget ran out): more may be on screen than is listed. */
  truncated?: boolean
}

export type ScreenAction =
  | { kind: 'click'; ref: string }
  /** Focus the control and replace its text with `text`. */
  | { kind: 'type'; ref: string; text: string }
  | { kind: 'press'; key: 'enter' | 'escape' | 'tab' | 'space' }
  /** `ref`: the list or area to scroll (a control of this observation); the window's middle when absent. */
  | { kind: 'scroll'; direction: 'up' | 'down' | 'top' | 'bottom'; ref?: string }

/**
 * How an action reached the app. `background`: without the person's real pointer, focus or keyboard (an accessibility
 * action such as AXPress, UIA Invoke or AT-SPI doAction; a value set through accessibility; input posted to the app's
 * own window or process). `pointer`: the real pointer and global keyboard, only after the person allowed it for that
 * step ("needs your mouse for a moment").
 */
export type InputMode = 'background' | 'pointer'

/**
 * What an action did. `mode` says how it reached the app (every action has one). `pointer`: where the real pointer was
 * left, when the action moved it (pointer mode only). `at`: where on screen it acted, for the visible virtual cursor
 * (it never moves the real one). `how`: how it was done, for the audit; never the typed text.
 */
export type ActResult = {
  mode: InputMode
  pointer?: Point
  at?: Point
  how?: string
}

export type ObserveOptions = { signal?: AbortSignal }

/**
 * `signal` aborts an action under way (a stop): long typing stops between chunks, a running script is killed.
 * `allowPointer`: the person allowed the real pointer and keyboard for this one step. Without it a backend acts in the
 * background or throws NeedsPointer, and never touches the real pointer or global keyboard.
 */
export type ActOptions = { signal?: AbortSignal; allowPointer?: boolean }

/**
 * Thrown by a backend when an action cannot be done in the background (a canvas or game with no accessible control, an
 * app that ignores input posted to its window). The controller then asks the person for their mouse for a moment.
 */
export class NeedsPointer extends Error {
  readonly why: string

  constructor(why: string) {
    super(`This step needs your mouse for a moment: ${why}`)
    this.name = 'NeedsPointer'
    this.why = why
  }
}

/** Whether an error is NeedsPointer, across module copies (the bundle and the tests). */
export function needsPointer(error: unknown): error is NeedsPointer {
  return error instanceof NeedsPointer || (typeof error === 'object' && error !== null && (error as { name?: unknown }).name === 'NeedsPointer')
}

/**
 * The front window as an image. `region` is the part of the screen it shows (global coordinates) and `scale` the
 * image pixels per global unit, so image pixel (px, py) is the global point (region.x + px / scale, region.y + py / scale).
 */
export type Screenshot = { png: Uint8Array; width: number; height: number; region: Rect; scale: number; app: string; title: string }

export type ScreenshotOptions = { maxEdge?: number; signal?: AbortSignal }

/** The longest image edge a screenshot is scaled to by default (what vision models read without resizing again). */
export const SCREENSHOT_MAX_EDGE = 1568

/** Reads and drives one surface. Errors are thrown with a message a person can act on. */
export interface ScreenBackend {
  readonly kind: 'browser' | 'desktop'
  observe(options?: ObserveOptions): Promise<Observation>
  /** Resolves with where the pointer was left when the action moved it (clicks, typing, scrolling). */
  act(action: ScreenAction, options?: ActOptions): Promise<ActResult | void>
  /** Where the pointer is now, for the local stop. */
  pointer?(): Promise<PointerReading>
  /** Every display's bounds, in global coordinates (the local stop's corners are worked out from them). */
  displays?(): Promise<Rect[]>
  /** The front window as an image, scaled down so its longer edge is at most `maxEdge` (SCREENSHOT_MAX_EDGE). */
  screenshot?(options?: ScreenshotOptions): Promise<Screenshot>
}

/** The scale a capture of `w`×`h` image pixels needs so its longer edge is at most `maxEdge` (never above 1). */
export function fitScale(w: number, h: number, maxEdge: number = SCREENSHOT_MAX_EDGE): number {
  const edge = Math.max(w, h)
  return edge > maxEdge && edge > 0 ? maxEdge / edge : 1
}

/** One open browser tab. */
export type BrowserTab = { window: number; index: number; title: string; url: string; active: boolean }

/** A named tab group and its tabs' titles as the tab strip shows them, in order. */
export type TabGroup = { name: string; tabs: string[] }

/** The browser's open tabs (read-only), and switching to one by clicking it in the tab strip with the cursor. */
export interface BrowserTabs {
  readonly browser: string
  list(): Promise<BrowserTab[]>
  /** Named tab groups, read from the tab strip (tab lists from the browser do not say which group a tab is in). */
  groups?(): Promise<TabGroup[]>
  /** Brings the browser forward and clicks the tab in its tab strip. */
  activate(tab: BrowserTab): Promise<void>
  /**
   * For when the browser's tabs cannot be read: switches with the browser's own tab search and returns
   * the title now in front, or null when nothing matched.
   */
  search?(name: string): Promise<string | null>
}

/**
 * The local stop as the controller sees it: checked before each action, told when the helper is about
 * to move the pointer itself and where it left it, so only a person's movement stops the task.
 */
export interface PointerWatch {
  /** Checks the pointer now; aborts the running task when the person took over. */
  check(): Promise<void>
  /**
   * The helper is about to move the pointer, or is waiting for the box (the person may tap Approve
   * in a browser on this Mac): movement is not a takeover until `left`. The corner still stops it.
   */
  moving(): void
  /**
   * Where the helper left the pointer. A point is the new expected position; undefined means the action
   * did not move it; null means it moved somewhere unknown (a Dock or tab-strip click), so the next
   * reading becomes the expected position.
   */
  left(point?: Point | null): void
}
