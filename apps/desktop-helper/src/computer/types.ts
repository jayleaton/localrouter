// Ported from SideBud (github.com/jayleaton/sidebud @70da540, apps/companion/src/computer/use/types.ts), reused in LocalRouter with the owner's permission.

/**
 * Computer use: the helper moves the real cursor and presses real keys on this Mac (a browser tab, an
 * app window) toward a goal, one small step at a time, with the box's decision provider choosing each
 * step from the controls the screen actually has, and the box granting each action before it happens.
 * Backends read the screen through accessibility and act with the cursor and keyboard; the controller
 * (controller.ts) asks the box and acts.
 */

/** A point in global, top-left-origin screen points (the space AX positions and CGEvent locations use). */
export type Point = { x: number; y: number }

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
}

export type ScreenAction =
  | { kind: 'click'; ref: string }
  /** Focus the control and replace its text with `text`. */
  | { kind: 'type'; ref: string; text: string }
  | { kind: 'press'; key: 'enter' | 'escape' | 'tab' | 'space' }
  | { kind: 'scroll'; direction: 'up' | 'down' | 'top' | 'bottom' }

/** What an action did to the pointer: where it left it, when it moved it. */
export type ActResult = { pointer?: Point }

/** Reads and drives one surface. Errors are thrown with a message a person can act on. */
export interface ScreenBackend {
  readonly kind: 'browser' | 'desktop'
  observe(): Promise<Observation>
  /** Resolves with where the pointer was left when the action moved it (clicks, typing, scrolling). */
  act(action: ScreenAction): Promise<ActResult | void>
  /** Where the pointer is now, for the local stop. */
  pointer?(): Promise<Point>
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
