import type { ComputerControl } from '../../../desktop-helper/src/computer/macControls.ts'
import type { BrowserTabs, PointerReading, Rect, ScreenBackend } from '../../../desktop-helper/src/computer/types.ts'

/**
 * What the executor drives on one operating system. The broker (policy, approvals, budgets, audit) and the local stop
 * are the same on every OS and live above this; a platform only reads the screen and acts, and says what it can do.
 * Every coordinate it takes or reports is in the one global space of `Point` (see types.ts in the helper).
 *
 *   macOS    AX through JXA, CGEvent input, Vision OCR fallback           (platform/mac.ts)
 *   Linux    AT-SPI through python3-gi; X11 XTest (xdotool), Wayland portals or ydotool  (platform/linux/)
 *   Windows  UI Automation and SendInput through a PowerShell worker       (platform/windows/)
 *   fake     the scripted Notes app, for tests and headless smokes         (platform/fake.ts)
 */
export type PlatformOs = 'darwin' | 'linux' | 'win32' | 'fake' | 'other'

/** A fixed control this platform cannot run, refused before the policy with the reason for the person. */
export type ControlSupport = Partial<Record<ComputerControl['kind'], string>>

export interface PlatformControls {
  /** Runs one fixed control; throws a ControlError (macControls.ts) with words for the person when it cannot. */
  run(control: ComputerControl): Promise<{ done: string }>
  /** A desktop notification (the "Jarvis is using this computer" notice). */
  notify?(title: string, text: string): Promise<void>
  /** Controls this OS has no way to run, with why ("Windows has no per-app volume for Music"). */
  readonly unsupported?: ControlSupport
}

/** One thing the platform needs from the person or the system, as Settings shows it. */
export type Requirement = {
  id: string
  /** "Accessibility", "AT-SPI accessibility bus", "Input (uinput or ydotoold)", "Screen capture portal". */
  label: string
  /** null: cannot be checked without asking (a portal prompt). */
  granted: boolean | null
  /** What it is needed for, in words. */
  for: string
  /** How to grant or install it, when it is missing. */
  fix?: string
}

/**
 * Whether this computer can be read and driven, in the shape the app and the box's legacy `hello` already use
 * (`accessibility`: the screen can be read; `screen_recording`: it can be captured), plus input and the details.
 */
export type AccessReport = {
  accessibility: boolean
  screen_recording: boolean
  /** Synthetic pointer and key input reaches other apps. */
  input: boolean
  /** "macOS (AX, CGEvent)", "Linux X11 (AT-SPI, XTest)", "Linux Wayland/Hyprland (AT-SPI, ydotool)", "Windows (UI Automation, SendInput)". */
  backend: string
  requirements: Requirement[]
}

export type PresenceOptions = { window?: number; waiting?: (text: string) => void; signal?: AbortSignal }

/** The person's presence for one approval, checked by the OS where it can be (Touch ID, Windows Hello). */
export interface PlatformPresence {
  readonly method: string
  /** After `available()` said no: why, in words ("Windows Hello is not set up for you"). */
  readonly why?: string
  /**
   * Resolves true only when the OS confirmed the person; false when they cancelled or it failed. `window`: the app's
   * window handle, so the OS prompt opens over it (Windows: an HWND); `waiting`: called with a sentence for the card when
   * the prompt is open but could not be brought in front of other windows; `signal`: aborted when the approval ends
   * another way, which cancels the OS prompt and resolves false.
   */
  verify(prompt: string, options?: PresenceOptions): Promise<boolean>
  /** Whether it can be used on this computer now (hardware enrolled, service running). */
  available(): Promise<boolean>
}

export type Platform = {
  readonly os: PlatformOs
  readonly backend: string
  /** The front window: observe and act (null when it cannot be read at all). */
  readonly screen: ScreenBackend | null
  /** A browser's tabs, when this OS can read them. */
  readonly tabs: BrowserTabs | null
  /** The person's own list of apps (the Dock, the taskbar's pinned apps, the dash favourites) and bringing one forward. */
  readonly launcher: { list(): Promise<string[]>; activate(name: string): Promise<unknown> } | null
  /** Running apps and bringing one to the front. */
  readonly apps: { running(): Promise<string[]>; activate(name: string): Promise<void> } | null
  readonly controls: PlatformControls
  /** The pointer, for the local stop; null where the OS does not let an app read it (GNOME on Wayland). */
  readonly pointer: (() => Promise<PointerReading>) | null
  /** Every display's bounds, for the local stop's corners; null when unknown (the stop then uses (0, 0)). */
  readonly displays: (() => Promise<Rect[]>) | null
  /** An OS presence check for approvals; null: the app's own confirmation dialog. */
  readonly presence: PlatformPresence | null
  access(): Promise<AccessReport>
  /**
   * macOS: the Automation grants computer use relies on, asked once from the app's setup (the one moment a "LocalRouter
   * wants to control …" prompt may appear; during tasks nothing asks). Resolves each bundle id asked to macOS's
   * status: 0 granted, -1743 denied, -1744 no answer, -600 not running (not asked), -1 could not be checked.
   */
  readonly automation?: { ask(): Promise<Record<string, number>> }
  /** Ends what the platform keeps running (Windows' PowerShell workers), when the executor stops. */
  close?(): void
}
