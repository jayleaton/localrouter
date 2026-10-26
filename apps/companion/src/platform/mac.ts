import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { ChromeTabs } from '../../../desktop-helper/src/computer/chromeTabs.ts'
import { DesktopBackend, macAccess } from '../../../desktop-helper/src/computer/desktop.ts'
import { DockApps } from '../../../desktop-helper/src/computer/dock.ts'
import { MacControls } from '../../../desktop-helper/src/computer/macControls.ts'
import { MacAutomation } from './macAutomation.ts'
import type { AccessReport, Platform } from './types.ts'

const BACKEND = 'macOS (AX, CGEvent, Vision OCR)'

/** The apps computer use and the controls send Apple Events to, asked once from setup (each only when it runs). */
export const AUTOMATION_TARGETS = ['com.apple.systemevents', 'com.google.Chrome', 'com.spotify.client', 'com.apple.Music'] as const

/** The helper in the packaged app's Resources; null in a dev run (no helper: no Apple Event is ever allowed). */
export function bundledAutomation(resources: string | undefined = process.resourcesPath): MacAutomation {
  const file = resources ? join(resources, 'localrouter-presence') : ''
  return new MacAutomation(file && existsSync(file) ? { file } : null)
}

/**
 * `automation`: macOS's Automation grants, checked without asking (default: the packaged helper). null: Apple Events
 * are sent unchecked, as the standalone desktop helper does (tests).
 */
export type MacPlatformOptions = { automation?: MacAutomation | null }

/**
 * macOS: the backend ported from SideBud into the desktop helper. Accessibility reads windows and drives input
 * (CGEvent, typed text as Unicode key events), Screen Recording is only for apps that hide their window from
 * accessibility and for screenshots. No system permission window appears during a task: the computer-use path sends
 * no Apple Event to System Events, and every remaining one (Chrome's tab addresses, dark mode, locking, Spotify and
 * Music, quitting an app) goes only where macOS already said yes, asked once from setup (`automation.ask`). Touch ID
 * for approvals is the app's, so `presence` is null here.
 */
export function createMacPlatform(options: MacPlatformOptions = {}): Platform {
  const automation = options.automation === undefined ? bundledAutomation() : options.automation
  const check = automation ? { automation } : {}
  const controls = new MacControls(check)
  const screen = new DesktopBackend(undefined, check)
  return {
    os: 'darwin',
    backend: BACKEND,
    screen,
    tabs: new ChromeTabs(undefined, check),
    // The Dock is read through accessibility and clicked with CGEvent; an app not in it is opened with `open -a`
    // (LaunchServices): no Apple Event either way.
    launcher: new DockApps(),
    apps: { running: () => controls.running(), activate: (name) => controls.activate(name) },
    controls,
    pointer: () => screen.pointer(),
    displays: () => screen.displays(),
    presence: null,
    ...(automation
      ? {
          automation: {
            // One at a time: macOS shows one prompt per app, and an app that is not running answers -600 without one.
            ask: async () => {
              const answers: Record<string, number> = {}
              for (const id of AUTOMATION_TARGETS) answers[id] = await automation.ask(id)
              return answers
            },
          },
        }
      : {}),
    access: async (): Promise<AccessReport> => {
      const a = await macAccess()
      const host = a.host || 'LocalRouter'
      const events = automation ? await automation.allowed('com.apple.systemevents') : null
      return {
        accessibility: a.accessibility,
        screen_recording: a.screenRecording,
        input: a.accessibility,
        backend: BACKEND,
        requirements: [
          { id: 'accessibility', label: 'Accessibility', granted: a.accessibility, for: 'reading windows, clicking and typing', fix: `Allow ${host} in System Settings › Privacy & Security › Accessibility, then reopen it.` },
          { id: 'screen_recording', label: 'Screen Recording', granted: a.screenRecording, for: 'screenshots, and apps that hide their window from accessibility', fix: `Allow ${host} in System Settings › Privacy & Security › Screen Recording, then reopen it.` },
          { id: 'automation', label: 'Automation (System Events)', granted: events, for: 'dark mode and locking the screen', fix: 'LocalRouter asks for it once from Settings › This computer (never during a task); or allow it under Privacy & Security › Automation.' },
        ],
      }
    },
  }
}
