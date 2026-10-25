import { ChromeTabs } from '../../../desktop-helper/src/computer/chromeTabs.ts'
import { DesktopBackend, macAccess } from '../../../desktop-helper/src/computer/desktop.ts'
import { DockApps } from '../../../desktop-helper/src/computer/dock.ts'
import { MacControls } from '../../../desktop-helper/src/computer/macControls.ts'
import type { AccessReport, Platform } from './types.ts'

const BACKEND = 'macOS (AX, CGEvent, Vision OCR)'

/**
 * macOS: the backend ported from SideBud into the desktop helper. Accessibility reads windows and drives input
 * (CGEvent, System Events keys), Screen Recording is only for apps that hide their window from accessibility and for
 * screenshots. Touch ID for approvals is the app's (Electron's systemPreferences), so `presence` is null here.
 */
export function createMacPlatform(): Platform {
  const controls = new MacControls()
  const screen = new DesktopBackend()
  return {
    os: 'darwin',
    backend: BACKEND,
    screen,
    tabs: new ChromeTabs(),
    launcher: new DockApps(),
    apps: { running: () => controls.running(), activate: (name) => controls.activate(name) },
    controls,
    pointer: () => screen.pointer(),
    displays: () => screen.displays(),
    presence: null,
    access: async (): Promise<AccessReport> => {
      const a = await macAccess()
      const host = a.host || 'LocalRouter'
      return {
        accessibility: a.accessibility,
        screen_recording: a.screenRecording,
        input: a.accessibility,
        backend: BACKEND,
        requirements: [
          { id: 'accessibility', label: 'Accessibility', granted: a.accessibility, for: 'reading windows, clicking and typing', fix: `Allow ${host} in System Settings › Privacy & Security › Accessibility, then reopen it.` },
          { id: 'screen_recording', label: 'Screen Recording', granted: a.screenRecording, for: 'screenshots, and apps that hide their window from accessibility', fix: `Allow ${host} in System Settings › Privacy & Security › Screen Recording, then reopen it.` },
          { id: 'automation', label: 'Automation (System Events)', granted: null, for: 'key presses', fix: 'macOS asks the first time; allow it under Privacy & Security › Automation.' },
        ],
      }
    },
  }
}
