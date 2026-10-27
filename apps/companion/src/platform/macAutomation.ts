/**
 * macOS Automation (Apple Events to another app) through `localrouter-presence` (native/presence-helper.m), shipped in
 * the app's Resources. The product rule: during a task or a control no system permission window appears, so before
 * every Apple Event the helper is asked `automation BUNDLE_ID`, which never prompts, and the event is sent only when
 * macOS already said yes. `automation-ask` is for the app's setup only, the one moment macOS may show its prompt.
 */
import { execFile } from 'node:child_process'
import type { AutomationCheck } from '../../../desktop-helper/src/computer/types.ts'

/** AEDeterminePermissionToAutomateTarget's answers. */
export const AUTOMATION_STATUS = { granted: 0, denied: -1743, wouldAsk: -1744, notRunning: -600 } as const
/** The helper could not be run, or said nothing readable. */
export const NOT_CHECKED = -1

/** The helper's command: its path (tests: node and a stand-in script). null: no helper (a dev run), never allowed. */
export type AutomationCommand = { file: string; args?: string[] }

const BUNDLE_ID = /^[A-Za-z0-9][A-Za-z0-9.-]{0,154}$/

export class MacAutomation implements AutomationCheck {
  private readonly command: AutomationCommand | null
  private readonly granted = new Set<string>()

  constructor(command: AutomationCommand | null) {
    this.command = command
  }

  /**
   * True only when macOS already allows LocalRouter to control that app (status 0). Never asks, never throws. A yes
   * is kept (macOS does not take it back on its own; a person who does gets the app's refusal, -1743, as before).
   */
  async allowed(bundleId: string): Promise<boolean> {
    if (this.granted.has(bundleId)) return true
    return (await this.status(bundleId, false)) === AUTOMATION_STATUS.granted
  }

  /** Asks the person through macOS when it has no answer yet (setup only), and returns macOS's status. */
  ask(bundleId: string): Promise<number> {
    return this.status(bundleId, true)
  }

  private async status(bundleId: string, ask: boolean): Promise<number> {
    if (!this.command || !BUNDLE_ID.test(bundleId)) return NOT_CHECKED
    // A check starts System Events hidden when it is not running (a few seconds at most); an ask waits for the person.
    const out = await run(this.command, [ask ? 'automation-ask' : 'automation', bundleId], ask ? 300_000 : 15_000).catch(() => '')
    let status: unknown
    try {
      status = (JSON.parse(out.trim()) as { status?: unknown }).status
    } catch {
      return NOT_CHECKED
    }
    if (typeof status !== 'number' || !Number.isInteger(status)) return NOT_CHECKED
    if (status === AUTOMATION_STATUS.granted) this.granted.add(bundleId)
    return status
  }
}

function run(command: AutomationCommand, args: string[], timeout: number): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(command.file, [...(command.args ?? []), ...args], { timeout, windowsHide: true }, (error, stdout) => (error ? reject(error) : resolve(String(stdout))))
  })
}
