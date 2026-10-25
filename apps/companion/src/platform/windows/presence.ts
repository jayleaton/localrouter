import { object, string } from '../../../../desktop-helper/src/validate.ts'
import type { PlatformPresence, PresenceOptions } from '../types.ts'
import type { WindowsCall } from './worker.ts'

/** UserConsentVerifierAvailability values: only `Available` means Windows Hello can confirm the person now. */
export const HELLO_AVAILABLE = 'Available'

/** On the card when the Windows Security window could not be brought in front of the others. */
export const HELLO_BEHIND = 'Windows Hello is waiting: look for the Windows Security window (it may be behind other windows or on another screen).'

/**
 * Windows Hello (face, fingerprint or PIN) for approvals, through WinRT's UserConsentVerifier. It runs in its own worker
 * (index.ts), because the prompt can stay open for the whole approval and must not hold up the pointer reads of the
 * local stop. Any failure is a "no": a person is confirmed only when Windows says `Verified`.
 */
export class WindowsHello implements PlatformPresence {
  readonly method = 'Windows Hello'
  private readonly worker: WindowsCall

  constructor(worker: WindowsCall) {
    this.worker = worker
  }

  /** False when Hello is not set up (no PIN or biometrics), disabled by policy, or the check fails. */
  async available(): Promise<boolean> {
    try {
      const reply = object(await this.worker.call('hello_available', {}, { timeoutMs: 20_000 }), 'Windows Hello')
      return string(reply.availability, 'Windows Hello availability') === HELLO_AVAILABLE
    } catch {
      return false
    }
  }

  /**
   * With the app's window: IUserConsentVerifierInterop::RequestVerificationForWindowAsync, so the prompt is owned by
   * it and opens over it, then the Windows Security window brought forward (the foreground ladder). When it still is
   * not in front, `waiting` tells the card where to look. Without a window, or when that call fails, the plain
   * RequestVerificationAsync, whose prompt can open behind: the card says so from the start.
   */
  async verify(prompt: string, options: PresenceOptions = {}): Promise<boolean> {
    const message = prompt.replace(/\s+/g, ' ').trim().slice(0, 200) || 'Approve this action'
    if (options.window !== undefined) {
      let started = false
      try {
        const start = object(await this.worker.call('hello_start', { prompt: message, hwnd: options.window }, { timeoutMs: 20_000 }), 'Windows Hello')
        started = true
        if (start.front !== true) options.waiting?.(HELLO_BEHIND)
        const reply = object(await this.worker.call('hello_wait', { ms: 120_000 }, { timeoutMs: 125_000 }), 'Windows Hello')
        return string(reply.result, 'Windows Hello result') === 'Verified'
      } catch {
        if (started) return false
      }
    }
    try {
      options.waiting?.(HELLO_BEHIND)
      const reply = object(await this.worker.call('hello_verify', { prompt: message }, { timeoutMs: 125_000 }), 'Windows Hello')
      return string(reply.result, 'Windows Hello result') === 'Verified'
    } catch {
      return false
    }
  }
}
