import { object, string } from '../../../../desktop-helper/src/validate.ts'
import type { PlatformPresence, PresenceOptions } from '../types.ts'
import type { WindowsCall } from './worker.ts'

/** UserConsentVerifierAvailability values: only `Available` means Windows Hello can confirm the person now. */
export const HELLO_AVAILABLE = 'Available'

/** On the card when the Windows Security window could not be brought in front of the others. */
export const HELLO_BEHIND = 'Windows Hello is waiting: look for the Windows Security window (it may be behind other windows or on another screen).'

/** How long one hello_wait waits before the app looks again (and can cancel). */
export const POLL_MS = 400

/** UserConsentVerifierAvailability, in words for the card when Hello cannot be used. */
export function helloUnavailable(availability: string): string {
  switch (availability) {
    case 'DeviceNotPresent':
      return 'this PC has no Windows Hello device'
    case 'NotConfiguredForUser':
      return 'Windows Hello (a PIN, face or fingerprint) is not set up for you'
    case 'DisabledByPolicy':
      return 'Windows Hello is turned off by policy on this PC'
    case 'DeviceBusy':
      return 'Windows Hello is busy right now'
    default:
      return `Windows Hello is not available (${availability || 'unknown'})`
  }
}

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

  /** After available() said no: why, for the card that names the fallback. */
  why = ''

  /** False when Hello is not set up (no PIN or biometrics), disabled by policy, or the check fails. */
  async available(): Promise<boolean> {
    try {
      const reply = object(await this.worker.call('hello_available', {}, { timeoutMs: 20_000 }), 'Windows Hello')
      const availability = string(reply.availability, 'Windows Hello availability')
      this.why = availability === HELLO_AVAILABLE ? '' : helloUnavailable(availability)
      return availability === HELLO_AVAILABLE
    } catch {
      this.why = 'Windows Hello could not be checked'
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
        // Short waits, so an approval that ends another way cancels the prompt (hello_cancel) between them.
        const deadline = Date.now() + 120_000
        for (;;) {
          if (options.signal?.aborted || Date.now() > deadline) {
            await this.worker.call('hello_cancel', {}, { timeoutMs: 10_000 }).catch(() => undefined)
            return false
          }
          const reply = object(await this.worker.call('hello_wait', { ms: POLL_MS }, { timeoutMs: POLL_MS + 10_000 }), 'Windows Hello')
          const result = string(reply.result, 'Windows Hello result')
          if (result !== 'Pending') return result === 'Verified'
        }
      } catch {
        if (started) return false
      }
    }
    if (options.signal?.aborted) return false
    try {
      options.waiting?.(HELLO_BEHIND)
      // The plain request cannot be cancelled from here: an abort ends its worker call (the worker is restarted).
      const reply = object(await this.worker.call('hello_verify', { prompt: message }, { timeoutMs: 125_000, ...(options.signal ? { signal: options.signal } : {}) }), 'Windows Hello')
      return string(reply.result, 'Windows Hello result') === 'Verified'
    } catch {
      return false
    }
  }
}
