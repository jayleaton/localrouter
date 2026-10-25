import { object, string } from '../../../../desktop-helper/src/validate.ts'
import type { PlatformPresence } from '../types.ts'
import type { WindowsCall } from './worker.ts'

/** UserConsentVerifierAvailability values: only `Available` means Windows Hello can confirm the person now. */
export const HELLO_AVAILABLE = 'Available'

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

  async verify(prompt: string): Promise<boolean> {
    try {
      const message = prompt.replace(/\s+/g, ' ').trim().slice(0, 200) || 'Approve this action'
      const reply = object(await this.worker.call('hello_verify', { prompt: message }, { timeoutMs: 125_000 }), 'Windows Hello')
      return string(reply.result, 'Windows Hello result') === 'Verified'
    } catch {
      return false
    }
  }
}
