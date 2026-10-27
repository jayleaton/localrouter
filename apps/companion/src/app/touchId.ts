/**
 * Touch ID through `localrouter-presence` (native/presence-helper.m), shipped in the app's Resources: unlike
 * Electron's promptTouchID, its prompt is dismissed when the approval ends another way (the process is sent SIGTERM),
 * and when Touch ID cannot be used it says why (LAError's code), so the card can name the fallback and the reason.
 */
import { spawn } from 'node:child_process'

export const PRESENCE_HELPER = 'localrouter-presence'

/** Whether Touch ID can be used now, and when not, why in words. */
export type TouchIdProbe = { available: boolean; why: string }

/** LAError's code from canEvaluatePolicy, as a reason a person can act on. */
export function touchIdUnavailable(code: number): string {
  switch (code) {
    case -5:
      return 'Touch ID is not available: this Mac has no login password'
    case -6:
      return 'Touch ID is not available on this Mac right now (no sensor, or the lid is closed with no Touch ID keyboard)'
    case -7:
      return 'Touch ID is not available: no fingerprint is enrolled'
    case -8:
      return 'Touch ID is locked after too many tries: unlock this Mac with its password once'
    case -12:
      return 'Touch ID is not available: no Magic Keyboard with Touch ID is paired'
    case -13:
      return 'Touch ID is not available: the Magic Keyboard with Touch ID is not connected'
    default:
      return `Touch ID is not available (macOS error ${code})`
  }
}

/** The helper's command: its path (tests: node and a stand-in script). */
export type HelperCommand = { file: string; args?: string[] }

export class TouchIdHelper {
  private readonly command: HelperCommand

  constructor(command: HelperCommand) {
    this.command = command
  }

  /** Never throws: a helper that cannot run counts as Touch ID not available, with that reason. */
  async check(): Promise<TouchIdProbe> {
    try {
      const out = await this.run('check')
      const v = JSON.parse(out.trim()) as { biometrics?: unknown; code?: unknown }
      if (v.biometrics === true) return { available: true, why: '' }
      return { available: false, why: touchIdUnavailable(typeof v.code === 'number' ? v.code : -6) }
    } catch {
      return { available: false, why: 'Touch ID could not be checked on this Mac' }
    }
  }

  /** True only when Touch ID confirmed the person. An abort dismisses the prompt (SIGTERM, then stdin closed) and is false. */
  verify(prompt: string, signal?: AbortSignal): Promise<boolean> {
    return new Promise((resolve) => {
      if (signal?.aborted) return resolve(false)
      const child = spawn(this.command.file, [...(this.command.args ?? []), 'verify'], { stdio: ['pipe', 'pipe', 'ignore'] })
      let settled = false
      const finish = (ok: boolean) => {
        if (settled) return
        settled = true
        signal?.removeEventListener('abort', onAbort)
        resolve(ok)
      }
      const onAbort = () => {
        child.kill('SIGTERM')
        child.stdin.end()
        finish(false)
      }
      signal?.addEventListener('abort', onAbort, { once: true })
      child.on('error', () => finish(false))
      child.on('exit', (code) => finish(code === 0))
      child.stdin.on('error', () => undefined)
      // The reason on one line; stdin stays open: closing it is the other way to dismiss the prompt.
      child.stdin.write(`${prompt.replace(/[\r\n]+/g, ' ').slice(0, 200)}\n`)
    })
  }

  private run(op: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const child = spawn(this.command.file, [...(this.command.args ?? []), op], { stdio: ['ignore', 'pipe', 'ignore'] })
      let out = ''
      const timer = setTimeout(() => child.kill('SIGKILL'), 10_000)
      child.stdout.setEncoding('utf8').on('data', (d: string) => (out += d))
      child.on('error', reject)
      child.on('close', (code) => {
        clearTimeout(timer)
        if (code === 0) resolve(out)
        else reject(new Error(`exit ${code}`))
      })
    })
  }
}
