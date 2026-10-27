/**
 * The standing grant for computer control as the app keeps it (BrokerOptions.standing): when the person allowed it,
 * stored with the Spark tokens (secrets.ts: the keychain helper on macOS, safeStorage elsewhere), so a restart or an
 * app update never asks again. `computer_grant` is empty (never decided), an ISO time (granted) or `revoked:<time>`
 * (turned off in Settings: an install is then never migrated back on). "Ask before each action" is a setting.
 */
import type { Standing } from '../executor/broker.ts'
import type { AccessReport } from '../platform/types.ts'
import type { Secrets } from './secrets.ts'

export type SecretsStore = { load(): Secrets; save(s: Secrets): void }

const REVOKED = 'revoked:'

export class StoredStanding implements Standing {
  private readonly store: SecretsStore
  private readonly askEach: () => boolean
  private readonly now: () => Date

  constructor(store: SecretsStore, askEach: () => boolean, now: () => Date = () => new Date()) {
    this.store = store
    this.askEach = askEach
    this.now = now
  }

  granted(): boolean {
    const v = this.store.load().computer_grant
    return Boolean(v) && !v.startsWith(REVOKED)
  }

  /** For the migration: only `revoked` (an explicit Revoke) blocks it. */
  get decision(): 'undecided' | 'granted' | 'revoked' {
    return this.granted() ? 'granted' : this.undecided ? 'undecided' : 'revoked'
  }

  /** Never granted nor revoked: an install the migration may grant. */
  get undecided(): boolean {
    return !this.store.load().computer_grant
  }

  set(granted: boolean): void {
    const at = this.now().toISOString()
    this.store.save({ ...this.store.load(), computer_grant: granted ? at : `${REVOKED}${at}` })
  }

  askEachAction(): boolean {
    return this.askEach()
  }

  /** When the grant was given (for Settings), or null. */
  get since(): string | null {
    return this.granted() ? this.store.load().computer_grant : null
  }
}

/**
 * An install that has been paired (ever: a token the box has since revoked still counts, Mac run 2026-10-27), with the
 * OS grants computer control needs in place (macOS: Accessibility and Screen Recording), and that never decided: that
 * setup was the consent (JJ, 2026-10-26), so it gets the standing grant with no card, at start and again after every
 * pairing. Revoked in Settings, it is not migrated back.
 */
export function migrates(i: { paired: boolean; undecided: boolean; os: string; access: Pick<AccessReport, 'accessibility' | 'input' | 'screen_recording'> | null }): boolean {
  const a = i.access
  return i.paired && i.undecided && a !== null && a.accessibility && a.input && (i.os !== 'darwin' || a.screen_recording)
}

type Access = Pick<AccessReport, 'accessibility' | 'input' | 'screen_recording'>

/**
 * The migration's decision and why, for the log (`consent.migrate`). Only an explicit Revoke blocks it: no earlier
 * "not eligible" is kept, so it is decided afresh at every launch, pairing and permission change.
 *
 * `access`: the executor's report (a helper process: on a Mac an osascript child, whose Screen Recording answer is not
 * always the app's). `app`: the app's own in-process answers (macOS: AXIsProcessTrusted and the Screen Recording
 * status), the same ones setup shows as granted (Mac run, 2026-10-27: both granted there, yet no migration). Either
 * saying yes counts.
 */
export function migrationDecision(i: { grant: 'undecided' | 'granted' | 'revoked'; everPaired: boolean; os: string; access: Access | null; app?: { accessibility: boolean; screen_recording: boolean } | null }): { eligible: boolean; reason: string } {
  if (i.grant === 'granted') return { eligible: false, reason: 'already granted' }
  if (i.grant === 'revoked') return { eligible: false, reason: 'revoked in Settings' }
  if (!i.everPaired) return { eligible: false, reason: 'never paired' }
  const ax = Boolean(i.access?.accessibility || i.app?.accessibility)
  const input = Boolean(i.access?.input || (i.os === 'darwin' && i.app?.accessibility))
  const sr = Boolean(i.access?.screen_recording || i.app?.screen_recording)
  if (!ax || !input) return { eligible: false, reason: 'Accessibility not granted' }
  if (i.os === 'darwin' && !sr) return { eligible: false, reason: 'Screen Recording not granted' }
  return { eligible: true, reason: 'paired before, OS permissions granted, never revoked' }
}

/** Paired at some point: an address kept, a token kept (even one revoked since), or a pairing recorded. */
export function everPaired(i: { base: string; token: string; pairedAt?: string }): boolean {
  return Boolean(i.base || i.token || i.pairedAt)
}

/**
 * Settings › "Let Jarvis control this computer": ticked only when control is on AND the person allowed it, so a ticked
 * box never sits next to "not allowed yet". Ticking it grants (`grant`) as well as turning control on; unticking pauses.
 */
export function controlBox(i: { enabled: boolean; granted: boolean }): boolean {
  return i.enabled && i.granted
}
export function controlTicked(on: boolean, granted: boolean): { enable: boolean; grant: boolean } {
  return { enable: on, grant: on && !granted }
}
