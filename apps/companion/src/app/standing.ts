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
 * An install that is already paired, with the OS grants computer control needs in place (macOS: Accessibility and
 * Screen Recording), and that never decided: that setup was the consent (JJ, 2026-10-26), so it gets the standing
 * grant at start with no card. Revoked in Settings, it is not migrated back.
 */
export function migrates(i: { paired: boolean; undecided: boolean; os: string; access: Pick<AccessReport, 'accessibility' | 'input' | 'screen_recording'> | null }): boolean {
  const a = i.access
  return i.paired && i.undecided && a !== null && a.accessibility && a.input && (i.os !== 'darwin' || a.screen_recording)
}
