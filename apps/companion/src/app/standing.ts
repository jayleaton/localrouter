/**
 * The standing grant for computer control as the app keeps it (BrokerOptions.standing): the time the person said yes
 * to "Let Jarvis control this computer?", stored with the Spark tokens (secrets.ts: the keychain helper on macOS,
 * safeStorage elsewhere), so a restart or an app update never asks again. "Ask before each action" is a setting.
 */
import type { Standing } from '../executor/broker.ts'
import type { Secrets } from './secrets.ts'

export type SecretsStore = { load(): Secrets; save(s: Secrets): void }

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
    return Boolean(this.store.load().computer_grant)
  }

  set(granted: boolean): void {
    this.store.save({ ...this.store.load(), computer_grant: granted ? this.now().toISOString() : '' })
  }

  askEachAction(): boolean {
    return this.askEach()
  }

  /** When the grant was given (for Settings), or null. */
  get since(): string | null {
    return this.store.load().computer_grant || null
  }
}
