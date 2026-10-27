import { describe, expect, it } from 'vitest'
import { CONTROL_OFF } from '../executor/broker.ts'
import { fakeBroker } from '../executor/testing.ts'
import { EMPTY, parseSecrets } from './secrets.ts'
import { controlBox, controlTicked, everPaired, migrates, migrationDecision, StoredStanding } from './standing.ts'

/** The stored blob as the keychain helper or safeStorage keeps it: JSON text, read back by whichever build runs next. */
function storedAs(text: string) {
  const box = { text }
  return { box, store: { load: () => parseSecrets(box.text), save: (s: typeof EMPTY) => void (box.text = JSON.stringify(s)) } }
}

const app = { caller: { id: 'app', name: 'LocalRouter', kind: 'app' as const } }
const granted = { accessibility: true, input: true, screen_recording: true }

/** One "send the Groceries note" task; the cards it showed and how it ended. */
async function task(standing: StoredStanding) {
  const { broker } = fakeBroker(['c4', 'c3', 'done'], { standing })
  const cards: string[] = []
  broker.on((e) => void (e.type === 'approval' && cards.push(e.card.action)))
  const started = broker.startUse(app, 'send the Groceries note', null)
  return { cards, outcome: started.ok ? (await started.finished).status : started.error, broker }
}

describe('the standing grant, stored with the tokens (JJ, 2026-10-26: no permission window during a task)', () => {
  it('an install already paired with the OS grants is migrated: granted at start with no card, and tasks just run', async () => {
    const { box, store } = storedAs(JSON.stringify({ token: 'lr_tok', legacy_token: '' }))
    const standing = new StoredStanding(store, () => false, () => new Date('2026-10-26T23:00:00Z'))
    expect(migrates({ paired: true, undecided: standing.undecided, os: 'darwin', access: granted })).toBe(true)
    standing.set(true)
    expect(JSON.parse(box.text)).toEqual({ token: 'lr_tok', legacy_token: '', computer_grant: '2026-10-26T23:00:00.000Z' })
    expect(await task(standing)).toMatchObject({ cards: [], outcome: 'done' })
    // Not paired, or a grant missing (Screen Recording on a Mac), or already decided: no migration.
    expect(migrates({ paired: false, undecided: true, os: 'darwin', access: granted })).toBe(false)
    expect(migrates({ paired: true, undecided: true, os: 'darwin', access: { ...granted, screen_recording: false } })).toBe(false)
    expect(migrates({ paired: true, undecided: true, os: 'win32', access: { ...granted, screen_recording: false } })).toBe(true)
    expect(migrates({ paired: true, undecided: false, os: 'darwin', access: granted })).toBe(false)
  })

  it('given once (onboarding): a restart and an update (a new build reading the same blob) never ask', async () => {
    const { box, store } = storedAs(JSON.stringify({ token: 'lr_tok', legacy_token: '' }))
    new StoredStanding(store, () => false).set(true)
    expect(await task(new StoredStanding(storedAs(box.text).store, () => false))).toMatchObject({ cards: [], outcome: 'done' })
    const newer = storedAs(JSON.stringify({ ...JSON.parse(box.text), some_later_field: 'x' }))
    expect(await task(new StoredStanding(newer.store, () => false))).toMatchObject({ cards: [], outcome: 'done' })
  })

  it('not given yet: a task is refused at once with no card; revoked: refused, and never migrated back on', async () => {
    const fresh = new StoredStanding(storedAs(JSON.stringify({ token: '', legacy_token: '' })).store, () => false)
    expect(await task(fresh)).toEqual(expect.objectContaining({ cards: [], outcome: CONTROL_OFF }))
    const { box, store } = storedAs(JSON.stringify({ token: 'lr_tok', legacy_token: '', computer_grant: '2026-10-26T21:40:00.000Z' }))
    const standing = new StoredStanding(store, () => false)
    expect(standing.since).toBe('2026-10-26T21:40:00.000Z')
    standing.set(false)
    expect(JSON.parse(box.text).token).toBe('lr_tok')
    expect(standing.granted()).toBe(false)
    expect(standing.since).toBeNull()
    expect(migrates({ paired: true, undecided: standing.undecided, os: 'darwin', access: granted })).toBe(false)
    expect(await task(standing)).toEqual(expect.objectContaining({ cards: [], outcome: CONTROL_OFF }))
  })

  it('Mac run 2026-10-27: the box revoked the token, so nothing migrated at start; the re-pair migrates, with no card', async () => {
    const { store } = storedAs(JSON.stringify({ token: '', legacy_token: '' }))
    const standing = new StoredStanding(store, () => false)
    // A fresh install (no address, no token, never paired) is not migrated.
    expect(migrates({ paired: everPaired({ base: '', token: '' }), undecided: standing.undecided, os: 'darwin', access: granted })).toBe(false)
    // Paired before (the address is kept), the token since revoked: still counts as paired, so it migrates at start.
    expect(everPaired({ base: 'https://spark.ts.net:10443', token: '' })).toBe(true)
    // The pairing records pairedAt and reruns the migration: the grant is given, and the task runs.
    const after = { base: 'https://spark.ts.net:10443', token: 'lr_new', pairedAt: '2026-10-27T09:25:00Z' }
    expect(migrates({ paired: everPaired(after), undecided: standing.undecided, os: 'darwin', access: granted })).toBe(true)
    standing.set(true)
    expect(await task(standing)).toMatchObject({ cards: [], outcome: 'done' })
    // A revoked grant is never migrated back by a pairing.
    standing.set(false)
    expect(migrates({ paired: everPaired(after), undecided: standing.undecided, os: 'darwin', access: granted })).toBe(false)
  })

  it('the Settings box is ticked only when control is allowed; ticking it once grants', () => {
    // The live bug: ticked (control on) next to "not allowed yet".
    expect(controlBox({ enabled: true, granted: false })).toBe(false)
    expect(controlBox({ enabled: true, granted: true })).toBe(true)
    expect(controlBox({ enabled: false, granted: true })).toBe(false)
    expect(controlTicked(true, false)).toEqual({ enable: true, grant: true })
    expect(controlTicked(true, true)).toEqual({ enable: true, grant: false })
    // Unticking pauses control; it is not a revoke.
    expect(controlTicked(false, true)).toEqual({ enable: false, grant: false })
  })

  it('Mac run 2026-10-27 (e634493): granted in setup but not in the helper’s check → migrated anyway; only a Revoke blocks it', () => {
    const helperSaysNo = { accessibility: true, input: true, screen_recording: false }
    const appSaysYes = { accessibility: true, screen_recording: true }
    expect(migrationDecision({ grant: 'undecided', everPaired: true, os: 'darwin', access: helperSaysNo, app: appSaysYes })).toEqual({ eligible: true, reason: 'paired before, OS permissions granted, never revoked' })
    // The helper could not answer at all: the app's own answers are enough.
    expect(migrationDecision({ grant: 'undecided', everPaired: true, os: 'darwin', access: null, app: appSaysYes }).eligible).toBe(true)
    // An earlier "not eligible" is not kept: decided afresh each time (here: not paired yet, then paired).
    expect(migrationDecision({ grant: 'undecided', everPaired: false, os: 'darwin', access: granted, app: appSaysYes })).toEqual({ eligible: false, reason: 'never paired' })
    expect(migrationDecision({ grant: 'undecided', everPaired: true, os: 'darwin', access: granted, app: appSaysYes }).eligible).toBe(true)
    // The reasons it does not, for the log.
    expect(migrationDecision({ grant: 'revoked', everPaired: true, os: 'darwin', access: granted, app: appSaysYes })).toEqual({ eligible: false, reason: 'revoked in Settings' })
    expect(migrationDecision({ grant: 'undecided', everPaired: true, os: 'darwin', access: helperSaysNo, app: { accessibility: true, screen_recording: false } }).reason).toBe('Screen Recording not granted')
    expect(migrationDecision({ grant: 'undecided', everPaired: true, os: 'win32', access: { accessibility: true, input: true, screen_recording: false }, app: null }).eligible).toBe(true)
  })

  it('the stored state the decision reads: only an explicit Revoke is "revoked"', () => {
    const { store } = storedAs(JSON.stringify({ token: 'lr_tok', legacy_token: '' }))
    const standing = new StoredStanding(store, () => false)
    expect(standing.decision).toBe('undecided')
    standing.set(true)
    expect(standing.decision).toBe('granted')
    standing.set(false)
    expect(standing.decision).toBe('revoked')
    // An older build's empty value (the 2026-10-26 consent card, never answered) is undecided, not a refusal.
    expect(new StoredStanding(storedAs(JSON.stringify({ token: 'x', legacy_token: '', computer_grant: '' })).store, () => false).decision).toBe('undecided')
  })
})
