import { describe, expect, it } from 'vitest'
import type { Card } from '../executor/broker.ts'
import { fakeBroker } from '../executor/testing.ts'
import { EMPTY, parseSecrets } from './secrets.ts'
import { StoredStanding } from './standing.ts'

/** The stored blob as the keychain helper or safeStorage keeps it: JSON text, read back by whichever build runs next. */
function storedAs(text: string) {
  const box = { text }
  return { box, store: { load: () => parseSecrets(box.text), save: (s: typeof EMPTY) => void (box.text = JSON.stringify(s)) } }
}

async function cardsFor(standing: StoredStanding) {
  const { broker } = fakeBroker(['c4', 'c3', 'done'], { standing })
  const shown: Card[] = []
  broker.on((e) => {
    if (e.type !== 'approval') return
    shown.push(e.card)
    setTimeout(() => void broker.answer(e.card.id, 'approve', 'app window'), 1)
  })
  const started = broker.startUse({ caller: { id: 'app', name: 'LocalRouter', kind: 'app' } }, 'send the Groceries note', null)
  if (!started.ok) throw new Error(started.error)
  expect((await started.finished).status).toBe('done')
  return shown.map((c) => c.action)
}

describe('the standing grant, stored with the tokens', () => {
  it('asked once; a restart and an update (a new build reading the same stored blob) do not ask again', async () => {
    // An install from before the grant existed: its blob has tokens only. The first computer use asks once.
    const { box, store } = storedAs(JSON.stringify({ token: 'lr_tok', legacy_token: '' }))
    expect(await cardsFor(new StoredStanding(store, () => false, () => new Date('2026-10-26T21:40:00Z')))).toEqual(['control this computer'])
    expect(JSON.parse(box.text)).toEqual({ token: 'lr_tok', legacy_token: '', computer_grant: '2026-10-26T21:40:00.000Z' })
    // Restart: a new StoredStanding over what was saved.
    expect(await cardsFor(new StoredStanding(storedAs(box.text).store, () => false))).toEqual([])
    // Update: a newer build that stores one more field still reads the grant.
    const newer = storedAs(JSON.stringify({ ...JSON.parse(box.text), some_later_field: 'x' }))
    expect(await cardsFor(new StoredStanding(newer.store, () => false))).toEqual([])
  })

  it('revoking clears only the grant (the tokens stay); "Ask before each action" asks per step', async () => {
    const { box, store } = storedAs(JSON.stringify({ token: 'lr_tok', legacy_token: '', computer_grant: '2026-10-26T21:40:00.000Z' }))
    const standing = new StoredStanding(store, () => false)
    expect(standing.since).toBe('2026-10-26T21:40:00.000Z')
    standing.set(false)
    expect(JSON.parse(box.text)).toEqual({ token: 'lr_tok', legacy_token: '', computer_grant: '' })
    expect(await cardsFor(standing)).toEqual(['control this computer'])
    expect(await cardsFor(new StoredStanding(storedAs(box.text).store, () => true))).toEqual(['click “Send” in Notes'])
  })
})
