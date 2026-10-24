import { describe, expect, it } from 'vitest'
import { grantStates } from './grants.ts'

const on = { accessibility: true, screen: true }
const off = { accessibility: false, screen: false }

describe('macOS grants across builds', () => {
  it('granted and working: recorded for this build', () => {
    expect(grantStates(on, on, {}, 'b2')).toEqual({ state: { accessibility: 'granted', screen: 'granted' }, grants: { accessibility: 'b2', screen: 'b2' } })
  })
  it('macOS says yes but the executor cannot read the screen: stale (the older build’s entry)', () => {
    expect(grantStates({ accessibility: true, screen: false }, { accessibility: false, screen: false }, {}, 'b2').state).toEqual({ accessibility: 'stale', screen: 'not-granted' })
  })
  it('worked in an older build, not in this one: stale', () => {
    expect(grantStates(off, off, { accessibility: 'b1', screen: 'b1' }, 'b2').state).toEqual({ accessibility: 'stale', screen: 'stale' })
  })
  it('worked in this build and is gone: revoked, not stale', () => {
    expect(grantStates(off, off, { accessibility: 'b2' }, 'b2').state.accessibility).toBe('not-granted')
  })
  it('never granted: not granted', () => {
    expect(grantStates(off, null, {}, 'b1').state).toEqual({ accessibility: 'not-granted', screen: 'not-granted' })
  })
})
