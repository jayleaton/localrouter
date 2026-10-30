import { describe, expect, it } from 'vitest'
import { grantStates, micAskAtStart, micResetBeforeAsk, signingId } from './grants.ts'

const all = (v: boolean) => ({ accessibility: v, screen: v, microphone: v })
const SIGNED = signingId('identifier "com.localrouter.companion" and certificate leaf = H"ab12"')
const ADHOC = signingId('cdhash H"77ee"')

describe('macOS grants across builds', () => {
  it('granted in this process: recorded under this signing identity', () => {
    expect(grantStates(all(true), { accessibility: true }, {}, SIGNED)).toEqual({ state: { accessibility: 'granted', screen: 'granted', microphone: 'granted' }, grants: { accessibility: SIGNED, screen: SIGNED, microphone: SIGNED } })
  })

  it('Screen Recording allowed in this process is granted, whatever another process’s probe says (JJ’s MacBook, 2026-10-25)', () => {
    // TCC: ScreenCapture Allowed (System Set); the osascript preflight said no. Only Accessibility's probe counts.
    const { state } = grantStates(all(true), { accessibility: true }, { screen: SIGNED }, SIGNED)
    expect(state.screen).toBe('granted')
  })

  it('a new build signed by the same certificate keeps its grants: an update is not a stale grant', () => {
    expect(grantStates(all(true), { accessibility: true }, { accessibility: SIGNED, screen: SIGNED, microphone: SIGNED }, SIGNED).state).toEqual({ accessibility: 'granted', screen: 'granted', microphone: 'granted' })
  })

  it('macOS says Accessibility is on but an AX read fails: stale (the older build’s entry)', () => {
    expect(grantStates({ accessibility: true, screen: false, microphone: true }, { accessibility: false }, {}, SIGNED).state).toEqual({ accessibility: 'stale', screen: 'not-granted', microphone: 'granted' })
  })

  it('missing now, seen working under another signature (the ad-hoc build): stale, Microphone included', () => {
    expect(grantStates(all(false), { accessibility: false }, { accessibility: ADHOC, screen: ADHOC, microphone: ADHOC }, SIGNED).state).toEqual({ accessibility: 'stale', screen: 'stale', microphone: 'stale' })
  })

  it('missing now, seen working under this same signature: revoked, not stale', () => {
    expect(grantStates(all(false), null, { microphone: SIGNED }, SIGNED).state.microphone).toBe('not-granted')
  })

  it('entries from before signatures were recorded, and an unreadable signature, call nothing stale', () => {
    expect(grantStates(all(false), null, { accessibility: '0.1.0@1761300000000', screen: '0.1.0@1761300000000' }, SIGNED)).toEqual({ state: { accessibility: 'not-granted', screen: 'not-granted', microphone: 'not-granted' }, grants: {} })
    expect(grantStates(all(false), null, { screen: ADHOC }, null).state.screen).toBe('not-granted')
  })

  it('no repeated prompts: a Microphone entry from an older signature is reset once and asked once; updates keep it (Mac run, 2026-10-26)', () => {
    // The c44379c install: macOS failed to match the old Microphone entry and reported it as not granted.
    expect(micResetBeforeAsk('not-determined')).toBe(true)
    expect(micResetBeforeAsk('denied')).toBe(true)
    // Asked once in setup: granted under the stable certificate, recorded under it.
    const after = grantStates({ accessibility: true, screen: true, microphone: true }, { accessibility: true }, {}, SIGNED)
    expect(after.grants.microphone).toBe(SIGNED)
    // An update (a new build, same certificate): still granted, so nothing is reset and nothing asks.
    expect(grantStates({ accessibility: true, screen: true, microphone: true }, { accessibility: true }, after.grants, SIGNED).state.microphone).toBe('granted')
    expect(micResetBeforeAsk('granted')).toBe(false)
  })
})

describe('the Microphone asked once at start (Mac run, 2026-10-27: the stale entry stayed and nothing asked)', () => {
  it('not granted: reset and ask once per signing identity, never again under it; granted or unsigned: never', () => {
    expect(micAskAtStart({ status: 'not-determined', identity: 'sig:a' })).toBe(true)
    expect(micAskAtStart({ status: 'denied', identity: 'sig:a', askedFor: 'sig:a' })).toBe(false)
    expect(micAskAtStart({ status: 'denied', identity: 'sig:b', askedFor: 'sig:a' })).toBe(true)
    expect(micAskAtStart({ status: 'granted', identity: 'sig:a' })).toBe(false)
    expect(micAskAtStart({ status: 'not-determined', identity: null })).toBe(false)
  })
})
