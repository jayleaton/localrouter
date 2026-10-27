import { createHash } from 'node:crypto'

/**
 * Whether a macOS permission is granted, missing, or stale. macOS ties a grant to the app's designated requirement
 * (its signing identity), so a grant given to a build signed differently (the ad-hoc builds before the stable
 * certificate) can still show as on in System Settings and do nothing for this one.
 *
 * The truth for each permission is what macOS answers this process now: Accessibility's trust check, and for Screen
 * Recording and the Microphone the in-process TCC status (`CGPreflightScreenCaptureAccess`, `AVCaptureDevice`), through
 * Electron. A probe from another process (osascript) is not the truth for Screen Recording: its preflight is not
 * always bridged and is attributed differently, which once called a working grant stale. Only Accessibility has a
 * working probe (an AX read), and it counts only to say a "yes" does not work.
 *
 * Stale means: macOS says yes and the AX probe fails (Accessibility only), or the grant is missing now and was seen
 * working under another signing identity. A grant seen under this same identity and gone now was revoked, not stale.
 * Pure, so it is tested without a Mac.
 */
export type Permission = 'accessibility' | 'screen' | 'microphone'
export type GrantState = 'granted' | 'not-granted' | 'stale'

export const PERMISSIONS = ['accessibility', 'screen', 'microphone'] as const

/** What a grant was last seen working under: `sig:` and a digest of the designated requirement. */
export function signingId(requirement: string): string {
  return `sig:${createHash('sha256').update(requirement.trim()).digest('hex').slice(0, 16)}`
}

/**
 * `said`: macOS's answer in this process. `works`: the Accessibility probe (null when it could not run). `seen`: the
 * signing identity each permission was last seen working under. `identity`: this build's (`signingId`), or null when
 * it cannot be read (then nothing is called stale from history).
 */
export function grantStates(said: Record<Permission, boolean>, works: { accessibility: boolean } | null, seen: Partial<Record<Permission, string>>, identity: string | null) {
  const grants: Partial<Record<Permission, string>> = {}
  // Entries from before identities were recorded (a version and a file time) say nothing about the signature.
  for (const p of PERMISSIONS) if (seen[p]?.startsWith('sig:')) grants[p] = seen[p]
  const state = {} as Record<Permission, GrantState>
  for (const p of PERMISSIONS) {
    const probeFails = p === 'accessibility' && works !== null && !works.accessibility
    if (said[p] && !probeFails) {
      if (identity) grants[p] = identity
      state[p] = 'granted'
      continue
    }
    const before = grants[p]
    const otherIdentity = Boolean(identity && before && before !== identity)
    state[p] = (said[p] && probeFails) || otherIdentity ? 'stale' : 'not-granted'
  }
  return { state, grants }
}

/**
 * Whether to reset this app's own Microphone entry before asking. macOS keeps an entry made under another signature
 * (the ad-hoc builds) and never prompts again while it fails to match this build ("Failed to match existing code
 * requirement … kTCCServiceMicrophone", Mac run, 2026-10-26), reporting it as not granted. So whenever the Microphone
 * is not granted now, the entry is reset first and macOS asks once; that grant is tied to the stable certificate's
 * requirement and survives every later update. A working grant is never touched.
 */
export function micResetBeforeAsk(status: string): boolean {
  return status !== 'granted'
}

/**
 * Whether to reset and ask for the Microphone by itself, once, at start (Mac run, 2026-10-27: the stale entry was still
 * there and nothing asked, because the Microphone was only asked from its button). The mismatch cannot be read without
 * the TCC database, so: not granted now, under a signing identity that can be read, and not asked yet under this one.
 * Asked once per identity, so a person who says no is not asked again until a differently signed build.
 */
export function micAskAtStart(i: { status: string; identity: string | null; askedFor?: string }): boolean {
  return micResetBeforeAsk(i.status) && i.identity !== null && i.askedFor !== i.identity
}
