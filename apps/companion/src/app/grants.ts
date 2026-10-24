/**
 * Whether a macOS permission is granted, missing, or stale: macOS ties a grant to the build's code requirement, so an
 * older build's grant can still show as on in System Settings and do nothing for this one. Stale: macOS says granted
 * but the executor's own check fails, or it worked in another build and does not in this one. A grant that worked in
 * this same build and is gone was revoked, not stale. Pure, so it is tested without a Mac.
 */
export type Permission = 'accessibility' | 'screen'
export type GrantState = 'granted' | 'not-granted' | 'stale'

export function grantStates(said: Record<Permission, boolean>, works: Record<Permission, boolean> | null, seen: Partial<Record<Permission, string>>, build: string) {
  const grants = { ...seen }
  const state = {} as Record<Permission, GrantState>
  for (const p of ['accessibility', 'screen'] as const) {
    const ok = said[p] && (works ? works[p] : true)
    if (ok) {
      grants[p] = build
      state[p] = 'granted'
    } else {
      const stale = (said[p] && works !== null && !works[p]) || (Boolean(seen[p]) && seen[p] !== build)
      state[p] = stale ? 'stale' : 'not-granted'
    }
  }
  return { state, grants }
}
