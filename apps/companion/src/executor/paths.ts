import { homedir } from 'node:os'
import { join } from 'node:path'

/**
 * Where the companion keeps its data on this computer: the app's folder on macOS (the same one Electron's
 * `userData` names), an XDG state folder elsewhere. `LOCALROUTER_COMPANION_HOME` overrides it (tests, a second copy).
 */
export function dataDir(env: NodeJS.ProcessEnv = process.env, platform: NodeJS.Platform = process.platform, home: string = homedir()): string {
  const set = env.LOCALROUTER_COMPANION_HOME?.trim()
  if (set) return set
  if (platform === 'darwin') return join(home, 'Library', 'Application Support', 'LocalRouter')
  return join(env.XDG_STATE_HOME?.trim() || join(home, '.local', 'state'), 'localrouter-companion')
}

export const executorSocket = (dir: string) => join(dir, 'executor.sock')
export const callersPath = (dir: string) => join(dir, 'callers.json')
export const auditPath = (dir: string) => join(dir, 'audit.jsonl')
export const identityPath = (dir: string) => join(dir, 'executor.json')
