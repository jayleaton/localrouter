import { createHash } from 'node:crypto'
import { homedir, userInfo } from 'node:os'
import { posix, win32 } from 'node:path'

/**
 * Where the companion keeps its data on this computer: the app's folder on macOS (the same one Electron's
 * `userData` names), `%APPDATA%\LocalRouter` on Windows, an XDG state folder elsewhere. `LOCALROUTER_COMPANION_HOME`
 * overrides it (tests, a second copy).
 */
export function dataDir(env: NodeJS.ProcessEnv = process.env, platform: NodeJS.Platform = process.platform, home: string = homedir()): string {
  const set = env.LOCALROUTER_COMPANION_HOME?.trim()
  if (set) return set
  // Joined for the OS asked about, not the one running (the tests ask about every OS from one).
  if (platform === 'win32') return win32.join(env.APPDATA?.trim() || win32.join(home, 'AppData', 'Roaming'), 'LocalRouter')
  const join = posix.join
  if (platform === 'darwin') return join(home, 'Library', 'Application Support', 'LocalRouter')
  return join(env.XDG_STATE_HOME?.trim() || join(home, '.local', 'state'), 'localrouter-companion')
}

/**
 * Where the executor listens for other harnesses: a Unix socket in the data folder (mode 0600), or on Windows, which
 * has no such socket for Node, a named pipe named after the user and the data folder. A pipe's default security lets
 * other local accounts open it for reading only, so the token every request carries (an enrolled caller's) is what
 * admits it, as on the socket.
 */
export function executorSocket(dir: string, platform: NodeJS.Platform = process.platform): string {
  if (platform !== 'win32') return posix.join(dir, 'executor.sock')
  const key = createHash('sha256').update(`${userInfo().username}\0${dir.toLowerCase()}`).digest('hex').slice(0, 16)
  return `\\\\.\\pipe\\localrouter-executor-${key}`
}
const local = process.platform === 'win32' ? win32 : posix
export const callersPath = (dir: string) => local.join(dir, 'callers.json')
export const auditPath = (dir: string) => local.join(dir, 'audit.jsonl')
export const identityPath = (dir: string) => local.join(dir, 'executor.json')
