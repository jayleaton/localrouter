import { chmod, mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { boolean, json, object, optional, string } from './validate.ts'

/**
 * The helper's pairing: which box it dials, the device token it shows there, the name the box shows
 * for this Mac, and whether the person switched computer control off here. Kept in
 * `~/Library/Application Support/LocalRouter Desktop/config.json` (directory 0700, file 0600), or under
 * `LOCALROUTER_DESKTOP_HOME`. The token is never logged or printed.
 */
export type Config = { url: string; token: string; name: string; disabled: boolean }

export const DISABLED_MESSAGE = 'Computer control is switched off on this Mac'

export function configDir(env: NodeJS.ProcessEnv = process.env, home: string = homedir()): string {
  const override = env.LOCALROUTER_DESKTOP_HOME?.trim()
  return override ? override : join(home, 'Library', 'Application Support', 'LocalRouter Desktop')
}

export const configPath = (dir: string) => join(dir, 'config.json')
export const controlSocketPath = (dir: string) => join(dir, 'control.sock')

/** The pairing, or null when this Mac has not been paired yet. */
export async function loadConfig(dir: string): Promise<Config | null> {
  let text: string
  try {
    text = await readFile(configPath(dir), 'utf8')
  } catch (error) {
    if (typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT') return null
    throw error
  }
  const raw = object(json(text, 'config.json'), 'config.json')
  return {
    url: checkUrl(string(raw.url, 'config url')),
    token: string(raw.token, 'config token'),
    name: string(raw.name, 'config name'),
    disabled: optional(raw.disabled, 'config disabled', boolean) ?? false,
  }
}

/** Writes the pairing atomically: the directory is 0700 and the file 0600 before the token is in it. */
export async function saveConfig(dir: string, config: Config): Promise<void> {
  await mkdir(dir, { recursive: true, mode: 0o700 })
  await chmod(dir, 0o700)
  const file = configPath(dir)
  const temporary = `${file}.${process.pid}.tmp`
  await writeFile(temporary, `${JSON.stringify(config, null, 2)}\n`, { mode: 0o600 })
  await chmod(temporary, 0o600)
  await rename(temporary, file)
}

/** Switches computer control off or on here, keeping the rest of the pairing. */
export async function setDisabled(dir: string, disabled: boolean): Promise<Config> {
  const config = await loadConfig(dir)
  if (!config) throw new Error('This Mac is not paired yet: run `node src/main.ts pair --url … --token …` first.')
  const next = { ...config, disabled }
  await saveConfig(dir, next)
  return next
}

const LOOPBACK = /^(localhost|127(?:\.\d{1,3}){3}|\[::1\])$/i

/**
 * The box's base address: https, or http only to this Mac itself (a box on loopback, for testing).
 * Returned without a trailing slash, query or fragment.
 */
export function checkUrl(value: string): string {
  let url: URL
  try {
    url = new URL(value.trim())
  } catch {
    throw new Error(`${value} is not a URL`)
  }
  if (url.username || url.password) throw new Error('The URL should not carry a user name or password')
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && LOOPBACK.test(url.hostname))) {
    throw new Error('The URL should be https:// (http:// only to this Mac itself)')
  }
  url.search = ''
  url.hash = ''
  return url.toString().replace(/\/+$/, '')
}

/** The WebSocket address of the box's desktop endpoint: `<url>/v1/desktop`, https → wss, http → ws. */
export function desktopSocketUrl(base: string): string {
  const url = new URL(`${checkUrl(base)}/v1/desktop`)
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
  return url.toString()
}

/** The pairing without its token, for printing and logs. */
export function describeConfig(config: Config) {
  return { url: config.url, name: config.name, disabled: config.disabled, token: config.token ? 'set' : 'missing' }
}
