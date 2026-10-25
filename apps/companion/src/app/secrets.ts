import { spawn } from 'node:child_process'
import { chmod, copyFile, mkdir, readFile, rename, rm, stat } from 'node:fs/promises'
// The keychain helper is macOS's (Linux and Windows have none, and never reach these paths): POSIX paths on any host.
import { dirname, join } from 'node:path/posix'

/**
 * Where the app keeps its Spark tokens on macOS: the login keychain, through `localrouter-keychain`
 * (native/keychain-helper.c), so an app update never brings a keychain prompt.
 *
 * A keychain item made by an app without an Apple Team ID is partitioned to the code hash of the process that made
 * it, and every LocalRouter build has a new one: Electron's safeStorage (its key is such an item, made by the app)
 * asks again after each update. The helper is copied once to `<data>/bin` and kept there across updates, so the
 * process that touches the keychain keeps its code hash. Before each use the app checks the installed helper is
 * signed like the one it ships (`codesign --verify -R` with the bundled helper's designated requirement); a missing,
 * altered or older one is replaced, which costs that Mac one last prompt. The helper itself answers only a parent
 * that satisfies LocalRouter's designated requirement.
 */
export type Secrets = { token: string; legacy_token: string }
export const EMPTY: Secrets = { token: '', legacy_token: '' }
export const SERVICE = 'LocalRouter'
export const ACCOUNT = 'spark-tokens'
export const HELPER = 'localrouter-keychain'

export type Run = (file: string, args: string[], input?: string, o?: { timeoutMs?: number; done?: (stdout: string) => boolean; env?: NodeJS.ProcessEnv }) => Promise<{ code: number; stdout: string; stderr: string }>

/** A command's own `LR_STATUS <code>` line (the keychain helper's, or the sh wrapper's), and stderr without it. */
export function statusOf(stderr: string): { code: number | null; stderr: string } {
  const m = /(?:^|\n)LR_STATUS (\d+)\n?$/.exec(stderr)
  return m ? { code: Number(m[1]), stderr: stderr.slice(0, m.index) } : { code: null, stderr }
}

/**
 * Runs a command and gives its exit code and output. It is done when the command's output pipes close, not when
 * Node is told the process exited: in Electron's main process on macOS that notification can be lost (another part
 * of the process collects the exit first), which left the app waiting for ever on a helper that had finished. The
 * code comes from the exit event when it came, else from the command's `LR_STATUS` line. `done` ends it early once
 * the output says so (a child that leaves its own children holding the pipe).
 */
export const run: Run = (file, args, input, o: { timeoutMs?: number; done?: (stdout: string) => boolean; env?: NodeJS.ProcessEnv } = {}) =>
  new Promise((resolve) => {
    // Long enough for a person to answer a keychain prompt (a replaced helper's first read), never forever.
    const timeoutMs = o.timeoutMs ?? 180_000
    const child = spawn(file, args, { stdio: ['pipe', 'pipe', 'pipe'], ...(o.env ? { env: o.env } : {}) })
    let stdout = ''
    let stderr = ''
    let exitCode: number | null = null
    let open = 2
    let settled = false
    const finish = (forced?: number) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      const s = statusOf(stderr)
      resolve({ code: forced ?? exitCode ?? s.code ?? 1, stdout, stderr: s.stderr })
    }
    const timer = setTimeout(() => {
      child.kill('SIGKILL')
      finish(137)
    }, timeoutMs)
    const closed = () => {
      open -= 1
      // Both pipes closed: the command is gone. Give the exit event a moment, then take the status line.
      if (open === 0) setTimeout(() => finish(), exitCode === null && statusOf(stderr).code === null ? 200 : 0)
    }
    child.stdout.setEncoding('utf8').on('data', (d: string) => {
      stdout += d
      if (o.done?.(stdout)) finish(0)
    }).on('close', closed)
    child.stderr.setEncoding('utf8').on('data', (d: string) => (stderr += d)).on('close', closed)
    child.on('exit', (code) => {
      exitCode = code ?? 137
    })
    child.on('error', (error) => {
      stderr += String(error.message)
      finish(127)
    })
    child.stdin.on('error', () => undefined)
    child.stdin.end(input ?? '')
  })

/** codesign through sh, which prints its exit code as `LR_STATUS` (codesign prints none of its own). */
const codesign = (exec: Run, args: string[]) => exec('/bin/sh', ['-c', '/usr/bin/codesign "$@"; echo "LR_STATUS $?" >&2', 'sh', ...args])

export type Fs = {
  exists(path: string): Promise<boolean>
  copy(from: string, to: string): Promise<void>
}

const realFs: Fs = {
  exists: (path) => stat(path).then(() => true, () => false),
  async copy(from, to) {
    await mkdir(dirname(to), { recursive: true, mode: 0o700 })
    const tmp = `${to}.${process.pid}.tmp`
    await copyFile(from, tmp)
    await chmod(tmp, 0o755)
    await rename(tmp, to)
  },
}

export class KeychainHelper {
  readonly bundled: string
  readonly installed: string
  private readonly exec: Run
  private readonly fs: Fs
  private ready: Promise<string | null> | null = null
  /** Why the helper could not be used, for the log and the state. */
  problem = ''

  constructor(o: { bundled: string; dataDir: string; run?: Run; fs?: Fs }) {
    this.bundled = o.bundled
    this.installed = join(o.dataDir, 'bin', HELPER)
    this.exec = o.run ?? run
    this.fs = o.fs ?? realFs
  }

  /** The installed helper's path, installing or replacing it when it is missing, altered or older; null without one. */
  path(): Promise<string | null> {
    this.ready ??= this.prepare()
    return this.ready
  }

  private async prepare(): Promise<string | null> {
    if (!(await this.fs.exists(this.bundled))) {
      this.problem = 'this build has no keychain helper'
      return null
    }
    const dr = await codesign(this.exec, ['-d', '-r-', this.bundled])
    const requirement = /designated => (.+)/.exec(`${dr.stdout}\n${dr.stderr}`)?.[1]?.trim()
    if (!requirement || !/certificate leaf/.test(requirement)) {
      this.problem = 'the keychain helper is not signed with LocalRouter’s certificate'
      return null
    }
    const wanted = (await this.exec(this.bundled, ['version'])).stdout.trim()
    if (await this.usable(requirement, wanted)) return this.installed
    await this.fs.copy(this.bundled, this.installed)
    if (await this.usable(requirement, wanted)) return this.installed
    this.problem = 'the installed keychain helper does not verify'
    return null
  }

  /** Installed, signed like the bundled one, and at least its version. */
  private async usable(requirement: string, wanted: string): Promise<boolean> {
    if (!(await this.fs.exists(this.installed))) return false
    const verified = await codesign(this.exec, ['--verify', '--strict', `-R=${requirement}`, this.installed])
    if (verified.code !== 0) return false
    const have = (await this.exec(this.installed, ['version'])).stdout.trim()
    return Number(have) >= Number(wanted)
  }

  async get(): Promise<Secrets | null> {
    const helper = await this.path()
    if (!helper) throw new Error(this.problem)
    const r = await this.exec(helper, ['get', SERVICE, ACCOUNT])
    if (r.code === 4) return null
    if (r.code !== 0) throw new Error(`keychain: ${r.stderr.trim() || `exit ${r.code}`}`)
    return { ...EMPTY, ...(JSON.parse(r.stdout) as Partial<Secrets>) }
  }

  async set(s: Secrets): Promise<void> {
    const helper = await this.path()
    if (!helper) throw new Error(this.problem)
    const r = await this.exec(helper, ['set', SERVICE, ACCOUNT], JSON.stringify(s))
    if (r.code !== 0) throw new Error(`keychain: ${r.stderr.trim() || `exit ${r.code}`}`)
  }
}

/** The tokens: from the keychain helper, moving a safeStorage file of an older build into it once (its last prompt). */
export async function loadSecrets(o: {
  helper: KeychainHelper | null
  legacyPath: string
  /** Decrypts the older build's file (Electron safeStorage); null when it cannot. */
  legacyDecrypt: (data: Buffer) => string | null
}): Promise<{ secrets: Secrets; store: 'keychain' | 'legacy' | 'none'; migrated: boolean }> {
  const legacy = await readFile(o.legacyPath).catch(() => null)
  if (!o.helper) {
    const text = legacy ? o.legacyDecrypt(legacy) : null
    return { secrets: text ? { ...EMPTY, ...(JSON.parse(text) as Partial<Secrets>) } : { ...EMPTY }, store: text ? 'legacy' : 'none', migrated: false }
  }
  if (legacy) {
    const text = o.legacyDecrypt(legacy)
    if (text) {
      const secrets = { ...EMPTY, ...(JSON.parse(text) as Partial<Secrets>) }
      await o.helper.set(secrets)
      // Only once it is in the keychain: a failed move keeps the old file for the next start.
      await rm(o.legacyPath, { force: true })
      return { secrets, store: 'keychain', migrated: true }
    }
  }
  const secrets = await o.helper.get()
  return { secrets: secrets ?? { ...EMPTY }, store: 'keychain', migrated: false }
}

/**
 * Moving an older build's safeStorage token into the keychain helper, once in the life of an install.
 *
 * The read goes through Electron safeStorage, whose keychain item belongs to older builds: macOS asks the person,
 * and that read blocks the process that makes it. So it runs in a short-lived child process (`--secrets-migrate`),
 * never on the app's main thread, and with a time limit. Whatever happens (moved, refused, no answer in time), the
 * outcome is recorded and the old file is gone afterwards: a refused or unanswered move is never asked again, the
 * person pairs again instead.
 */
export type Migration = { state: 'done' | 'failed'; at: number; reason?: string }
export type MigrationRun = (timeoutMs: number) => Promise<{ ok: boolean; reason: string }>

export const MIGRATION_TIMEOUT_MS = 120_000

export async function migrateLegacy(o: {
  legacyPath: string
  previous: Migration | undefined
  /** The token already in the keychain: the old file is only a leftover then. */
  haveToken: boolean
  run: MigrationRun
  timeoutMs?: number
  now?: () => number
  log?: (event: string, fields?: Record<string, unknown>) => void
}): Promise<{ outcome: 'none' | 'migrated' | 'failed' | 'dropped'; migration?: Migration }> {
  const log = o.log ?? (() => undefined)
  const now = o.now ?? Date.now
  const exists = await stat(o.legacyPath).then(() => true, () => false)
  if (!exists) return { outcome: 'none' }
  if (o.haveToken || o.previous?.state === 'failed') {
    await rm(o.legacyPath, { force: true })
    log('secrets.migration.dropped', { why: o.haveToken ? 'the token is in the keychain already' : 'an earlier move failed' })
    return { outcome: 'dropped' }
  }
  log('secrets.migration.start', {})
  let r: { ok: boolean; reason: string }
  try {
    r = await o.run(o.timeoutMs ?? MIGRATION_TIMEOUT_MS)
  } catch (error) {
    r = { ok: false, reason: error instanceof Error ? error.message : String(error) }
  }
  // Moved or not, the old file is never read again.
  await rm(o.legacyPath, { force: true })
  const migration: Migration = r.ok ? { state: 'done', at: now() } : { state: 'failed', at: now(), reason: r.reason.slice(0, 300) }
  log(r.ok ? 'secrets.migration.done' : 'secrets.migration.failed', r.ok ? {} : { reason: migration.reason })
  return { outcome: r.ok ? 'migrated' : 'failed', migration }
}
