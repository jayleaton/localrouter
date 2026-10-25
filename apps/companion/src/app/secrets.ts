import { execFile } from 'node:child_process'
import { chmod, copyFile, mkdir, readFile, rename, rm, stat } from 'node:fs/promises'
import { dirname, join } from 'node:path'

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

export type Run = (file: string, args: string[], input?: string) => Promise<{ code: number; stdout: string; stderr: string }>

export const run: Run = (file, args, input) =>
  new Promise((resolve) => {
    // Long enough for a person to answer a keychain prompt (a replaced helper's first read), never forever.
    const child = execFile(file, args, { timeout: 180_000, maxBuffer: 1 << 20 }, (error, stdout, stderr) => {
      const code = error ? (typeof (error as { code?: unknown }).code === 'number' ? ((error as { code: number }).code) : 1) : 0
      resolve({ code, stdout: String(stdout), stderr: String(stderr) })
    })
    child.stdin?.end(input ?? '')
  })

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
    const dr = await this.exec('/usr/bin/codesign', ['-d', '-r-', this.bundled])
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
    const verified = await this.exec('/usr/bin/codesign', ['--verify', '--strict', `-R=${requirement}`, this.installed])
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
