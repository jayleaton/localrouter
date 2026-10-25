import { mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { KeychainHelper, loadSecrets, migrateLegacy } from './secrets.ts'
import type { Fs, Run } from './secrets.ts'

const REQ = 'identifier "com.localrouter.companion.keychain" and certificate leaf = H"ca6edcf5f3b24de8cb6c84be2cae1d79aa15c36d"'

/** A Mac in miniature: files that exist, what codesign says of each, each helper's version, and the keychain. */
function mac(o: { bundledVersion?: string; installed?: { version: string; signed: boolean } | null; bundledSigned?: boolean } = {}) {
  const files = new Map<string, { version: string; signed: boolean }>()
  files.set('/app/Resources/localrouter-keychain', { version: o.bundledVersion ?? '1', signed: o.bundledSigned ?? true })
  if (o.installed) files.set('/data/bin/localrouter-keychain', o.installed)
  let item: string | null = null
  const calls: string[][] = []
  const exec: Run = async (file, args, input) => {
    calls.push([file, ...args])
    if (file === '/usr/bin/codesign' && args[0] === '-d') {
      const f = files.get(args[2])
      return { code: 0, stdout: f?.signed ? `designated => ${REQ}\n` : 'designated => cdhash H"0123"\n', stderr: 'Executable=…\n' }
    }
    if (file === '/usr/bin/codesign' && args[0] === '--verify') {
      const f = files.get(args[3])
      return { code: f?.signed && args[2] === `-R=${REQ}` ? 0 : 3, stdout: '', stderr: '' }
    }
    const f = files.get(file)
    if (!f) return { code: 127, stdout: '', stderr: 'no such file' }
    if (args[0] === 'version') return { code: 0, stdout: `${f.version}\n`, stderr: '' }
    if (args[0] === 'get') return item === null ? { code: 4, stdout: '', stderr: '' } : { code: 0, stdout: item, stderr: '' }
    if (args[0] === 'set') {
      item = input ?? ''
      return { code: 0, stdout: '', stderr: '' }
    }
    return { code: 2, stdout: '', stderr: 'usage' }
  }
  const fs: Fs = {
    exists: async (p) => files.has(p),
    copy: async (from, to) => void files.set(to, { ...files.get(from)! }),
  }
  const helper = new KeychainHelper({ bundled: '/app/Resources/localrouter-keychain', dataDir: '/data', run: exec, fs })
  return { helper, files, calls, item: () => item }
}

const dirs: string[] = []
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
})

describe('the keychain helper survives app updates', () => {
  it('first start: copies the bundled helper out of the app, then uses that copy', async () => {
    const m = mac()
    expect(await m.helper.path()).toBe('/data/bin/localrouter-keychain')
    expect(m.files.has('/data/bin/localrouter-keychain')).toBe(true)
  })

  it('an update keeps the installed helper (same version, signed by the same certificate): its code hash, and the item, stay', async () => {
    const m = mac({ installed: { version: '1', signed: true } })
    let copied = false
    const keep = m.files.get('/data/bin/localrouter-keychain')
    expect(await m.helper.path()).toBe('/data/bin/localrouter-keychain')
    copied = m.files.get('/data/bin/localrouter-keychain') !== keep
    expect(copied).toBe(false)
  })

  it('an altered or foreign helper is replaced, never used', async () => {
    const m = mac({ installed: { version: '9', signed: false } })
    expect(await m.helper.path()).toBe('/data/bin/localrouter-keychain')
    expect(m.files.get('/data/bin/localrouter-keychain')).toEqual({ version: '1', signed: true })
  })

  it('a newer bundled helper version replaces the installed one (that Mac’s one last prompt)', async () => {
    const m = mac({ bundledVersion: '2', installed: { version: '1', signed: true } })
    await m.helper.path()
    expect(m.files.get('/data/bin/localrouter-keychain')?.version).toBe('2')
  })

  it('a build whose helper is not signed with the certificate has none (and says why)', async () => {
    const m = mac({ bundledSigned: false })
    expect(await m.helper.path()).toBeNull()
    expect(m.helper.problem).toMatch(/certificate/)
  })
})

describe('loading the tokens', () => {
  const tokens = { token: 'lr_tok', legacy_token: '' }

  it('moves an older build’s safeStorage file into the keychain once, then deletes the file', async () => {
    const d = mkdtempSync(join(tmpdir(), 'lr-sec-'))
    dirs.push(d)
    const legacy = join(d, 'secrets.bin')
    writeFileSync(legacy, 'ENCRYPTED')
    const m = mac()
    const first = await loadSecrets({ helper: m.helper, legacyPath: legacy, legacyDecrypt: () => JSON.stringify(tokens) })
    expect(first).toEqual({ secrets: tokens, store: 'keychain', migrated: true })
    expect(existsSync(legacy)).toBe(false)
    expect(JSON.parse(m.item()!)).toEqual(tokens)
    // Next start (and every update after): the keychain only, safeStorage never asked again.
    let asked = 0
    const again = await loadSecrets({ helper: m.helper, legacyPath: legacy, legacyDecrypt: () => (asked++, null) })
    expect(again).toEqual({ secrets: tokens, store: 'keychain', migrated: false })
    expect(asked).toBe(0)
  })

  it('a file that cannot be decrypted (prompt refused) is kept for the next start', async () => {
    const d = mkdtempSync(join(tmpdir(), 'lr-sec-'))
    dirs.push(d)
    const legacy = join(d, 'secrets.bin')
    writeFileSync(legacy, 'ENCRYPTED')
    const r = await loadSecrets({ helper: mac().helper, legacyPath: legacy, legacyDecrypt: () => null })
    expect(r.secrets.token).toBe('')
    expect(readFileSync(legacy, 'utf8')).toBe('ENCRYPTED')
  })

  it('without a helper (Linux, Windows, an unsigned dev build) the older store is used as before', async () => {
    const d = mkdtempSync(join(tmpdir(), 'lr-sec-'))
    dirs.push(d)
    const legacy = join(d, 'secrets.bin')
    writeFileSync(legacy, 'ENCRYPTED')
    expect(await loadSecrets({ helper: null, legacyPath: legacy, legacyDecrypt: () => JSON.stringify(tokens) })).toEqual({ secrets: tokens, store: 'legacy', migrated: false })
  })
})

describe('moving the old token: once in the life of an install', () => {
  const legacyFile = () => {
    const d = mkdtempSync(join(tmpdir(), 'lr-mig-'))
    dirs.push(d)
    const p = join(d, 'secrets.bin')
    writeFileSync(p, 'ENCRYPTED')
    return p
  }

  it('moves it, then the old file is gone and nothing runs again', async () => {
    const p = legacyFile()
    let runs = 0
    const run = async () => (runs++, { ok: true, reason: '' })
    const first = await migrateLegacy({ legacyPath: p, previous: undefined, haveToken: false, run, now: () => 5 })
    expect(first).toEqual({ outcome: 'migrated', migration: { state: 'done', at: 5 } })
    expect(existsSync(p)).toBe(false)
    expect((await migrateLegacy({ legacyPath: p, previous: first.migration, haveToken: true, run })).outcome).toBe('none')
    expect(runs).toBe(1)
  })

  it('refused (Deny) or unanswered (time out): recorded, the old file dropped, never asked again', async () => {
    const p = legacyFile()
    let runs = 0
    const r = await migrateLegacy({ legacyPath: p, previous: undefined, haveToken: false, run: async () => (runs++, { ok: false, reason: 'no answer within 120 s' }) })
    expect(r).toMatchObject({ outcome: 'failed', migration: { state: 'failed', reason: 'no answer within 120 s' } })
    expect(existsSync(p)).toBe(false)
    // Even if an old file turned up again (a restored backup), a failed move is not retried.
    writeFileSync(p, 'ENCRYPTED')
    expect((await migrateLegacy({ legacyPath: p, previous: r.migration, haveToken: false, run: async () => (runs++, { ok: true, reason: '' }) })).outcome).toBe('dropped')
    expect(runs).toBe(1)
    expect(existsSync(p)).toBe(false)
  })

  it('a crash of the move counts as failed, not as "try again next launch"', async () => {
    const p = legacyFile()
    const r = await migrateLegacy({ legacyPath: p, previous: undefined, haveToken: false, run: async () => { throw new Error('spawn failed') } })
    expect(r.migration).toMatchObject({ state: 'failed', reason: 'spawn failed' })
    expect(existsSync(p)).toBe(false)
  })

  it('a token already in the keychain makes the old file a leftover: dropped without asking', async () => {
    const p = legacyFile()
    let runs = 0
    expect((await migrateLegacy({ legacyPath: p, previous: undefined, haveToken: true, run: async () => (runs++, { ok: true, reason: '' }) })).outcome).toBe('dropped')
    expect(runs).toBe(0)
  })
})
