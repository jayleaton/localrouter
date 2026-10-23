import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { checkUrl, configDir, configPath, controlSocketPath, describeConfig, desktopSocketUrl, loadConfig, saveConfig, setDisabled } from './config.ts'
import type { Config } from './config.ts'
import { createLogger } from './log.ts'

const CONFIG: Config = { url: 'https://dgx-01.example.ts.net:10443', token: 'lr_dev_0123456789', name: "JJ's MacBook Pro", disabled: false }

describe('config', () => {
  const dirs: string[] = []
  const fresh = async () => {
    const parent = await mkdtemp(join(tmpdir(), 'lrd-'))
    dirs.push(parent)
    return join(parent, 'LocalRouter Desktop')
  }
  afterEach(async () => {
    await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })))
  })

  it('lives in Application Support, or under LOCALROUTER_DESKTOP_HOME', () => {
    expect(configDir({}, '/Users/jj')).toBe('/Users/jj/Library/Application Support/LocalRouter Desktop')
    expect(configDir({ LOCALROUTER_DESKTOP_HOME: '/tmp/lrd' }, '/Users/jj')).toBe('/tmp/lrd')
    expect(controlSocketPath('/tmp/lrd')).toBe('/tmp/lrd/control.sock')
  })

  it('keeps the token in a 0600 file inside a 0700 directory, and reads it back', async () => {
    const dir = await fresh()
    expect(await loadConfig(dir)).toBeNull()
    await saveConfig(dir, CONFIG)
    expect((await stat(dir)).mode & 0o777).toBe(0o700)
    expect((await stat(configPath(dir))).mode & 0o777).toBe(0o600)
    expect(await loadConfig(dir)).toEqual(CONFIG)
    // Saving again (as disable does) keeps the permissions.
    await setDisabled(dir, true)
    expect((await stat(configPath(dir))).mode & 0o777).toBe(0o600)
    expect(await loadConfig(dir)).toEqual({ ...CONFIG, disabled: true })
    expect(JSON.parse(await readFile(configPath(dir), 'utf8'))).toMatchObject({ disabled: true, token: CONFIG.token })
  })

  it('refuses a config that is not the right shape', async () => {
    const dir = await fresh()
    await saveConfig(dir, CONFIG)
    await writeFile(configPath(dir), JSON.stringify({ url: 'http://box.example', token: 'x', name: 'Mac' }))
    await expect(loadConfig(dir)).rejects.toThrow(/https/)
    await writeFile(configPath(dir), '{"url":')
    await expect(loadConfig(dir)).rejects.toThrow(/JSON/)
  })

  it('cannot disable a Mac that was never paired', async () => {
    await expect(setDisabled(await fresh(), true)).rejects.toThrow(/not paired/)
  })

  it('accepts https, or http only to this Mac itself, and builds the desktop socket address', () => {
    expect(checkUrl('https://dgx-01.example.ts.net:10443/')).toBe('https://dgx-01.example.ts.net:10443')
    expect(checkUrl('http://127.0.0.1:8080')).toBe('http://127.0.0.1:8080')
    expect(checkUrl('http://localhost:8080/')).toBe('http://localhost:8080')
    expect(checkUrl('http://[::1]:8080')).toBe('http://[::1]:8080')
    expect(() => checkUrl('http://dgx-01.example.ts.net:10443')).toThrow(/https/)
    expect(() => checkUrl('http://192.168.1.116:10443')).toThrow(/https/)
    expect(() => checkUrl('ftp://box')).toThrow(/https/)
    expect(() => checkUrl('https://user:pass@box')).toThrow(/user name/)
    expect(() => checkUrl('not a url')).toThrow(/not a URL/)
    expect(desktopSocketUrl('https://dgx-01.example.ts.net:10443')).toBe('wss://dgx-01.example.ts.net:10443/v1/desktop')
    expect(desktopSocketUrl('http://127.0.0.1:8080')).toBe('ws://127.0.0.1:8080/v1/desktop')
  })

  it('never shows or logs the token', () => {
    expect(JSON.stringify(describeConfig(CONFIG))).not.toContain(CONFIG.token)
    const lines: string[] = []
    createLogger((line) => lines.push(line)).info('helper.started', { token: CONFIG.token, authorization: `Bearer ${CONFIG.token}` })
    expect(lines.join()).not.toContain(CONFIG.token)
  })
})
