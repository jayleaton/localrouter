import { describe, expect, it } from 'vitest'
import { dataDir, executorSocket } from '../executor/paths.ts'
import { fakeBroker } from '../executor/testing.ts'
import { createPlatform, unsupportedPlatform } from './index.ts'

describe('the platform each OS gets', () => {
  it('macOS gets its AX backend; the fake is the same Notes app everywhere; another OS reads nothing and says why', async () => {
    expect((await createPlatform('native', { os: 'darwin' })).platform).toMatchObject({ os: 'darwin', backend: expect.stringMatching(/^macOS/) })
    // Windows: its UI Automation platform, whose PowerShell workers start only on first use (none here).
    const windows = (await createPlatform('native', { os: 'win32' })).platform
    expect(windows).toMatchObject({ os: 'win32', backend: 'Windows (UI Automation, SendInput)' })
    windows.close?.()
    const fake = await createPlatform('fake', { os: 'win32' })
    expect(fake.platform.os).toBe('fake')
    expect((await fake.platform.screen?.observe())?.app).toBe('Notes')
    const none = unsupportedPlatform('freebsd')
    expect(none).toMatchObject({ os: 'other', screen: null, pointer: null })
    await expect(none.controls.run({ kind: 'lock_screen' })).rejects.toThrow('Computer control is not available on freebsd yet.')
  })

  it('refuses a control its OS cannot run before the policy, and leaves it out of the capabilities', async () => {
    const { broker, mac } = fakeBroker([], {})
    const limited = fakeBroker([], { machine: { screen: mac.desktop, tabs: null, launcher: null, apps: null, controls: { run: mac.controls.run.bind(mac.controls), unsupported: { dark_mode: 'This desktop has no dark mode switch.' } } } }).broker
    const who = { caller: { id: 'a', name: 'app', kind: 'app' as const } }
    expect(await limited.control(who, { kind: 'dark_mode', action: 'on' })).toEqual({ ok: false, text: 'This desktop has no dark mode switch.' })
    expect(limited.capabilities.controls).not.toContain('dark_mode')
    expect(broker.capabilities.controls).toContain('dark_mode')
    expect(mac.controls.calls).toEqual([])
  })
})

describe('where the companion keeps its data and listens', () => {
  it('per OS', () => {
    expect(dataDir({}, 'darwin', '/Users/jj')).toBe('/Users/jj/Library/Application Support/LocalRouter')
    expect(dataDir({ XDG_STATE_HOME: '/s' }, 'linux', '/home/jj')).toBe('/s/localrouter-companion')
    expect(dataDir({ APPDATA: 'C:\\Users\\jj\\AppData\\Roaming' }, 'win32', 'C:\\Users\\jj')).toMatch(/Roaming[\\/]LocalRouter$/)
    expect(executorSocket('/s/localrouter-companion', 'linux')).toBe('/s/localrouter-companion/executor.sock')
    // Windows: a named pipe, the same for the same folder and user, different for another folder.
    const pipe = executorSocket('C:\\Users\\jj\\AppData\\Roaming\\LocalRouter', 'win32')
    expect(pipe).toMatch(/^\\\\\.\\pipe\\localrouter-executor-[0-9a-f]{16}$/)
    expect(executorSocket('c:\\users\\jj\\appdata\\roaming\\localrouter', 'win32')).toBe(pipe)
    expect(executorSocket('D:\\other', 'win32')).not.toBe(pipe)
  })
})
