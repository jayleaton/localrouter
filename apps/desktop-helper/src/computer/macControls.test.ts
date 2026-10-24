// Ported from SideBud (github.com/jayleaton/sidebud @70da540, apps/companion/src/computer/macControls.test.ts), reused in LocalRouter with the owner's permission.
import { describe, expect, it } from 'vitest'
import { ControlError, MacControls, parseControl, plistNames, resolveInstalled } from './macControls.ts'
import type { CommandRunner, InstalledApp } from './macControls.ts'

type Call = { file: string; args: string[] }

const TREE: Record<string, string[]> = {
  '/Applications': ['Spotify.app', 'Slack.app', 'Utilities', '.DS_Store', 'Google Chrome.app'],
  '/Applications/Utilities': ['Terminal.app', 'notes.txt'],
  '/System/Applications': ['Music.app', 'Utilities', 'Spotify.app'],
  '/System/Applications/Utilities': ['Terminal.app', 'Google-Chrome.app'],
}

/** Fake runner: `respond` picks stdout from the joined script text; every call is recorded. */
function harness(respond: (script: string, argv: string[]) => string | Error = () => '') {
  const calls: Call[] = []
  const reads: string[] = []
  let now = 0
  const runner: CommandRunner = {
    async run(file, args) {
      calls.push({ file, args })
      const lines = args.filter((_, index) => args[index - 1] === '-e')
      const argv = args.slice(args.lastIndexOf('-e') + 2)
      const result = respond(lines.join('\n'), argv)
      if (result instanceof Error) throw result
      return result
    },
  }
  const controls = new MacControls({
    runner,
    platform: 'darwin',
    appDirectories: ['/Applications', '/System/Applications', '/System/Applications/Utilities', '/missing'],
    readDir: async (path) => {
      reads.push(path)
      const entries = TREE[path]
      if (!entries) throw new Error(`ENOENT: ${path}`)
      return entries
    },
    now: () => now,
  })
  return { controls, calls, reads, advance: (ms: number) => (now += ms) }
}

describe('MacControls', () => {
  it('scans app bundles one folder deep, dedupes by name, and assigns unique slugs', async () => {
    const { controls } = harness()
    expect(await controls.apps()).toEqual([
      { id: 'google_chrome', name: 'Google Chrome' },
      { id: 'google_chrome_2', name: 'Google-Chrome' },
      { id: 'music', name: 'Music' },
      { id: 'slack', name: 'Slack' },
      { id: 'spotify', name: 'Spotify' },
      { id: 'terminal', name: 'Terminal' },
    ])
  })

  it('caches the scan for five minutes', async () => {
    const { controls, reads, advance } = harness()
    await controls.apps()
    const firstScan = reads.length
    advance(4 * 60_000)
    await controls.apps()
    expect(reads.length).toBe(firstScan)
    advance(2 * 60_000)
    await controls.apps()
    expect(reads.length).toBe(firstScan * 2)
  })

  it('opens an installed app by its canonical name', async () => {
    const { controls, calls } = harness()
    expect(await controls.run({ kind: 'open_app', app: 'spotify' })).toEqual({ done: 'Opened Spotify.' })
    expect(calls).toEqual([{ file: 'open', args: ['-a', 'Spotify'] }])
  })

  it('rejects apps that are not installed without running anything', async () => {
    const { controls, calls } = harness()
    const error = await controls.run({ kind: 'open_app', app: 'Evil"; do shell script "rm' }).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ControlError)
    expect((error as ControlError).code).toBe('not_found')
    expect(calls).toEqual([])
  })

  it('quits through argv so the name never enters the script source', async () => {
    const { controls, calls } = harness((script) => (script.includes('is running') ? 'true\n' : ''))
    expect(await controls.run({ kind: 'quit_app', app: 'Google Chrome' })).toEqual({ done: 'Quit Google Chrome.' })
    const quit = calls[1]
    expect(quit.file).toBe('osascript')
    expect(quit.args).toEqual([
      '-e', 'on run argv',
      '-e', 'tell application (item 1 of argv) to quit',
      '-e', 'end run',
      'Google Chrome',
    ])
    expect(calls.flatMap((call) => call.args.filter((_, i) => call.args[i - 1] === '-e')).join()).not.toContain('Chrome')
  })

  it("reports an app that isn't running instead of quitting it", async () => {
    const { controls, calls } = harness((script) => (script.includes('is running') ? 'false\n' : ''))
    expect(await controls.run({ kind: 'quit_app', app: 'Slack' })).toEqual({ done: "Slack wasn't open." })
    expect(calls).toHaveLength(1)
  })

  it('steps system volume by 20 and clamps to 0..100', async () => {
    let level = '90'
    const { controls, calls } = harness((script, argv) => {
      if (script.includes('get volume settings')) return `${level}\n`
      if (script.includes('set volume output volume')) level = argv[0]
      return ''
    })
    expect(await controls.run({ kind: 'volume', action: 'up', target: 'system' })).toEqual({ done: 'Volume is at 100%.' })
    level = '10'
    expect(await controls.run({ kind: 'volume', action: 'down', target: 'system' })).toEqual({ done: 'Volume is at 0%.' })
    expect(await controls.run({ kind: 'volume', action: 'set', level: 140, target: 'system' })).toEqual({
      done: 'Volume is at 100%.',
    })
    expect(calls.at(-1)?.args.at(-1)).toBe('100')
  })

  it("does not launch a player that isn't running for pause or volume", async () => {
    const { controls, calls } = harness((script) => (script.includes('is running') ? 'false' : ''))
    expect(await controls.run({ kind: 'media', action: 'pause', player: 'Spotify' })).toEqual({ done: "Spotify isn't playing." })
    expect(await controls.run({ kind: 'volume', action: 'up', target: 'Spotify' })).toEqual({ done: "Spotify isn't playing." })
    expect(calls.every((call) => call.args.join(' ').includes('is running'))).toBe(true)
  })

  it('is unavailable off macOS', async () => {
    const controls = new MacControls({ platform: 'linux', runner: { run: async () => '' } })
    expect(controls.available).toBe(false)
    await expect(controls.run({ kind: 'lock_screen' })).rejects.toMatchObject({ code: 'unavailable' })
  })

  it('maps an accessibility refusal on lock to a settings hint', async () => {
    const { controls } = harness(
      () => new Error('Command failed: osascript\nSystem Events got an error: osascript is not allowed to send keystrokes. (1002)')
    )
    const error = await controls.run({ kind: 'lock_screen' }).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ControlError)
    expect((error as ControlError).message).toContain('Privacy & Security → Accessibility')
  })

  it('describes controls for the audit log', () => {
    const { controls } = harness()
    expect(controls.describe({ kind: 'open_app', app: 'Spotify' })).toBe('open Spotify')
    expect(controls.describe({ kind: 'volume', action: 'set', level: 40, target: 'system' })).toBe('set system volume to 40%')
  })

  it('counts the user at the computer only with the screen unlocked and recent input', async () => {
    const presence = (idleSeconds: number, locked: boolean) =>
      new MacControls({
        platform: 'darwin',
        runner: { run: async (_file, args) => (args.includes('IOHIDSystem') ? `  |   "HIDIdleTime" = ${idleSeconds * 1e9}` : locked ? '"CGSSessionScreenIsLocked"=Yes' : '"IOConsoleUsers" = ()') },
      })
    expect(await presence(40, false).atComputer()).toBe(true)
    expect(await presence(900, false).atComputer()).toBe(false)
    expect(await presence(5, true).atComputer()).toBe(false)
    expect(await new MacControls({ platform: 'linux' }).atComputer()).toBe(false)
  })

  it('shows the task-start notice with its words only as argv', async () => {
    const calls: Call[] = []
    const controls = new MacControls({ platform: 'darwin', runner: { run: async (file, args) => { calls.push({ file, args }); return 'ok' } } })
    await controls.notify('LocalRouter Desktop', '--Jarvis is using your Mac: "search"')
    expect(calls[0]?.file).toBe('osascript')
    expect(calls[0]?.args.slice(-3)).toEqual(['--', 'LocalRouter Desktop', 'Jarvis is using your Mac: "search"'])
    expect(calls[0]?.args[3]).not.toContain('Jarvis')
  })

  it('brings an app forward by name without the Dock, and never passes a flag-like name', async () => {
    const { controls, calls } = harness()
    await controls.activate('Notes')
    expect(calls).toEqual([{ file: 'open', args: ['-a', 'Notes'] }])
    await expect(controls.activate('-n')).rejects.toMatchObject({ code: 'invalid' })
  })
})

describe('parseControl', () => {
  it('accepts exactly the protocol controls', () => {
    expect(parseControl({ kind: 'open_app', app: 'Safari' })).toEqual({ kind: 'open_app', app: 'Safari' })
    expect(parseControl({ kind: 'volume', action: 'set', level: 30, target: 'system' })).toEqual({ kind: 'volume', action: 'set', level: 30, target: 'system' })
    expect(parseControl({ kind: 'volume', action: 'down', target: 'Spotify', level: null })).toEqual({ kind: 'volume', action: 'down', target: 'Spotify' })
    expect(parseControl({ kind: 'media', action: 'next', player: 'Music' })).toEqual({ kind: 'media', action: 'next', player: 'Music' })
    expect(parseControl({ kind: 'dark_mode', action: 'toggle' })).toEqual({ kind: 'dark_mode', action: 'toggle' })
    expect(parseControl({ kind: 'lock_screen' })).toEqual({ kind: 'lock_screen' })
  })

  it('refuses anything else as invalid', () => {
    for (const value of [null, 'open_app', { kind: 'shell', command: 'rm -rf /' }, { kind: 'open_app', app: '' }, { kind: 'volume', action: 'set', level: 101, target: 'system' }, { kind: 'media', action: 'play', player: 'VLC' }]) {
      expect(() => parseControl(value)).toThrow(ControlError)
      try {
        parseControl(value)
      } catch (error) {
        expect((error as ControlError).code).toBe('invalid')
      }
    }
  })
})

// A Mac like the owner's: what the scan finds in /Applications and /System/Applications (JJ's #79 test asked for "Chrome").
const MAC: InstalledApp[] = [
  'Google Chrome', 'Safari', 'Mail', 'Messages', 'Notes', 'Music', 'System Settings', 'Visual Studio Code', 'Microsoft Word',
  'Microsoft Excel', 'Microsoft Outlook', 'Adobe Photoshop 2026', 'zoom.us', 'iTerm', 'Finder', 'Slack', 'Spotify', 'Calendar',
].map((name, i) => ({ id: `a${i}`, name }))

describe('open_app: the app a person means', () => {
  it('“Chrome” opens Google Chrome (the #79 failure), on the real control path', async () => {
    const { controls, calls } = harness()
    expect(await controls.run({ kind: 'open_app', app: 'Chrome' })).toEqual({ done: 'Opened Google Chrome.' })
    expect(calls).toEqual([{ file: 'open', args: ['-a', 'Google Chrome'] }])
  })

  it.each([
    ['chrome', 'Google Chrome'], ['Google chrome', 'Google Chrome'], ['VS Code', 'Visual Studio Code'], ['vscode', 'Visual Studio Code'],
    ['Code', 'Visual Studio Code'], ['mail', 'Mail'], ['Email', 'Mail'], ['Word', 'Microsoft Word'], ['excel', 'Microsoft Excel'],
    ['settings', 'System Settings'], ['System Preferences', 'System Settings'], ['iTunes', 'Music'], ['zoom', 'zoom.us'],
    ['photoshop', 'Adobe Photoshop 2026'], ['googlechrome', 'Google Chrome'], ['SLACK', 'Slack'], ['spot', 'Spotify'],
  ])('“%s” is %s', (asked, app) => {
    expect(resolveInstalled(asked, MAC)).toBe(app)
  })

  it('an alias only opens what is installed, preferring the usual one', () => {
    const canaryOnly = [{ id: 'c', name: 'Google Chrome Canary' }, { id: 's', name: 'Safari' }]
    expect(resolveInstalled('chrome', canaryOnly)).toBe('Google Chrome Canary')
    expect(resolveInstalled('chrome', [...canaryOnly, { id: 'g', name: 'Google Chrome' }])).toBe('Google Chrome')
    expect(() => resolveInstalled('vs code', [{ id: 's', name: 'Safari' }])).toThrow(/isn't installed/)
  })

  it('asks when several apps match equally, and never guesses', () => {
    const e = (() => { try { resolveInstalled('microsoft', MAC) } catch (error) { return error } })() as ControlError
    expect(e).toBeInstanceOf(ControlError)
    expect(e.code).toBe('ambiguous')
    expect(e.message).toMatch(/Which one do you mean: .*Microsoft Excel.*Microsoft Word/)
  })

  it('bundle and display names from Info.plist count too', () => {
    const plist = '<?xml version="1.0"?><plist><dict><key>CFBundleDisplayName</key><string>Code</string><key>CFBundleName</key><string>Code</string></dict></plist>'
    expect(plistNames(plist)).toEqual(['Code'])
    expect(plistNames('bplist00…')).toEqual([])
    expect(resolveInstalled('Teams for work', [{ id: 't', name: 'Microsoft Teams (work or school)', aliases: ['Teams for work'] }])).toBe('Microsoft Teams (work or school)')
  })

  it('a made-up name is still refused without running anything', () => {
    expect(() => resolveInstalled('Evil"; do shell script "rm', MAC)).toThrow(/isn't installed/)
  })
})
