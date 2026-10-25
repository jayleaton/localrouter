import { describe, expect, it } from 'vitest'
import { ControlError } from '../../../../desktop-helper/src/computer/macControls.ts'
import type { CommandRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import { MUSIC_VOLUME, sessionApp, shortcuts, startAppsShape, WindowsControls } from './controls.ts'
import type { DirEntry } from './controls.ts'
import { accessReport, createWindowsPlatform } from './index.ts'
import { WindowsHello } from './presence.ts'
import type { WindowsCall } from './worker.ts'

type Call = { op: string; args: Record<string, unknown> }

function fakeWorker(respond: (call: Call) => unknown) {
  const calls: Call[] = []
  const worker: WindowsCall = {
    async call(op, args = {}) {
      calls.push({ op, args })
      const result = respond({ op, args })
      if (result instanceof Error) throw result
      return result
    },
  }
  return { worker, calls }
}

function fakeRunner(fail?: Error) {
  const runs: { file: string; args: string[] }[] = []
  const runner: CommandRunner = {
    async run(file, args) {
      runs.push({ file, args })
      if (fail) throw fail
      return ''
    },
  }
  return { runner, runs }
}

/** What Get-StartApps lists on a typical Windows 11 machine (trimmed). */
const START_APPS = [
  { name: 'Google Chrome', id: 'Chrome' },
  { name: 'Microsoft Edge', id: 'MSEdge' },
  { name: 'Visual Studio Code', id: 'Microsoft.VisualStudioCode' },
  { name: 'Word', id: 'Microsoft.Office.WINWORD.EXE.15' },
  { name: 'Settings', id: 'windows.immersivecontrolpanel_cw5n1h2txyewy!microsoft.windows.immersivecontrolpanel' },
  { name: 'File Explorer', id: 'Microsoft.Windows.Explorer' },
  { name: 'Terminal', id: 'Microsoft.WindowsTerminal_8wekyb3d8bbwe!App' },
  { name: 'Calculator', id: 'Microsoft.WindowsCalculator_8wekyb3d8bbwe!App' },
  { name: 'Spotify', id: 'SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify' },
  { name: 'Uninstall Spotify', id: '{7C5A40EF-A0FB-4BFC-874A-C0F2E0B9FA8E}\\Spotify\\uninstall.exe' },
  { name: 'Adobe Photoshop 2026', id: '{6D809377-6AF0-444B-8957-A3773F02200E}\\Adobe\\Photoshop.exe' },
  { name: 'Adobe Photoshop Express', id: 'AdobeSystemsIncorporated.AdobePhotoshopExpress_ynb6jyjzte8ga!App' },
  { name: 'Evil', id: '"C:\\x.exe" --flag' },
]

const WINDOWS = [
  { hwnd: 11, pid: 1, app: 'Google Chrome', process: 'chrome', title: 'Inbox', minimized: false },
  { hwnd: 12, pid: 1, app: 'Google Chrome', process: 'chrome', title: 'Docs', minimized: true },
  { hwnd: 21, pid: 2, app: 'Notepad', process: 'Notepad', title: 'notes.txt', minimized: false },
]

function controls(respond: (call: Call) => unknown = () => ({}), options: { lnk?: Record<string, DirEntry[]>; runnerFail?: Error } = {}) {
  const fake = fakeWorker((call) => (call.op === 'start_apps' ? START_APPS : call.op === 'windows' ? WINDOWS : respond(call)))
  const run = fakeRunner(options.runnerFail)
  const lnk = options.lnk ?? {}
  const c = new WindowsControls({
    worker: fake.worker,
    runner: run.runner,
    env: { SystemRoot: 'C:\\Windows', ProgramData: 'C:\\ProgramData', APPDATA: 'C:\\Users\\jj\\AppData\\Roaming' },
    listDir: async (path) => {
      const entries = lnk[path.replace(/\//g, '\\')]
      if (!entries) throw new Error('ENOENT')
      return entries
    },
  })
  return { c, calls: fake.calls, runs: run.runs }
}

const explorer = (path: string) => path.replace(/\//g, '\\')

describe('WindowsControls open_app', () => {
  it('opens a Start menu app through the shell AppsFolder with its AppUserModelID as one argument', async () => {
    const { c, runs } = controls()
    expect(await c.run({ kind: 'open_app', app: 'chrome' })).toEqual({ done: 'Opened Google Chrome.' })
    expect(await c.run({ kind: 'open_app', app: 'calculator' })).toEqual({ done: 'Opened Calculator.' })
    expect(runs.map((r) => [explorer(r.file), r.args])).toEqual([
      ['C:\\Windows\\explorer.exe', ['shell:AppsFolder\\Chrome']],
      ['C:\\Windows\\explorer.exe', ['shell:AppsFolder\\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App']],
    ])
  })

  it('knows what people call apps on Windows', async () => {
    const { c } = controls()
    for (const [said, app] of [['vs code', 'Visual Studio Code'], ['Edge', 'Microsoft Edge'], ['files', 'File Explorer'], ['explorer', 'File Explorer'], ['Terminal', 'Terminal'], ['word', 'Word'], ['settings', 'Settings'], ['photoshop 2026', 'Adobe Photoshop 2026']]) {
      expect((await c.resolve(said)).name).toBe(app)
    }
  })

  it('asks which one when two apps match equally well, and never finds an uninstaller', async () => {
    const { c, runs } = controls()
    const error = await c.run({ kind: 'open_app', app: 'photoshop' }).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ControlError)
    expect((error as ControlError).code).toBe('ambiguous')
    expect((error as ControlError).message).toBe('Which one do you mean: Adobe Photoshop 2026, Adobe Photoshop Express?')
    expect((await c.resolve('spotify')).name).toBe('Spotify')
    expect(runs).toHaveLength(0)
  })

  it('drops Start entries whose id could carry quotes or flags', () => {
    const apps = startAppsShape(START_APPS)
    expect(apps.map((a) => a.name)).not.toContain('Evil')
    expect(apps.map((a) => a.name)).not.toContain('Uninstall Spotify')
    expect(() => startAppsShape([{ name: 1, id: 'x' }])).toThrow('start apps[0] name should be a string')
  })

  it('adds Start menu shortcuts Get-StartApps missed, opened through explorer.exe by their path', async () => {
    const programs = 'C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs'
    const { c, runs } = controls(() => ({}), { lnk: { [programs]: [{ name: 'Tools', directory: true }, { name: 'Google Chrome.lnk', directory: false }], [`${programs}\\Tools`]: [{ name: 'Old Tool.lnk', directory: false }, { name: 'Uninstall Old Tool.lnk', directory: false }, { name: 'readme.txt', directory: false }] } })
    expect(await c.run({ kind: 'open_app', app: 'old tool' })).toEqual({ done: 'Opened Old Tool.' })
    expect(explorer(runs[0]!.args[0]!)).toBe(`${programs}\\Tools\\Old Tool.lnk`)
    // The Get-StartApps entry wins over a shortcut of the same name.
    expect((await c.resolve('Google Chrome')).launch).toEqual({ kind: 'app', id: 'Chrome' })
    expect((await shortcuts(programs, async () => [{ name: '-x.lnk', directory: false }])).length).toBe(0)
  })

  it('treats explorer.exe’s exit code 1 as opened, but not a missing explorer.exe', async () => {
    expect(await controls(() => ({}), { runnerFail: new Error('Command failed: explorer.exe shell:AppsFolder\\Chrome') }).c.run({ kind: 'open_app', app: 'chrome' })).toEqual({ done: 'Opened Google Chrome.' })
    await expect(controls(() => ({}), { runnerFail: new Error('spawn C:\\Windows\\explorer.exe ENOENT') }).c.run({ kind: 'open_app', app: 'chrome' })).rejects.toThrow('Windows Explorer could not be started')
  })

  it('says an unknown app is not installed', async () => {
    await expect(controls().c.run({ kind: 'open_app', app: 'Final Cut Pro' })).rejects.toThrow("Final Cut Pro isn't installed on this computer")
  })
})

describe('WindowsControls quit_app', () => {
  it('asks each of the app’s windows to close (WM_CLOSE), never killing it', async () => {
    const { c, calls } = controls()
    expect(await c.run({ kind: 'quit_app', app: 'chrome' })).toEqual({ done: 'Quit Google Chrome.' })
    expect(calls.filter((call) => call.op === 'close').map((call) => call.args)).toEqual([{ hwnd: 11 }, { hwnd: 12 }])
    expect(calls.map((call) => call.op)).not.toContain('kill')
  })

  it('says an installed app that has no window was not open', async () => {
    const { c, calls } = controls()
    expect(await c.run({ kind: 'quit_app', app: 'calculator' })).toEqual({ done: "Calculator wasn't open." })
    expect(calls.filter((call) => call.op === 'close')).toHaveLength(0)
  })
})

describe('WindowsControls volume, media, dark mode, lock', () => {
  it('steps the system volume by 20 from where it is, and sets a level', async () => {
    const { c, calls } = controls((call) => (call.op === 'volume_read' ? { level: 50, muted: false } : { level: Number(call.args.level ?? 0), muted: false }))
    expect(await c.run({ kind: 'volume', action: 'up', target: 'system' })).toEqual({ done: 'Volume is at 70%.' })
    expect(await c.run({ kind: 'volume', action: 'set', level: 15, target: 'system' })).toEqual({ done: 'Volume is at 15%.' })
    expect(calls.filter((call) => call.op === 'volume_write').map((call) => call.args)).toEqual([{ target: 'system', level: 70, muted: false }, { target: 'system', level: 15, muted: false }])
    expect(await c.run({ kind: 'volume', action: 'mute', target: 'system' })).toEqual({ done: 'Muted.' })
  })

  it('sets Spotify’s own volume session, and says when it is not playing', async () => {
    const playing = controls((call) => (call.op === 'volume_read' ? { level: 10, muted: false } : { level: 0, muted: false }))
    expect(await playing.c.run({ kind: 'volume', action: 'down', target: 'Spotify' })).toEqual({ done: 'Spotify volume is at 0%.' })
    const silent = controls(() => new Error('LR_NOT_PLAYING|Spotify|'))
    expect(await silent.c.run({ kind: 'volume', action: 'up', target: 'Spotify' })).toEqual({ done: "Spotify isn't playing." })
  })

  it('refuses Music volume: Windows has no Music app', async () => {
    const { c, calls } = controls()
    const error = await c.run({ kind: 'volume', action: 'up', target: 'Music' }).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ControlError)
    expect((error as ControlError).code).toBe('unavailable')
    expect((error as ControlError).message).toBe(MUSIC_VOLUME)
    expect(calls.filter((call) => call.op.startsWith('volume'))).toHaveLength(0)
  })

  it('controls the media session, naming the app that played, and falls back to the media keys', async () => {
    const session = controls((call) => (call.op === 'media' ? { session: true, done: true, app: 'Spotify.exe' } : {}))
    expect(await session.c.run({ kind: 'media', action: 'next', player: 'Spotify' })).toEqual({ done: 'Skipped to the next track on Spotify.' })
    expect(session.calls.find((call) => call.op === 'media')?.args).toEqual({ action: 'next', player: 'Spotify' })
    const browser = controls((call) => (call.op === 'media' ? { session: true, done: true, app: 'Chrome' } : {}))
    expect(await browser.c.run({ kind: 'media', action: 'pause', player: 'Music' })).toEqual({ done: 'Paused Google Chrome.' })
    const old = controls((call) => (call.op === 'media' ? new Error('Unable to find type [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager]') : {}))
    expect(await old.c.run({ kind: 'media', action: 'play', player: 'Music' })).toEqual({ done: 'Pressed the play/pause media key.' })
    expect(old.calls.find((call) => call.op === 'media_key')?.args).toEqual({ action: 'play' })
  })

  it('does not press a media key for a player that has no session', async () => {
    const none = controls((call) => (call.op === 'media' ? { session: false, done: false, app: '' } : {}))
    expect(await none.c.run({ kind: 'media', action: 'play', player: 'Spotify' })).toEqual({ done: "Spotify isn't playing." })
    expect(await none.c.run({ kind: 'media', action: 'next', player: 'Music' })).toEqual({ done: 'Nothing is playing.' })
    expect(none.calls.filter((call) => call.op === 'media_key')).toHaveLength(0)
  })

  it('switches dark mode and locks the screen', async () => {
    const { c, calls } = controls((call) => (call.op === 'dark_mode' ? { dark: true } : {}))
    expect(await c.run({ kind: 'dark_mode', action: 'toggle' })).toEqual({ done: 'Dark mode is on.' })
    expect(await c.run({ kind: 'lock_screen' })).toEqual({ done: 'Locked the screen.' })
    expect(calls.filter((call) => call.op !== 'start_apps').map((call) => call.op)).toEqual(['dark_mode', 'lock'])
  })

  it('names media session apps', () => {
    expect(sessionApp('SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify')).toBe('Spotify')
    expect(sessionApp('Microsoft.ZuneMusic_8wekyb3d8bbwe!Microsoft.ZuneMusic')).toBe('Media Player')
    expect(sessionApp('MSEdge')).toBe('Microsoft Edge')
  })
})

describe('Windows Hello', () => {
  it('is available only when Windows says Available, and confirms only on Verified', async () => {
    const ready = fakeWorker((call) => (call.op === 'hello_available' ? { availability: 'Available' } : { result: 'Verified' }))
    const hello = new WindowsHello(ready.worker)
    expect(hello.method).toBe('Windows Hello')
    expect(await hello.available()).toBe(true)
    expect(await hello.verify('Approve:\n  send “hi” to Sam')).toBe(true)
    expect(ready.calls[1]!.args).toEqual({ prompt: 'Approve: send “hi” to Sam' })
    const unset = fakeWorker((call) => (call.op === 'hello_available' ? { availability: 'NotConfiguredForUser' } : { result: 'Canceled' }))
    expect(await new WindowsHello(unset.worker).available()).toBe(false)
    expect(await new WindowsHello(unset.worker).verify('x')).toBe(false)
    const broken = fakeWorker(() => new Error('Windows took too long to answer (timed out).'))
    expect(await new WindowsHello(broken.worker).available()).toBe(false)
    expect(await new WindowsHello(broken.worker).verify('x')).toBe(false)
  })
})

describe('createWindowsPlatform', () => {
  const facts = { uia: true, secure: false, session: 1, integrity: 0x2000, dpi: 'per_monitor_v2', interactive: true }

  function platform(system: (call: Call) => unknown, pinned: DirEntry[] = []) {
    const screen = fakeWorker(() => ({ x: 1, y: 2 }))
    const sys = fakeWorker((call) => (call.op === 'start_apps' ? START_APPS : call.op === 'windows' ? WINDOWS : system(call)))
    const run = fakeRunner()
    const p = createWindowsPlatform({
      workers: { screen: screen.worker, system: sys.worker },
      runner: run.runner,
      env: { SystemRoot: 'C:\\Windows', APPDATA: 'C:\\Users\\jj\\AppData\\Roaming' },
      listDir: async (path) => (path.replace(/\//g, '\\').endsWith('User Pinned\\TaskBar') ? pinned : []),
    })
    return { p, screen: screen.calls, system: sys.calls, runs: run.runs }
  }

  it('reports a usable session, and what to fix when it is not', async () => {
    const { p } = platform((call) => (call.op === 'access' ? facts : {}))
    expect(p.os).toBe('win32')
    expect(await p.access()).toMatchObject({ accessibility: true, screen_recording: true, input: true, backend: 'Windows (UI Automation, SendInput)' })
    const report = accessReport({ ...facts, integrity: 0x3000, secure: true, session: 0 })
    expect(report).toMatchObject({ accessibility: false, input: false })
    expect(report.requirements.find((r) => r.id === 'not_elevated')).toMatchObject({ granted: false, fix: 'Run LocalRouter as the signed-in user, not as administrator, unless the apps you control run as administrator.' })
    expect(report.requirements.find((r) => r.id === 'desktop')?.granted).toBe(false)
    const broken = platform(() => new Error('spawn powershell.exe ENOENT'))
    expect(await broken.p.access()).toMatchObject({ accessibility: false, input: false, requirements: [{ id: 'powershell', granted: false }] })
  })

  it('lists the taskbar’s pinned apps and running apps, and brings one forward or opens it', async () => {
    const { p, system, runs } = platform(() => ({}), [{ name: 'Google Chrome.lnk', directory: false }, { name: 'File Explorer.lnk', directory: false }, { name: 'desktop.ini', directory: false }])
    expect(await p.launcher!.list()).toEqual(['Google Chrome', 'File Explorer'])
    expect(await p.apps!.running()).toEqual(['Google Chrome', 'Notepad'])
    await p.apps!.activate('chrome')
    expect(system.find((call) => call.op === 'foreground')?.args).toEqual({ hwnd: 11 })
    await p.launcher!.activate('calculator')
    expect(runs[0]!.args).toEqual(['shell:AppsFolder\\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App'])
  })

  it('reads the pointer and displays from the screen worker, and Chrome tabs from the system worker', async () => {
    const { p, screen, system } = platform((call) => (call.op === 'tabs' ? { app: 'Microsoft Edge', tabs: [{ index: 0, title: 'Inbox', active: true }, { index: 1, title: 'News', active: false }] } : {}))
    expect(await p.pointer!()).toEqual({ x: 1, y: 2 })
    expect(screen.map((call) => call.op)).toEqual(['pointer'])
    const tabs = await p.tabs!.list()
    expect(tabs[1]).toEqual({ window: 0, index: 1, title: 'News', url: '', active: false })
    expect(p.tabs!.browser).toBe('Microsoft Edge')
    await p.tabs!.activate(tabs[1]!)
    expect(system.find((call) => call.op === 'tab_select')?.args).toEqual({ index: 1, title: 'News' })
    expect(p.presence?.method).toBe('Windows Hello')
  })
})
