// Ported from SideBud (github.com/jayleaton/sidebud @70da540, apps/companion/src/computer/use/backends.test.ts), reused in LocalRouter with the owner's permission.
import { describe, expect, it } from 'vitest'
import { ChromeTabs, chooseInstance } from './chromeTabs.ts'
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { chunks, DesktopBackend, jxaRunner, landed, macAccess, SCRIPTS } from './desktop.ts'
import { DockApps } from './dock.ts'
import type { CommandRunner } from './macControls.ts'
import type { BrowserTab } from './types.ts'

type Call = { file: string; script: string; argv: string[] }

/** Fake runner: `respond` returns stdout (or an Error to throw) for each call; every call is recorded. */
function fakeRunner(respond: (call: Call) => string | Error) {
  const calls: Call[] = []
  const runner: CommandRunner = {
    async run(file, args) {
      const call = { file, script: args[args.indexOf('-e') + 1] ?? '', argv: args.slice(args.indexOf('--') + 1) }
      calls.push(call)
      const result = respond(call)
      if (result instanceof Error) throw result
      return result
    },
  }
  return { runner, calls }
}

const TAB: BrowserTab = { window: 0, index: 2, title: '-e Inbox (3) "quoted"', url: 'https://mail.example.com/', active: false }

/**
 * The owner's Mac on 2026-09-26: a headless Chrome that Playwright launched (background-only, its
 * scripting answers with one tab) next to the owner's Chrome, whose window is in front.
 */
const HEADLESS = { pid: 22752, regular: false, frontmost: false, frontWindow: null }
const ON_SCREEN = { pid: 502, regular: true, frontmost: true, frontWindow: 0 }
const isInstances = (call: Call) => call.script.includes('runningApplicationsWithBundleIdentifier')

/** Chrome processes as `instances`, then `respond` for every other script. */
function chromeRunner(respond: (call: Call) => string | Error, instances: object[] = [HEADLESS, ON_SCREEN]) {
  return fakeRunner((call) => (isInstances(call) ? JSON.stringify(instances) : respond(call)))
}

describe('ChromeTabs', () => {
  it('reads and clicks tabs in the Chrome process that owns the window on screen, not whichever one the name "Google Chrome" reaches', async () => {
    const tabs = [{ window: 0, index: 14, title: 'blink-182 - YouTube', url: 'https://www.youtube.com/watch?v=1', active: false }]
    const { runner, calls } = chromeRunner(() => JSON.stringify(tabs))
    const chrome = new ChromeTabs(runner)
    expect(await chrome.list()).toEqual(tabs)
    await chrome.activate(tabs[0])
    const scripts = calls.filter((call) => !isInstances(call))
    expect(scripts.map((call) => call.argv[0])).toEqual(['502', '502'])
    // Nothing is addressed to Chrome by name, which is what reached the headless process.
    for (const call of scripts) expect(call.script).not.toContain("Application('Google Chrome')")
  })

  it('picks the frontmost process, else the one with the frontmost window, and never a background-only one', () => {
    expect(chooseInstance([HEADLESS, ON_SCREEN])?.pid).toBe(502)
    const second = { pid: 900, regular: true, frontmost: false, frontWindow: 3 }
    const behind = { pid: 901, regular: true, frontmost: false, frontWindow: 7 }
    expect(chooseInstance([behind, second, HEADLESS])?.pid).toBe(900)
    expect(chooseInstance([{ ...behind, frontWindow: null }, HEADLESS])?.pid).toBe(901)
    expect(chooseInstance([HEADLESS])).toBeNull()
  })

  it('returns no tabs when no Chrome window can be on screen, without asking any process', async () => {
    const { runner, calls } = chromeRunner(() => new Error('should not be asked'), [HEADLESS])
    expect(await new ChromeTabs(runner).list()).toEqual([])
    expect(calls).toHaveLength(1)
  })

  it('passes the tab as argv after "--" and never splices its title into the script', async () => {
    const { runner, calls } = chromeRunner(() => '{"x":10,"y":20}')
    await new ChromeTabs(runner).activate(TAB)
    expect(calls[1]?.argv).toEqual(['502', TAB.title, '2', 'click'])
    expect(calls[1]?.script).not.toContain('Inbox')
  })

  it('explains a tab that is not in the tab strip, and missing Accessibility access for the app running the helper', async () => {
    const missing = chromeRunner(() => new Error('execution error: Error: Error: LOCALROUTER_NO_TAB (-2700)'))
    await expect(new ChromeTabs(missing.runner).activate(TAB)).rejects.toThrow("Could not find that tab in Chrome's tab strip.")
    const denied = chromeRunner(() => new Error('execution error: Error: Error: LOCALROUTER_NO_ACCESSIBILITY:Terminal: (-2700)'))
    await expect(new ChromeTabs(denied.runner).activate(TAB)).rejects.toThrow('Allow Terminal in System Settings › Privacy & Security › Accessibility')
    const launchAgent = chromeRunner(() => new Error('execution error: Error: Error: LOCALROUTER_NO_ACCESSIBILITY:: (-2700)'))
    await expect(new ChromeTabs(launchAgent.runner).activate(TAB)).rejects.toThrow('Allow the node binary that runs LocalRouter Desktop')
  })

  it('says Chrome itself is refusing, rather than blaming the tab or a permission', async () => {
    const { runner } = chromeRunner(() => new Error('execution error: Error: Error: LOCALROUTER_UNREACHABLE:Google Chrome: (-2700)'))
    const error = await new ChromeTabs(runner).list().then(() => new Error('listed'), (caught: Error) => caught)
    expect(error.message).toMatch(/^Google Chrome is refusing to show its window to other apps.*Quitting and reopening Google Chrome fixes this/)
    expect(error.message).not.toContain('Privacy & Security')
  })

  it('switches with Chrome’s tab search, typing the name only as argv, and reports the title now in front', async () => {
    const found = chromeRunner(() => JSON.stringify({ before: 'Sidebud · Computer', after: 'blink-182 - YouTube', opened: true }))
    expect(await new ChromeTabs(found.runner).search('YouTube')).toBe('blink-182 - YouTube')
    expect(found.calls[1]?.argv).toEqual(['502', 'YouTube'])
    expect(found.calls[1]?.script).not.toContain('YouTube')
    const nothing = chromeRunner(() => JSON.stringify({ before: 'Sidebud · Computer', after: 'Sidebud · Computer', opened: true }))
    expect(await new ChromeTabs(nothing.runner).search('YouTube')).toBeNull()
    const closed = chromeRunner(() => JSON.stringify({ before: 'x', after: 'x', opened: false }))
    await expect(new ChromeTabs(closed.runner).search('YouTube')).rejects.toThrow("Chrome's tab search did not open.")
  })
})

describe('DesktopBackend', () => {
  const observation = {
    app: 'Mail',
    title: 'Inbox',
    text: 'Inbox 3 unread',
    controls: [
      { ref: '0.1', role: 'button', label: 'Compose' },
      { ref: '0.2.4', role: 'textbox', label: 'Search', value: '', focused: true },
      { ref: '0.3.0', role: 'row', label: 'Weekly report' },
    ],
  }

  it('parses an observation of the frontmost window', async () => {
    const { runner, calls } = fakeRunner(() => JSON.stringify(observation))
    expect(await new DesktopBackend(runner).observe()).toEqual({ backend: 'desktop', ...observation })
    expect(calls[0]?.argv).toEqual([])
  })

  it('refuses an observation with a role outside the closed set', async () => {
    const { runner } = fakeRunner(() => JSON.stringify({ ...observation, controls: [{ ref: '0.1', role: 'slider', label: 'Volume' }] }))
    await expect(new DesktopBackend(runner).observe()).rejects.toThrow(/role should be one of/)
  })

  it('sends refs, text and key codes only as argv', async () => {
    const { runner, calls } = fakeRunner((call) => (call.argv[0] === 'value' ? JSON.stringify({ value: "'); Application('Finder').delete(); ('" }) : 'ok'))
    const desktop = new DesktopBackend(runner, { sleep: async () => {} })
    const text = "'); Application('Finder').delete(); ('"
    await desktop.act({ kind: 'type', ref: '0.2.4', text })
    await desktop.act({ kind: 'click', ref: '0.1' })
    await desktop.act({ kind: 'press', key: 'enter' })
    await desktop.act({ kind: 'scroll', direction: 'bottom' })
    await desktop.act({ kind: 'scroll', direction: 'down', ref: '0.3' })
    await desktop.act({ kind: 'click', ref: 'v:1850,72' })
    await desktop.act({ kind: 'press', key: 'space' })
    expect(calls.map((call) => call.argv)).toEqual([
      ['focus', JSON.stringify({ ref: '0.2.4' })],
      ['keys', JSON.stringify({ text })],
      ['value', JSON.stringify({ ref: '0.2.4' })],
      ['click', JSON.stringify({ ref: '0.1' })],
      ['press', JSON.stringify({ code: 36 })],
      ['scroll', JSON.stringify({ direction: 'bottom' })],
      ['scroll', JSON.stringify({ direction: 'down', ref: '0.3' })],
      ['click', JSON.stringify({ ref: 'v:1850,72' })],
      ['press', JSON.stringify({ code: 49 })],
    ])
    expect(calls[1]?.script).not.toContain('Finder')
  })

  it('types plain text in chunks a stop can come between, and pastes anything the keyboard cannot type as is', async () => {
    const typed: string[] = []
    const { runner, calls } = fakeRunner((call) => {
      const payload = JSON.parse(call.argv[1] ?? '{}') as { text?: string }
      if (call.argv[0] === 'keys' || call.argv[0] === 'paste') typed.push(payload.text ?? '')
      return call.argv[0] === 'value' ? JSON.stringify({ value: typed.join('') }) : '{"x":10,"y":20}'
    })
    const desktop = new DesktopBackend(runner, { sleep: async () => {} })
    const long = 'a'.repeat(120)
    expect(await desktop.act({ kind: 'type', ref: '0.1', text: long })).toEqual({ pointer: { x: 10, y: 20 } })
    expect(calls.filter((call) => call.argv[0] === 'keys').map((call) => (JSON.parse(call.argv[1]) as { text: string }).text.length)).toEqual([50, 50, 20])
    // Line breaks (a typed Return could send), other scripts and emoji go through the pasteboard, whole.
    for (const text of ['line one\nline two', 'Grüße, 你好 👋']) {
      typed.length = 0
      calls.length = 0
      await desktop.act({ kind: 'type', ref: '0.1', text })
      expect(calls.map((call) => call.argv[0])).toEqual(['focus', 'paste', 'value'])
    }
    const stop = new AbortController()
    stop.abort('You stopped it on the Mac.')
    await expect(desktop.act({ kind: 'type', ref: '0.1', text: long }, { signal: stop.signal })).rejects.toThrow('You stopped it on the Mac.')
  })

  it('presses a button through accessibility, leaving the pointer alone, unless the pointer is asked for', async () => {
    const { runner, calls } = fakeRunner((call) => (JSON.parse(call.argv[1] ?? '{}').pointer ? '{"x":5,"y":6}' : '{"background":true}'))
    const desktop = new DesktopBackend(runner)
    expect(await desktop.act({ kind: 'click', ref: '0.1' })).toEqual({ background: true })
    expect(await desktop.act({ kind: 'click', ref: '0.1' }, { pointer: true })).toEqual({ pointer: { x: 5, y: 6 } })
    expect(calls.map((call) => call.argv[1])).toEqual([JSON.stringify({ ref: '0.1' }), JSON.stringify({ ref: '0.1', pointer: true })])
    expect(SCRIPTS.act).toContain("AXUIElementPerformAction(el, $('AXPress'))")
  })

  it('fails a typing step whose text did not land, without repeating the text', async () => {
    const { runner } = fakeRunner((call) => (call.argv[0] === 'value' ? '{"value":"something else"}' : 'ok'))
    const error = await new DesktopBackend(runner, { sleep: async () => {} }).act({ kind: 'type', ref: '0.1', text: 'groceries' }).catch((e: unknown) => e as Error)
    expect(error).toBeInstanceOf(Error)
    expect((error as Error).message).toBe('The text did not land in the text box, so it stopped.')
    expect(landed(null, 'x')).toBe(true)
    expect(landed('4111 1111 1111 1111', '4111111111111111')).toBe(true)
    expect(landed('', 'x')).toBe(false)
  })

  it('never splits a surrogate pair between chunks', () => {
    const text = `${'a'.repeat(49)}👋b`
    expect(chunks(text, 50)).toEqual(['a'.repeat(49), '👋b'])
    expect(chunks('', 50)).toEqual([])
  })

  it('explains a text box that would not take the focus, and a control off every display', async () => {
    const focus = fakeRunner(() => new Error('execution error: Error: LOCALROUTER_NOT_FOCUSED (-2700)'))
    await expect(new DesktopBackend(focus.runner).act({ kind: 'type', ref: '0.1', text: 'x' })).rejects.toThrow('The text box did not take the typing focus, so nothing was typed.')
    const off = fakeRunner(() => new Error('execution error: Error: LOCALROUTER_OFFSCREEN (-2700)'))
    await expect(new DesktopBackend(off.runner).act({ kind: 'click', ref: '0.1' })).rejects.toThrow('That control is off screen; scroll it into view first.')
  })

  it('selects a field’s text through the Edit menu by its character, never with key code 0 (⌘Q on AZERTY)', () => {
    expect(SCRIPTS.act).toContain("menuCommand(app, letter)")
    expect(SCRIPTS.act).not.toMatch(/keyCode\(0,/)
  })

  it('parses every script it runs as JavaScript, as osascript would', () => {
    for (const [name, script] of Object.entries(SCRIPTS)) {
      expect(() => new Function(script), name).not.toThrow()
    }
  })

  it('reads the displays, and the page a browser shows, the window and whether the walk was cut short', async () => {
    const displays = [{ x: 0, y: 0, w: 1512, h: 982 }, { x: -1920, y: -98, w: 1920, h: 1080 }]
    expect(await new DesktopBackend(fakeRunner(() => JSON.stringify(displays)).runner).displays()).toEqual(displays)
    const seen = { ...observation, url: 'https://mail.example.com/', window: { x: 0, y: 25, w: 1200, h: 800 }, truncated: true }
    expect(await new DesktopBackend(fakeRunner(() => JSON.stringify(seen)).runner).observe()).toEqual({ backend: 'desktop', ...seen })
  })

  it('scales a Retina window capture to the longest edge asked for, and says how image pixels map to points', async () => {
    const files: string[] = []
    let resized = false
    const { runner, calls } = fakeRunner((call) => {
      if (call.script.includes('CGPreflightScreenCaptureAccess')) return '{"accessibility":true,"screenRecording":true,"host":""}'
      if (call.script.includes('CGWindowListCopyWindowInfo')) return JSON.stringify({ app: 'Safari', window: { id: 7, x: 100, y: 50, w: 1440, h: 900, title: 'Docs' } })
      if (call.file === 'screencapture') return ''
      if (call.file === 'sips' && call.argv.includes('--resampleWidth')) { resized = true; return '' }
      if (call.file === 'sips') return resized ? 'pixelWidth: 1568\n  pixelHeight: 980\n' : 'pixelWidth: 2880\n  pixelHeight: 1800\n'
      return ''
    })
    const dir = mkdtempSync(join(tmpdir(), 'lr-shot-'))
    const file = join(dir, 'shot.png')
    writeFileSync(file, Buffer.from([0x89, 0x50, 0x4e, 0x47]))
    files.push(file)
    const shot = await new DesktopBackend(runner, { tmp: () => file }).screenshot({ maxEdge: 1568 })
    expect(shot).toMatchObject({ width: 1568, height: 980, region: { x: 100, y: 50, w: 1440, h: 900 }, app: 'Safari', title: 'Docs' })
    expect(shot.scale).toBeCloseTo(1568 / 1440)
    expect([...shot.png.slice(0, 4)]).toEqual([0x89, 0x50, 0x4e, 0x47])
    expect(calls.find((call) => call.argv.includes('--resampleWidth'))?.argv).toEqual(['--resampleWidth', '1568', file])
    // The capture is deleted at once.
    expect(existsSync(file)).toBe(false)
    rmSync(dir, { recursive: true, force: true })
  })

  it('refuses a screenshot without Screen Recording access instead of returning the desktop', async () => {
    const { runner } = fakeRunner(() => '{"accessibility":true,"screenRecording":false,"host":""}')
    await expect(new DesktopBackend(runner).screenshot()).rejects.toThrow(/Screen Recording/)
  })

  it('reports where an action left the pointer, for the local stop', async () => {
    const { runner } = fakeRunner((call) => (call.argv[0] === 'press' ? 'ok' : '{"x":640.5,"y":212}'))
    const desktop = new DesktopBackend(runner)
    expect(await desktop.act({ kind: 'click', ref: '0.1' })).toEqual({ pointer: { x: 640.5, y: 212 } })
    expect(await desktop.act({ kind: 'press', key: 'enter' })).toEqual({})
  })

  it('reads screenshot spots ("v:x,y") with a real digit class in the JXA source', async () => {
    const { runner, calls } = fakeRunner(() => 'ok')
    await new DesktopBackend(runner).act({ kind: 'click', ref: 'v:1850,72' })
    // A bare \d inside the TypeScript template literal would reach osascript as a plain "d".
    expect(calls[0]?.script).toContain('/^v:(-?[\\d.]+),(-?[\\d.]+)$/')
  })

  it('reads the pointer without moving it', async () => {
    const { runner, calls } = fakeRunner(() => '{"x":1.5,"y":2}\n')
    expect(await new DesktopBackend(runner).pointer()).toEqual({ x: 1.5, y: 2 })
    expect(calls[0]?.script).toContain('CGEventGetLocation')
  })

  it('says a control is gone when its ref no longer resolves', async () => {
    const { runner } = fakeRunner(() => new Error('execution error: Error: Error: LOCALROUTER_GONE (-2700)'))
    await expect(new DesktopBackend(runner).act({ kind: 'click', ref: '9.9' })).rejects.toThrow('That control is gone; look again.')
  })

  it('maps accessibility and automation refusals to the setting to change', async () => {
    const accessibility = fakeRunner(() => new Error('System Events got an error: osascript is not allowed assistive access. (-1719)'))
    await expect(new DesktopBackend(accessibility.runner).observe()).rejects.toThrow(/Privacy & Security › Accessibility/)
    const automation = fakeRunner(() => new Error('Not authorized to send Apple events to System Events. (-1743)'))
    await expect(new DesktopBackend(automation.runner).act({ kind: 'press', key: 'tab' })).rejects.toThrow(/Privacy & Security › Automation/)
  })

  it('reads an app that refuses accessibility from a screenshot of its front window, as text spots to click', async () => {
    const window = { id: 86, x: 50, y: 30, w: 2000, h: 1400, title: 'blink-182 - YouTube' }
    const { runner, calls } = fakeRunner((call) => {
      if (call.file === 'screencapture') return ''
      if (call.script.includes('LOCALROUTER_UNREACHABLE') && call.script.includes('AXMainWindow') && !call.script.includes('CGWindowListCopyWindowInfo')) return new Error('execution error: Error: Error: LOCALROUTER_UNREACHABLE:Google Chrome: (-2700)')
      if (call.script.includes('VNRecognizeTextRequest')) return JSON.stringify([{ text: 'Search', x: 0.25, y: 0.88, w: 0.1, h: 0.02 }, { text: '•', x: 0.1, y: 0.5, w: 0.01, h: 0.01 }, { text: 'blink-182 - YouTube', x: 0.8, y: 0.97, w: 0.1, h: 0.02 }])
      return JSON.stringify({ app: 'Google Chrome', window })
    })
    const screen = await new DesktopBackend(runner).observe()
    expect(screen).toMatchObject({ backend: 'desktop', app: 'Google Chrome', title: 'blink-182 - YouTube', visual: true, text: 'blink-182 - YouTube Search' })
    // Spots are window-relative boxes turned into screen points (bottom-left origin to top-left).
    expect(screen.controls).toEqual([{ ref: 'v:1750,58', role: 'text', label: 'blink-182 - YouTube' }, { ref: 'v:650,184', role: 'text', label: 'Search' }])
    const capture = calls.find((call) => call.file === 'screencapture')
    expect(capture).toBeDefined()
  })

  it('says why when an app refuses accessibility and its screenshot shows nothing', async () => {
    const { runner } = fakeRunner((call) => {
      if (call.file === 'screencapture') return ''
      if (call.script.includes('VNRecognizeTextRequest')) return '[]'
      if (call.script.includes('AXMainWindow') && !call.script.includes('CGWindowListCopyWindowInfo')) return new Error('LOCALROUTER_UNREACHABLE:Google Chrome:')
      return JSON.stringify({ app: 'Google Chrome', window: { id: 1, x: 0, y: 0, w: 10, h: 10, title: '' } })
    })
    await expect(new DesktopBackend(runner).observe()).rejects.toThrow(/^Google Chrome is refusing to show its window/)
  })

  it('passes other failures through unchanged', async () => {
    const { runner } = fakeRunner(() => new Error('The Mac (osascript) took too long to answer (timed out)'))
    await expect(new DesktopBackend(runner).observe()).rejects.toThrow('took too long to answer')
  })
})

describe('macAccess', () => {
  it('reports Accessibility, Screen Recording, and the app macOS attributes the helper to', async () => {
    const { runner } = fakeRunner(() => '{"accessibility":true,"screenRecording":false,"host":""}\n')
    expect(await macAccess(runner)).toEqual({ accessibility: true, screenRecording: false, host: '' })
  })
})

describe('DockApps', () => {
  it("lists the Dock's apps, clicks an app's icon to bring it forward, and opens one that is not in the Dock", async () => {
    const calls: Array<{ file: string; args: string[] }> = []
    const replies = ['["Finder","Google Chrome","T3 Agents"]', 'clicked', 'missing', '']
    const dock = new DockApps({ run: async (file, args) => { calls.push({ file, args }); return replies.shift() ?? '' } })
    expect(await dock.list()).toEqual(['Finder', 'Google Chrome', 'T3 Agents'])
    expect(await dock.activate('Google Chrome')).toBe('clicked')
    // The app name reaches the script as an argument, never inside its source.
    expect(calls[1].args.slice(-2)).toEqual(['Google Chrome', 'act'])
    expect(calls[1].args[3]).not.toContain('Google Chrome')
    expect(await dock.activate('Zed')).toBe('opened')
    expect(calls.at(-1)).toEqual({ file: 'open', args: ['-a', 'Zed'] })
  })
})

describe('ChromeTabs.groups', () => {
  it("reads named tab groups from the tab strip's order", async () => {
    const tabs = new ChromeTabs(chromeRunner(() => JSON.stringify([{ name: 'Work', tabs: ['Sidebud · Computer', 'Internal testing | Sidebud'] }])).runner)
    expect(await tabs.groups()).toEqual([{ name: 'Work', tabs: ['Sidebud · Computer', 'Internal testing | Sidebud'] }])
  })
})

/** On a Mac (the macOS CI runner): every script compiles as osascript compiles it, and the reads that need no permission run. */
describe.runIf(process.platform === 'darwin')('on macOS', () => {
  it('compiles every script with osacompile', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'lr-jxa-'))
    try {
      for (const [name, script] of Object.entries(SCRIPTS)) {
        const file = join(dir, `${name}.js`)
        writeFileSync(file, script)
        await expect(jxaRunner.run('osacompile', ['-l', 'JavaScript', '-o', join(dir, `${name}.scpt`), file]), name).resolves.toBeDefined()
      }
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('reads the displays, the pointer and the grants for real', async () => {
    const desktop = new DesktopBackend()
    const displays = await desktop.displays()
    expect(displays.length).toBeGreaterThan(0)
    expect(displays[0]).toMatchObject({ x: 0, y: 0 })
    const pointer = await desktop.pointer()
    expect(Number.isFinite(pointer.x) && Number.isFinite(pointer.y)).toBe(true)
    const access = await macAccess()
    expect(typeof access.accessibility).toBe('boolean')
  })
})
