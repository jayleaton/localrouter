import { describe, expect, it } from 'vitest'
import { fitScale } from '../../../../desktop-helper/src/computer/types.ts'
import { chunks, displaysShape, explain, NEWLINE_REFUSED, screenshotShape, WindowsScreen } from './screen.ts'
import type { CallOptions, WindowsCall } from './worker.ts'
import { HELPER_CS } from './helper.cs.ts'

type Call = { op: string; args: Record<string, unknown>; options: CallOptions }

/** A fake worker: `respond` answers each op (an Error is thrown, as the worker rejects); every call is recorded. */
function fakeWorker(respond: (call: Call) => unknown) {
  const calls: Call[] = []
  const worker: WindowsCall = {
    async call(op, args = {}, options = {}) {
      const call = { op, args, options }
      calls.push(call)
      const result = respond(call)
      if (result instanceof Error) throw result
      return result
    },
  }
  return { worker, calls }
}

const REF = '1311234:0.2.1#42.65552.4'

const observation = {
  app: 'Google Chrome',
  title: 'Inbox - Gmail',
  url: 'mail.google.com/mail/u/0/#inbox',
  text: 'Inbox 3 unread',
  window: { x: -8, y: 0, w: 1936, h: 1048 },
  truncated: false,
  controls: [
    { ref: '1311234:0.1', role: 'button', label: 'Compose', bounds: { x: 10, y: 100, w: 80, h: 30 } },
    { ref: REF, role: 'textbox', label: 'Search mail', value: '', focused: true, bounds: { x: 200, y: 60, w: 600, h: 40 } },
    { ref: '1311234:0.3.0', role: 'textbox', label: 'Password', password: true, value: 'hunter2' },
  ],
}

describe('WindowsScreen observe', () => {
  it('parses the front window, and never keeps a password field’s value even if the helper sent one', async () => {
    const { worker, calls } = fakeWorker(() => observation)
    const seen = await new WindowsScreen(worker, { observeBudgetMs: 2_000 }).observe()
    expect(seen).toMatchObject({ backend: 'desktop', app: 'Google Chrome', title: 'Inbox - Gmail', url: 'mail.google.com/mail/u/0/#inbox', window: { x: -8, y: 0, w: 1936, h: 1048 } })
    expect(seen.truncated).toBeUndefined()
    expect(seen.controls[1]).toEqual({ ref: REF, role: 'textbox', label: 'Search mail', value: '', focused: true, bounds: { x: 200, y: 60, w: 600, h: 40 } })
    expect(seen.controls[2]).toEqual({ ref: '1311234:0.3.0', role: 'textbox', label: 'Password' })
    expect(JSON.stringify(seen)).not.toContain('hunter2')
    expect(calls[0]).toMatchObject({ op: 'observe', args: { budgetMs: 2_000 } })
  })

  it('marks a walk that ran out of budget, and refuses output of the wrong shape', async () => {
    const truncated = fakeWorker(() => ({ ...observation, truncated: true }))
    expect((await new WindowsScreen(truncated.worker).observe()).truncated).toBe(true)
    const wrong = fakeWorker(() => ({ ...observation, controls: [{ ref: 'x', role: 'slider', label: '' }] }))
    await expect(new WindowsScreen(wrong.worker).observe()).rejects.toThrow('observation controls[0] role should be one of')
  })

  it('explains elevated apps (UIPI) and the secure desktop in words a person can act on', async () => {
    const elevated = fakeWorker(() => new Error('LR_ELEVATED|Task Manager|'))
    await expect(new WindowsScreen(elevated.worker).observe()).rejects.toThrow('Task Manager runs as administrator, so LocalRouter cannot see, click or type in it. Run Task Manager normally (not as administrator), or do this step yourself.')
    const secure = fakeWorker(() => new Error('Exception calling "Run": "LR_SECURE_DESKTOP||"'))
    await expect(new WindowsScreen(secure.worker).observe()).rejects.toThrow(/secure screen \(a User Account Control prompt, or the lock or sign-in screen\)/)
    expect(explain('LR_GONE||')?.message).toBe('That control is gone; look again.')
    expect(explain('something else')).toBeNull()
  })
})

describe('WindowsScreen act', () => {
  it('clicks through a UI Automation pattern first: no pointer, reported as background', async () => {
    const { worker, calls } = fakeWorker(() => ({ done: true, how: 'invoke' }))
    expect(await new WindowsScreen(worker).act({ kind: 'click', ref: REF })).toEqual({ background: true })
    expect(calls.map((c) => c.op)).toEqual(['semantic'])
    expect(calls[0]!.args).toEqual({ kind: 'click', ref: REF })
  })

  it('counts a pattern click still running (a modal dialog it opened holds Invoke) as done, without a pointer fallback', async () => {
    const { worker, calls } = fakeWorker(() => ({ done: true, how: 'invoke', pending: true }))
    expect(await new WindowsScreen(worker).act({ kind: 'click', ref: REF })).toEqual({ background: true })
    expect(calls.map((c) => c.op)).toEqual(['semantic'])
  })

  it('falls back to a real click at the control when it has no pattern, and reports where the pointer was left', async () => {
    const { worker, calls } = fakeWorker((call) => {
      if (call.op === 'semantic') return { done: false, how: '' }
      if (call.op === 'point') return { hwnd: 1311234, x: 240, y: 80 }
      return { x: 240, y: 80 }
    })
    expect(await new WindowsScreen(worker).act({ kind: 'click', ref: REF })).toEqual({ pointer: { x: 240, y: 80 } })
    expect(calls.map((c) => c.op)).toEqual(['semantic', 'point', 'click'])
    expect(calls[2]!.args).toEqual({ hwnd: 1311234, x: 240, y: 80 })
  })

  it('clicks with the real pointer, skipping the pattern, when the pointer is asked for', async () => {
    const { worker, calls } = fakeWorker((call) => (call.op === 'point' ? { hwnd: 1311234, x: 240, y: 80 } : { x: 240, y: 80 }))
    expect(await new WindowsScreen(worker).act({ kind: 'click', ref: REF }, { pointer: true })).toEqual({ pointer: { x: 240, y: 80 } })
    expect(calls.map((c) => c.op)).toEqual(['point', 'click'])
  })

  it('sets a value semantically; a value that did not land is an error, not a retry with keys', async () => {
    const set = fakeWorker(() => ({ done: true, how: 'value' }))
    expect(await new WindowsScreen(set.worker).act({ kind: 'type', ref: REF, text: 'line one\nline two' })).toEqual({ background: true })
    expect(set.calls[0]!.args).toEqual({ kind: 'type', ref: REF, text: 'line one\nline two' })
    const missed = fakeWorker(() => new Error('LR_DID_NOT_TAKE|Search mail|'))
    await expect(new WindowsScreen(missed.worker).act({ kind: 'type', ref: REF, text: 'x' })).rejects.toThrow('The text did not land in “Search mail”')
    expect(missed.calls.map((c) => c.op)).toEqual(['semantic'])
  })

  it('types with keys in chunks after clicking into the field and selecting its text, when there is no value pattern', async () => {
    const text = 'a'.repeat(450)
    const { worker, calls } = fakeWorker((call) => {
      if (call.op === 'semantic') return { done: false, how: '' }
      if (call.op === 'point') return { hwnd: 77, x: 10, y: 20 }
      if (call.op === 'click') return { x: 10, y: 20 }
      return {}
    })
    expect(await new WindowsScreen(worker).act({ kind: 'type', ref: REF, text })).toEqual({ pointer: { x: 10, y: 20 } })
    expect(calls.map((c) => c.op)).toEqual(['semantic', 'point', 'click', 'select_all', 'type', 'type', 'type'])
    expect(calls.filter((c) => c.op === 'type').map((c) => (c.args.text as string).length)).toEqual([200, 200, 50])
    expect(calls.filter((c) => c.op === 'type').every((c) => c.args.hwnd === 77)).toBe(true)
  })

  it('never types a line break as keys: nothing is clicked or typed', async () => {
    const { worker, calls } = fakeWorker(() => ({ done: false, how: '' }))
    await expect(new WindowsScreen(worker).act({ kind: 'type', ref: REF, text: 'Hi\r\nthere' })).rejects.toThrow(NEWLINE_REFUSED)
    expect(calls.map((c) => c.op)).toEqual(['semantic'])
  })

  it('stops typing between chunks when the task is stopped', async () => {
    const controller = new AbortController()
    const { worker, calls } = fakeWorker((call) => {
      if (call.op === 'semantic') return { done: false, how: '' }
      if (call.op === 'point') return { hwnd: 77, x: 10, y: 20 }
      if (call.op === 'click') return { x: 10, y: 20 }
      if (call.op === 'type') controller.abort('You stopped it on the computer.')
      return {}
    })
    await expect(new WindowsScreen(worker).act({ kind: 'type', ref: REF, text: 'b'.repeat(500) }, { signal: controller.signal })).rejects.toThrow('You stopped it on the computer.')
    expect(calls.filter((c) => c.op === 'type')).toHaveLength(1)
    expect(calls.every((c) => c.options.signal === controller.signal)).toBe(true)
  })

  it('clears a field typed empty with Delete after selecting its text', async () => {
    const { worker, calls } = fakeWorker((call) => (call.op === 'semantic' ? { done: false, how: '' } : call.op === 'point' ? { hwnd: 5, x: 1, y: 1 } : { x: 1, y: 1 }))
    await new WindowsScreen(worker).act({ kind: 'type', ref: REF, text: '' })
    expect(calls.map((c) => c.op)).toEqual(['semantic', 'point', 'click', 'select_all', 'key'])
    expect(calls[4]!.args).toEqual({ hwnd: 5, vk: 0x2e })
  })

  it('scrolls with ScrollPattern when it can, else with the wheel over the ref or the window', async () => {
    const semantic = fakeWorker(() => ({ done: true, how: 'scroll' }))
    expect(await new WindowsScreen(semantic.worker).act({ kind: 'scroll', direction: 'bottom' })).toEqual({ background: true })
    expect(semantic.calls[0]!.args).toEqual({ kind: 'scroll', direction: 'bottom' })
    const wheel = fakeWorker((call) => (call.op === 'semantic' ? { done: false, how: '' } : call.op === 'point' ? { hwnd: 9, x: 500, y: 400 } : { x: 500, y: 400 }))
    expect(await new WindowsScreen(wheel.worker).act({ kind: 'scroll', direction: 'down', ref: REF })).toEqual({ pointer: { x: 500, y: 400 } })
    expect(wheel.calls[1]!.args).toEqual({ ref: REF })
    expect(wheel.calls[2]).toMatchObject({ op: 'wheel', args: { hwnd: 9, x: 500, y: 400, notches: -3 } })
  })

  it('presses keys as virtual-key codes, without moving the pointer', async () => {
    const { worker, calls } = fakeWorker(() => ({}))
    expect(await new WindowsScreen(worker).act({ kind: 'press', key: 'enter' })).toEqual({})
    expect(calls[0]).toMatchObject({ op: 'key', args: { vk: 0x0d } })
  })

  it('refuses a ref that is not one of its own before asking the helper', async () => {
    const { worker, calls } = fakeWorker(() => ({ done: true }))
    await expect(new WindowsScreen(worker).act({ kind: 'click', ref: '0.1"; Remove-Item' })).rejects.toThrow('That control is gone; look again.')
    expect(calls).toHaveLength(0)
  })

  it('says why a pointer fallback could not run on an elevated window or one that will not come forward', async () => {
    const elevated = fakeWorker((call) => (call.op === 'semantic' ? new Error('LR_ELEVATED|Registry Editor|') : {}))
    await expect(new WindowsScreen(elevated.worker).act({ kind: 'click', ref: REF })).rejects.toThrow('Registry Editor runs as administrator')
    const behind = fakeWorker((call) => (call.op === 'semantic' ? { done: false } : new Error('LR_NOT_FRONT|Notepad|')))
    await expect(new WindowsScreen(behind.worker).act({ kind: 'click', ref: REF })).rejects.toThrow('Windows did not let LocalRouter bring Notepad to the front. Click Notepad in the taskbar, then try again.')
  })
})

describe('chunks', () => {
  it('splits at the size and never between the halves of a surrogate pair', () => {
    expect(chunks('abcdef', 4)).toEqual(['abcd', 'ef'])
    expect(chunks('abc😀de', 4)).toEqual(['abc', '😀de'])
    expect(chunks('', 4)).toEqual([])
    expect(chunks('😀😀', 2).join('')).toBe('😀😀')
  })
})

const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13]).toString('base64')

describe('screenshots and displays', () => {
  it('scales the window to the longest edge and maps image pixels back to global coordinates', async () => {
    const region = { x: -1928, y: 8, w: 3840, h: 2160 }
    const scale = fitScale(3840, 2160, 1568)
    const { worker, calls } = fakeWorker(() => ({ png: PNG, width: 1568, height: Math.round(2160 * scale), region, app: 'Notepad', title: 'notes.txt' }))
    const shot = await new WindowsScreen(worker).screenshot()
    expect(calls[0]!.args).toEqual({ maxEdge: 1568 })
    expect(shot.scale).toBeCloseTo(1568 / 3840)
    expect(shot.height).toBe(882)
    expect([...shot.png.slice(0, 4)]).toEqual([0x89, 0x50, 0x4e, 0x47])
    // Image pixel (784, 441) is the middle of the window in global coordinates.
    expect(region.x + 784 / shot.scale).toBeCloseTo(region.x + region.w / 2)
    expect(region.y + 441 / shot.scale).toBeCloseTo(region.y + region.h / 2)
  })

  it('keeps a small window at scale 1, and refuses an image whose size does not match its scale or is not a PNG', () => {
    const region = { x: 0, y: 0, w: 800, h: 600 }
    expect(screenshotShape({ png: PNG, width: 800, height: 600, region, app: 'a', title: 't' }, 1568).scale).toBe(1)
    expect(() => screenshotShape({ png: PNG, width: 400, height: 300, region, app: 'a', title: 't' }, 1568)).toThrow('not the 800×600 its scale gives')
    expect(() => screenshotShape({ png: Buffer.from('GIF89a').toString('base64'), width: 800, height: 600, region, app: 'a', title: 't' }, 1568)).toThrow('not a PNG')
  })

  it('lists displays primary first, in global physical pixels, including ones left of and above the primary', () => {
    expect(displaysShape([
      { x: -2560, y: -300, w: 2560, h: 1440, primary: false },
      { x: 0, y: 0, w: 3840, h: 2160, primary: true },
    ])).toEqual([{ x: 0, y: 0, w: 3840, h: 2160 }, { x: -2560, y: -300, w: 2560, h: 1440 }])
    expect(() => displaysShape([{ x: 0, y: 0, w: 'wide', h: 1 }])).toThrow('displays[0] w should be a number')
  })

  it('reads the pointer without a click time (Windows has no reliable one)', async () => {
    const { worker } = fakeWorker(() => ({ x: -100, y: 50 }))
    expect(await new WindowsScreen(worker).pointer()).toEqual({ x: -100, y: 50 })
  })
})

describe('bringing a window to the front (Windows run, 2026-10-25: "Windows did not let LocalRouter bring Google Chrome to the front")', () => {
  it('tries the usual ways in order, least invasive first, before saying to click the taskbar', () => {
    const front = HELPER_CS.slice(HELPER_CS.indexOf('private static void Front(IntPtr hwnd)'), HELPER_CS.indexOf('private static bool IsFront'))
    const steps = ['Native.SetForegroundWindow(hwnd);', 'el.SetFocus();', 'Native.AttachThreadInput(me, them, true)', 'KeyInput(0xE8, 0, 0)', 'throw new LrError("NOT_FRONT"']
    const at = steps.map((step) => front.indexOf(step))
    expect(at.every((i) => i >= 0)).toBe(true)
    expect([...at].sort((x, y) => x - y)).toEqual(at)
    // The thread input is always detached again, and no Alt key (it opens the window's menu) is pressed.
    expect(front).toContain('Native.AttachThreadInput(me, them, false);')
    expect(front).not.toMatch(/0x12\b|VK_MENU/)
  })
})

describe('setting a field’s value gives it the focus first (Windows run, 2026-10-25: an Enter went to the page)', () => {
  it('SetFocus comes before ValuePattern.SetValue, so the Enter that follows reaches the field', () => {
    const typing = HELPER_CS.slice(HELPER_CS.indexOf('private static Dictionary<string, object> SemanticType'), HELPER_CS.indexOf('private static ScrollPattern Scrollable'))
    expect(typing.indexOf('el.SetFocus();')).toBeGreaterThan(0)
    expect(typing.indexOf('el.SetFocus();')).toBeLessThan(typing.indexOf('vp.SetValue(text);'))
  })
})

