import { describe, expect, it } from 'vitest'
import { ATSPI_SCRIPT, LinuxScreen, NO_ORIGIN, addressUrl } from './atspi.ts'
import { MULTILINE, chooseInjector } from './input.ts'
import type { Session } from './session.ts'
import { atspiOp, fakeRunner, fakeSession, tool } from './testing.ts'
import type { Call } from './testing.ts'

/** An observation as the script prints it: raw extents, the window's own frame and its process. */
const RAW = {
  app: 'gedit',
  title: 'notes.txt',
  pid: 4242,
  frame: [0, 0, 1260, 800],
  controls: [
    { ref: '0.1', role: 'button', label: 'Save', bounds: [1100, 10, 80, 30] },
    { ref: '0.2', role: 'textbox', label: 'Search', value: 'groceries', focused: true, bounds: [10, 10, 300, 30] },
    { ref: '0.3', role: 'textbox', label: 'Password' },
  ],
  text: 'Milk, eggs',
  truncated: true,
}

const HYPR = { address: '0x1', at: [1930, 52], size: [1260, 800], pid: 4242, class: 'org.gnome.gedit', title: 'notes.txt' }

/** A screen over a fake runner: `atspi` answers each script op, `other` every other tool. */
function screen(session: Session, atspi: (op: string, payload: Record<string, unknown>) => unknown, other: (call: Call) => string | Error = () => '') {
  const fake = fakeRunner((call) => {
    const op = atspiOp(call)
    if (!op) return other(call)
    const out = atspi(op, JSON.parse(call.input ?? '{}'))
    return out instanceof Error ? out : JSON.stringify(out)
  })
  return { ...fake, screen: new LinuxScreen({ session, runner: fake.runner, injector: chooseInjector(session, fake.runner) }) }
}

const X11 = fakeSession({ server: 'x11', tools: ['xdotool', 'python3'] })
const HYPRLAND = fakeSession({ server: 'wayland', compositor: 'hyprland', ydotoold: true, tools: ['hyprctl', 'ydotool', 'wtype', 'python3'] })
const GNOME = fakeSession({ server: 'wayland', compositor: 'gnome', tools: ['python3'] })

describe('LinuxScreen observe', () => {
  it('on X11 keeps AT-SPI’s screen extents as global bounds, and says the walk was cut short', async () => {
    const { screen: x11, calls } = screen(X11, () => RAW)
    const seen = await x11.observe()
    expect(seen).toMatchObject({ backend: 'desktop', app: 'gedit', title: 'notes.txt', text: 'Milk, eggs', truncated: true, window: { x: 0, y: 0, w: 1260, h: 800 } })
    expect(seen.controls[0]).toEqual({ ref: '0.1', role: 'button', label: 'Save', bounds: { x: 1100, y: 10, w: 80, h: 30 } })
    expect(seen.controls[2]).toEqual({ ref: '0.3', role: 'textbox', label: 'Password' })
    // Nothing but the script ran: X11 needs no compositor answer.
    expect(calls.map(atspiOp)).toEqual(['observe'])
  })

  it('on Hyprland adds the compositor’s window origin to the window-relative extents, and picks the window by its process', async () => {
    const { screen: hypr, calls } = screen(HYPRLAND, () => RAW, () => JSON.stringify(HYPR))
    const seen = await hypr.observe()
    expect(seen.window).toEqual({ x: 1930, y: 52, w: 1260, h: 800 })
    expect(seen.controls[0]?.bounds).toEqual({ x: 3030, y: 62, w: 80, h: 30 })
    expect(JSON.parse(calls[1]?.input ?? '')).toEqual({ hint: { pid: 4242, title: 'notes.txt' } })
  })

  it('leaves bounds out when the compositor’s focused window is another process, or the compositor cannot say (GNOME)', async () => {
    const other = screen(HYPRLAND, () => RAW, () => JSON.stringify({ ...HYPR, pid: 1 }))
    const mismatched = await other.screen.observe()
    expect(mismatched.window).toBeUndefined()
    expect(mismatched.controls.every((control) => control.bounds === undefined)).toBe(true)
    const gnome = await screen(GNOME, () => RAW).screen.observe()
    expect(gnome.controls.every((control) => control.bounds === undefined)).toBe(true)
  })

  it('sets the observation URL from the address bar as shown, and not while words are being typed there', async () => {
    const seen = await screen(X11, () => ({ ...RAW, url: 'github.com/jayleaton/localrouter' })).screen.observe()
    expect(seen.url).toBe('github.com/jayleaton/localrouter')
    expect(addressUrl('http://localhost:8080/x')).toBe('http://localhost:8080/x')
    expect(addressUrl('localhost:3000')).toBe('localhost:3000')
    expect(addressUrl('about:blank')).toBe('about:blank')
    expect(addressUrl('what is a fox')).toBeUndefined()
    expect(addressUrl('fox')).toBeUndefined()
  })

  it('refuses output with a role outside the closed set', async () => {
    await expect(screen(X11, () => ({ ...RAW, controls: [{ ref: '0', role: 'slider', label: 'Volume' }] })).screen.observe()).rejects.toThrow(/role should be one of/)
  })

  it('explains script failures in words, naming the app the compositor says is in front', async () => {
    await expect(screen(X11, () => new Error('LOCALROUTER_GONE')).screen.act({ kind: 'click', ref: '9.9' })).rejects.toThrow('That control is gone; look again.')
    await expect(screen(HYPRLAND, () => new Error('LOCALROUTER_UNREACHABLE'), () => JSON.stringify(HYPR)).screen.observe()).rejects.toThrow(/^org\.gnome\.gedit is not showing its window to accessibility.*toolkit-accessibility true/)
    await expect(screen(fakeSession({ distro: 'fedora', tools: ['python3'] }), () => new Error('LOCALROUTER_NO_ATSPI')).screen.observe()).rejects.toThrow('sudo dnf install python3-gobject at-spi2-core')
  })
})

describe('LinuxScreen act', () => {
  const FALLBACK = { done: false, point: [140, 70], frame: [0, 0, 1260, 800], pid: 4242 }

  it('acts through AT-SPI first and reports it as background (no pointer, no keys)', async () => {
    const { screen: x11, calls } = screen(X11, () => ({ done: true }))
    const text = "'); import os; os.system('rm -rf ~') #"
    expect(await x11.act({ kind: 'click', ref: '0.1' })).toEqual({ background: true })
    expect(await x11.act({ kind: 'type', ref: '0.2', text })).toEqual({ background: true })
    expect(await x11.act({ kind: 'scroll', direction: 'down' })).toEqual({ background: true })
    expect(calls.every((call) => atspiOp(call) === 'act')).toBe(true)
    // The action travels on stdin; argv carries only the op.
    expect(JSON.parse(calls[1]?.input ?? '')).toEqual({ kind: 'type', ref: '0.2', text, hint: {} })
    expect(calls[1]?.args.join(' ')).not.toContain('rm -rf')
  })

  it('falls back to the pointer at the element’s centre when it has no action, and reports where it was left', async () => {
    const { screen: x11, lines } = screen(X11, () => FALLBACK)
    expect(await x11.act({ kind: 'click', ref: '0.4' })).toEqual({ pointer: { x: 140, y: 70 } })
    expect(lines()).toEqual(['atspi act', 'xdotool mousemove --sync 140 70 click 1'])
  })

  it('asks the script for the pointer path when the pointer is asked for (a background click changed nothing)', async () => {
    const { screen: x11, calls } = screen(X11, () => FALLBACK)
    expect(await x11.act({ kind: 'click', ref: '0.4' }, { pointer: true })).toEqual({ pointer: { x: 140, y: 70 } })
    expect(JSON.parse(calls[0]?.input ?? '')).toMatchObject({ kind: 'click', ref: '0.4', pointer: true })
    expect(ATSPI_SCRIPT).toContain("if op == 'click' and req.get('pointer'):")
  })

  it('on Hyprland clicks at the window origin plus the element’s offset', async () => {
    const { screen: hypr, lines } = screen(HYPRLAND, () => FALLBACK, (call) => (tool(call) === 'hyprctl' && call.args[0] === 'activewindow' ? JSON.stringify(HYPR) : ''))
    expect(await hypr.act({ kind: 'click', ref: '0.4' })).toEqual({ pointer: { x: 2070, y: 122 } })
    expect(lines().slice(-2)).toEqual(['hyprctl dispatch movecursor 2070 122', 'ydotool click 0xC0'])
  })

  it('refuses a pointer fallback where the window’s global position is unknown, rather than clicking a guess', async () => {
    const unknown = screen(HYPRLAND, () => FALLBACK, () => new Error('Hyprland IPC: no socket'))
    await expect(unknown.screen.act({ kind: 'click', ref: '0.4' })).rejects.toThrow(NO_ORIGIN)
    expect(unknown.lines().filter((line) => line.startsWith('ydotool') || line.includes('movecursor'))).toEqual([])
    const gnome = screen(GNOME, () => FALLBACK)
    await expect(gnome.screen.act({ kind: 'click', ref: '0.4' })).rejects.toThrow(/does not let another app place the pointer/)
  })

  it('types through the keyboard when the box cannot be set directly, then reads it back', async () => {
    const { screen: x11, lines } = screen(X11, (op) => (op === 'read' ? { text: 'milk' } : FALLBACK))
    expect(await x11.act({ kind: 'type', ref: '0.2', text: 'milk' })).toEqual({ pointer: { x: 140, y: 70 } })
    expect(lines()).toEqual(['atspi act', 'xdotool mousemove --sync 140 70 click 1', 'xdotool key --clearmodifiers ctrl+a', 'xdotool type --clearmodifiers --delay 12 -- milk', 'atspi read'])
    const wrong = screen(X11, (op) => (op === 'read' ? { text: 'mi' } : FALLBACK))
    await expect(wrong.screen.act({ kind: 'type', ref: '0.2', text: 'milk' })).rejects.toThrow('The text did not land in that box as typed')
  })

  it('does not read a password box back', async () => {
    const { screen: x11 } = screen(X11, (op) => (op === 'read' ? new Error('read must not run') : { ...FALLBACK, secret: true }))
    expect(await x11.act({ kind: 'type', ref: '0.3', text: 'hunter2' })).toEqual({ pointer: { x: 140, y: 70 } })
  })

  it('refuses multi-line text on the keyboard path before touching the pointer', async () => {
    const { screen: x11, lines } = screen(X11, () => FALLBACK)
    await expect(x11.act({ kind: 'type', ref: '0.2', text: 'Hi,\nsee you' })).rejects.toThrow(MULTILINE)
    expect(lines()).toEqual(['atspi act'])
  })

  it('presses keys through the injector; on X11 without xdotool through AT-SPI’s own key events', async () => {
    const withTool = screen(X11, () => ({ done: true }))
    expect(await withTool.screen.act({ kind: 'press', key: 'enter' })).toEqual({})
    expect(withTool.lines()).toEqual(['xdotool key --clearmodifiers Return'])
    const bare = screen(fakeSession({ server: 'x11', tools: ['python3'] }), () => ({ done: true }))
    expect(await bare.screen.act({ kind: 'press', key: 'escape' })).toEqual({})
    expect(JSON.parse(bare.calls[0]?.input ?? '')).toEqual({ keysym: 0xff1b })
    await expect(screen(GNOME, () => ({ done: true })).screen.act({ kind: 'press', key: 'tab' })).rejects.toThrow(/Pressing keys on this desktop needs/)
  })
})
