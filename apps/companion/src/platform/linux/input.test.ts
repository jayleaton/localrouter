import { describe, expect, it } from 'vitest'
import { CHUNK, MULTILINE, chooseInjector, chunks, keysym } from './input.ts'
import type { TokenStore } from './input.ts'
import { fakeRunner, fakeSession, tool } from './testing.ts'

const memoryTokens = (start: string | null = null) => {
  const store = { token: start, writes: 0 }
  const tokens: TokenStore = { read: async () => store.token, write: async (token) => { store.token = token; store.writes += 1 } }
  return { store, tokens }
}

describe('injector choice', () => {
  it('drives X11 through xdotool: exact pointer, keys, and text after "--"', async () => {
    const { runner, lines } = fakeRunner()
    const input = chooseInjector(fakeSession({ server: 'x11', tools: ['xdotool'] }), runner)
    expect(input).toMatchObject({ name: 'xdotool' })
    expect(input.noPointer).toBeUndefined()
    await input.click({ x: 100.4, y: 200.6 })
    await input.key('enter')
    await input.key('select_all')
    await input.type('-rf hello')
    await input.scroll({ x: 5, y: 6 }, 'up')
    expect(lines()).toEqual([
      'xdotool mousemove --sync 100 201 click 1',
      'xdotool key --clearmodifiers Return',
      'xdotool key --clearmodifiers ctrl+a',
      'xdotool type --clearmodifiers --delay 12 -- -rf hello',
      'xdotool mousemove --sync 5 6 click --repeat 5 --delay 20 4',
    ])
  })

  it('on Hyprland places the pointer with hyprctl and clicks with ydotool, and types with wtype', async () => {
    const { runner, lines } = fakeRunner()
    const input = chooseInjector(fakeSession({ server: 'wayland', compositor: 'hyprland', ydotoold: true, tools: ['hyprctl', 'ydotool', 'wtype'] }), runner)
    expect(input.name).toBe('hyprctl + ydotool + wtype')
    await input.click({ x: 10, y: 20 })
    await input.scroll({ x: 10, y: 20 }, 'bottom')
    await input.key('tab')
    await input.type('hi')
    expect(lines()).toEqual([
      'hyprctl dispatch movecursor 10 20',
      'ydotool click 0xC0',
      'hyprctl dispatch movecursor 10 20',
      'ydotool mousemove --wheel -x 0 -y -25',
      'wtype -k Tab',
      'wtype -d 12 -- hi',
    ])
  })

  it('on Hyprland without ydotoold has keys but refuses the pointer, saying what is missing', async () => {
    const { runner, calls } = fakeRunner()
    const input = chooseInjector(fakeSession({ server: 'wayland', compositor: 'hyprland', ydotoold: false, tools: ['hyprctl', 'ydotool', 'wtype'] }), runner)
    expect(input.noPointer).toMatch(/ydotoold/)
    await expect(input.click({ x: 1, y: 1 })).rejects.toThrow(/ydotoold/)
    expect(calls).toHaveLength(0)
  })

  it('on sway uses its own seat commands for the pointer (no uinput needed)', async () => {
    const { runner, lines } = fakeRunner()
    const input = chooseInjector(fakeSession({ server: 'wayland', compositor: 'sway', tools: ['swaymsg', 'wtype'] }), runner)
    await input.click({ x: 3, y: 4 })
    expect(lines()).toEqual(['swaymsg seat - cursor set 3 4', 'swaymsg seat - cursor press button1', 'swaymsg seat - cursor release button1'])
  })

  it('on GNOME Wayland has no pointer at all, and keys through ydotool when its daemon runs', async () => {
    const { runner, lines } = fakeRunner()
    const input = chooseInjector(fakeSession({ server: 'wayland', compositor: 'gnome', ydotoold: true, tools: ['ydotool', 'wtype', 'python3'] }), runner)
    expect(input.noPointer).toMatch(/does not let another app place the pointer/)
    await input.key('escape')
    expect(lines()).toEqual(['ydotool key 1:1 1:0'])
  })

  it('on GNOME Wayland without ydotool sends keys through the RemoteDesktop portal and keeps its restore token', async () => {
    const { store, tokens } = memoryTokens()
    const { runner, calls } = fakeRunner(() => '{"token":"tok-1"}\n')
    const input = chooseInjector(fakeSession({ server: 'wayland', compositor: 'gnome', tools: ['python3'] }), runner, { tokens })
    expect(input.name).toBe('RemoteDesktop portal')
    await input.key('enter')
    expect(JSON.parse(calls[0]?.input ?? '')).toEqual({ token: null, events: [[0xff0d, 1], [0xff0d, 0]] })
    // The first run waits for the person's answer to the portal's dialog; later ones restore silently.
    expect(calls[0]?.options.timeoutMs).toBe(120_000)
    expect(store).toEqual({ token: 'tok-1', writes: 1 })
    await input.type('é')
    expect(JSON.parse(calls[1]?.input ?? '')).toEqual({ token: 'tok-1', events: [[0xe9, 1], [0xe9, 0]] })
    expect(calls[1]?.options.timeoutMs).toBe(15_000)
    expect(store.writes).toBe(1)
  })

  it('has no keys on GNOME Wayland with neither ydotool nor a token store, and says how to fix it', async () => {
    const input = chooseInjector(fakeSession({ server: 'wayland', compositor: 'gnome', tools: ['python3'] }), fakeRunner().runner)
    expect(input.noKeys).toMatch(/ydotool with ydotoold, or the RemoteDesktop portal/)
    await expect(input.key('enter')).rejects.toThrow(input.noKeys)
  })
})

describe('typing', () => {
  it('types long text in chunks of at most 200 characters and stops between chunks on a stop', async () => {
    const controller = new AbortController()
    const { runner, calls } = fakeRunner((call) => {
      if (tool(call) === 'xdotool' && calls.length === 1) controller.abort('You stopped it.')
      return ''
    })
    const input = chooseInjector(fakeSession({ server: 'x11', tools: ['xdotool'] }), runner)
    await expect(input.type('a'.repeat(CHUNK * 3), { signal: controller.signal })).rejects.toThrow('You stopped it.')
    expect(calls).toHaveLength(1)
    expect(calls[0]?.args.at(-1)).toHaveLength(CHUNK)
  })

  it('never splits a surrogate pair across chunks', () => {
    const text = `${'a'.repeat(CHUNK - 1)}😀b`
    expect(chunks(text)).toEqual([`${'a'.repeat(CHUNK - 1)}😀`, 'b'])
  })

  it('refuses line breaks rather than pressing Enter mid-text', async () => {
    const { runner, calls } = fakeRunner()
    const input = chooseInjector(fakeSession({ server: 'x11', tools: ['xdotool'] }), runner)
    await expect(input.type('line one\nline two')).rejects.toThrow(MULTILINE)
    expect(calls).toHaveLength(0)
  })

  it('maps characters to X keysyms: Latin-1 as itself, the rest as Unicode keysyms', () => {
    expect(keysym('a')).toBe(0x61)
    expect(keysym('ß')).toBe(0xdf)
    expect(keysym('€')).toBe(0x010020ac)
  })
})
