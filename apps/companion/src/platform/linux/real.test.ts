import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import type { Observation, ScreenControl } from '../../../../desktop-helper/src/computer/types.ts'
import type { Platform } from '../types.ts'
import { createLinuxPlatform } from './index.ts'

/**
 * The real Linux backend against a real session: only in CI (`LOCALROUTER_REAL_LINUX=1`), where the job starts Xvfb,
 * openbox, the at-spi bus and scripts/smoke/gtk-window.py inside its own D-Bus session. Never on a person's desktop:
 * it clicks and types.
 */
const REAL = process.env.LOCALROUTER_REAL_LINUX === '1' && process.platform === 'linux'

const find = (seen: Observation, role: ScreenControl['role'], label: RegExp) => seen.controls.find((c) => c.role === role && label.test(c.label))

describe.runIf(REAL)('Linux, for real (Xvfb, openbox, at-spi, the GTK smoke window)', () => {
  const dir = mkdtempSync(join(tmpdir(), 'lr-linux-real-'))
  let platform: Platform
  afterAll(() => rmSync(dir, { recursive: true, force: true }))

  const observe = async () => {
    const seen = await platform.screen!.observe()
    return seen
  }

  it('detects the session and is ready', async () => {
    platform = await createLinuxPlatform({ dataDir: dir })
    const access = await platform.access()
    console.log(`backend: ${access.backend}; ${JSON.stringify(access.requirements.map((r) => [r.id, r.granted]))}`)
    expect(access.accessibility).toBe(true)
    expect(access.input).toBe(true)
    expect(platform.screen).not.toBeNull()
  })

  it('reads the window: its controls with bounds, and never the password', async () => {
    const seen = await observe()
    console.log(JSON.stringify({ app: seen.app, title: seen.title, controls: seen.controls.map((c) => `${c.role}:${c.label}`) }))
    expect(seen.title).toBe('LocalRouter smoke')
    expect(find(seen, 'button', /^Press me$/)).toBeDefined()
    expect(find(seen, 'textbox', /^Search$/)).toBeDefined()
    expect(find(seen, 'tab', /^Second$/)).toBeDefined()
    const secret = seen.controls.find((c) => /^Password$/.test(c.label))
    expect(secret?.value ?? '').not.toContain('hunter2')
    expect(JSON.stringify(seen)).not.toContain('hunter2')
    const press = find(seen, 'button', /^Press me$/)!
    expect(press.bounds && press.bounds.w > 0 && press.bounds.h > 0).toBe(true)
  })

  it('clicks a button through its accessible action, without the pointer', async () => {
    const press = find(await observe(), 'button', /^Press me$/)!
    const result = await platform.screen!.act({ kind: 'click', ref: press.ref })
    expect(result).toMatchObject({ mode: 'background' })
    await new Promise((resolve) => setTimeout(resolve, 300))
    expect((await observe()).text).toContain('Pressed')
  })

  it('types into a text box and reads it back', async () => {
    const search = find(await observe(), 'textbox', /^Search$/)!
    try {
      await platform.screen!.act({ kind: 'type', ref: search.ref, text: 'groceries list' })
    } catch (error) {
      console.log(`typing failed; the box now holds ${JSON.stringify(find(await observe(), 'textbox', /^Search$/)?.value)}`)
      throw error
    }
    await new Promise((resolve) => setTimeout(resolve, 300))
    expect(find(await observe(), 'textbox', /^Search$/)?.value).toBe('groceries list')
  })

  it('switches a tab', async () => {
    const second = find(await observe(), 'tab', /^Second$/)!
    await platform.screen!.act({ kind: 'click', ref: second.ref })
    await new Promise((resolve) => setTimeout(resolve, 300))
    expect((await observe()).text).toContain('Second page')
  })

  it('acts in the background on the GTK window while a “user” moves the real pointer, and never moves it itself', async () => {
    // xdotool is the "user" here: it parks the pointer, and we assert the agent's background clicks never move it.
    const run = (args: string[]) => execFileSync('xdotool', args, { stdio: 'pipe' }).toString().trim()
    run(['mousemove', '--sync', '1100', '740'])
    const before = run(['getmouselocation', '--shell'])
    for (let i = 0; i < 3; i += 1) {
      const press = find(await observe(), 'button', /^Press me$/)!
      const result = await platform.screen!.act({ kind: 'click', ref: press.ref })
      expect(result).toMatchObject({ mode: 'background' }) // never the real pointer
    }
    // A click on the text box focuses it through AT-SPI (grab_focus), not the pointer and not its "activate" (Enter).
    const search = find(await observe(), 'textbox', /^Search$/)!
    expect(await platform.screen!.act({ kind: 'click', ref: search.ref })).toMatchObject({ mode: 'background', how: 'focused through AT-SPI', at: { x: expect.any(Number), y: expect.any(Number) } })
    await new Promise((resolve) => setTimeout(resolve, 200))
    expect((await observe()).text).toContain('Pressed')
    // The real pointer is exactly where the "user" left it: the agent did not move it.
    expect(run(['getmouselocation', '--shell'])).toBe(before)
  })

  it('reads the pointer and the display, and takes a scaled screenshot of the window', async () => {
    const pointer = await platform.pointer!()
    expect(Number.isFinite(pointer.x) && Number.isFinite(pointer.y)).toBe(true)
    const displays = await platform.displays!()
    expect(displays).toEqual(expect.arrayContaining([expect.objectContaining({ w: 1280, h: 800 })]))
    const shot = await platform.screen!.screenshot!({ maxEdge: 400 })
    expect([...shot.png.slice(0, 4)]).toEqual([0x89, 0x50, 0x4e, 0x47])
    expect(Math.max(shot.width, shot.height)).toBeLessThanOrEqual(400)
    expect(shot.region.w).toBeGreaterThan(300)
    console.log(JSON.stringify({ width: shot.width, height: shot.height, region: shot.region, scale: shot.scale }))
  })
})
