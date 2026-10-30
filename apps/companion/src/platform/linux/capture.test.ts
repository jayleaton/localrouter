import { describe, expect, it } from 'vitest'
import { pngSize, screenCapture } from './capture.ts'
import type { CaptureOptions } from './capture.ts'
import { fakeRunner, fakeSession } from './testing.ts'
import type { Call } from './testing.ts'

/** The first 24 bytes of a PNG of `w`×`h` (the signature and IHDR), which is all pngSize reads. */
function png(w: number, h: number): Uint8Array {
  const bytes = new Uint8Array(24)
  bytes.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52])
  const view = new DataView(bytes.buffer)
  view.setUint32(16, w)
  view.setUint32(20, h)
  return bytes
}

/** Capture options over fake files: the image read back is `image`; every directory made is removed. */
function files(image: Uint8Array) {
  const made: string[] = []
  const removed: string[] = []
  const reads: string[] = []
  const options: Pick<CaptureOptions, 'makeDir' | 'read' | 'remove'> = {
    makeDir: async () => {
      made.push(`/tmp/shot-${made.length}`)
      return made.at(-1)!
    },
    read: async (path) => {
      reads.push(path)
      return image
    },
    remove: async (path) => {
      removed.push(path)
    },
  }
  return { options, made, removed, reads }
}

const X11_ANSWERS: Record<string, string> = { getactivewindow: '42\n', getwindowgeometry: 'X=100\nY=50\nWIDTH=3136\nHEIGHT=1000\n', getwindowname: 'Report\n', getwindowpid: '7\n', getwindowclassname: 'libreoffice\n' }

describe('screenshots', () => {
  it('captures the X11 front window with import, scaled in the same command, and maps image pixels to the global space', async () => {
    const { runner, calls } = fakeRunner((call: Call) => X11_ANSWERS[call.args[0] ?? ''] ?? '')
    const fs = files(png(1568, 500))
    const capture = screenCapture({ session: fakeSession({ server: 'x11', tools: ['xdotool', 'import'] }), runner, displays: async () => [], ...fs.options })
    const shot = await capture!()
    expect(calls.at(-1)?.args).toEqual(['-window', '42', '-resize', '1568x500!', 'png:/tmp/shot-0/window.png'])
    expect(shot).toMatchObject({ width: 1568, height: 500, region: { x: 100, y: 50, w: 3136, h: 1000 }, scale: 0.5, app: 'libreoffice', title: 'Report' })
    expect(fs.removed).toEqual(['/tmp/shot-0'])
  })

  it('captures a Hyprland window’s global rectangle with grim at the fitted scale', async () => {
    const { runner, calls } = fakeRunner(() => JSON.stringify({ address: '0x1', at: [1440, 30], size: [800, 600], pid: 9, class: 'kitty', title: 'shell' }))
    const fs = files(png(800, 600))
    const shot = await screenCapture({ session: fakeSession({ server: 'wayland', compositor: 'hyprland', tools: ['hyprctl', 'grim'] }), runner, displays: async () => [], ...fs.options })!({ maxEdge: 1000 })
    expect(calls.at(-1)?.args).toEqual(['-g', '1440,30 800x600', '-s', '1', '-t', 'png', '/tmp/shot-0/window.png'])
    expect(shot).toMatchObject({ region: { x: 1440, y: 30, w: 800, h: 600 }, scale: 1, app: 'kitty', title: 'shell' })
  })

  it('on GNOME Wayland takes the whole desktop through the portal, its region the displays and its scale from the image', async () => {
    const { runner, calls } = fakeRunner(() => '{"width":1568,"height":441,"source_width":3840,"source_height":1080}')
    const fs = files(png(1568, 441))
    const displays = async () => [{ x: 0, y: 0, w: 1920, h: 1080 }, { x: 1920, y: 0, w: 1920, h: 1080 }]
    const shot = await screenCapture({ session: fakeSession({ server: 'wayland', compositor: 'gnome', tools: ['python3'] }), runner, displays, ...fs.options })!()
    expect(JSON.parse(calls[0]?.input ?? '')).toEqual({ out: '/tmp/shot-0/screen.png', max_edge: 1568 })
    expect(shot.region).toEqual({ x: 0, y: 0, w: 3840, h: 1080 })
    expect(shot.scale).toBeCloseTo(1568 / 3840)
    expect(fs.removed).toEqual(['/tmp/shot-0'])
  })

  it('removes the temporary directory when the capture fails', async () => {
    const { runner } = fakeRunner((call: Call) => (call.args[0] === '-window' ? new Error('import: unable to read X window image') : X11_ANSWERS[call.args[0] ?? ''] ?? ''))
    const fs = files(png(1, 1))
    await expect(screenCapture({ session: fakeSession({ server: 'x11', tools: ['xdotool', 'import'] }), runner, displays: async () => [], ...fs.options })!()).rejects.toThrow('unable to read')
    expect(fs.removed).toEqual(['/tmp/shot-0'])
  })

  it('has no capture where nothing can take one', () => {
    expect(screenCapture({ session: fakeSession({ server: 'x11', tools: ['xdotool'] }), runner: fakeRunner().runner, displays: async () => [] })).toBeNull()
    expect(screenCapture({ session: fakeSession({ server: 'wayland', compositor: 'kde', gio: false, tools: ['python3'] }), runner: fakeRunner().runner, displays: async () => [] })).toBeNull()
  })

  it('refuses bytes that are not a PNG', () => {
    expect(pngSize(png(3, 4))).toEqual({ width: 3, height: 4 })
    expect(() => pngSize(new Uint8Array(30))).toThrow('not a PNG')
  })
})
