// Renders the app's icons into build/ from the LocalRouter mark (apps/mobile/assets/images/icon.png, 1024 px: the
// artwork of the web favicon). Needs ImageMagick 7 (`magick`); the outputs are committed, so a build never runs this.
// `node scripts/make-icons.mjs`, after the mark changes.
//   build/icon.png   1024 px, a rounded square on the macOS icon grid (824 px body); Linux, and the window icon
//   build/icon.icns  macOS (scripts/package-mac.sh), PNG entries 16 to 1024 px
//   build/icon.ico   Windows (electron-builder), 16 to 256 px
//   build/tray.png, tray@2x.png  the Windows and Linux tray (32, 64 px; macOS shows a menu bar title)
//   build/favicon.png  64 px, inlined as the /voice page's favicon (src/voice/voice.html)
import { Buffer } from 'node:buffer'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const out = join(root, 'build')
const source = join(root, '..', 'mobile', 'assets', 'images', 'icon.png')
const tmp = mkdtempSync(join(tmpdir(), 'lr-icons-'))
const magick = (...args) => execFileSync('magick', args, { stdio: ['ignore', 'ignore', 'inherit'] })

try {
  // The mark sits mid-canvas with wide margins: crop to it, then round the corners (macOS grid: 824 px body, 185 px radius).
  const master = join(out, 'icon.png')
  magick(source, '-crop', '840x840+92+92', '+repage', '-resize', '824x824', '-alpha', 'set',
    '(', '-size', '824x824', 'xc:none', '-fill', 'white', '-draw', 'roundrectangle 0,0,823,823,185,185', ')',
    '-compose', 'DstIn', '-composite', '-compose', 'Over', '-background', 'none', '-gravity', 'center', '-extent', '1024x1024', '-strip', master)

  const png = (size) => {
    const path = join(tmp, `${size}.png`)
    magick(master, '-resize', `${size}x${size}`, '-strip', path)
    return readFileSync(path)
  }

  // ICNS: 'icns', the file length, then (type, length, PNG) entries.
  const entries = [['icp4', 16], ['icp5', 32], ['ic11', 32], ['icp6', 64], ['ic12', 64], ['ic07', 128], ['ic08', 256], ['ic13', 256], ['ic09', 512], ['ic14', 512], ['ic10', 1024]]
  const chunks = entries.map(([type, size]) => {
    const data = png(size)
    const head = Buffer.alloc(8)
    head.write(type, 0, 'ascii')
    head.writeUInt32BE(data.length + 8, 4)
    return Buffer.concat([head, data])
  })
  const header = Buffer.alloc(8)
  header.write('icns', 0, 'ascii')
  header.writeUInt32BE(8 + chunks.reduce((n, c) => n + c.length, 0), 4)
  writeFileSync(join(out, 'icon.icns'), Buffer.concat([header, ...chunks]))

  magick(master, '-define', 'icon:auto-resize=256,128,64,48,32,24,16', join(out, 'icon.ico'))
  magick(master, '-resize', '64x64', '-strip', join(out, 'favicon.png'))
  // The tray: the mark alone on transparency (a dark square is a hole in a dark taskbar).
  const mark = join(root, '..', 'mobile', 'assets', 'images', 'sidebud-mark.png')
  for (const [name, size] of [['tray.png', 32], ['tray@2x.png', 64]]) {
    magick(mark, '-trim', '+repage', '-resize', `${size}x${size}`, '-background', 'none', '-gravity', 'center', '-extent', `${size}x${size}`, '-strip', join(out, name))
  }
  process.stdout.write(`make-icons: build/icon.png, icon.icns, icon.ico, tray.png, tray@2x.png, favicon.png from ${source}\n`)
} finally {
  rmSync(tmp, { recursive: true, force: true })
}
