// Bundles the companion app into dist/: the Electron main and preload, the executor's MCP CLI, the static UI and the
// voice page (the repo's single web UI, src/voice/voice.html, with the app's shim injected). `npm run build`.
import { build } from 'esbuild'
import { cp, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const root = dirname(dirname(fileURLToPath(import.meta.url)))
const dist = join(root, 'dist')
const voiceSource = join(root, '..', '..', 'src', 'voice', 'voice.html')
const shimTag = '<script src="voice-shim.js"></script>'

const fail = (message) => {
  process.stderr.write(`build: ${message}\n`)
  process.exit(1)
}
const exists = (path) => stat(path).then(() => true, () => false)

const pkg = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'))

// The cjs bundles have no import.meta: `import.meta.url` becomes the bundle file's own URL.
const shim = "'use strict';const __import_meta_url = require('node:url').pathToFileURL(__filename).href;"
const common = {
  absWorkingDir: root,
  bundle: true,
  platform: 'node',
  format: 'cjs',
  target: 'node22',
  external: ['electron'],
  sourcemap: false,
  logLevel: 'warning',
  define: { 'import.meta.url': '__import_meta_url' },
  banner: { js: shim },
}

await rm(dist, { recursive: true, force: true })
await mkdir(dist, { recursive: true })

await Promise.all([
  build({ ...common, entryPoints: ['src/app/main.ts'], outfile: join(dist, 'main.cjs') }),
  // The preload runs sandboxed: no Node modules, so no import.meta shim (it needs none).
  build({ ...common, define: {}, banner: { js: "'use strict';" }, entryPoints: ['src/app/preload.ts'], outfile: join(dist, 'preload.cjs') }),
  build({ ...common, entryPoints: ['src/executor/mcpMain.ts'], outfile: join(dist, 'mcp.cjs') }),
])

// Exactly one hashbang on the MCP CLI, first line (esbuild keeps the entry's own, if it has one).
const mcpPath = join(dist, 'mcp.cjs')
const mcp = (await readFile(mcpPath, 'utf8')).replace(/^(#![^\n]*\n)+/, '')
await writeFile(mcpPath, `#!/usr/bin/env node\n${mcp}`, { mode: 0o755 })

const ui = join(root, 'src', 'app', 'ui')
if (!(await exists(join(ui, 'voice-shim.js')))) fail(`missing ${join(ui, 'voice-shim.js')}`)
if (await exists(join(ui, 'voice.html'))) fail('src/app/ui/voice.html would shadow the voice page built from src/voice/voice.html')
await cp(ui, join(dist, 'ui'), { recursive: true })

const voice = await readFile(voiceSource, 'utf8')
const at = voice.search(/<script\b/i)
if (at < 0) fail(`no <script> tag in ${voiceSource}: refusing to ship the voice page without its shim`)
const patched = `${voice.slice(0, at)}${shimTag}\n${voice.slice(at)}`
if (!patched.includes(shimTag)) fail('voice shim not injected')
await writeFile(join(dist, 'ui', 'voice.html'), patched)

await writeFile(
  join(dist, 'package.json'),
  `${JSON.stringify({ name: 'localrouter-companion', productName: 'LocalRouter', version: pkg.version, description: pkg.description, author: { name: 'LocalRouter', email: 'noreply@localrouter.invalid' }, homepage: 'https://github.com/jayleaton/localrouter', license: pkg.license, main: 'main.cjs' }, null, 2)}\n`,
)

process.stdout.write(`build: dist/ (main.cjs, preload.cjs, mcp.cjs, ui/, package.json) for ${pkg.version}\n`)
