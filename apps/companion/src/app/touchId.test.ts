import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { TouchIdHelper, touchIdUnavailable } from './touchId.ts'

/**
 * localrouter-presence stood in for by a Node script with the same command line: `check` prints its answer, `verify`
 * reads the reason and waits like an open Touch ID prompt. The script records the reason and how it ended.
 */
const dir = mkdtempSync(join(tmpdir(), 'lr-presence-'))
afterAll(() => rmSync(dir, { recursive: true, force: true }))
const script = join(dir, 'fake-presence.mjs')
writeFileSync(
  script,
  `import { appendFileSync } from 'node:fs'
const [log, mode, op] = process.argv.slice(2)
if (op === 'check') { process.stdout.write(mode === 'enrolled' ? '{"biometrics":true,"code":0}\\n' : '{"biometrics":false,"code":-7}\\n'); process.exit(0) }
process.stdin.setEncoding('utf8')
process.stdin.once('data', (d) => {
  appendFileSync(log, 'reason ' + d)
  if (mode === 'yes') process.exit(0)
  if (mode === 'no') process.exit(1)
})
process.on('SIGTERM', () => { appendFileSync(log, 'dismissed\\n'); process.exit(3) })
setInterval(() => undefined, 1000)
`,
)
const helper = (mode: string) => {
  const log = join(dir, `${mode}-${Math.random().toString(36).slice(2)}.log`)
  writeFileSync(log, '')
  return { touch: new TouchIdHelper({ file: process.execPath, args: [script, log, mode] }), log: () => readFileSync(log, 'utf8') }
}

describe('Touch ID through localrouter-presence (Mac run, 2026-10-26: a plain sheet, and a sheet that stayed after ⌘⇧⎋)', () => {
  it('says whether Touch ID can be used now, and why not', async () => {
    expect(await helper('enrolled').touch.check()).toEqual({ available: true, why: '' })
    expect(await helper('none').touch.check()).toEqual({ available: false, why: 'Touch ID is not available: no fingerprint is enrolled' })
    expect(await new TouchIdHelper({ file: join(dir, 'missing') }).check()).toEqual({ available: false, why: 'Touch ID could not be checked on this Mac' })
    expect(touchIdUnavailable(-6)).toMatch(/lid is closed/)
  })

  it('confirms only on the helper’s yes; the reason goes on stdin, one line', async () => {
    const yes = helper('yes')
    expect(await yes.touch.verify('let Jarvis type “hi”\ninto Notes')).toBe(true)
    expect(yes.log()).toBe('reason let Jarvis type “hi” into Notes\n')
    expect(await helper('no').touch.verify('x')).toBe(false)
  })

  it('an approval that ends another way dismisses the open prompt: the helper is told and goes, and it is a no', async () => {
    const open = helper('wait')
    const stop = new AbortController()
    const answer = open.touch.verify('let Jarvis send it', stop.signal)
    await new Promise((resolve) => setTimeout(resolve, 300))
    stop.abort('cancelled')
    expect(await answer).toBe(false)
    await new Promise((resolve) => setTimeout(resolve, 300))
    // Windows has no SIGTERM handler to run (the process is ended outright); elsewhere the helper sees it.
    if (process.platform !== 'win32') expect(open.log()).toContain('dismissed')
  })
})
