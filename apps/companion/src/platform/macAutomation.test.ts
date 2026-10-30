import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import { MacAutomation, NOT_CHECKED } from './macAutomation.ts'
import { bundledAutomation, createMacPlatform } from './mac.ts'

/**
 * localrouter-presence stood in for by a Node script with the same command line: `automation ID` and
 * `automation-ask ID` print the status the test set for that id, and every call is logged.
 */
const dir = mkdtempSync(join(tmpdir(), 'lr-automation-'))
afterAll(() => rmSync(dir, { recursive: true, force: true }))
const script = join(dir, 'fake-presence.mjs')
writeFileSync(
  script,
  `import { appendFileSync } from 'node:fs'
const [log, answers, op, id] = process.argv.slice(2)
appendFileSync(log, op + ' ' + id + '\\n')
const status = JSON.parse(answers)[id]
process.stdout.write(status === 'junk' ? 'not json\\n' : JSON.stringify({ status }) + '\\n')
`,
)
const helper = (answers: Record<string, number | string>) => {
  const log = join(dir, `${Math.random().toString(36).slice(2)}.log`)
  writeFileSync(log, '')
  return { automation: new MacAutomation({ file: process.execPath, args: [script, log, JSON.stringify(answers)] }), log: () => readFileSync(log, 'utf8').trim().split('\n').filter(Boolean) }
}

describe('macOS Automation through localrouter-presence: checked without asking, asked only from setup', () => {
  it('allows only what macOS already granted (status 0), and keeps a yes', async () => {
    const { automation, log } = helper({ 'com.apple.systemevents': 0, 'com.google.Chrome': -1744, 'com.apple.Music': -1743, 'com.spotify.client': -600, 'com.example.junk': 'junk' })
    expect(await automation.allowed('com.apple.systemevents')).toBe(true)
    expect(await automation.allowed('com.apple.systemevents')).toBe(true)
    expect(await automation.allowed('com.google.Chrome')).toBe(false)
    expect(await automation.allowed('com.apple.Music')).toBe(false)
    expect(await automation.allowed('com.spotify.client')).toBe(false)
    expect(await automation.allowed('com.example.junk')).toBe(false)
    // The yes is cached; a no is asked again next time (the person may have granted it since). Never the asking form.
    expect(log()).toEqual(['automation com.apple.systemevents', 'automation com.google.Chrome', 'automation com.apple.Music', 'automation com.spotify.client', 'automation com.example.junk'])
  })

  it('asks with automation-ask and returns macOS’s status; a bad id or a missing helper is never run and never allowed', async () => {
    const { automation, log } = helper({ 'com.google.Chrome': 0 })
    expect(await automation.ask('com.google.Chrome')).toBe(0)
    expect(await automation.allowed('com.google.Chrome')).toBe(true)
    expect(await automation.ask('-flag')).toBe(NOT_CHECKED)
    expect(log()).toEqual(['automation-ask com.google.Chrome'])
    const none = new MacAutomation(null)
    expect(await none.allowed('com.apple.systemevents')).toBe(false)
    expect(await none.ask('com.apple.systemevents')).toBe(NOT_CHECKED)
    expect(await new MacAutomation({ file: join(dir, 'missing') }).allowed('com.apple.systemevents')).toBe(false)
  })

  it('the Mac platform asks each target once, one at a time, and without the packaged helper allows no Apple Event', async () => {
    const { automation, log } = helper({ 'com.apple.systemevents': 0, 'com.google.Chrome': -600, 'com.spotify.client': -600, 'com.apple.Music': -600 })
    const platform = createMacPlatform({ automation })
    expect(await platform.automation?.ask()).toEqual({ 'com.apple.systemevents': 0, 'com.google.Chrome': -600, 'com.spotify.client': -600, 'com.apple.Music': -600 })
    expect(log()).toEqual(['automation-ask com.apple.systemevents', 'automation-ask com.google.Chrome', 'automation-ask com.spotify.client', 'automation-ask com.apple.Music'])
    // No check configured (the standalone helper's behaviour): no `automation` to ask.
    expect(createMacPlatform({ automation: null }).automation).toBeUndefined()
    // A dev run has no Resources: nothing is allowed rather than macOS asking mid-task.
    expect(await bundledAutomation(undefined).allowed('com.apple.systemevents')).toBe(false)
    const empty = mkdtempSync(join(dir, 'resources-'))
    expect(await bundledAutomation(empty).allowed('com.apple.systemevents')).toBe(false)
  })
})
