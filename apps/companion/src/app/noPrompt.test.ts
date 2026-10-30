import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

/**
 * window.prompt() does not exist in Electron: in the app it throws "prompt() is not supported" (Windows run,
 * 2026-10-27: "Could not reach the models"). No page the app shows may call it; they ask in the page instead.
 */
const here = import.meta.dirname
const pages = [
  ...readdirSync(join(here, 'ui')).filter((f) => /\.(js|html)$/.test(f)).map((f) => join(here, 'ui', f)),
  join(here, '../../../../src/voice/voice.html'),
]

describe('no window.prompt() in any page the app shows', () => {
  it.each(pages)('%s', (file) => {
    const calls = readFileSync(file, 'utf8').split('\n').filter((line) => /(^|[^.\w$])(window\.)?prompt\s*\(/.test(line) && !/^\s*(\/\/|\*)/.test(line))
    expect(calls).toEqual([])
  })
})
