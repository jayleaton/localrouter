import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { TOOLS } from '../executor/mcp.ts'
import { BACKGROUND_NOTE } from '../executor/broker.ts'
import { fakeBroker } from '../executor/testing.ts'
import { HARNESS_INSTRUCTIONS } from './runtime.ts'
import { computerTools } from './tools.ts'

/**
 * Every text a model reads about computer use (Mac run, 2026-10-26: Jarvis declined, "it runs in the foreground using
 * your real pointer"). It runs in the background as a virtual mouse and the computer picks how; nothing approves each step.
 */
const STALE = [/foreground/i, /real (pointer|mouse)/i, /your (pointer|mouse)/i, /user['’]s pointer/i, /tap approve/i, /wait(s)? for the (user|person) to approve/i, /asks? the (user|person) (on screen|on this computer) first/i, /approve (it|them) on/i]

const repo = join(import.meta.dirname, '../../../..')
/** The server's own tool definitions, read from its source (the Zig MCP tools and the voice chat tools). */
const zig = (file: string, from: string, to: string) => {
  const text = readFileSync(join(repo, file), 'utf8')
  return text.slice(text.indexOf(from), text.indexOf(to, text.indexOf(from)))
}

describe('what models are told about computer use', () => {
  const caller = { id: 'app', name: 'LocalRouter', kind: 'app' as const }
  const texts: Record<string, string> = {
    'harness computer_use': computerTools(fakeBroker([]).broker, caller).find((t) => t.name === 'computer_use')!.description,
    'harness computer_control': computerTools(fakeBroker([]).broker, caller).find((t) => t.name === 'computer_control')!.description,
    'executor MCP computer_use': TOOLS.find((t) => t.name === 'computer_use')!.description,
    'executor MCP computer_control': TOOLS.find((t) => t.name === 'computer_control')!.description,
    'harness instructions': HARNESS_INSTRUCTIONS,
    'server MCP tools (src/api/mcp.zig)': zig('src/api/mcp.zig', '.name = "computer_control"', '.name = "computer_status"'),
    'server voice tools (src/voice/respond.zig)': zig('src/voice/respond.zig', 'const computer_tools =', '\n;'),
  }

  it.each(Object.entries(texts))('%s: no foreground, real-pointer or per-step approval claims', (_name, text) => {
    expect(text.length).toBeGreaterThan(40)
    for (const stale of STALE) expect(text, String(stale)).not.toMatch(stale)
  })

  it('every computer_use description says it runs in the background as a virtual mouse, chosen on the computer', () => {
    for (const name of ['harness computer_use', 'executor MCP computer_use', 'harness instructions']) expect(texts[name]).toContain(BACKGROUND_NOTE)
    for (const name of ['server MCP tools (src/api/mcp.zig)', 'server voice tools (src/voice/respond.zig)']) expect(texts[name]).toContain(BACKGROUND_NOTE.replace(/’/g, "'"))
    expect(BACKGROUND_NOTE).toMatch(/background as a virtual mouse/)
    expect(BACKGROUND_NOTE).toMatch(/never decline/)
  })
})
