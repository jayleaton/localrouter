import { randomBytes } from 'node:crypto'
import { appendFileSync, mkdirSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

/**
 * The harness's durable history on this computer: one JSON-lines file per conversation, appended as things happen
 * (what the person said, what the assistant said, every tool call with its complete arguments and its outcome). The
 * canonical record of a conversation is here, not on the Spark: a new realtime session is reseeded from it. `export`
 * writes the neutral form another harness can import when the person switches (no tokens, no live approvals).
 */
export type Entry =
  | { type: 'session'; mode: string; tools: string[]; model?: string }
  | { type: 'user'; text: string; via: 'voice' | 'text' }
  | { type: 'assistant'; text: string; item_id?: string }
  | { type: 'tool_call'; call_id: string; name: string; arguments: string }
  | { type: 'tool_result'; call_id: string; ok: boolean; output: string; ms: number }
  | { type: 'tool_cancelled'; call_id: string; reason: string }
  | { type: 'note'; text: string }

export type Line = Entry & { ts: number }

export type ExportedMessage =
  | { role: 'user' | 'assistant'; text: string; at: number }
  | { role: 'tool_call'; call_id: string; name: string; arguments: string; at: number }
  | { role: 'tool_result'; call_id: string; ok: boolean; output: string; at: number }

export const EXPORT_FORMAT = 'localrouter-conversation/1'
const MAX_OUTPUT = 8000

export class Journal {
  readonly dir: string
  private readonly now: () => number

  constructor(dir: string, now: () => number = Date.now) {
    this.dir = dir
    this.now = now
  }

  /** A new conversation's id. */
  create(): string {
    const id = `conv_${new Date(this.now()).toISOString().slice(0, 10).replace(/-/g, '')}_${randomBytes(4).toString('hex')}`
    this.append(id, { type: 'note', text: 'conversation started' })
    return id
  }

  append(id: string, entry: Entry) {
    if (!/^conv_[\w-]+$/.test(id)) throw new Error('not a conversation id')
    const e = entry.type === 'tool_result' && entry.output.length > MAX_OUTPUT ? { ...entry, output: `${entry.output.slice(0, MAX_OUTPUT)}…` } : entry
    mkdirSync(this.dir, { recursive: true, mode: 0o700 })
    appendFileSync(join(this.dir, `${id}.jsonl`), `${JSON.stringify({ ts: this.now(), ...e })}\n`, { mode: 0o600 })
  }

  read(id: string): Line[] {
    let text = ''
    try {
      text = readFileSync(join(this.dir, `${id}.jsonl`), 'utf8')
    } catch {
      return []
    }
    return text.split('\n').filter(Boolean).flatMap((l) => {
      try {
        return [JSON.parse(l) as Line]
      } catch {
        return [] // a line cut by a crash
      }
    })
  }

  list(): string[] {
    try {
      return readdirSync(this.dir).filter((f) => f.endsWith('.jsonl')).map((f) => f.slice(0, -6)).sort().reverse()
    } catch {
      return []
    }
  }

  /** The neutral export: messages and tool receipts, in order. */
  export(id: string) {
    const lines = this.read(id)
    return {
      format: EXPORT_FORMAT,
      id,
      exported: new Date(this.now()).toISOString(),
      messages: lines.flatMap((l): ExportedMessage[] => {
        switch (l.type) {
          case 'user':
          case 'assistant':
            return [{ role: l.type, text: l.text, at: l.ts }]
          case 'tool_call':
            return [{ role: 'tool_call', call_id: l.call_id, name: l.name, arguments: l.arguments, at: l.ts }]
          case 'tool_result':
            return [{ role: 'tool_result', call_id: l.call_id, ok: l.ok, output: l.output, at: l.ts }]
          default:
            return []
        }
      }),
    }
  }
}
