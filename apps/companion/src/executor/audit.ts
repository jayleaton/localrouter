import { createHash } from 'node:crypto'
import { appendFileSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

/**
 * The executor's audit log, as the box's `src/computer/audit.zig` keeps it: one JSON object a line for every task,
 * grant, approval, action, refusal and outcome, in `<data>/audit.jsonl` on this computer. The person's screen text
 * stays out of it: typed text is its length and a SHA-256 prefix, labels and titles are clipped to 80 characters. The
 * file keeps its newest half past 8 MiB. Best effort: a full disk loses lines, never a task.
 */
export const CLIP = 80
const MAX_BYTES = 8 << 20

export type TextDigest = { len: number; sha256: string }

export function textDigest(text: string): TextDigest {
  return { len: Buffer.byteLength(text, 'utf8'), sha256: createHash('sha256').update(text, 'utf8').digest('hex').slice(0, 12) }
}

/** `s` cut to 80 characters (whole code points). */
export function clip(s: string): string {
  const chars = [...s]
  return chars.length <= CLIP ? s : chars.slice(0, CLIP).join('')
}

export interface Audit {
  record(event: string, fields?: Record<string, unknown>): void
  tail(limit: number): Array<Record<string, unknown>>
}

export class FileAudit implements Audit {
  readonly path: string
  private readonly now: () => number

  constructor(path: string, now: () => number = Date.now) {
    this.path = path
    this.now = now
  }

  record(event: string, fields: Record<string, unknown> = {}) {
    const line = `${JSON.stringify({ ts: this.now(), event, ...fields })}\n`
    try {
      mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 })
      let size = 0
      try {
        size = statSync(this.path).size
      } catch {
        // No file yet.
      }
      if (size > MAX_BYTES) {
        const data = readFileSync(this.path, 'utf8')
        let keep = data.slice(Math.floor(data.length / 2))
        const nl = keep.indexOf('\n')
        if (nl >= 0) keep = keep.slice(nl + 1)
        writeFileSync(this.path, keep + line, { mode: 0o600 })
      } else appendFileSync(this.path, line, { mode: 0o600 })
    } catch {
      // Best effort.
    }
  }

  tail(limit: number) {
    let data = ''
    try {
      data = readFileSync(this.path, 'utf8')
    } catch {
      return []
    }
    return data.split('\n').filter(Boolean).slice(-limit).flatMap((line) => {
      try {
        return [JSON.parse(line) as Record<string, unknown>]
      } catch {
        return []
      }
    })
  }
}

/** An audit in memory, for tests and the headless runner's report. */
export class MemoryAudit implements Audit {
  readonly lines: Array<Record<string, unknown>> = []
  private readonly now: () => number

  constructor(now: () => number = Date.now) {
    this.now = now
  }

  record(event: string, fields: Record<string, unknown> = {}) {
    this.lines.push(JSON.parse(JSON.stringify({ ts: this.now(), event, ...fields })) as Record<string, unknown>)
  }

  tail(limit: number) {
    return this.lines.slice(-limit)
  }
}
