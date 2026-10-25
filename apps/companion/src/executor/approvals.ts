import { createHash } from 'node:crypto'
import type { Action } from './policy.ts'

/**
 * Approvals: a person's OK on this computer for one consequential action. A port of the box's
 * `src/computer/approvals.zig` with the same binding: the executor installation, the task, the step, what was on
 * screen and the exact action, by a SHA-256 digest of all of it. It expires (60 s), is used once, and only the step it
 * was made for can take it. Kept in memory: quitting the app forgets every pending approval, which then never runs.
 * Time is passed in, so the rules are tested without a clock.
 */
export type Binding = {
  /** This executor installation: an approval made here is never valid on another computer. */
  executor: string
  task_id: string
  step: number
  app: string
  title: string
  url: string
  action: Action
  /** The policy's reason, for the card. */
  why: string
  /** What the person asked for. */
  goal: string
  /** Who started the task: "Jarvis (voice)", "MCP: opencode", "LocalRouter box (legacy)". */
  requester: string
}

export type State = 'pending' | 'approved' | 'denied' | 'expired' | 'used' | 'cancelled'

const ACTION_FIELDS = ['kind', 'ref', 'role', 'label', 'text', 'key', 'field_role', 'field_label', 'direction', 'what', 'name'] as const

/** SHA-256 over the step and every field, each length-prefixed, so no two different bindings share a digest. */
export function digest(b: Binding): string {
  const h = createHash('sha256')
  const step = Buffer.alloc(4)
  step.writeUInt32LE(b.step >>> 0)
  h.update(step)
  const fields = [b.executor, b.task_id, b.app, b.title, b.url, ...ACTION_FIELDS.map((f) => b.action[f] ?? '')]
  for (const f of fields) {
    const data = Buffer.from(f, 'utf8')
    const n = Buffer.alloc(8)
    n.writeBigUInt64LE(BigInt(data.length))
    h.update(n)
    h.update(data)
  }
  return h.digest('hex')
}

export type Approval = {
  id: string
  binding: Binding
  digest: string
  created_ms: number
  expires_ms: number
  state: State
  decided_by: string
}

export const DEFAULT_TTL_MS = 60_000
/** Finished approvals are kept this long for the card's last state, then dropped. */
const KEEP_MS = 10 * 60_000
const MAX_ITEMS = 256

export type Decided = 'ok' | 'unknown' | 'not_pending' | 'expired'
export type Taken = 'approved' | 'denied' | 'expired' | 'pending' | 'unknown' | 'mismatch' | 'cancelled'

export class Store {
  readonly ttlMs: number
  private items: Approval[] = []

  constructor(ttlMs = DEFAULT_TTL_MS) {
    this.ttlMs = ttlMs
  }

  /** A pending approval for `b`; `id` is its random id. */
  create(b: Binding, nowMs: number, id: string): Approval {
    this.prune(nowMs)
    if (this.items.length >= MAX_ITEMS) throw new Error('too many approvals are waiting')
    const x: Approval = { id, binding: structuredClone(b), digest: digest(b), created_ms: nowMs, expires_ms: nowMs + this.ttlMs, state: 'pending', decided_by: '' }
    this.items.push(x)
    return x
  }

  /** The person's answer. Only a pending, unexpired approval can be answered, once. */
  decide(id: string, approve: boolean, by: string, nowMs: number): Decided {
    const x = this.find(id)
    if (!x) return 'unknown'
    expire(x, nowMs)
    if (x.state === 'expired') return 'expired'
    if (x.state !== 'pending') return 'not_pending'
    x.state = approve ? 'approved' : 'denied'
    x.decided_by = by
    return 'ok'
  }

  /**
   * The person pressed Approve and the OS is checking it is them (Windows Hello, Touch ID): a pending approval stays
   * open until `untilMs` so that check's own time does not run the card out. Never shortens it; nothing else changes.
   */
  hold(id: string, untilMs: number, nowMs: number): boolean {
    const x = this.find(id)
    if (!x) return false
    expire(x, nowMs)
    if (x.state !== 'pending') return false
    x.expires_ms = Math.max(x.expires_ms, untilMs)
    return true
  }

  /** The waiting step's check: `approved` once, for the same binding only (then the approval is used up). */
  take(id: string, d: string, nowMs: number): Taken {
    const x = this.find(id)
    if (!x) return 'unknown'
    if (x.digest !== d) return 'mismatch'
    expire(x, nowMs)
    switch (x.state) {
      case 'approved':
        x.state = 'used'
        return 'approved'
      case 'used':
        return 'unknown' // used up: a second take is not a second approval
      default:
        return x.state
    }
  }

  get(id: string, nowMs: number): Approval | null {
    const x = this.find(id)
    if (x) expire(x, nowMs)
    return x ? structuredClone(x) : null
  }

  /** The task ended or was stopped: its pending approvals can no longer be granted. */
  cancelTask(taskId: string) {
    for (const x of this.items) if (x.state === 'pending' && x.binding.task_id === taskId) x.state = 'cancelled'
  }

  /** Every approval still pending, then the ones that ended in the last ten minutes, newest first. */
  list(nowMs: number): Approval[] {
    this.prune(nowMs)
    const out = [...this.items].reverse().map((x) => {
      expire(x, nowMs)
      return structuredClone(x)
    })
    return out.sort((l, r) => Number(r.state === 'pending') - Number(l.state === 'pending'))
  }

  private find(id: string) {
    return this.items.find((x) => x.id === id) ?? null
  }

  private prune(nowMs: number) {
    this.items = this.items.filter((x) => {
      expire(x, nowMs)
      return x.state === 'pending' || nowMs - x.created_ms <= KEEP_MS
    })
  }
}

function expire(x: Approval, nowMs: number) {
  if ((x.state === 'pending' || x.state === 'approved') && nowMs >= x.expires_ms) x.state = 'expired'
}
