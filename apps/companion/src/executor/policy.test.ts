import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { afterJudge, control, screen } from './policy.ts'
import type { Action, Verdict } from './policy.ts'

// The same file the box's policy.zig tests read: one set of cases, two implementations.
const golden = JSON.parse(readFileSync(new URL('../../../../src/computer/policy_golden.json', import.meta.url), 'utf8')) as {
  control: Array<{ kind: string; verdict: Verdict }>
  screen: Array<{ action: Action; verdict: Verdict }>
  after_judge: Array<{ p: number | null; verdict: Verdict }>
}

describe('policy parity with the box (src/computer/policy_golden.json)', () => {
  it.each(golden.control)('control $kind is $verdict', ({ kind, verdict }) => {
    expect(control(kind).verdict).toBe(verdict)
  })
  it.each(golden.screen.map((c) => ({ ...c, name: `${c.action.kind} ${c.action.role ?? ''} ${JSON.stringify(c.action.label ?? c.action.key ?? c.action.direction ?? c.action.name ?? '')}` })))('$name is $verdict', ({ action, verdict }) => {
    expect(screen(action).verdict).toBe(verdict)
  })
  it.each(golden.after_judge)('judged p=$p is $verdict', ({ p, verdict }) => {
    expect(afterJudge(p).verdict).toBe(verdict)
  })
})

describe('policy edges the fixture cannot carry', () => {
  it('a NaN judgment asks', () => {
    expect(afterJudge(Number.NaN).verdict).toBe('tap')
  })
  it('an over-long label is refused rather than clipped past its consequential word', () => {
    expect(screen({ kind: 'click', role: 'button', label: `${' '.repeat(600)}Send` }).verdict).toBe('refused')
  })
  it('limits count UTF-8 bytes, as the box does', () => {
    expect(screen({ kind: 'click', role: 'button', label: 'é'.repeat(201) }).verdict).toBe('refused') // 402 bytes
    expect(screen({ kind: 'type', role: 'textbox', label: 'Search', text: 'é'.repeat(2001) }).verdict).toBe('refused')
  })
  it('every approval and refusal says why', () => {
    for (const c of golden.screen) {
      const d = screen(c.action)
      if (d.verdict === 'tap' || d.verdict === 'refused') expect(d.why.length).toBeGreaterThan(0)
    }
  })
})

describe('a drag (Windows run, 2026-10-27: Paint)', () => {
  it('is the person’s to allow, as any real-pointer step: the standing grant allows it, "Ask before each action" asks', () => {
    expect(screen({ kind: 'drag', role: 'other', label: 'canvas (it draws itself)', why: 'to a stroke inside it' })).toEqual({ verdict: 'tap', why: 'it drags with your mouse: to a stroke inside it' })
  })
})
