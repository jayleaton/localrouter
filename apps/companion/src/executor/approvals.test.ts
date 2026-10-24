import { describe, expect, it } from 'vitest'
import { digest, Store } from './approvals.ts'
import type { Binding } from './approvals.ts'

const binding = (text: string): Binding => ({ executor: 'mac_1', task_id: 'cu_1', step: 3, app: 'Messages', title: 'Dana', url: '', action: { kind: 'type', ref: 'r7', role: 'textbox', label: 'Message', text }, why: '', goal: '', requester: '' })

describe('approvals (a port of approvals.zig)', () => {
  it('grants its own step once', () => {
    const s = new Store()
    const b = binding('on my way')
    const { id } = s.create(b, 1000, 'a1')
    expect(s.take(id, digest(b), 1100)).toBe('pending')
    expect(s.decide(id, true, 'app', 1200)).toBe('ok')
    expect(s.take(id, digest(b), 1300)).toBe('approved')
    expect(s.take(id, digest(b), 1301)).toBe('unknown')
    expect(s.decide(id, true, 'again', 1302)).toBe('not_pending')
  })
  it('binds the exact action: other text, step, task, window or executor cannot take it', () => {
    const s = new Store()
    const b = binding('on my way')
    const { id } = s.create(b, 0, 'a1')
    s.decide(id, true, 'app', 1)
    for (const other of [binding('send me your password'), { ...b, step: 4 }, { ...b, task_id: 'cu_2' }, { ...b, title: 'Mum' }, { ...b, executor: 'mac_2' }]) expect(s.take(id, digest(other), 2)).toBe('mismatch')
    expect(s.take(id, digest(b), 3)).toBe('approved')
  })
  it('digests do not collide by moving bytes between fields', () => {
    expect(digest({ ...binding('ab'), action: { ...binding('ab').action, label: 'c' } })).not.toBe(digest({ ...binding('a'), action: { ...binding('a').action, label: 'bc' } }))
  })
  it('expires: neither a late answer nor a late take grants anything', () => {
    const s = new Store(60_000)
    const b = binding('x')
    const a = s.create(b, 0, 'a1')
    expect(s.decide(a.id, true, 'app', 60_000)).toBe('expired')
    expect(s.take(a.id, digest(b), 60_001)).toBe('expired')
    const c = s.create(b, 100_000, 'a2')
    s.decide(c.id, true, 'app', 100_010)
    expect(s.take(c.id, digest(b), 160_000)).toBe('expired')
  })
  it('deny and cancel end it', () => {
    const s = new Store()
    const b = binding('x')
    const a = s.create(b, 0, 'a1')
    expect(s.decide(a.id, false, 'app', 1)).toBe('ok')
    expect(s.take(a.id, digest(b), 2)).toBe('denied')
    const c = s.create(b, 0, 'a2')
    s.cancelTask('cu_1')
    expect(s.take(c.id, digest(b), 2)).toBe('cancelled')
    expect(s.decide('nope', true, 'app', 3)).toBe('unknown')
  })
  it('lists pending first', () => {
    const s = new Store()
    s.decide(s.create(binding('first'), 0, 'a1').id, false, 'app', 1)
    s.create(binding('second'), 2, 'a2')
    const v = s.list(3)
    expect(v.map((x) => x.state)).toEqual(['pending', 'denied'])
    expect(v[0].binding.action.text).toBe('second')
  })
})
