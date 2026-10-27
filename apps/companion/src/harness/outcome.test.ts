import { describe, expect, it } from 'vitest'
import { CONTROL_OFF } from '../executor/broker.ts'
import { fakeBroker, MemoryStanding } from '../executor/testing.ts'
import { Registry } from './registry.ts'
import { HARNESS_INSTRUCTIONS, VOICE_CALLER } from './runtime.ts'
import { computerTools, tellUser } from './tools.ts'

/**
 * Cody's Windows run (989e6b1): the Notepad approval expired (audit: approval.end expired → deny → task.end blocked
 * "Nobody approved it in time."), nothing was typed, and the reply said the text "has been typed". What the model is
 * given is the outcome as it happened, led by whether it did, and the sentence to say.
 */
describe('a computer task’s result tells the reply what really happened', () => {
  it('blocked by an approval nobody gave: done is false and the sentence says it did not happen, and why', async () => {
    // Send needs an approval; nobody answers within the (shortened) window.
    const { broker, mac } = fakeBroker(['c4', 'c3', 'done'], { limits: { approvalTimeoutMs: 80 }, presenceMs: 80 })
    const registry = new Registry().add(...computerTools(broker, VOICE_CALLER, 10_000))
    const ran = await registry.run('computer_use', JSON.stringify({ goal: 'send the Groceries note' }), { callId: 'call_1', utterance: 'send the groceries note', signal: new AbortController().signal })
    const out = JSON.parse(ran.output) as { done: boolean; tell_user: string; status: string }
    expect(mac.desktop.sent).toEqual([])
    expect(out.status).toBe('blocked')
    expect(out.done).toBe(false)
    expect(out.tell_user).toMatch(/^It did not do it: Nobody approved it in time\./)
    // What the model reads first is whether it happened.
    expect(Object.keys(out).slice(0, 2)).toEqual(['done', 'tell_user'])
    expect(HARNESS_INSTRUCTIONS).toMatch(/When `done` is false the action did not happen: never say or imply that it did/)
  })

  it('only done is a success; each other ending says it did not happen', () => {
    const typed = ['clicked the textbox “Text editor”']
    expect(tellUser('type hello into Notepad', { status: 'done', reason: '', steps: ['typed the text into the textbox'] })).toBe('Done: typed the text into the textbox.')
    expect(tellUser('type hello into Notepad', { status: 'stuck', reason: 'It did not finish within 15 steps.', steps: typed })).toBe('It did not manage to type hello into Notepad: It did not finish within 15 steps. What it did before that: clicked the textbox “Text editor”. Nothing is waiting for you on screen.')
    expect(tellUser('x', { status: 'stopped', reason: 'You stopped it.', steps: [] })).toBe('It stopped before finishing: You stopped it. Nothing is waiting for you on screen.')
    expect(tellUser('x', { status: 'failed', reason: 'boom', steps: [] })).toMatch(/^It did not manage to x: boom/)
    for (const status of ['blocked', 'stuck', 'stopped', 'failed', 'unavailable']) expect(tellUser('type hello', { status, reason: 'r', steps: [] })).not.toMatch(/^Done/)
    // An ended task leaves no card behind, and the sentence says so (Mac run, 2026-10-26: "an approval dialog is waiting").
    for (const status of ['blocked', 'stuck', 'stopped', 'failed', 'unavailable']) expect(tellUser('type hello', { status, reason: 'Nobody approved it in time.', steps: [] })).toMatch(/Nothing is waiting for you on screen\.$/)
  })

  it('computer control not given yet: refused at once, the reply says it is off and where to turn it on', async () => {
    const { broker } = fakeBroker(['c4', 'c3', 'done'], { standing: new MemoryStanding() })
    const registry = new Registry().add(...computerTools(broker, VOICE_CALLER, 10_000))
    const ran = await registry.run('computer_use', JSON.stringify({ goal: 'send the Groceries note' }), { callId: 'call_2', utterance: 'send the groceries note', signal: new AbortController().signal })
    const out = JSON.parse(ran.output) as { done: boolean; tell_user: string; status: string }
    expect(out).toMatchObject({ done: false, status: 'control_off' })
    expect(out.tell_user).toBe(`I could not do that on your computer: ${CONTROL_OFF} Nothing is waiting for you on screen.`)
  })
})
