import { describe, expect, it } from 'vitest'
import type { Card } from '../executor/broker.ts'
import { fakeBroker } from '../executor/testing.ts'
import { assistantNameProblem, assistantOf, DEFAULT_ASSISTANT, tidyAssistantName, voiceHint } from './assistant.ts'
import { Registry } from './registry.ts'
import { VOICE_CALLER, voiceCaller } from './runtime.ts'
import { Spark } from './spark.ts'
import { computerTools } from './tools.ts'

/** A box that answers `/v1/assistant` with `status` and `body`, and records what it was asked. */
function box(status: number, body: unknown) {
  const asked: Array<{ method: string; url: string; auth: string; body: string }> = []
  const f = (async (url: string, init?: RequestInit) => {
    asked.push({ method: init?.method ?? 'GET', url, auth: new Headers(init?.headers).get('authorization') ?? '', body: typeof init?.body === 'string' ? init.body : '' })
    return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
  }) as typeof fetch
  return { spark: new Spark('https://box.ts.net:10443', 'device-token', f), asked }
}

const FRIDAY = { object: 'assistant', name: 'Friday', default_name: 'Jarvis', custom: true, voice: { model: 'fishaudio/drama-3', voice: 'default' } }

describe('the assistant’s name, kept on the Spark', () => {
  it('is read with the device token, and is null on a box from before the setting', async () => {
    const b = box(200, FRIDAY)
    expect(await b.spark.assistant()).toEqual({ name: 'Friday', default_name: 'Jarvis', custom: true, voice: { model: 'fishaudio/drama-3', voice: 'default' } })
    expect(b.asked).toEqual([{ method: 'GET', url: 'https://box.ts.net:10443/v1/assistant', auth: 'Bearer device-token', body: '' }])
    expect(await box(404, { error: { message: 'not found' } }).spark.assistant()).toBeNull()
    expect(await box(403, { error: { message: 'forbidden' } }).spark.assistant()).toBeNull()
    await expect(box(500, { error: { message: 'boom' } }).spark.assistant()).rejects.toThrow('boom')
  })

  it('is renamed with PATCH; the box’s refusal comes back in its words', async () => {
    const b = box(200, FRIDAY)
    expect((await b.spark.renameAssistant('Friday')).name).toBe('Friday')
    expect(b.asked[0]).toMatchObject({ method: 'PATCH', url: 'https://box.ts.net:10443/v1/assistant', auth: 'Bearer device-token', body: '{"name":"Friday"}' })
    await expect(box(403, { error: { message: 'this device needs the manage scope' } }).spark.renameAssistant('Friday')).rejects.toThrow('this device needs the manage scope')
  })

  it('reads a partial or broken object as the default, voice as a hint only', () => {
    expect(DEFAULT_ASSISTANT).toBe('Jarvis')
    expect(assistantOf({})).toEqual({ name: 'Jarvis', default_name: 'Jarvis', custom: false, voice: null })
    expect(assistantOf(null).name).toBe('Jarvis')
    expect(assistantOf({ name: ' ', voice: 'x' })).toMatchObject({ name: 'Jarvis', voice: null })
    expect(voiceHint(assistantOf(FRIDAY).voice)).toBe('Speaks as default (fishaudio/drama-3)')
    expect(voiceHint(null)).toBe('')
  })

  it('checks a name as the box does: trimmed, spaces collapsed, 1–32 characters, letters of any script', () => {
    expect(tidyAssistantName('  Mary   Jane ')).toBe('Mary Jane')
    for (const ok of ['Friday', "O'Brien", 'Dr. Who', 'R2-D2', 'Zoë', 'ジャービス', '']) expect(assistantNameProblem(ok), ok).toBeNull()
    expect(assistantNameProblem('x'.repeat(33))).toMatch(/at most 32/)
    expect(assistantNameProblem('<script>')).toMatch(/letters, digits/)
    expect(assistantNameProblem("'.-")).toMatch(/at least one letter or digit/)
  })

  it('a renamed assistant is who the task-start notice and an approval name', async () => {
    const { broker, mac } = fakeBroker(['c4', 'c3', 'done'], { limits: { approvalTimeoutMs: 80 }, presenceMs: 80 })
    broker.setAssistant('Friday')
    const card = new Promise<Card>((resolve) => broker.on((e) => e.type === 'approval' && resolve(e.card)))
    const registry = new Registry().add(...computerTools(broker, voiceCaller(() => 'Friday'), 10_000))
    const ran = registry.run('computer_use', JSON.stringify({ goal: 'send the Groceries note' }), { callId: 'call_1', utterance: 'send the groceries note', signal: new AbortController().signal })
    expect((await card).requester).toBe('Friday (voice)')
    await ran
    expect(mac.controls.notices[0]?.text).toMatch(/^Friday is using Test Mac: send the Groceries note\./)
    expect(VOICE_CALLER.name).toBe('Jarvis (voice)')
  })
})
