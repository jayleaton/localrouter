import { describe, expect, it } from 'vitest'
import { assistantName, DEFAULT_ASSISTANT, startNotice } from './assistant.ts'

/** A box that answers `GET /v1/assistant` with `status` and `body`, and records what it was asked. */
function box(status: number, body: unknown) {
  const asked: Array<{ url: string; auth: string }> = []
  const f = (async (url: string, init?: RequestInit) => {
    asked.push({ url, auth: new Headers(init?.headers).get('authorization') ?? '' })
    return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
  }) as typeof fetch
  return { f, asked }
}

describe('the assistant’s name, from the box', () => {
  it('reads GET /v1/assistant with the device token', async () => {
    const b = box(200, { object: 'assistant', name: 'Friday', default_name: 'Jarvis', custom: true, voice: null })
    expect(await assistantName('https://box.example:10443/', 'secret-token', b.f)).toBe('Friday')
    expect(b.asked).toEqual([{ url: 'https://box.example:10443/v1/assistant', auth: 'Bearer secret-token' }])
  })

  it('is Jarvis on a box from before the setting, one that refuses, or one that does not answer', async () => {
    expect(DEFAULT_ASSISTANT).toBe('Jarvis')
    expect(await assistantName('https://box', 't', box(404, { error: { message: 'not found' } }).f)).toBe('Jarvis')
    expect(await assistantName('https://box', 't', box(403, { error: { message: 'scope' } }).f)).toBe('Jarvis')
    expect(await assistantName('https://box', 't', box(200, { name: '  ' }).f)).toBe('Jarvis')
    expect(await assistantName('https://box', 't', (async () => Promise.reject(new Error('ECONNREFUSED'))) as typeof fetch)).toBe('Jarvis')
  })

  it('names it in the task-start notice', () => {
    expect(startNotice('Friday', 'Search my notes')).toBe('Friday is using your Mac: Search my notes. Move the pointer to the top-left corner to stop.')
  })
})
