import { describe, expect, it } from 'vitest'

import { Conversation } from './conversation'
import { parseServerEvent, type ServerEvent } from './protocol'

const ev = (event: Record<string, unknown>) => parseServerEvent(JSON.stringify(event)) as ServerEvent

function spokenTurn(c: Conversation, item: string, transcript: string) {
  c.apply(ev({ type: 'input_audio_buffer.speech_started', item_id: item }))
  c.apply(ev({ type: 'localrouter.transcription.partial', item_id: item, transcript: transcript.slice(0, 3) }))
  c.apply(ev({ type: 'conversation.item.input_audio_transcription.completed', item_id: item, transcript }))
}

describe('the conversation', () => {
  it('a spoken turn and its streamed reply', () => {
    const c = new Conversation()
    spokenTurn(c, 'u1', 'What is the capital of France?')
    c.apply(ev({ type: 'response.output_item.added', item: { id: 'a1', type: 'message', role: 'assistant' } }))
    c.apply(ev({ type: 'response.audio_transcript.delta', item_id: 'a1', delta: 'Paris' }))
    c.apply(ev({ type: 'response.audio_transcript.delta', item_id: 'a1', delta: '.' }))
    c.apply(ev({ type: 'response.done', response: { status: 'completed', output: [{ id: 'a1', type: 'message' }] } }))
    expect(c.messages.map((m) => [m.role, 'text' in m ? m.text : ''])).toEqual([['user', 'What is the capital of France?'], ['assistant', 'Paris.']])
    expect(c.messages[1]).toMatchObject({ streaming: false, interrupted: false })
  })

  it('an empty transcript leaves no message; a cancelled reply is kept as interrupted', () => {
    const c = new Conversation()
    spokenTurn(c, 'u1', '  ')
    c.apply(ev({ type: 'response.text.delta', item_id: 'a1', delta: 'Well' }))
    c.apply(ev({ type: 'response.done', response: { status: 'cancelled', output: [{ id: 'a1', type: 'message' }] } }))
    expect(c.messages).toHaveLength(1)
    expect(c.messages[0]).toMatchObject({ role: 'assistant', text: 'Well', interrupted: true })
  })

  it('a picture from call to job done, in the reel newest first, with URLs rebased', () => {
    const c = new Conversation((url) => `https://box${url}`)
    c.apply(ev({ type: 'response.output_item.done', item: { type: 'function_call', name: 'generate_image', call_id: 'c1', arguments: '{"prompt":"a fox"}' } }))
    c.apply(ev({ type: 'localrouter.job.created', call_id: 'c1', job_id: 'j1', kind: 'image' }))
    expect(c.generations()[0]).toMatchObject({ kind: 'image', prompt: 'a fox', jobId: 'j1', status: 'working' })
    c.apply(ev({ type: 'localrouter.job.done', call_id: 'c1', job_id: 'j1', kind: 'image', status: 'completed', urls: ['/v1/files/j1/0.png'], error: null }))
    c.apply(ev({ type: 'localrouter.job.created', call_id: 'c2', job_id: 'j2', kind: 'video' }))
    expect(c.generations().map((g) => [g.kind, g.status, g.urls])).toEqual([['video', 'working', []], ['image', 'completed', ['https://box/v1/files/j1/0.png']]])
    // A late failure (a poll racing job.done) never replaces a finished picture.
    c.finish(c.generations()[1].id, { status: 'failed', error: 'late' })
    expect(c.generations()[1].status).toBe('completed')
  })

  it('reseeds a new session with what was said and made, and a new connection settles streaming items', () => {
    const c = new Conversation()
    c.typed('make a fox')
    c.apply(ev({ type: 'response.output_item.done', item: { type: 'function_call', name: 'generate_image', call_id: 'c1', arguments: '{"prompt":"a fox"}' } }))
    c.apply(ev({ type: 'localrouter.job.done', call_id: 'c1', job_id: 'j1', kind: 'image', status: 'completed', urls: ['/v1/files/j1/0.png'], error: null }))
    c.apply(ev({ type: 'response.text.delta', item_id: 'a9', delta: 'Here it' }))
    c.newConnection()
    expect(c.messages.at(-1)).toMatchObject({ role: 'assistant', streaming: false, interrupted: true })
    const seed = c.seed()
    expect(seed.map((e) => (e.type === 'conversation.item.create' ? e.item.type : e.type))).toEqual(['message', 'function_call', 'function_call_output', 'message'])
    const output = seed[2]
    expect(output.type === 'conversation.item.create' && output.item.type === 'function_call_output' && JSON.parse(output.item.output)).toEqual({ status: 'completed', urls: ['/v1/files/j1/0.png'] })
    // After a reconnect the server numbers items from 1 again: item a9 is a new reply, not the old one.
    c.apply(ev({ type: 'response.text.delta', item_id: 'a9', delta: 'New' }))
    expect(c.messages.at(-1)).toMatchObject({ text: 'New', streaming: true })
  })
})
