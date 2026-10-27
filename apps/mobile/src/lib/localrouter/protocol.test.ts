import { describe, expect, it } from 'vitest'

import { parseServerEvent, refusedVoice, sessionUpdate } from './protocol'

describe('session settings', () => {
  it('a call sets audio and server VAD with the user wait, clamped to 0.3–5 s', () => {
    const call = sessionUpdate({ audio: true, waitMs: 12_000, voice: 'Mae', speechModel: 'drama' }).session
    expect(call.modalities).toEqual(['audio', 'text'])
    expect(call.turn_detection).toMatchObject({ type: 'server_vad', silence_duration_ms: 5_000, interrupt_response: true, create_response: true })
    expect(call.voice).toBe('Mae')
    expect(sessionUpdate({ audio: true, waitMs: 100 }).session.turn_detection).toMatchObject({ silence_duration_ms: 300 })
    expect(sessionUpdate({ audio: true, waitMs: Number.NaN }).session.turn_detection).toMatchObject({ silence_duration_ms: 800 })
  })

  it('chat without a call is text only and has no VAD', () => {
    const chat = sessionUpdate({ audio: false, waitMs: 800 }).session
    expect(chat.modalities).toEqual(['text'])
    expect(chat.turn_detection).toBeNull()
    expect('voice' in chat).toBe(false)
  })
})

describe('server events', () => {
  it('parses what the app uses, ignores the rest and anything malformed', () => {
    expect(parseServerEvent('{"type":"localrouter.job.done","call_id":"c","job_id":"j","kind":"image","status":"completed","urls":["/v1/files/j/0.png"],"error":null}')).toMatchObject({ type: 'localrouter.job.done', urls: ['/v1/files/j/0.png'] })
    expect(parseServerEvent('{"type":"localrouter.response.text_done","item_id":"i","tool_calls":false,"chars":40,"speech_rate":null}')).toMatchObject({ speech_rate: null })
    expect(parseServerEvent('{"type":"localrouter.response.speech_first","response_id":"r","item_id":"i"}')).toMatchObject({ item_id: 'i' })
    expect(parseServerEvent('{"type":"rate_limits.updated"}')).toBeNull()
    expect(parseServerEvent('{"type":"response.audio.delta"}')).toBeNull()
    expect(parseServerEvent('not json')).toBeNull()
  })

  it('recognizes a refused voice', () => {
    expect(refusedVoice("model 'drama' has no voice 'Rosie'")).toEqual({ model: 'drama', voice: 'Rosie' })
    expect(refusedVoice('something else')).toBeNull()
  })
})
