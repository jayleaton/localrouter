import { describe, expect, it } from 'vitest'
import { CONFIRM_CHUNKS, EchoGate, echoReference, INITIAL_COUPLING, OPEN_HOLD_MS, REPLY_GAP_BRIDGE_MS, SPEECH_OVER_ECHO } from './echoGate'

const play = 0.2
/** The phone's microphone chunk. */
const CHUNK = 40

describe('echo gate', () => {
  it('passes everything while nothing is playing', () => {
    const gate = new EchoGate()
    expect(gate.process({ data: 'a', rms: 0.01, now: 0, playLevel: 0 }, 'S')).toEqual(['a'])
    expect(gate.process({ data: 'b', rms: 0.4, now: 100, playLevel: 0 }, 'S')).toEqual(['b'])
  })

  it('keeps a strong early echo out, and keeps it out as the canceller adapts', () => {
    const gate = new EchoGate()
    // Start of a call: the echo is nearly as loud as what plays (it would pass a fixed threshold).
    for (let at = 0; at < 1_000; at += CHUNK) expect(gate.process({ data: 'echo', rms: play * 0.8, now: at, playLevel: play }, 'S')).toEqual(['S'])
    // The canceller settles and the echo drops; still silence.
    for (let at = 1_000; at < 3_000; at += CHUNK) expect(gate.process({ data: 'echo', rms: play * 0.1, now: at, playLevel: play }, 'S')).toEqual(['S'])
  })

  it('lets the user talk over a reply once their voice stays above the echo, without losing the first chunk', () => {
    const gate = new EchoGate()
    // Learn a settled echo first.
    for (let at = 0; at < 2_000; at += CHUNK) gate.process({ data: 'echo', rms: play * 0.1, now: at, playLevel: play }, 'S')
    const voice = play * 0.5
    // 200 ms of speech confirms it, as it did with two 100 ms chunks.
    expect(CONFIRM_CHUNKS * CHUNK).toBe(200)
    const words = ['no', 'wait', 'stop', 'that', 'now']
    const confirmedAt = 2_000 + (CONFIRM_CHUNKS - 1) * CHUNK
    for (const [index, word] of words.slice(0, -1).entries()) {
      expect(gate.process({ data: word, rms: voice, now: 2_000 + index * CHUNK, playLevel: play }, 'S')).toEqual(['S'])
      expect(gate.bargedIn).toBe(false)
    }
    expect(gate.process({ data: 'now', rms: voice, now: confirmedAt, playLevel: play }, 'S')).toEqual(words)
    // Confirmed once: the phone stops playback there.
    expect(gate.bargedIn).toBe(true)
    // A pause between words still passes while the gate is held open.
    expect(gate.process({ data: 'pause', rms: 0.01, now: confirmedAt + CHUNK, playLevel: play }, 'S')).toEqual(['pause'])
    expect(gate.bargedIn).toBe(false)
    expect(gate.process({ data: 'later', rms: play * 0.1, now: confirmedAt + OPEN_HOLD_MS + 1, playLevel: play }, 'S')).toEqual(['S'])
  })

  it('never lets the echo itself open the gate or hold it open', () => {
    const gate = new EchoGate()
    // As loud as the initial coupling predicts (under the speech margin): silence, however long.
    const echo = play * INITIAL_COUPLING * (SPEECH_OVER_ECHO - 1)
    for (let at = 0; at < 2_000; at += CHUNK) expect(gate.process({ data: 'echo', rms: echo, now: at, playLevel: play }, 'S')).toEqual(['S'])
  })

  it('does not open on a short loud sound (a knock or a cough)', () => {
    const gate = new EchoGate()
    for (let at = 0; at < 2_000; at += CHUNK) gate.process({ data: 'echo', rms: play * 0.1, now: at, playLevel: play }, 'S')
    // 160 ms loud, just under the 200 ms that confirms speech.
    for (let at = 2_000; at < 2_000 + (CONFIRM_CHUNKS - 1) * CHUNK; at += CHUNK) {
      expect(gate.process({ data: 'knock', rms: 0.5, now: at, playLevel: play }, 'S')).toEqual(['S'])
    }
    expect(gate.process({ data: 'echo', rms: play * 0.1, now: 2_200, playLevel: play }, 'S')).toEqual(['S'])
    expect(gate.bargedIn).toBe(false)
  })

  it('learns the echo at the same pace per second as with 100 ms chunks', () => {
    const gate = new EchoGate()
    // One second (25 chunks) of a settled echo (coupling 0.1) after the strong initial guess.
    for (let chunk = 0; chunk < 25; chunk++) gate.process({ data: 'echo', rms: play * 0.1, now: chunk * CHUNK, playLevel: play }, 'S')
    // 30% of the way per 100 ms: ten 100 ms steps.
    expect(gate.coupling).toBeCloseTo(0.1 + (INITIAL_COUPLING - 0.1) * 0.7 ** 10, 5)
  })

  describe('a gap inside a reply (its audio ran dry, more is coming)', () => {
    const window = { chunkMs: CHUNK, tailMs: 300 }
    // "We are" played from 0 to 600 ms; the next chunk is late.
    const played = [{ from: 0, to: 600, rms: play }]

    it('still expects the echo of its last words while the reply is streaming, and not after it ended', () => {
      expect(echoReference(played, 1_200, { ...window, replyOpen: true })).toEqual({ level: play, bridged: true })
      expect(echoReference(played, 1_200, { ...window, replyOpen: false })).toEqual({ level: 0, bridged: false })
      // Within the tail it is plain playback, not a bridge.
      expect(echoReference(played, 800, { ...window, replyOpen: true })).toEqual({ level: play, bridged: false })
      // A stalled reply does not gate the microphone for long.
      expect(echoReference(played, 600 + REPLY_GAP_BRIDGE_MS + 1, { ...window, replyOpen: true }).level).toBe(0)
    })

    it('keeps a lagging echo out in the gap, without learning from the silence', () => {
      const gate = new EchoGate()
      // Early in a call: a late echo of the first words, as loud as the canceller lets through at the start.
      for (let at = 1_000; at < 1_600; at += CHUNK) expect(gate.process({ data: 'we are', rms: play * 0.8, now: at, playLevel: play, bridged: true }, 'S')).toEqual(['S'])
      // Quiet chunks in the gap would teach it a near-zero coupling; bridged, they teach nothing.
      for (let at = 1_600; at < 2_000; at += CHUNK) gate.process({ data: 'quiet', rms: 0.001, now: at, playLevel: play, bridged: true }, 'S')
      expect(gate.coupling).toBe(INITIAL_COUPLING)
    })

    it('still lets the user barge in during the gap to change the subject', () => {
      const gate = new EchoGate()
      for (let at = 0; at < 2_000; at += CHUNK) gate.process({ data: 'echo', rms: play * 0.1, now: at, playLevel: play }, 'S')
      const words = ['actually', 'let', 'us', 'talk', 'about']
      const sent = words.map((word, index) => gate.process({ data: word, rms: play * 0.5, now: 2_000 + index * CHUNK, playLevel: play, bridged: true }, 'S'))
      expect(sent.at(-1)).toEqual(words)
      expect(gate.bargedIn).toBe(true)
    })
  })
})
