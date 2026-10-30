import { describe, expect, it } from 'vitest'

import { WakeGate } from './wakeGate'

const CHUNK_MS = 160
const seconds = (s: number) => Math.round((s * 1_000) / CHUNK_MS)

/** Runs levels through a gate; returns every chunk index fed to the recognizer and each close. */
function run(levels: number[], gate = new WakeGate<number>({ chunkMs: CHUNK_MS })) {
  const fed: number[] = []
  const closes: number[] = []
  levels.forEach((rms, index) => {
    const step = gate.process(index, rms)
    fed.push(...step.feed)
    if (step.closed) closes.push(index)
  })
  return { fed, closes }
}

const quietRoom = (s: number) => Array.from({ length: seconds(s) }, (_, i) => 0.001 + (i % 3) * 0.0002)

describe('WakeGate', () => {
  it('feeds the recognizer nothing in a quiet room', () => {
    expect(run(quietRoom(60)).fed).toEqual([])
  })

  it('feeds a phrase with the audio just before it, then closes once after the hang time', () => {
    const before = quietRoom(5)
    const phrase = Array.from({ length: seconds(1) }, () => 0.05)
    const after = quietRoom(5)
    const { fed, closes } = run([...before, ...phrase, ...after])
    const start = before.length
    // The pre-roll (800 ms = 5 chunks) comes first, so "hey" is not clipped.
    expect(fed.slice(0, 6)).toEqual([start - 5, start - 4, start - 3, start - 2, start - 1, start])
    const end = start + phrase.length
    // 1.5 s of quiet after the phrase (about 9 chunks) still reaches the recognizer, then the gate closes.
    expect(closes).toEqual([end + seconds(1.5) - 1])
    expect(fed.at(-1)).toBe(closes[0])
    expect(fed.filter((index) => index >= end + seconds(1.5))).toEqual([])
  })

  it('hears a quiet voice in a quiet room', () => {
    const { fed } = run([...quietRoom(5), ...Array.from({ length: seconds(1) }, () => 0.008)])
    expect(fed.length).toBeGreaterThan(0)
  })

  it('stops opening for a steady noise once the floor has learned it', () => {
    const fan = Array.from({ length: seconds(60) }, (_, i) => 0.03 + (i % 2) * 0.002)
    const { fed } = run([...quietRoom(5), ...fan])
    const lastFed = Math.max(...fed)
    // It opens as the fan starts, then learns it within the floor's rise time plus the hang.
    expect(lastFed - seconds(5)).toBeLessThan(seconds(15))
  })

  it('still opens for speech over a steady noise it has learned', () => {
    const fan = Array.from({ length: seconds(30) }, () => 0.02)
    const speech = Array.from({ length: seconds(1) }, () => 0.08)
    const { fed } = run([...quietRoom(5), ...fan, ...speech])
    expect(fed).toContain(seconds(35))
  })

  // Review of #173: a learned background must never make an ordinary voice over it inaudible.
  it('hears a voice over music it has learned, at the music\'s own level or louder', () => {
    const rms = (...parts: number[]) => Math.sqrt(parts.reduce((sum, part) => sum + part * part, 0))
    const steadyMusic = Array.from({ length: seconds(60) }, () => 0.03)
    // Background 0.03 plus a voice at 0.04 (combined 0.05), for 1.28 s.
    const phrase = Array.from({ length: 8 }, () => rms(0.03, 0.04))
    expect(run([...steadyMusic, ...phrase]).fed).toContain(seconds(60))

    // Music that swells and fades (0.02–0.04 every 2 s), then a voice at its average level.
    const music = (i: number) => 0.03 + 0.01 * Math.sin((2 * Math.PI * i * CHUNK_MS) / 2_000)
    const song = Array.from({ length: seconds(60) }, (_, i) => music(i))
    const voice = Array.from({ length: 8 }, (_, i) => rms(music(seconds(60) + i), 0.03))
    const { fed } = run([...song, ...voice])
    const phraseChunks = voice.map((_, i) => seconds(60) + i)
    expect(phraseChunks.filter((index) => fed.includes(index)).length).toBeGreaterThanOrEqual(6)
  })

  it('forgets the room on reset', () => {
    const gate = new WakeGate<number>({ chunkMs: CHUNK_MS })
    run(Array.from({ length: seconds(30) }, () => 0.04), gate)
    gate.reset()
    expect(run(quietRoom(5), gate).fed).toEqual([])
    expect(gate.isOpen).toBe(false)
  })
})
