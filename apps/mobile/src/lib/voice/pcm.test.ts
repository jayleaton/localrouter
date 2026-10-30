import { describe, expect, it } from 'vitest'
import { base64ToBytes, bytesToBase64, floatToPcm16Base64, pcm16Base64ToFloat, resample, silenceBase64 } from './pcm'

describe('pcm helpers', () => {
  it('matches the platform base64 for every padding length', () => {
    for (const length of [0, 1, 2, 3, 4, 5, 257]) {
      const bytes = Uint8Array.from({ length }, (_, index) => (index * 37) & 255)
      const expected = btoa(String.fromCharCode(...bytes))
      expect(bytesToBase64(bytes)).toBe(expected)
      expect(Array.from(base64ToBytes(expected))).toEqual(Array.from(bytes))
    }
  })

  it('decodes base64 with line breaks or spaces', () => {
    expect(Array.from(base64ToBytes('AQID\nBA =='))).toEqual([1, 2, 3, 4])
  })

  it('encodes silence once per length', () => {
    expect(silenceBase64(3)).toBe(bytesToBase64(new Uint8Array(6)))
    expect(silenceBase64(1_600)).toBe(silenceBase64(1_600))
    expect(pcm16Base64ToFloat(silenceBase64(1_600)).every((sample) => sample === 0)).toBe(true)
  })

  it('round-trips PCM16 within quantization error', () => {
    const samples = Float32Array.from({ length: 160 }, (_, index) => Math.sin(index / 5) * 0.8)
    const decoded = pcm16Base64ToFloat(floatToPcm16Base64(samples, 16_000, 16_000))
    expect(decoded.length).toBe(160)
    decoded.forEach((value, index) => expect(Math.abs(value - samples[index])).toBeLessThan(1e-3))
  })

  it('resamples 48 kHz microphone audio to 16 kHz for the model', () => {
    const input = new Float32Array(4_800).fill(0.25)
    const output = resample(input, 48_000, 16_000)
    expect(output.length).toBe(1_600)
    expect(output[800]).toBeCloseTo(0.25)
  })
})
