/** pcm16 mono at `from` Hz to 24 kHz (the realtime rate), linear interpolation. */
export function to24k(pcm: Buffer, from: number): Buffer {
  if (from === 24000) return pcm
  const n = Math.floor(pcm.length / 2)
  const out = Math.floor((n * 24000) / from)
  const buf = Buffer.alloc(out * 2)
  for (let i = 0; i < out; i++) {
    const x = (i * from) / 24000
    const j = Math.floor(x)
    const f = x - j
    const a = pcm.readInt16LE(Math.min(j, n - 1) * 2)
    const b = pcm.readInt16LE(Math.min(j + 1, n - 1) * 2)
    buf.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(a + (b - a) * f))), i * 2)
  }
  return buf
}
