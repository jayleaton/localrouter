const ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
const LOOKUP = new Uint8Array(256)
for (let index = 0; index < ALPHABET.length; index += 1) LOOKUP[ALPHABET.charCodeAt(index)] = index

/** Base64 without Buffer/atob (not reliably available on React Native). */
export function bytesToBase64(bytes: Uint8Array) {
  let output = ''
  let index = 0
  for (; index + 2 < bytes.length; index += 3) {
    const chunk = (bytes[index] << 16) | (bytes[index + 1] << 8) | bytes[index + 2]
    output += ALPHABET[(chunk >> 18) & 63] + ALPHABET[(chunk >> 12) & 63] + ALPHABET[(chunk >> 6) & 63] + ALPHABET[chunk & 63]
  }
  const rest = bytes.length - index
  if (rest === 1) {
    const chunk = bytes[index] << 16
    output += `${ALPHABET[(chunk >> 18) & 63]}${ALPHABET[(chunk >> 12) & 63]}==`
  } else if (rest === 2) {
    const chunk = (bytes[index] << 16) | (bytes[index + 1] << 8)
    output += `${ALPHABET[(chunk >> 18) & 63]}${ALPHABET[(chunk >> 12) & 63]}${ALPHABET[(chunk >> 6) & 63]}=`
  }
  return output
}

const NOT_BASE64 = /[^A-Za-z0-9+/=]/

export function base64ToBytes(base64: string) {
  // Audio from the computer is clean base64: skip the copy unless something needs stripping.
  const clean = NOT_BASE64.test(base64) ? base64.replace(/[^A-Za-z0-9+/]/g, '') : base64
  let length = clean.length
  while (length > 0 && clean.charCodeAt(length - 1) === 61) length -= 1 // '=' padding
  const bytes = new Uint8Array(Math.floor((length * 3) / 4))
  let byte = 0
  for (let index = 0; index < length; index += 4) {
    const chunk =
      (LOOKUP[clean.charCodeAt(index)] << 18) |
      (LOOKUP[clean.charCodeAt(index + 1)] << 12) |
      (LOOKUP[clean.charCodeAt(index + 2)] << 6) |
      LOOKUP[clean.charCodeAt(index + 3)]
    bytes[byte++] = (chunk >> 16) & 255
    if (byte < bytes.length) bytes[byte++] = (chunk >> 8) & 255
    if (byte < bytes.length) bytes[byte++] = chunk & 255
  }
  return bytes
}

/** Linear resampling, adequate for speech (as in True Reply's browser client). */
export function resample(samples: Float32Array, fromRate: number, toRate: number) {
  if (fromRate === toRate) return samples
  const length = Math.max(1, Math.round((samples.length * toRate) / fromRate))
  const output = new Float32Array(length)
  const ratio = fromRate / toRate
  for (let index = 0; index < length; index += 1) {
    const position = index * ratio
    const left = Math.floor(position)
    const right = Math.min(left + 1, samples.length - 1)
    const fraction = position - left
    output[index] = samples[left] * (1 - fraction) + samples[right] * fraction
  }
  return output
}

/** Float samples in [-1, 1] → PCM16 mono at `toRate`. */
export function floatToPcm16(samples: Float32Array, fromRate: number, toRate: number) {
  const resampled = resample(samples, fromRate, toRate)
  const pcm = new Int16Array(resampled.length)
  for (let index = 0; index < resampled.length; index += 1) {
    const clamped = Math.max(-1, Math.min(1, resampled[index]))
    pcm[index] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
  }
  return pcm
}

/** PCM16 samples → base64 little-endian bytes. */
export function pcm16ToBase64(pcm: Int16Array) {
  const bytes = new Uint8Array(pcm.length * 2)
  const view = new DataView(bytes.buffer)
  for (let index = 0; index < pcm.length; index += 1) view.setInt16(index * 2, pcm[index], true)
  return bytesToBase64(bytes)
}

/** Float samples in [-1, 1] → base64 PCM16 little-endian mono at `toRate`. */
export function floatToPcm16Base64(samples: Float32Array, fromRate: number, toRate: number) {
  return pcm16ToBase64(floatToPcm16(samples, fromRate, toRate))
}

/** Base64 PCM16 little-endian mono → float samples in [-1, 1]. */
export function pcm16Base64ToFloat(base64: string) {
  const bytes = base64ToBytes(base64)
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const samples = new Float32Array(Math.floor(bytes.length / 2))
  for (let index = 0; index < samples.length; index += 1) samples[index] = view.getInt16(index * 2, true) / 0x8000
  return samples
}

const silence = new Map<number, string>()

/** Base64 PCM16 silence of `samples` samples, encoded once per length (the mic sends it on every gated chunk). */
export function silenceBase64(samples: number) {
  let data = silence.get(samples)
  if (data === undefined) {
    data = pcm16ToBase64(new Int16Array(samples))
    if (silence.size >= 8) silence.clear()
    silence.set(samples, data)
  }
  return data
}
