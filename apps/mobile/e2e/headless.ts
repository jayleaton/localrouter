/*
 * The app's protocol layer against a live box, without a phone: one spoken turn and one picture request, through
 * the same modules the app uses (RealtimeClient, Conversation, the HTTP API), under Node 22+ (global WebSocket and
 * fetch).
 *
 *   LR_URL=https://box.tailnet.ts.net:10443 [LR_TOKEN=…] [LR_PAIR_CODE=XXXX-XXXX] pnpm e2e
 *
 * The spoken prompt is synthesized by the box itself (/v1/audio/speech), resampled to 24 kHz and streamed in real
 * time in 40 ms chunks, as the phone's microphone does. Prints one JSON summary; exits 1 if any check fails.
 */
import { normalizeBaseUrl } from '../src/lib/localrouter/address'
import { createApi, type Fetch } from '../src/lib/localrouter/api'
import { Conversation, isGeneration } from '../src/lib/localrouter/conversation'
import { AUDIO_RATE, sessionUpdate, type ServerEvent } from '../src/lib/localrouter/protocol'
import { RealtimeClient, type SocketLike } from '../src/lib/localrouter/realtime'
import { floatToPcm16, pcm16ToBase64, silenceBase64 } from '../src/lib/voice/pcm'

const PROMPT = process.env.LR_PROMPT ?? 'What is the capital of France? Answer in one short sentence.'
const PICTURE = process.env.LR_PICTURE ?? 'Make a picture of a red fox sitting in the snow.'
const CHUNK = 960 // 40 ms at 24 kHz
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/** Mono float samples and their rate from a PCM16 or float32 WAV. */
function decodeWav(buffer: ArrayBuffer) {
  const view = new DataView(buffer)
  const tag = (at: number) => String.fromCharCode(...new Uint8Array(buffer, at, 4))
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') throw new Error('speech is not a WAV')
  let at = 12
  let format = 0, channels = 1, rate = 0, bits = 16
  while (at + 8 <= buffer.byteLength) {
    const id = tag(at)
    const size = view.getUint32(at + 4, true)
    if (id === 'fmt ') {
      format = view.getUint16(at + 8, true)
      channels = view.getUint16(at + 10, true)
      rate = view.getUint32(at + 12, true)
      bits = view.getUint16(at + 22, true)
    } else if (id === 'data') {
      const bytes = Math.min(size, buffer.byteLength - at - 8)
      const step = (bits / 8) * channels
      const samples = new Float32Array(Math.floor(bytes / step))
      for (let i = 0; i < samples.length; i++) {
        const p = at + 8 + i * step
        samples[i] = format === 3 ? view.getFloat32(p, true) : view.getInt16(p, true) / 0x8000
      }
      return { samples, rate }
    }
    at += 8 + size + (size & 1)
  }
  throw new Error('the WAV has no data')
}

type Check = { name: string; ok: boolean; detail?: unknown }

async function main() {
  const baseUrl = normalizeBaseUrl(process.env.LR_URL ?? '')
  if (!baseUrl) throw new Error('set LR_URL to the box, e.g. https://box.tailnet.ts.net:10443')
  const fetchFn: Fetch = (input, init) => fetch(input, init)
  const checks: Check[] = []
  const check = (name: string, ok: boolean, detail?: unknown) => checks.push({ name, ok, ...(detail === undefined ? {} : { detail }) })

  let token = process.env.LR_TOKEN || null
  if (process.env.LR_PAIR_CODE) {
    const paired = await createApi({ baseUrl, fetch: fetchFn }).pair(process.env.LR_PAIR_CODE, 'headless check')
    token = paired.token
    check('pairing code exchanged for a token', !!paired.token, { device: paired.device })
  }
  const api = createApi({ baseUrl, token, fetch: fetchFn })
  const status = await api.pairStatus().catch((error: unknown) => ({ error: String(error) }))
  check('pair status answers', !('error' in status), status)

  // The prompt, spoken by the box's own speech model.
  const wav = decodeWav(await api.speech(PROMPT))
  const pcm = floatToPcm16(wav.samples, wav.rate, AUDIO_RATE)
  check('prompt synthesized', pcm.length > AUDIO_RATE / 2, { seconds: +(pcm.length / AUDIO_RATE).toFixed(2), source_rate: wav.rate })

  const conversation = new Conversation(api.fileUrl)
  const events: ServerEvent[] = []
  const waiters: { test: (e: ServerEvent) => boolean; resolve: (e: ServerEvent) => void }[] = []
  const at = new Map<string, number>()
  const waitFor = (test: (e: ServerEvent) => boolean, ms: number, what: string) =>
    new Promise<ServerEvent>((resolve, reject) => {
      const found = events.find(test)
      if (found) return resolve(found)
      const timer = setTimeout(() => reject(new Error(`timed out waiting for ${what}`)), ms)
      waiters.push({ test, resolve: (e) => { clearTimeout(timer); resolve(e) } })
    })
  let call = true
  const client = new RealtimeClient({
    baseUrl,
    token,
    socket: (url, protocols) => new WebSocket(url, protocols) as unknown as SocketLike,
    onEvent: (event) => {
      events.push(event)
      if (!at.has(event.type)) at.set(event.type, Date.now())
      conversation.apply(event)
      for (const w of [...waiters]) if (w.test(event)) { waiters.splice(waiters.indexOf(w), 1); w.resolve(event) }
    },
    onOpen: () => [sessionUpdate({ audio: call, waitMs: 800 }), ...conversation.seed()],
  })
  client.connect()
  await waitFor((e) => e.type === 'session.created', 15_000, 'session.created')
  check('realtime session opened with the app protocol layer', client.isOpen)

  // ---- one voice turn: real-time 40 ms chunks, then silence until the box answers
  let audioMs = 0
  let firstAudioAt = 0
  const onAudio = (e: ServerEvent) => {
    if (e.type !== 'response.audio.delta') return false
    if (!firstAudioAt) firstAudioAt = Date.now()
    audioMs += (e.delta.length * 3) / 4 / 2 / (AUDIO_RATE / 1000)
    return false
  }
  waiters.push({ test: onAudio, resolve: () => undefined })
  const started = Date.now()
  for (let i = 0; i < pcm.length; i += CHUNK) {
    client.send({ type: 'input_audio_buffer.append', audio: pcm16ToBase64(pcm.subarray(i, i + CHUNK)) })
    await sleep(Math.max(0, started + ((i + CHUNK) / AUDIO_RATE) * 1000 - Date.now()))
  }
  const speechEnd = Date.now()
  const quiet = silenceBase64(CHUNK)
  const silenceLoop = (async () => {
    while (call) {
      client.send({ type: 'input_audio_buffer.append', audio: quiet })
      await sleep(40)
    }
  })()
  const transcript = await waitFor((e) => e.type === 'conversation.item.input_audio_transcription.completed', 30_000, 'the transcript')
  const done = await waitFor((e) => e.type === 'response.done', 60_000, 'the spoken reply')
  call = false
  await silenceLoop
  const said = transcript.type === 'conversation.item.input_audio_transcription.completed' ? transcript.transcript : ''
  const reply = conversation.messages.filter((m) => m.role === 'assistant').at(-1)
  check('transcript of the spoken turn', /capital/i.test(said) && /france/i.test(said), said)
  check('reply completed', done.type === 'response.done' && done.response.status === 'completed', done.type === 'response.done' ? done.response.status : null)
  check('reply audio streamed (24 kHz PCM16)', audioMs > 500, { audio_ms: Math.round(audioMs), first_audio_after_speech_ms: firstAudioAt ? firstAudioAt - speechEnd : null })
  check('reply text says Paris', !!reply && /paris/i.test(reply.text), reply && reply.text)

  // ---- one picture request, typed in a text-only session as the app's chat does
  client.send(sessionUpdate({ audio: false, waitMs: 800 }))
  for (const event of conversation.typed(PICTURE)) client.send(event)
  const jobDone = await waitFor((e) => e.type === 'localrouter.job.done', 300_000, 'the picture')
  const generation = conversation.generations()[0]
  check('picture job done', jobDone.type === 'localrouter.job.done' && jobDone.status === 'completed', jobDone.type === 'localrouter.job.done' ? { status: jobDone.status, error: jobDone.error } : null)
  check('the reel has the picture with its prompt', !!generation && isGeneration(generation) && generation.status === 'completed' && generation.urls.length > 0, generation && { prompt: generation.prompt, urls: generation.urls })
  if (generation?.urls[0]) {
    const response = await fetch(generation.urls[0], { headers: token ? { authorization: `Bearer ${token}` } : {} })
    const bytes = new Uint8Array(await response.arrayBuffer())
    const png = bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47
    check('the picture downloads through the rebased URL and is a PNG', response.ok && png, { status: response.status, bytes: bytes.length, url: generation.urls[0] })
  }
  await waitFor((e) => e.type === 'response.done' && events.filter((x) => x.type === 'response.done').length >= 3, 120_000, 'the spoken confirmation').catch(() => undefined)
  const confirmation = conversation.messages.filter((m) => m.role === 'assistant').at(-1)
  check('the assistant confirms the picture', !!confirmation && confirmation.text.length > 0, confirmation && confirmation.text)
  client.close()

  const ok = checks.every((c) => c.ok)
  console.log(JSON.stringify({ ok, base_url: baseUrl, paired: !!token, checks }, null, 2))
  process.exit(ok ? 0 : 1)
}

main().catch((error: unknown) => {
  console.log(JSON.stringify({ ok: false, error: error instanceof Error ? error.message : String(error) }))
  process.exit(1)
})
