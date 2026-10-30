import { Platform } from 'react-native'
import { AudioContext, AudioManager, AudioRecorder, type AudioBufferSourceNode } from 'react-native-audio-api'

import { conversation, getBox, onVoiceEvent, sendEvent, setCall } from '@/lib/app/box'
import { isGeneration } from '@/lib/localrouter/conversation'
import { AUDIO_RATE, type ServerEvent } from '@/lib/localrouter/protocol'
import { CallSession } from '../../../modules/call-session/src'
import { VoiceCommands } from '../../../modules/voice-commands/src'
import { EchoGate, echoReference, type PlayedSegment } from './echoGate'
import { floatToPcm16, pcm16Base64ToFloat, pcm16ToBase64, silenceBase64 } from './pcm'
import { ReplyHold, type VoicePriors } from './replyHold'
import { DEFAULT_VOICE_COMMAND_PHRASES, lastCommand, voiceCommandPhrases, type VoiceCommandPhrases } from './voiceCommands'
import { getVoiceState, setVoiceState as setState, type CallActivity } from './voiceState'
import { WakeGate } from './wakeGate'

/*
 * Voice on this phone, against LocalRouter's realtime session (docs/dev/VOICE.md). The recorder runs while voice is
 * on. In standby only the on-device recognizer hears it, behind an energy gate, listening for the wake phrase; nothing
 * leaves the phone. A call switches the box's session to audio: the microphone streams PCM16 at 24 kHz, the box's
 * VAD ends each turn after the user's own wait (0.3 to 5 s, restarted by every pause that speech interrupts), and
 * the reply streams back as 24 kHz PCM16 that is held at its start until it can play through (replyHold.ts).
 * Talking over a reply stops it here at once and tells the box how much was heard.
 */

/** LocalRouter takes and sends PCM16 mono at 24 kHz. */
const RATE = AUDIO_RATE
/** The on-device recognizer listens at 16 kHz (what both platforms' recognizers are built for). */
const RECOGNIZER_RATE = 16_000
/** 40 ms a chunk in a call: the end of speech reaches the box sooner than with 100 ms chunks. */
const CHUNK_SAMPLES = 960
/** Standby takes 160 ms chunks: JavaScript wakes 6 times a second instead of 25. */
const STANDBY_CHUNK_SAMPLES = 3_840
/** A call that has no open session by then fails back to standby. */
const CONNECT_TIMEOUT_MS = 10_000
/** What the user says while the call opens is kept, up to 10 s. */
const MAX_WAKE_CHUNKS = 250
/** Kept speech goes out at twice real time: up to two chunks for each new one. */
const FLUSH_PER_CHUNK = 2
/** Microphone level (RMS) that counts as the user speaking. */
const SPEECH_RMS = 0.015
/** A call ends after this long with neither side speaking (the phone goes back to listening for the wake phrase). */
const IDLE_MS = 120_000
/** After the mic closes, silence goes on this much past the wait, so the box ends the turn the user was in. */
const CLOSE_TAIL_MS = 600
/** The speaker and echo canceller lag the schedule; echo can still arrive this long after audio plays. */
const ECHO_TAIL_MS = 300
/** A held reply is re-checked this often. */
const HOLD_TICK_MS = 25
/** A reply that ran dry restarts with a lead that grows by how late it was, up to this; it shrinks after each reply. */
const MIN_LEAD_S = 0.1
const MAX_LEAD_S = 0.5

/** A stalled permission or native audio operation must not trap the voice UI. */
function waitFor<T>(promise: Promise<T>, message: string, timeoutMs = 15_000): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(message)), timeoutMs)
    promise.then((value) => { clearTimeout(timer); resolve(value) }, (error: unknown) => { clearTimeout(timer); reject(error) })
  })
}

let recorder: AudioRecorder | null = null
let playback: AudioContext | null = null
let chunkSamples = 0
let opening: Promise<void> | null = null
let listeningInBackground = false
/** Bumped whenever standby starts or stops, so late native callbacks from an earlier run are ignored. */
let generation = 0
/** The spoken commands for this standby: "hey <name>" with the assistant's name when it started (a rename restarts it). */
let commandPhrases: VoiceCommandPhrases = DEFAULT_VOICE_COMMAND_PHRASES
let commands: { remove(): void; audio(peak: number): void; newStretch(): void } | null = null
const wakeGate = new WakeGate<Float32Array>({ chunkMs: (STANDBY_CHUNK_SAMPLES / RATE) * 1_000 })

// ---------------------------------------------------------------- the call

let micOpen = false
let voiceMuted = false
let wakeChunks: string[] = []
let outgoing: string[] = []
let lastSpeechAt = 0
let closedAt = 0
let activity: CallActivity = 'listening'
const echoGate = new EchoGate()

/** One scheduled piece of reply audio. */
type Live = { source: AudioBufferSourceNode; item: string; start: number; duration: number }
type Piece = { item: string; samples: Float32Array<ArrayBuffer> }
const live: Live[] = []
/** Ms of each reply item already played through (for `conversation.item.truncate`). */
const doneMs = new Map<string, number>()
/** Replies the user cut off: their remaining audio is dropped. */
const cut = new Set<string>()
let played: PlayedSegment[] = []
let nextTime = 0
/** The reply item scheduled last: a piece of another item starts a new reply, not an underrun. */
let lastItem: string | null = null
/** The reply item whose audio came last (held or playing). */
let lastDelta: string | null = null
let lead = MIN_LEAD_S
let playbackEndsAt = 0
/** A reply is still coming (between its first audio and `response.done`): a gap in it is still the reply. */
let replyOpen = false
/** What the reply start learned of each voice (its arrival rate and audio a character), kept while the app runs. */
const voicePriors: VoicePriors = new Map()
const voiceKey = () => {
  const { speechModel, voices } = getBox().settings
  return `${speechModel}:${voices[speechModel] ?? ''}`
}
const hold = new ReplyHold<Piece>(() => conversation.messages.some((m) => isGeneration(m) && m.status === 'working'), voiceKey, voicePriors)
let holdTimer: ReturnType<typeof setInterval> | null = null

function setActivity(next: CallActivity) {
  activity = next
  const state = getVoiceState()
  if (state.status === 'live' && state.activity !== next) setState({ ...state, activity: next })
}

const callActivity = (): CallActivity => (live.length ? 'speaking' : activity)

function schedulePlayed(startInMs: number, durationMs: number, samples: Float32Array) {
  let energy = 0
  for (const sample of samples) energy += sample * sample
  const from = Date.now() + startInMs
  played = played.filter((segment) => segment.to > Date.now() - 2_000)
  played.push({ from, to: from + durationMs, rms: Math.sqrt(energy / Math.max(samples.length, 1)) })
}

/** A time on the context's frame grid, so pieces start exactly where the one before ends. */
const frame = (t: number) => Math.ceil(t * RATE - 1e-6) / RATE

function schedule({ item, samples }: Piece) {
  if (!playback || cut.has(item)) return
  const now = playback.currentTime
  const sameReply = item === lastItem
  lastItem = item
  if (nextTime < now) {
    // Ran dry mid-reply: restart with a lead grown by how late it was, so a slow stream pauses rarely, and the
    // session's next replies start with more in hand (the pause heard). A new reply starts at once: replyHold already
    // waited for enough of it.
    if (sameReply && nextTime > 0) {
      hold.gap(Math.round((now - nextTime + lead) * 1_000))
      lead = Math.min(MAX_LEAD_S, lead + (now - nextTime))
    }
    nextTime = frame(now + (sameReply && nextTime > 0 ? lead : 0.01))
  }
  const buffer = playback.createBuffer(1, samples.length, RATE)
  buffer.copyToChannel(samples, 0)
  const source = playback.createBufferSource()
  source.buffer = buffer
  source.connect(playback.destination)
  const start = nextTime
  source.start(start)
  nextTime = frame(start + buffer.duration)
  schedulePlayed((start - now) * 1_000, buffer.duration * 1_000, samples)
  playbackEndsAt = Date.now() + (nextTime - now) * 1_000
  const entry: Live = { source, item, start, duration: buffer.duration }
  live.push(entry)
  source.onEnded = () => {
    const index = live.indexOf(entry)
    if (index >= 0) live.splice(index, 1)
    doneMs.set(item, (doneMs.get(item) ?? 0) + entry.duration * 1_000)
    setActivity(live.length ? 'speaking' : activity === 'speaking' ? 'listening' : activity)
  }
  setActivity('speaking')
}

function release(pieces: Piece[]) {
  for (const piece of pieces) schedule(piece)
  if (!hold.holding && holdTimer) {
    clearInterval(holdTimer)
    holdTimer = null
  }
}

function holdTick() {
  holdTimer ??= setInterval(() => release(hold.tick(Date.now())), HOLD_TICK_MS)
}

/** Stops all reply audio now. */
function silence() {
  for (const entry of live) {
    entry.source.onEnded = null
    try {
      entry.source.stop()
    } catch {
      // Already finished.
    }
  }
  live.length = 0
  played = []
  nextTime = 0
  lastItem = null
  playbackEndsAt = 0
  replyOpen = false
  hold.reset()
  if (holdTimer) clearInterval(holdTimer)
  holdTimer = null
}

/** Ms of `item` heard so far. */
function heardMs(item: string) {
  let ms = doneMs.get(item) ?? 0
  const now = playback?.currentTime ?? 0
  for (const entry of live) if (entry.item === item) ms += Math.min(Math.max(now - entry.start, 0), entry.duration) * 1_000
  return Math.floor(ms)
}

/**
 * The user talked over the reply (or tapped to stop it): playback stops here at once, the box is told how much was
 * heard (`conversation.item.truncate`, so the model sees what the user actually heard), and with `cancel` the
 * response ends. Speech alone needs no cancel: the box's VAD cancels the response itself.
 */
function bargeIn(cancel: boolean) {
  const playing = live.find((entry) => entry.start <= (playback?.currentTime ?? 0)) ?? live[0]
  // A reply whose start is still held was not heard at all: it is cut at 0 ms.
  const item = playing?.item ?? (hold.holding ? lastDelta : null)
  if (item) {
    const ms = heardMs(item)
    silence()
    cut.add(item)
    sendEvent({ type: 'conversation.item.truncate', item_id: item, content_index: 0, audio_end_ms: ms })
    conversation.interrupted(item)
  } else {
    silence()
  }
  if (cancel) sendEvent({ type: 'response.cancel' })
}

function onServerEvent(event: ServerEvent) {
  if (event.type === 'session.created') {
    // A new session numbers its items from 1 again: what was kept per item belongs to the old one.
    silence()
    hold.reset(true) // and starts at the usual margin
    cut.clear()
    doneMs.clear()
    lastDelta = null
    return
  }
  if (getVoiceState().status !== 'live') return
  const now = Date.now()
  switch (event.type) {
    case 'input_audio_buffer.speech_started':
      lastSpeechAt = now
      if (live.length || hold.holding) bargeIn(false)
      setActivity('hearing')
      break
    case 'input_audio_buffer.speech_stopped':
      setActivity('thinking')
      break
    case 'input_audio_buffer.committed':
      setActivity('thinking')
      break
    case 'response.audio.delta': {
      if (cut.has(event.item_id) || voiceMuted) break
      const samples = pcm16Base64ToFloat(event.delta)
      if (!samples.length) break
      replyOpen = true
      lastDelta = event.item_id
      release(hold.piece(event.item_id, { item: event.item_id, samples }, (samples.length / RATE) * 1_000, now))
      if (hold.holding) holdTick()
      break
    }
    case 'response.audio_transcript.delta':
      hold.written(event.item_id, event.delta.length)
      break
    case 'localrouter.response.text_done':
      release(hold.text(event.item_id, event.chars, event.speech_rate, event.tool_calls, now))
      break
    case 'localrouter.response.speech_done':
      release(hold.speech(event.item_id, now))
      break
    case 'response.done':
      replyOpen = false
      lead = Math.max(MIN_LEAD_S, lead * 0.8)
      if (!live.length) setActivity('listening')
      break
  }
}

onVoiceEvent(onServerEvent)

// ---------------------------------------------------------------- cues

const CUE_REPEAT_MS = 600
let lastCueAt = 0

/** Two short notes through the phone; the echo gate keeps them from the box. */
function playNotes(pitches: [number, number]) {
  if (!playback) return
  const note = Math.floor(RATE * 0.09)
  const samples = new Float32Array(note * 2)
  for (let i = 0; i < samples.length; i++) {
    const at = i % note
    const pitch = i < note ? pitches[0] : pitches[1]
    samples[i] = 0.25 * Math.sin((Math.PI * at) / note) * Math.sin((2 * Math.PI * pitch * at) / RATE)
  }
  const buffer = playback.createBuffer(1, samples.length, RATE)
  buffer.copyToChannel(samples, 0)
  const source = playback.createBufferSource()
  source.buffer = buffer
  source.connect(playback.destination)
  source.start(playback.currentTime)
  schedulePlayed(0, buffer.duration * 1_000, samples)
}

/** A short rising chime when the call is ready (never before, so talking straight after it reaches the box). */
function playListeningCue() {
  if (voiceMuted || Date.now() - lastCueAt < CUE_REPEAT_MS) return
  lastCueAt = Date.now()
  playNotes([880, 1_320])
}

/** A short falling tone: the call could not open. */
const playFailureCue = () => playNotes([660, 440])

// ---------------------------------------------------------------- the recognizer

/** Microphone peak that counts as someone speaking, for the recognizer watchdog. */
const SPEECH_PEAK = 0.05
const RECOGNIZER_STALL_MS = 3_000
const RECOGNIZER_RESTART_GAP_MS = 10_000
const RECOGNIZER_RESTART_PAUSE_MS = 600

/**
 * The on-device recognizer, for as long as voice is on: the wake phrase in standby, and spoken commands in a call
 * (they work over the assistant's speech and while the mic is closed). Without speech permission or on-device
 * recognition the buttons still work, and the wake phrase is off.
 */
async function listenForCommands(phrases: VoiceCommandPhrases) {
  if (!VoiceCommands || !getBox().settings.wakeWord) return false
  let heard = ''
  let unansweredSince: number | null = null
  let lastRestart = 0
  const mine = generation
  const all = [...phrases.mute, ...phrases.muteMic, ...phrases.muteVoice, ...phrases.unmute, ...phrases.endCall]
  const permission = await VoiceCommands.requestPermission().catch(() => 'denied')
  if (permission !== 'granted' || !(await VoiceCommands.start(RECOGNIZER_RATE, all).catch(() => false))) return false
  if (mine !== generation || getVoiceState().status === 'off') {
    VoiceCommands.stop()
    return false
  }
  const subscription = VoiceCommands.addListener('onTranscript', ({ text }) => {
    // One continuous recognition session; the part already acted on is skipped (restarting after each command
    // sometimes left iOS recognizing nothing more).
    unansweredSince = null
    if (!text.startsWith(heard)) heard = ''
    const command = lastCommand(text.slice(heard.length), phrases)
    if (!command) return
    heard = text
    const state = getVoiceState()
    if (command === 'unmute') {
      if (state.status === 'standby') void wake()
      else if (state.status === 'live') {
        setVoiceMuted(false)
        setMicOpen(true)
      }
      return
    }
    if (command === 'mute') return muteAll()
    if (command === 'endCall') return endSession()
    if (state.status !== 'live') return
    if (command === 'muteVoice') setVoiceMuted(true)
    if (command === 'muteMic') setMicOpen(false)
  })
  const errors = VoiceCommands.addListener('onError', ({ domain, code, message }) => console.warn('[voice-commands] recognition error', domain, code, message))
  /** The recognizer can stop producing results; speech with no transcript for a few seconds restarts it. */
  const restart = () => {
    lastRestart = Date.now()
    unansweredSince = null
    heard = ''
    VoiceCommands?.stop()
    setTimeout(() => {
      if (commands === listener && mine === generation) void VoiceCommands?.start(RECOGNIZER_RATE, all).catch(() => false)
    }, RECOGNIZER_RESTART_PAUSE_MS)
  }
  const listener = {
    remove: () => { subscription.remove(); errors.remove(); VoiceCommands?.stop() },
    newStretch: () => { unansweredSince = null },
    audio: (peak: number) => {
      const now = Date.now()
      if (peak >= SPEECH_PEAK) unansweredSince ??= now
      if (unansweredSince !== null && now - unansweredSince >= RECOGNIZER_STALL_MS && now - lastRestart >= RECOGNIZER_RESTART_GAP_MS) restart()
    },
  }
  commands = listener
  return true
}

const recognizerAudio = (samples: Float32Array, sampleRate: number) => pcm16ToBase64(floatToPcm16(samples, sampleRate, RECOGNIZER_RATE))

// ---------------------------------------------------------------- audio sessions

/**
 * The iOS audio session while only listening (standby): no voice processing, which ducks other apps' audio, and no
 * Bluetooth HFP, which drops AirPods to call quality. Music or a video in another app plays on.
 */
const LISTENING_SESSION = { iosCategory: 'playAndRecord', iosMode: 'default', iosOptions: ['defaultToSpeaker', 'allowBluetoothA2DP', 'mixWithOthers'] } as const
/** In a call: voice processing cancels the assistant's own voice (so it does not barge in on itself), and a headset's microphone through HFP. */
const TALKING_SESSION = { iosCategory: 'playAndRecord', iosMode: 'voiceChat', iosOptions: ['defaultToSpeaker', 'allowBluetoothHFP', 'mixWithOthers'] } as const

let talkingSession = false

function setTalkingSession(talking: boolean) {
  if (!recorder || talkingSession === talking) return
  talkingSession = talking
  const session = talking ? TALKING_SESSION : LISTENING_SESSION
  AudioManager.setAudioSessionOptions({ ...session, iosOptions: [...session.iosOptions] })
}

function setChunkSize(samples: number) {
  if (!recorder || chunkSamples === samples) return
  chunkSamples = samples
  recorder.onAudioReady({ sampleRate: RATE, bufferLength: samples, channelCount: 1 }, onMicrophone)
}

function listenForWake() {
  setTalkingSession(false)
  setChunkSize(STANDBY_CHUNK_SAMPLES)
  wakeGate.reset()
  VoiceCommands?.idle()
}

// ---------------------------------------------------------------- the microphone

type MicrophoneEvent = { buffer: { sampleRate: number; getChannelData(channel: number): Float32Array }; numFrames: number }

function onMicrophone(event: MicrophoneEvent) {
  const samples = event.buffer.getChannelData(0).slice(0, event.numFrames)
  const rate = event.buffer.sampleRate
  let peak = 0
  let energy = 0
  for (const sample of samples) {
    peak = Math.max(peak, Math.abs(sample))
    energy += sample * sample
  }
  const rms = Math.sqrt(energy / Math.max(samples.length, 1))
  const state = getVoiceState()
  if (state.status === 'standby') return hearGated(samples, rate, rms, peak)
  const pcm = floatToPcm16(samples, rate, RATE)
  const data = pcm16ToBase64(pcm)
  // In a call the recognizer hears everything (spoken commands); that audio stays on the phone.
  if (commands) {
    VoiceCommands?.append(recognizerAudio(samples, rate))
    commands.audio(peak)
  }
  if (state.status === 'connecting') {
    if (wakeChunks.length < MAX_WAKE_CHUNKS) wakeChunks.push(data)
    return
  }
  if (state.status !== 'live') return
  const now = Date.now()
  if (micOpen && rms >= SPEECH_RMS) lastSpeechAt = now
  if (now - Math.max(lastSpeechAt, playbackEndsAt) >= IDLE_MS) return endSession()
  if (!micOpen) {
    // Closed: silence until the box's wait has ended the turn the user was in, then nothing.
    if (now - closedAt < getBox().settings.waitMs + CLOSE_TAIL_MS) sendEvent({ type: 'input_audio_buffer.append', audio: silenceBase64(pcm.length) })
    return
  }
  const { level, bridged } = echoReference(played, now, { chunkMs: (CHUNK_SAMPLES / RATE) * 1_000, tailMs: ECHO_TAIL_MS, replyOpen })
  const chunks = echoGate.process({ data, rms, now, playLevel: level, bridged }, level > 0 ? silenceBase64(pcm.length) : data)
  // The user is talking over the reply: playback stops here at once, whatever the box does.
  if (echoGate.bargedIn) bargeIn(false)
  outgoing.push(...chunks)
  for (let sent = 0; sent < Math.max(chunks.length, FLUSH_PER_CHUNK) && outgoing.length; sent++) {
    sendEvent({ type: 'input_audio_buffer.append', audio: outgoing.shift()! })
  }
}

/** Standby: the recognizer hears only what the gate lets through (sound, with the chunks just before it). */
function hearGated(samples: Float32Array, rate: number, rms: number, peak: number) {
  if (!commands) return
  const wasOpen = wakeGate.isOpen
  const { feed, closed } = wakeGate.process(samples, rms)
  if (!wasOpen && feed.length) commands.newStretch()
  for (const chunk of feed) VoiceCommands?.append(recognizerAudio(chunk, rate))
  if (feed.length) commands.audio(peak)
  if (closed) {
    VoiceCommands?.idle()
    commands.newStretch()
  }
}

// ---------------------------------------------------------------- standby

function standby(muted: boolean, error: string | null) {
  return { status: 'standby' as const, muted, error, wakePhrase: commands ? commandPhrases.unmute[0] ?? null : null }
}

function resetCall() {
  micOpen = false
  voiceMuted = false
  wakeChunks = []
  outgoing = []
  activity = 'listening'
  silence()
  cut.clear()
  doneMs.clear()
  echoGate.reset()
  listenForWake()
}

CallSession?.addListener('onHeadsetMuteGesture', () => {
  const state = getVoiceState()
  if (state.status === 'live') muteAll()
  else if (state.status === 'standby') void wake()
})

/**
 * Starts standby: the microphone and the on-device recognizer. Asks for microphone permission the first time.
 * Must be called while the app is on screen (neither platform lets recording start from the background).
 * `background`: keep listening once the app is in the background (Android runs a microphone foreground service).
 */
export function startStandby(options: { background?: boolean } = {}) {
  if (recorder) return Promise.resolve()
  opening ??= openStandby(options.background ?? false).finally(() => { opening = null })
  return opening
}

async function openStandby(background: boolean) {
  const mine = ++generation
  try {
    const permission = await waitFor(AudioManager.requestRecordingPermissions(), 'Microphone permission did not respond. Close and reopen the app, then try again.', 30_000)
    if (permission !== 'Granted') throw new Error('Microphone access is off. Enable it in Settings to talk.')
    AudioManager.setAudioSessionOptions({ ...LISTENING_SESSION, iosOptions: [...LISTENING_SESSION.iosOptions] })
    await waitFor(AudioManager.setAudioSessionActivity(true), 'Microphone setup did not respond. Close and reopen the app, then try again.')
    if (mine !== generation) return
    // Playback starts before capture and stays running, so a reply never reconfigures the engine while recording.
    playback = new AudioContext({ sampleRate: RATE })
    await waitFor(playback.resume(), 'Audio playback did not start. Try again.')
    const starting = new AudioRecorder()
    starting.onAudioReady({ sampleRate: RATE, bufferLength: STANDBY_CHUNK_SAMPLES, channelCount: 1 }, onMicrophone)
    const started = await waitFor(starting.start(), 'Audio recording did not start. Try again.')
    if (started.status === 'error') throw new Error(started.message)
    if (mine !== generation) {
      await starting.stop().catch(() => undefined)
      return
    }
    recorder = starting
    chunkSamples = STANDBY_CHUNK_SAMPLES
    talkingSession = false
    wakeGate.reset()
    listeningInBackground = background
    if (background && Platform.OS === 'android') await CallSession?.startStandby?.().catch(() => undefined)
    commandPhrases = voiceCommandPhrases(getBox().assistant.name)
    setState(standby(false, null))
    if (await listenForCommands(commandPhrases)) {
      const state = getVoiceState()
      if (state.status === 'standby') setState(standby(state.muted, state.error))
    }
  } catch (error) {
    await teardown()
    setState({ status: 'off', error: error instanceof Error ? error.message : 'Could not start voice' })
  }
}

/** Turns voice off on this phone: any call ends, and the microphone and recognizer stop. */
export async function stopStandby(error: string | null = null) {
  if (getVoiceState().status === 'live' || getVoiceState().status === 'connecting') setCall(false)
  resetCall()
  generation++
  await teardown()
  setState({ status: 'off', error })
}

async function teardown() {
  commands?.remove()
  commands = null
  const oldRecorder = recorder
  const oldPlayback = playback
  recorder = null
  playback = null
  chunkSamples = 0
  const cleanup = async () => {
    if (oldRecorder) {
      await oldRecorder.stop().catch(() => undefined)
      oldRecorder.clearOnAudioReady()
    }
    await oldPlayback?.close().catch(() => undefined)
    await AudioManager.setAudioSessionActivity(false).catch(() => undefined)
    if (listeningInBackground && Platform.OS === 'android') await CallSession?.stopStandby?.().catch(() => undefined)
    listeningInBackground = false
  }
  await waitFor(cleanup(), 'Audio cleanup timed out', 2_000).catch(() => undefined)
}

// ---------------------------------------------------------------- calls

function waitForSession(timeoutMs: number) {
  return new Promise<boolean>((resolve) => {
    const started = Date.now()
    const check = () => {
      const connection = getBox().connection
      if (connection === 'open') return resolve(true)
      if (connection === 'failed' || Date.now() - started > timeoutMs || getVoiceState().status !== 'connecting') return resolve(false)
      setTimeout(check, 100)
    }
    check()
  })
}

/** The wake phrase or Talk: opens a call. */
export async function wake() {
  const state = getVoiceState()
  if (state.status !== 'standby') return
  if (getBox().status !== 'paired') {
    playFailureCue()
    return setState(standby(state.muted, 'Pair with your box first.'))
  }
  wakeChunks = []
  outgoing = []
  setChunkSize(CHUNK_SAMPLES)
  setState({ status: 'connecting' })
  setCall(true)
  if (!(await waitForSession(CONNECT_TIMEOUT_MS))) {
    if (getVoiceState().status !== 'connecting') return
    setCall(false)
    resetCall()
    playFailureCue()
    return setState(standby(false, "Couldn't reach your box. Check that Tailscale is on."))
  }
  if (getVoiceState().status !== 'connecting') return
  // What the user said while the call opened goes first, then live audio (onMicrophone paces it).
  outgoing = wakeChunks
  wakeChunks = []
  micOpen = true
  voiceMuted = false
  lastSpeechAt = Date.now()
  lastCueAt = 0
  activity = 'listening'
  setTalkingSession(true)
  setState({ status: 'live', micOpen, voiceMuted, activity: callActivity() })
  playListeningCue()
}

/** "End call" / "hang up", or End: the session goes back to text, and the phone to listening for the wake phrase. */
export function endSession() {
  const state = getVoiceState()
  if (state.status === 'off') return
  if (state.status === 'live' || state.status === 'connecting') setCall(false)
  resetCall()
  setState(standby(false, null))
}

/** Tapping the orb while the box thinks or speaks: the reply stops. */
export function interrupt() {
  if (getVoiceState().status !== 'live') return
  bargeIn(true)
  setActivity('listening')
}

/** "Mute audio": the call ends and standby is muted until the wake phrase or Talk. */
export function muteAll() {
  endSession()
  setState(standby(true, null))
}

/** Opens or closes the microphone in a call (mute, or hold to talk). */
export function setMicOpen(open: boolean, options: { quiet?: boolean } = {}) {
  const state = getVoiceState()
  if (state.status !== 'live' || state.micOpen === open) return
  if (open) lastSpeechAt = Date.now()
  else closedAt = Date.now()
  micOpen = open
  setState({ ...state, micOpen: open })
  if (open && !options.quiet) playListeningCue()
}

/** Mutes or unmutes the assistant's voice on this phone; replies still arrive as text. */
export function setVoiceMuted(muted: boolean) {
  const state = getVoiceState()
  if (state.status !== 'live' || state.voiceMuted === muted) return
  voiceMuted = muted
  if (muted) silence()
  setState({ ...state, voiceMuted: muted })
  if (!muted) playListeningCue()
}
