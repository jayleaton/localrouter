import { useSyncExternalStore } from 'react'
import { AppState, Platform } from 'react-native'
import { fetch as expoFetch } from 'expo/fetch'

import { normalizeBaseUrl, normalizeCode } from '@/lib/localrouter/address'
import { defaultAssistant, fetchAssistant, saveAssistantName, type Assistant } from '@/lib/localrouter/assistant'
import { ApiError, createApi, speechEngines, type Api, type Fetch, type SpeechEngine } from '@/lib/localrouter/api'
import { Conversation, GENERATION_LIMITS, isGeneration } from '@/lib/localrouter/conversation'
import { refusedVoice, sessionUpdate, type ClientEvent, type ServerEvent } from '@/lib/localrouter/protocol'
import { RealtimeClient, type ConnectionStatus, type SocketFactory } from '@/lib/localrouter/realtime'

import { assistantStorage, conversationStorage, defaultSettings, pairingStorage, settingsStorage, type Pairing, type Settings } from './storage'

/*
 * The box this phone talks to: its pairing, the realtime session, the conversation and this phone's settings. One
 * session carries both chat and voice. Without a call it is text only; a call (the voice engine) switches it to
 * audio with server VAD, so typed messages in a call get spoken replies, and the same server-side tools make
 * pictures and videos for both.
 */

export type BoxState = {
  status: 'loading' | 'unpaired' | 'paired'
  pairing: Pairing | null
  connection: ConnectionStatus
  /** A voice model was unloaded to make room for a job (`localrouter.voice.paused`); it comes back by itself. */
  paused: boolean
  /** The latest thing to tell the user (an error from the box, a refused voice), shown until dismissed. */
  notice: string | null
  engines: SpeechEngine[]
  /** The assistant's name (a setting on the box, the last one it said while offline) and its default voice. */
  assistant: Assistant
  settings: Settings
  /** The session is in a call (audio and server VAD). */
  call: boolean
}

let state: BoxState = { status: 'loading', pairing: null, connection: 'closed', paused: false, notice: null, engines: [], assistant: defaultAssistant(), settings: defaultSettings(), call: false }
const listeners = new Set<() => void>()
const voiceListeners = new Set<(event: ServerEvent) => void>()

function set(patch: Partial<BoxState>) {
  state = { ...state, ...patch }
  for (const listener of listeners) listener()
}

export const getBox = () => state

/** Box state changes, for code outside React (the standby controller). */
export function subscribeBox(listener: () => void) {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

export function useBox<T>(select: (state: BoxState) => T): T {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener)
      return () => listeners.delete(listener)
    },
    () => select(state)
  )
}

/** The assistant's name, as the box says it ("Jarvis" until it says otherwise). */
export const useAssistantName = () => useBox((box) => box.assistant.name)

/** The conversation; its file URLs are rebased onto the paired box's address. */
export const conversation = new Conversation((url) => api?.fileUrl(url) ?? url)

export function useConversation() {
  return useSyncExternalStore(
    (listener) => conversation.subscribe(listener),
    () => conversation.messages
  )
}

let api: Api | null = null
let client: RealtimeClient | null = null
let saveTimer: ReturnType<typeof setTimeout> | null = null

const fetchJson: Fetch = (input, init) => expoFetch(input, init)

/** The token goes as a subprotocol entry (see realtime.ts). React Native's event types are wider than `SocketLike`'s. */
const socket: SocketFactory = (url, protocols) => new WebSocket(url, protocols) as unknown as ReturnType<SocketFactory>

/** The server events the voice engine needs (audio, VAD, the reply's text and speech ends). */
export function onVoiceEvent(listener: (event: ServerEvent) => void) {
  voiceListeners.add(listener)
  return () => {
    voiceListeners.delete(listener)
  }
}

export const sendEvent = (event: ClientEvent) => client?.send(event) ?? false

function save() {
  if (saveTimer) return
  saveTimer = setTimeout(() => {
    saveTimer = null
    conversationStorage.write(conversation.toJSON())
  }, 500)
}

/** The session's settings: audio and the wait only in a call; the voice only when its model lists it. */
function settingsEvent(): ClientEvent {
  const { settings, engines, call } = state
  const engine = engines.find((e) => e.id === settings.speechModel) ?? engines.find((e) => e.isDefault) ?? engines[0]
  const voice = engine ? settings.voices[engine.id] : undefined
  return sessionUpdate({
    audio: call,
    waitMs: settings.waitMs,
    threshold: settings.threshold,
    speechModel: settings.speechModel || undefined,
    voice: engine && voice && engine.voices.some((v) => v.id === voice) ? voice : undefined,
    instructions: settings.instructions || undefined,
  })
}

function onEvent(event: ServerEvent) {
  conversation.apply(event)
  if (event.type === 'localrouter.voice.paused') set({ paused: true })
  else if (event.type === 'localrouter.voice.resumed') set({ paused: false })
  else if (event.type === 'error') {
    const refused = refusedVoice(event.error.message)
    if (refused) {
      // A voice the box no longer has: the default instead, from now on.
      const voices = { ...state.settings.voices }
      if (voices[refused.model] === refused.voice) delete voices[refused.model]
      updateSettings({ voices, ...(state.settings.speechModel === refused.model ? { speechModel: '' } : {}) })
      set({ notice: `The voice “${refused.voice}” is no longer available; using the default.` })
    } else {
      set({ notice: event.error.message })
    }
  }
  for (const listener of voiceListeners) listener(event)
  save()
}

function connect() {
  const pairing = state.pairing
  if (!pairing || client) return
  client = new RealtimeClient({
    baseUrl: pairing.baseUrl,
    token: pairing.token,
    socket,
    onEvent,
    onStatus: (connection, detail) => {
      set({ connection, ...(connection === 'failed' ? { notice: `Lost the connection to the box${detail.reason ? `: ${detail.reason}` : ''}. Pull to retry.` } : {}) })
    },
    onOpen: () => {
      conversation.newConnection()
      return [settingsEvent(), ...conversation.seed()]
    },
  })
  client.connect()
}

function disconnect() {
  client?.close()
  client = null
}

/** Reconnects now (after a failure, or once the app is back on screen). */
export function reconnect() {
  disconnect()
  if (state.status === 'paired') connect()
}

async function loadEngines() {
  if (!api) return
  try {
    const engines = speechEngines(await api.models())
    // Choices the box no longer lists (a removed model or voice) are dropped, or every sentence would fail.
    const voices = Object.fromEntries(Object.entries(state.settings.voices).filter(([model, voice]) => engines.find((e) => e.id === model)?.voices.some((v) => v.id === voice)))
    const speechModel = engines.some((e) => e.id === state.settings.speechModel) ? state.settings.speechModel : ''
    set({ engines })
    if (speechModel !== state.settings.speechModel || Object.keys(voices).length !== Object.keys(state.settings.voices).length) updateSettings({ voices, speechModel })
    else client?.send(settingsEvent())
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) set({ notice: 'The box no longer knows this phone. Pair it again in Settings.' })
  }
}

function setAssistant(assistant: Assistant) {
  set({ assistant })
  assistantStorage.write(assistant)
}

/** The assistant's name from the box (an older box: the default; out of reach: the one already known). */
export async function loadAssistant() {
  const asked = api
  if (!asked) return state.assistant
  const assistant = await fetchAssistant(asked, state.assistant)
  // A box forgotten or replaced meanwhile keeps its own name.
  if (asked === api && JSON.stringify(assistant) !== JSON.stringify(state.assistant)) setAssistant(assistant)
  return assistant
}

/**
 * Renames the assistant on the box (an empty name restores the default). Throws what to tell the user: a refused
 * name, a phone without the `manage` permission, an older box, or no connection.
 */
export async function renameAssistant(name: string) {
  if (!api) throw new Error('No box is paired.')
  const assistant = await saveAssistantName(api, name)
  setAssistant(assistant)
  return assistant
}

/** Restores the pairing, settings and conversation; connects if paired. Call once at start. */
export async function initialize() {
  const [pairing, settings, saved, assistant] = await Promise.all([pairingStorage.read(), settingsStorage.read(), conversationStorage.read(), assistantStorage.read()])
  conversation.load(saved)
  set({ settings, pairing, assistant, status: pairing ? 'paired' : 'unpaired' })
  if (pairing) {
    api = createApi({ baseUrl: pairing.baseUrl, token: pairing.token, fetch: fetchJson })
    connect()
    void loadEngines()
    void loadAssistant()
  }
  AppState.addEventListener('change', (next) => {
    if (state.status !== 'paired') return
    if (next === 'active') {
      if (!client || client.status === 'failed') reconnect()
      // Renamed on the box or from another device while this phone was away.
      void loadAssistant()
    } else if (!state.call) {
      // Off screen without a call: nothing to listen to; the conversation is reseeded on return.
      disconnect()
    }
  })
  setInterval(watchGenerations, 10_000)
}

/**
 * Pairs with a box: `code` from `localrouter pair add` on it. A box that does not require pairing yet still issues
 * a token, so the phone keeps working once the operator turns pairing on. Without a code the phone connects
 * unpaired, which only works while the box does not require pairing.
 */
export async function pair(baseUrlInput: string, codeInput: string, deviceName: string) {
  const baseUrl = normalizeBaseUrl(baseUrlInput)
  if (!baseUrl) throw new Error('That is not a web address. Use the one your box is reached at, e.g. https://box.tailnet.ts.net:10443')
  let pairing: Pairing
  if (codeInput.trim()) {
    const code = normalizeCode(codeInput)
    if (!code) throw new Error('A pairing code is 8 letters and digits, like K7QM-3XPD.')
    const result = await createApi({ baseUrl, fetch: fetchJson }).pair(code, deviceName || Platform.OS)
    pairing = { baseUrl, token: result.token, deviceId: result.device.id, deviceName: result.device.name }
  } else {
    const status = await createApi({ baseUrl, fetch: fetchJson }).pairStatus()
    if (status.required) throw new Error('This box requires pairing. Run `localrouter pair add "My phone"` on it and enter the code.')
    pairing = { baseUrl, token: null, deviceId: '', deviceName: '' }
  }
  await pairingStorage.write(pairing)
  disconnect()
  api = createApi({ baseUrl, token: pairing.token, fetch: fetchJson })
  // Another box has its own name: the default until it answers.
  set({ pairing, status: 'paired', notice: null, ...(state.pairing?.baseUrl !== baseUrl ? { assistant: defaultAssistant() } : {}) })
  connect()
  void loadEngines()
  // The setup step after pairing shows (and may change) the box's name, so it is known before pairing returns.
  await loadAssistant()
}

/** Forgets the box (the token is deleted from the keychain; the operator revokes it with `localrouter pair revoke`). */
export async function forget() {
  disconnect()
  api = null
  await pairingStorage.remove()
  assistantStorage.remove()
  set({ pairing: null, status: 'unpaired', engines: [], assistant: defaultAssistant() })
}

export function sendText(text: string) {
  const trimmed = text.trim()
  if (!trimmed) return
  if (!client) connect()
  const events = conversation.typed(trimmed)
  if (!client?.isOpen) {
    // The reconnect reseeds the conversation, typed message included, but asks for no reply: ask once it is open.
    set({ notice: 'Not connected to the box yet; it will answer once it is.' })
    const stop = setInterval(() => {
      if (!client) return clearInterval(stop)
      if (client.isOpen) {
        clearInterval(stop)
        client.send({ type: 'response.create' })
      }
    }, 500)
    return
  }
  for (const event of events) client.send(event)
  save()
}

/** The voice engine starts or ends a call: the same session switches between audio and text. */
export function setCall(call: boolean) {
  if (state.call === call) return
  set({ call })
  if (call && !client) connect()
  client?.send(settingsEvent())
}

export function updateSettings(patch: Partial<Settings>) {
  const settings = { ...state.settings, ...patch }
  set({ settings })
  settingsStorage.write(settings)
  client?.send(settingsEvent())
}

export const dismissNotice = () => set({ notice: null })

export function newConversation() {
  conversation.clear()
  save()
  reconnect()
}

/** Makes a failed picture or video again over HTTP, with the same prompt. */
export async function retryGeneration(id: string) {
  const m = conversation.messages.find((x) => x.id === id)
  if (!m || !isGeneration(m) || !api) return
  conversation.retrying(id)
  try {
    if (m.kind === 'image') conversation.finish(id, { status: 'completed', urls: await api.image(m.prompt) })
    else {
      const jobId = await api.video(m.prompt)
      const started = Date.now()
      while (Date.now() - started < GENERATION_LIMITS.videoMs) {
        await new Promise((resolve) => setTimeout(resolve, 3_000))
        const job = await api.job(jobId)
        if (job.status === 'completed') return conversation.finish(id, { status: 'completed', urls: job.urls })
        if (job.status === 'failed') return conversation.finish(id, { status: 'failed', error: job.error ?? 'failed' })
      }
      conversation.finish(id, { status: 'failed', error: 'the video took too long' })
    }
  } catch (error) {
    conversation.finish(id, { status: 'failed', error: error instanceof Error ? error.message : 'failed' })
  } finally {
    save()
  }
}

/**
 * Generations never spin for ever: a late `job.done` (the socket was replaced, the phone slept) is asked of the job
 * itself, and a job that never starts, or makes no progress, fails with its reason.
 */
async function watchGenerations() {
  if (!api) return
  const now = Date.now()
  for (const m of conversation.messages) {
    if (!isGeneration(m) || m.status !== 'working') continue
    if (!m.jobId) {
      if (now - m.touchedAt > GENERATION_LIMITS.noJobMs) conversation.finish(m.id, { status: 'failed', error: 'the job never started' })
      continue
    }
    if (now - m.at > (m.kind === 'video' ? GENERATION_LIMITS.videoMs : GENERATION_LIMITS.imageMs)) {
      conversation.finish(m.id, { status: 'failed', error: 'it took too long' })
      continue
    }
    if (now - m.touchedAt < GENERATION_LIMITS.pollAfterMs) continue
    try {
      const job = await api.job(m.jobId)
      if (job.status === 'completed') conversation.finish(m.id, { status: 'completed', urls: job.urls })
      else if (job.status === 'failed') conversation.finish(m.id, { status: 'failed', error: job.error ?? 'failed' })
      else conversation.touched(m.id)
    } catch (error) {
      if (error instanceof ApiError && error.status === 404) conversation.finish(m.id, { status: 'failed', error: 'the box no longer has this job' })
    }
  }
  save()
}
