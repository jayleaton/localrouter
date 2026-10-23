import { useSyncExternalStore } from 'react'

/**
 * Voice on this phone.
 *
 * - `off`: not listening (voice turned off, no mic permission, or the app is in the background without background
 *   listening). `error` says why when it was a failure.
 * - `standby`: listening on this phone for the wake phrase with the on-device recognizer. Nothing leaves the phone.
 *   `muted`: "mute audio" silenced it until the next wake phrase. `wakePhrase`: what to say ("hey jarvis");
 *   null when the wake phrase is off (or the recognizer is unavailable) and only Talk opens a call.
 * - `connecting`: a call is opening; what the user says meanwhile is kept for it.
 * - `live`: a call. `micOpen` is the mic gate (mute, hold-to-talk); `voiceMuted`: the assistant's voice is not
 *   played (replies still arrive as text).
 */
export type VoiceState =
  | { status: 'off'; error: string | null }
  | { status: 'standby'; muted: boolean; error: string | null; wakePhrase: string | null }
  | { status: 'connecting' }
  | { status: 'live'; micOpen: boolean; voiceMuted: boolean; activity: CallActivity }

/** What the call is doing, for the orb and the hint: the user talking, the box thinking, the reply playing. */
export type CallActivity = 'listening' | 'hearing' | 'thinking' | 'speaking'

let state: VoiceState = { status: 'off', error: null }
const listeners = new Set<() => void>()

export const getVoiceState = () => state

export function subscribeVoice(listener: () => void) {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

export function setVoiceState(next: VoiceState) {
  state = next
  for (const listener of listeners) listener()
}

/** Voice state without loading the native audio engine (safe in Expo Go). */
export function useVoice() {
  return useSyncExternalStore(subscribeVoice, () => state)
}

/** Loads the native audio engine on first use. Expo Go lacks the audio module, so voice there needs a development build. */
export async function loadVoiceEngine() {
  try {
    return await import('./voiceEngine')
  } catch {
    setVoiceState({ status: 'off', error: 'Voice needs the development build of the app.' })
    return null
  }
}
