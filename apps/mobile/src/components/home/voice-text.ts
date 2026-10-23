import type { VoiceState } from '@/lib/voice/voiceState'

/** "hey jarvis" → "Hey Jarvis". */
const title = (phrase: string) => phrase.replace(/\b\w/g, (letter) => letter.toUpperCase())

/** What a call is doing, in a word or two: the line under the voice wave and in the lightbox's call pill. */
export function callLine(voice: VoiceState) {
  if (voice.status === 'connecting') return 'Connecting…'
  if (voice.status !== 'live') return ''
  if (!voice.micOpen) return 'Muted'
  switch (voice.activity) {
    case 'hearing':
      return 'Hearing you…'
    case 'thinking':
      return 'Thinking…'
    case 'speaking':
      return 'Speaking'
    default:
      return 'Listening'
  }
}

/** What voice is doing outside a call, for the hint above the dock. */
export function voiceHint(voice: VoiceState, enabled: boolean) {
  if (voice.status === 'connecting' || voice.status === 'live') return callLine(voice)
  if (voice.status === 'standby') {
    const say = voice.wakePhrase ? `Say “${title(voice.wakePhrase)}” or tap Talk` : 'Tap Talk'
    if (voice.muted) return `Muted. ${say} when you want me again.`
    return voice.error ?? `${say} to start a call.`
  }
  if (voice.error) return voice.error
  return enabled ? 'Starting voice…' : 'Turn on voice to talk hands-free.'
}

/** Thinking or speaking: tapping the wave (or Interrupt) stops the reply. */
export const canInterrupt = (voice: VoiceState) => voice.status === 'live' && (voice.activity === 'thinking' || voice.activity === 'speaking')
