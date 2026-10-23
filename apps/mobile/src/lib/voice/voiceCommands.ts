/** Phrases the phone hears on-device (never sent anywhere): the wake phrase and the call's spoken controls. */
export type VoiceCommandPhrases = {
  /** Mutes both the microphone and the assistant's voice. */
  mute: string[]
  /** Mutes only the microphone. */
  muteMic: string[]
  /** Mutes only the assistant's voice on the phone: replies still arrive as text. */
  muteVoice: string[]
  /** The wake phrase: opens a call from standby; during one, unmutes everything. */
  unmute: string[]
  endCall: string[]
}
export type VoiceCommand = keyof VoiceCommandPhrases

export const DEFAULT_VOICE_COMMAND_PHRASES: VoiceCommandPhrases = {
  mute: ['mute audio', 'mute the audio'],
  muteMic: ['mute mic', 'mute microphone', 'mute my mic'],
  muteVoice: ['mute voice', 'mute speaker'],
  unmute: ['hey jarvis'],
  endCall: ['end call', 'end the call', 'hang up'],
}

const words = (text: string) => ` ${text.toLowerCase().replace(/[^a-z'\s]/g, ' ').replace(/\s+/g, ' ').trim()} `

/**
 * The most recent spoken command in an on-device transcript, if any. Phrases
 * match as whole words ("mute audio" is not inside "unmute audio"); when two
 * end at the same place, the longer one wins.
 */
export function lastCommand(transcript: string, phrases: VoiceCommandPhrases): VoiceCommand | null {
  const heard = words(transcript)
  let best: { end: number; length: number; command: VoiceCommand } | null = null
  for (const command of ['mute', 'muteMic', 'muteVoice', 'unmute', 'endCall'] as const) {
    for (const phrase of phrases[command]) {
      const needle = words(phrase)
      const at = heard.lastIndexOf(needle)
      if (at < 0) continue
      const end = at + needle.length
      if (!best || end > best.end || (end === best.end && needle.length > best.length)) best = { end, length: needle.length, command }
    }
  }
  return best?.command ?? null
}
