import { DEFAULT_ASSISTANT_NAME } from '@/lib/localrouter/assistant'

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
  /**
   * The assistant's name: a greeting and the name ("hey Friday", "okay Friday") also wakes it, with the name heard
   * a letter off (see `wakeAt`). The `unmute` phrases are matched exactly.
   */
  wakeName?: string
}
export type VoiceCommand = Exclude<keyof VoiceCommandPhrases, 'wakeName'>

/** The spoken controls, and "hey <name>" for the assistant's name. */
export function voiceCommandPhrases(name: string): VoiceCommandPhrases {
  return {
    mute: ['mute audio', 'mute the audio'],
    muteMic: ['mute mic', 'mute microphone', 'mute my mic'],
    muteVoice: ['mute voice', 'mute speaker'],
    unmute: [`hey ${name}`],
    endCall: ['end call', 'end the call', 'hang up'],
    wakeName: name,
  }
}

export const DEFAULT_VOICE_COMMAND_PHRASES: VoiceCommandPhrases = voiceCommandPhrases(DEFAULT_ASSISTANT_NAME)

/** Lower case, accents dropped, anything but letters, digits and apostrophes a space, padded with one space each side. */
const words = (text: string) =>
  ` ${text.normalize('NFD').replace(/\p{M}/gu, '').toLowerCase().replace(/[^\p{L}\p{N}'\s]/gu, ' ').replace(/\s+/g, ' ').trim()} `

/** What the recognizer writes before the name. "Hay" is how it often hears "hey". */
const GREETINGS = new Set(['hey', 'hay', 'hi', 'okay', 'ok'])
/** A name this long or longer may be heard one letter off (added, dropped or changed); a shorter one must be exact. */
const FUZZY_FROM = 4

function editDistance(a: string, b: string) {
  let previous = Array.from({ length: b.length + 1 }, (_, i) => i)
  for (let i = 1; i <= a.length; i++) {
    const current = [i]
    for (let j = 1; j <= b.length; j++) current[j] = Math.min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1))
    previous = current
  }
  return previous[b.length]
}

/**
 * Where the last greeting-and-name in `heard` (as `words` makes it) ends, in the same terms as an exact phrase, and
 * how long it is. The name is compared without its spaces, apostrophes, hyphens and periods, over one more word than
 * it has ("Mary Jane" heard as "maryjane", "Jarvis" as "jar vis"); the closest reading wins.
 */
function wakeAt(heard: string, name: string): { end: number; length: number } | null {
  const target = words(name).replace(/[\s']/g, '')
  if (!target) return null
  const allowed = [...target].length >= FUZZY_FROM ? 1 : 0
  const span = words(name).trim().split(' ').length + 1
  const tokens: { text: string; start: number }[] = []
  for (let at = 1; at < heard.length - 1; ) {
    const next = heard.indexOf(' ', at)
    tokens.push({ text: heard.slice(at, next), start: at })
    at = next + 1
  }
  for (let i = tokens.length - 2; i >= 0; i--) {
    if (!GREETINGS.has(tokens[i].text)) continue
    let best: { distance: number; last: number } | null = null
    let joined = ''
    for (let k = i + 1; k < tokens.length && k <= i + span; k++) {
      joined += tokens[k].text.replace(/'/g, '')
      const distance = editDistance(joined, target)
      if (distance <= allowed && (!best || distance < best.distance)) best = { distance, last: k }
    }
    if (best) {
      const last = tokens[best.last]
      const end = last.start + last.text.length + 1
      return { end, length: end - (tokens[i].start - 1) }
    }
  }
  return null
}

/**
 * The most recent spoken command in an on-device transcript, if any. Phrases
 * match as whole words ("mute audio" is not inside "unmute audio"); when two
 * end at the same place, the longer one wins.
 */
export function lastCommand(transcript: string, phrases: VoiceCommandPhrases): VoiceCommand | null {
  const heard = words(transcript)
  const found: { end: number; length: number; command: VoiceCommand }[] = []
  for (const command of ['mute', 'muteMic', 'muteVoice', 'unmute', 'endCall'] as const) {
    for (const phrase of phrases[command]) {
      const needle = words(phrase)
      const at = heard.lastIndexOf(needle)
      if (at >= 0) found.push({ end: at + needle.length, length: needle.length, command })
    }
  }
  const wake = phrases.wakeName ? wakeAt(heard, phrases.wakeName) : null
  if (wake) found.push({ ...wake, command: 'unmute' })
  let best: (typeof found)[number] | null = null
  for (const f of found) if (!best || f.end > best.end || (f.end === best.end && f.length > best.length)) best = f
  return best?.command ?? null
}
