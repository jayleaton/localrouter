/**
 * The assistant's name: a setting on the Spark (`GET`/`PATCH /v1/assistant`), the one place it is kept, so every
 * device paired with the box shows and speaks the same name. "Jarvis" until someone changes it, and on a box from
 * before the setting. The box checks a new name; the rules are mirrored here only to say what is wrong before asking.
 */
import { DEFAULT_ASSISTANT } from '../../../desktop-helper/src/assistant.ts'

export { DEFAULT_ASSISTANT }

/** The box's default voice for its voice loop: a hint for Settings only. */
export type AssistantVoice = { model: string; voice: string }
export type Assistant = { name: string; default_name: string; custom: boolean; voice: AssistantVoice | null }

export const MAX_ASSISTANT_NAME = 32

/** `/v1/assistant`'s object, with anything missing or malformed as the default. */
export function assistantOf(j: unknown): Assistant {
  const o = (j && typeof j === 'object' ? j : {}) as { name?: unknown; default_name?: unknown; custom?: unknown; voice?: unknown }
  const text = (v: unknown) => (typeof v === 'string' && v.trim() ? v.trim() : '')
  const fallback = text(o.default_name) || DEFAULT_ASSISTANT
  const v = (o.voice && typeof o.voice === 'object' ? o.voice : null) as { model?: unknown; voice?: unknown } | null
  const voice = v && (text(v.voice) || text(v.model)) ? { model: text(v.model), voice: text(v.voice) } : null
  return { name: text(o.name) || fallback, default_name: fallback, custom: o.custom === true, voice }
}

/** A typed name as the box keeps it: trimmed, runs of spaces as one. */
export function tidyAssistantName(raw: string): string {
  return raw.trim().replace(/\s+/g, ' ')
}

/**
 * Why the box would refuse `raw` as the assistant's name, or null when it reads as one (empty: back to the default).
 * 1 to 32 characters: letters (any script), digits, spaces, apostrophes, hyphens and periods, one letter or digit at least.
 */
export function assistantNameProblem(raw: string): string | null {
  const name = tidyAssistantName(raw)
  if (!name) return null
  if ([...name].length > MAX_ASSISTANT_NAME) return `A name has at most ${MAX_ASSISTANT_NAME} characters.`
  if (!/^[\p{L}\p{M}\p{N} '.-]+$/u.test(name)) return 'A name uses letters, digits, spaces, apostrophes, hyphens and periods only.'
  if (!/[\p{L}\p{N}]/u.test(name)) return 'A name needs at least one letter or digit.'
  return null
}

/** "Speaks as default (fishaudio/drama-3)": the box's default voice, as Settings shows it. */
export function voiceHint(voice: AssistantVoice | null): string {
  if (!voice) return ''
  return `Speaks as ${[voice.voice, voice.model && `(${voice.model})`].filter(Boolean).join(' ')}`
}
