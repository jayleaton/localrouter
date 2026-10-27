import { z } from 'zod'

import { ApiError, type Api } from './api'

/*
 * The assistant's name, a setting on the box (`/v1/assistant`): the box is the one place it is kept, and every client
 * shows and speaks with it. The phone keeps the last name it was told, so it shows offline too. A box older than the
 * setting has no route for it: the name is then the default and cannot be changed from the phone.
 */

export const DEFAULT_ASSISTANT_NAME = 'Jarvis'
export const ASSISTANT_NAME_MAX = 32

/**
 * What the phone knows of the assistant. `voice`: the box's default voice for the voice loop (a hint in Settings).
 * `supported`: the box has the setting (an older one does not, and the name stays the default).
 */
export const assistantSchema = z.object({
  name: z.string().min(1).default(DEFAULT_ASSISTANT_NAME),
  defaultName: z.string().min(1).default(DEFAULT_ASSISTANT_NAME),
  custom: z.boolean().default(false),
  voice: z.object({ model: z.string(), voice: z.string() }).nullable().default(null),
  supported: z.boolean().default(true),
})
export type Assistant = z.infer<typeof assistantSchema>

export const defaultAssistant = (): Assistant => assistantSchema.parse({})

/** A cached assistant, or the default for anything unreadable. */
export function parseAssistant(raw: unknown): Assistant {
  const parsed = assistantSchema.safeParse(raw ?? {})
  return parsed.success ? parsed.data : defaultAssistant()
}

/** A name as the box keeps it: trimmed, runs of spaces made one (and a typed curly apostrophe a plain one). */
export const normalizeAssistantName = (input: string) => input.normalize('NFC').replace(/’/g, "'").trim().replace(/\s+/g, ' ')

/**
 * Why the box would refuse a name, or null (the box checks again; this only saves a round trip). 1 to 32
 * characters of letters (any alphabet), digits, spaces, apostrophes, hyphens and periods, with a letter or digit.
 */
export function assistantNameError(input: string): string | null {
  const name = normalizeAssistantName(input)
  if (!name) return 'Give your assistant a name.'
  if ([...name].length > ASSISTANT_NAME_MAX) return `A name is at most ${ASSISTANT_NAME_MAX} characters.`
  if (!/^[\p{L}\p{M}0-9 '.-]+$/u.test(name)) return 'Use letters, digits, spaces, apostrophes, hyphens or periods.'
  if (!/[\p{L}0-9]/u.test(name)) return 'A name needs at least one letter or digit.'
  return null
}

/** The box has no assistant setting: an older box answers 404, or 403 for a route it does not know. */
const olderBox = (error: unknown) => error instanceof ApiError && (error.status === 404 || error.status === 403)

/**
 * The assistant as the box says now. An older box gives the default (not editable); a box that cannot be reached
 * keeps what the phone knew (`cached`).
 */
export async function fetchAssistant(api: Pick<Api, 'assistant'>, cached: Assistant): Promise<Assistant> {
  try {
    return await api.assistant()
  } catch (error) {
    if (olderBox(error)) return { ...defaultAssistant(), supported: false }
    return cached
  }
}

/**
 * Renames the assistant on the box (an empty name restores the default) and returns what the box now says. Throws
 * an error to show the user: a refused name, a phone without the `manage` permission, or an older box.
 */
export async function saveAssistantName(api: Pick<Api, 'setAssistantName'>, input: string): Promise<Assistant> {
  const name = normalizeAssistantName(input)
  const invalid = name ? assistantNameError(name) : null
  if (invalid) throw new Error(invalid)
  try {
    return await api.setAssistantName(name)
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) throw new Error('This box is too old to rename the assistant. Update LocalRouter on it first.')
    if (error instanceof ApiError && error.status === 403) throw new Error('This phone may not change the box’s settings. Pair it with the manage permission, or rename the assistant on the box.')
    throw error
  }
}
