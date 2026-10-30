import { File, Paths } from 'expo-file-system'
import * as SecureStore from 'expo-secure-store'
import { z } from 'zod'

import { parseAssistant, type Assistant } from '@/lib/localrouter/assistant'
import { WAIT_DEFAULT_MS } from '@/lib/localrouter/protocol'

/** The box this phone is paired with. The token stays in the keychain, never in the documents folder. */
const pairingSchema = z.object({ baseUrl: z.string(), token: z.string().nullable(), deviceName: z.string().default(''), deviceId: z.string().default('') })
export type Pairing = z.infer<typeof pairingSchema>

const PAIRING_KEY = 'localrouter.pairing'

export const pairingStorage = {
  async read(): Promise<Pairing | null> {
    const text = await SecureStore.getItemAsync(PAIRING_KEY).catch(() => null)
    if (!text) return null
    try {
      const parsed = pairingSchema.safeParse(JSON.parse(text))
      return parsed.success ? parsed.data : null
    } catch {
      return null
    }
  },
  write: (pairing: Pairing) => SecureStore.setItemAsync(PAIRING_KEY, JSON.stringify(pairing), { keychainAccessible: SecureStore.AFTER_FIRST_UNLOCK_THIS_DEVICE_ONLY }),
  remove: () => SecureStore.deleteItemAsync(PAIRING_KEY),
}

const jsonFile = <T>(name: string, parse: (raw: unknown) => T) => {
  const file = new File(Paths.document, name)
  return {
    async read(): Promise<T> {
      try {
        return parse(file.exists ? JSON.parse(await file.text()) : null)
      } catch {
        return parse(null)
      }
    },
    write(value: unknown) {
      file.write(JSON.stringify(value))
    },
    remove() {
      if (file.exists) file.delete()
    },
  }
}

/**
 * This phone's own settings. `waitMs`: the wait before a reply (0.3 to 5 s). `voices`: the voice chosen per speech
 * model, kept fixed until the user changes it (a voice the box no longer lists is dropped when the models load).
 * `enabled`: voice is on (listening for the wake phrase); `background`: keep listening with the app in the background.
 */
const settingsSchema = z.object({
  waitMs: z.number().default(WAIT_DEFAULT_MS),
  threshold: z.number().min(0).max(1).nullable().default(null),
  speechModel: z.string().default(''),
  voices: z.record(z.string(), z.string()).default({}),
  instructions: z.string().default(''),
  enabled: z.boolean().default(false),
  background: z.boolean().default(false),
  wakeWord: z.boolean().default(true),
})
export type Settings = z.infer<typeof settingsSchema>
export const defaultSettings = (): Settings => settingsSchema.parse({})

export const settingsStorage = jsonFile<Settings>('settings.json', (raw) => {
  const parsed = settingsSchema.safeParse(raw ?? {})
  return parsed.success ? parsed.data : defaultSettings()
})

/** The assistant's name as the box last said it (the box keeps the setting), so the app shows it offline. */
export const assistantStorage = jsonFile<Assistant>('assistant.json', parseAssistant)

/** The conversation is kept on the phone (the box keeps a session's history only while it is open). */
export const conversationStorage = jsonFile<unknown>('conversation.json', (raw) => raw)
