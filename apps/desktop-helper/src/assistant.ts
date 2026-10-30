/**
 * The assistant's name, a setting on the box (`GET /v1/assistant`): every client shows and speaks it. A box from
 * before the setting (404, or 403 for the route) and one that does not answer mean the default.
 */
export const DEFAULT_ASSISTANT = 'Jarvis'

const LOOKUP_MS = 8_000

/** The box's assistant name with this device's token; the default when the box has none or cannot say. */
export async function assistantName(url: string, token: string, f: typeof fetch = fetch): Promise<string> {
  try {
    const r = await f(`${url.replace(/\/+$/, '')}/v1/assistant`, { headers: token ? { authorization: `Bearer ${token}` } : {}, signal: AbortSignal.timeout(LOOKUP_MS) })
    if (!r.ok) return DEFAULT_ASSISTANT
    const name = ((await r.json()) as { name?: unknown }).name
    return typeof name === 'string' && name.trim() ? name.trim().slice(0, 64) : DEFAULT_ASSISTANT
  } catch {
    return DEFAULT_ASSISTANT
  }
}

/** The notification at a task's start. */
export function startNotice(assistant: string, goal: string): string {
  return `${assistant} is using your Mac: ${goal}. Move the pointer to the top-left corner to stop.`
}
