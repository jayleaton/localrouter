import { describe, expect, it } from 'vitest'

import { createApi, type Fetch } from './api'
import { assistantNameError, defaultAssistant, fetchAssistant, normalizeAssistantName, parseAssistant, saveAssistantName } from './assistant'

type Call = { url: string; method: string; headers: Record<string, string>; body: unknown }

/** A box that answers `/v1/assistant` like the real one, or with `status` for every call. */
function box(options: { name?: string; status?: number; canManage?: boolean; offline?: boolean } = {}) {
  let name = options.name ?? 'Jarvis'
  const calls: Call[] = []
  const answer = (status: number, body: unknown) => ({
    ok: status < 400,
    status,
    json: async () => body,
    text: async () => JSON.stringify(body),
    arrayBuffer: async () => new ArrayBuffer(0),
  })
  const object = () => ({ object: 'assistant', name, default_name: 'Jarvis', custom: name !== 'Jarvis', voice: { model: 'fishaudio/drama-3', voice: 'default' } })
  const fetch: Fetch = async (url, init) => {
    const body = init?.body ? JSON.parse(init.body) : undefined
    calls.push({ url, method: init?.method ?? 'GET', headers: init?.headers ?? {}, body })
    if (options.offline) throw new Error('Network request failed')
    if (options.status) return answer(options.status, { error: { message: 'Not found' } })
    if (init?.method === 'PATCH') {
      if (!options.canManage) return answer(403, { error: { message: 'this device lacks the manage scope' } })
      const wanted = normalizeAssistantName(String(body.name))
      if (wanted.length > 32) return answer(400, { error: { message: 'name too long' } })
      name = wanted || 'Jarvis'
    }
    return answer(200, object())
  }
  return { api: createApi({ baseUrl: 'https://box:10443', token: 'tok', fetch }), calls }
}

describe('the assistant name', () => {
  it('reads the box’s name with the device token', async () => {
    const { api, calls } = box({ name: 'Friday' })
    const assistant = await fetchAssistant(api, defaultAssistant())
    expect(assistant).toEqual({ name: 'Friday', defaultName: 'Jarvis', custom: true, voice: { model: 'fishaudio/drama-3', voice: 'default' }, supported: true })
    expect(calls[0]).toMatchObject({ url: 'https://box:10443/v1/assistant', method: 'GET', headers: { authorization: 'Bearer tok' } })
  })

  it('renames it on the box and back to the default', async () => {
    const { api, calls } = box({ canManage: true })
    expect((await saveAssistantName(api, '  Mary   Jane ')).name).toBe('Mary Jane')
    expect(calls[0]).toMatchObject({ method: 'PATCH', body: { name: 'Mary Jane' } })
    expect((await fetchAssistant(api, defaultAssistant())).name).toBe('Mary Jane')
    const reset = await saveAssistantName(api, '')
    expect(reset).toMatchObject({ name: 'Jarvis', custom: false })
  })

  it('says why a rename is refused', async () => {
    await expect(saveAssistantName(box({ canManage: false }).api, 'Friday')).rejects.toThrow(/manage permission/)
    await expect(saveAssistantName(box({ status: 404 }).api, 'Friday')).rejects.toThrow(/too old/)
    // Checked on the phone first: the box is not asked.
    const { api, calls } = box({ canManage: true })
    await expect(saveAssistantName(api, '!!!')).rejects.toThrow(/letters/)
    expect(calls).toHaveLength(0)
  })

  it('is Jarvis on an older box, and the last known name when the box is out of reach', async () => {
    const cached = parseAssistant({ name: 'Friday', defaultName: 'Jarvis', custom: true, voice: null, supported: true })
    expect(await fetchAssistant(box({ status: 404 }).api, cached)).toMatchObject({ name: 'Jarvis', supported: false })
    expect(await fetchAssistant(box({ status: 403 }).api, cached)).toMatchObject({ name: 'Jarvis', supported: false })
    expect(await fetchAssistant(box({ offline: true }).api, cached)).toBe(cached)
    expect(await fetchAssistant(box({ status: 500 }).api, cached)).toBe(cached)
  })

  it('reads a cached name, or the default for anything else', () => {
    const assistant = { name: 'Friday', defaultName: 'Jarvis', custom: true, voice: { model: 'm', voice: 'v' }, supported: true }
    expect(parseAssistant(JSON.parse(JSON.stringify(assistant)))).toEqual(assistant)
    expect(parseAssistant(null)).toEqual(defaultAssistant())
    expect(parseAssistant({ name: '' })).toEqual(defaultAssistant())
    expect(defaultAssistant().name).toBe('Jarvis')
  })

  it('checks a name lightly, as the box does', () => {
    expect(normalizeAssistantName('  hal   9000 ')).toBe('hal 9000')
    expect(normalizeAssistantName('O’Brien')).toBe("O'Brien")
    for (const ok of ['Friday', 'Mary-Jane', 'J.A.R.V.I.S.', "O'Brien", 'Zoë', 'ヒカリ', 'R2']) expect(assistantNameError(ok)).toBeNull()
    expect(assistantNameError('   ')).toMatch(/name/)
    expect(assistantNameError('x'.repeat(33))).toMatch(/32/)
    expect(assistantNameError('Hey!')).toMatch(/letters/)
    expect(assistantNameError("-.'")).toMatch(/letter or digit/)
  })
})
