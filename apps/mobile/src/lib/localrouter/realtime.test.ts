import { afterEach, describe, expect, it, vi } from 'vitest'

import { RealtimeClient, type SocketLike } from './realtime'

class FakeSocket implements SocketLike {
  readyState = 0
  sent: string[] = []
  onopen: SocketLike['onopen'] = null
  onmessage: SocketLike['onmessage'] = null
  onclose: SocketLike['onclose'] = null
  onerror: SocketLike['onerror'] = null
  constructor(readonly url: string, readonly protocols: string[]) {}
  send(data: string) { this.sent.push(data) }
  close() { this.readyState = 3 }
  open() { this.readyState = 1; this.onopen?.({}) }
  drop(code = 1006, reason = '') { this.readyState = 3; this.onclose?.({ code, reason }) }
}

afterEach(() => vi.useRealTimers())

describe('the realtime client', () => {
  it('sends the token as a subprotocol, seeds every new session and reconnects with backoff', () => {
    vi.useFakeTimers()
    const sockets: FakeSocket[] = []
    const events: string[] = []
    let opens = 0
    const client = new RealtimeClient({
      baseUrl: 'https://box:10443',
      token: 'tok',
      socket: (url, protocols) => { const s = new FakeSocket(url, protocols); sockets.push(s); return s },
      onEvent: (e) => events.push(e.type),
      onOpen: () => { opens++; return [{ type: 'response.create' }] },
    })
    client.connect()
    expect(sockets[0].url).toBe('wss://box:10443/v1/realtime')
    expect(sockets[0].protocols).toEqual(['realtime', 'openai-insecure-api-key.tok'])
    sockets[0].open()
    expect(sockets[0].sent).toEqual(['{"type":"response.create"}'])
    sockets[0].onmessage?.({ data: '{"type":"session.created"}' })
    expect(events).toEqual(['session.created'])

    sockets[0].drop()
    expect(client.status).toBe('reconnecting')
    vi.advanceTimersByTime(999)
    expect(sockets).toHaveLength(1)
    vi.advanceTimersByTime(1)
    sockets[1].open()
    expect(opens).toBe(2) // the new session got the settings and the conversation again
    expect(client.status).toBe('open')

    // The server ending the session on purpose is not retried.
    sockets[1].drop(1000, 'the server is shutting down')
    expect(client.status).toBe('closed')
    vi.advanceTimersByTime(60_000)
    expect(sockets).toHaveLength(2)
  })

  it('gives up after 8 failures in a row', () => {
    vi.useFakeTimers()
    const sockets: FakeSocket[] = []
    const client = new RealtimeClient({ baseUrl: 'http://box', socket: (u, p) => { const s = new FakeSocket(u, p); sockets.push(s); return s }, onEvent: () => undefined, onOpen: () => [] })
    client.connect()
    for (let i = 0; i < 9; i++) {
      sockets.at(-1)!.drop()
      vi.advanceTimersByTime(15_000)
    }
    expect(client.status).toBe('failed')
    expect(sockets).toHaveLength(9)
  })
})
