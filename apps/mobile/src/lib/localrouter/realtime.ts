import { realtimeUrl } from './address'
import { parseServerEvent, type ClientEvent, type ServerEvent } from './protocol'

/*
 * One realtime session with the box. A close nobody asked for (the service restarting, the tailnet dropping) is
 * followed by a new session after 1, 2, 4 … 15 s; the new session gets the settings and the conversation again
 * (`onOpen` returns them), since a new server session starts empty and numbers its items from 1. After 8 failures
 * in a row it gives up and reports `failed`.
 */

/** The part of WebSocket the client uses: React Native's, the browser's and Node's all fit. */
export type SocketLike = {
  readyState: number
  send(data: string): void
  close(code?: number, reason?: string): void
  onopen: ((event: unknown) => void) | null
  onmessage: ((event: { data: unknown }) => void) | null
  onclose: ((event: { code: number; reason: string }) => void) | null
  onerror: ((event: unknown) => void) | null
}

export type SocketFactory = (url: string, protocols: string[]) => SocketLike

/**
 * The token goes as a subprotocol entry (`openai-insecure-api-key.<token>`, the OpenAI browser SDK's convention):
 * every WebSocket implementation can send it, and unlike a query string it never lands in a URL or a log.
 */
export function protocolsFor(token?: string | null) {
  return token ? ['realtime', `openai-insecure-api-key.${token}`] : ['realtime']
}

export type ConnectionStatus = 'connecting' | 'open' | 'reconnecting' | 'failed' | 'closed'

export type RealtimeOptions = {
  baseUrl: string
  token?: string | null
  model?: string
  socket: SocketFactory
  onEvent(event: ServerEvent): void
  onStatus?(status: ConnectionStatus, detail: { retries: number; code?: number; reason?: string }): void
  /** Events to send on every (re)connect, before anything else: the session settings, then the conversation. */
  onOpen(): ClientEvent[]
}

const OPEN = 1
const MAX_RETRIES = 8

export class RealtimeClient {
  private socket: SocketLike | null = null
  private retries = 0
  private timer: ReturnType<typeof setTimeout> | null = null
  private wanted = false
  status: ConnectionStatus = 'closed'

  constructor(private readonly options: RealtimeOptions) {}

  connect() {
    if (this.wanted) return
    this.wanted = true
    this.retries = 0
    this.open()
  }

  /** Ends the session; no reconnect follows. */
  close() {
    this.wanted = false
    if (this.timer) clearTimeout(this.timer)
    this.timer = null
    const socket = this.socket
    this.socket = null
    if (socket) {
      socket.onclose = null
      socket.onmessage = null
      try {
        socket.close(1000, '')
      } catch {
        // Already closed.
      }
    }
    this.setStatus('closed')
  }

  get isOpen() {
    return this.socket?.readyState === OPEN && this.status === 'open'
  }

  /** Sends one event; nothing is queued while the socket is down (a reconnect reseeds the conversation instead). */
  send(event: ClientEvent) {
    if (!this.isOpen) return false
    this.socket!.send(JSON.stringify(event))
    return true
  }

  private setStatus(status: ConnectionStatus, detail: { code?: number; reason?: string } = {}) {
    this.status = status
    this.options.onStatus?.(status, { retries: this.retries, ...detail })
  }

  private open() {
    const { baseUrl, token, model, socket: create } = this.options
    this.setStatus(this.retries ? 'reconnecting' : 'connecting')
    let socket: SocketLike
    try {
      socket = create(realtimeUrl(baseUrl, model), protocolsFor(token))
    } catch {
      return this.retry({ code: 0, reason: 'could not open the connection' })
    }
    this.socket = socket
    socket.onopen = () => {
      if (this.socket !== socket) return
      this.retries = 0
      this.status = 'open'
      for (const event of this.options.onOpen()) socket.send(JSON.stringify(event))
      this.setStatus('open')
    }
    socket.onmessage = (message) => {
      if (this.socket !== socket || typeof message.data !== 'string') return
      const event = parseServerEvent(message.data)
      if (event) this.options.onEvent(event)
    }
    socket.onerror = () => undefined // a close follows
    socket.onclose = (event) => {
      if (this.socket !== socket) return
      this.socket = null
      if (!this.wanted) return this.setStatus('closed')
      // The server ended the session on purpose (code 1000 with a reason): no reconnect.
      if (event.code === 1000 && event.reason) {
        this.wanted = false
        return this.setStatus('closed', event)
      }
      this.retry(event)
    }
  }

  private retry(detail: { code: number; reason: string }) {
    if (++this.retries > MAX_RETRIES) {
      this.wanted = false
      return this.setStatus('failed', detail)
    }
    this.setStatus('reconnecting', detail)
    const wait = Math.min(15_000, 1_000 * 2 ** (this.retries - 1))
    this.timer = setTimeout(() => {
      this.timer = null
      if (this.wanted) this.open()
    }, wait)
  }
}
