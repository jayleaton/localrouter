/*
 * Where the box is. The phone reaches LocalRouter over the tailnet, normally through `tailscale serve` (HTTPS in
 * front of the daemon's loopback port), so a base URL is an origin such as https://box.tailnet.ts.net:10443.
 * Parsed by hand: React Native's URL is incomplete (no setters, partial getters), and this runs on both.
 */

const ORIGIN = /^(https?):\/\/([^/?#\s]+)([^?#\s]*)/i

/** "box.tailnet.ts.net:10443", "https://box:10443/v1/", "http://10.0.0.5:8190" → an origin without a trailing slash or /v1. */
export function normalizeBaseUrl(input: string): string | null {
  let text = input.trim()
  if (!text || /\s/.test(text)) return null
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(text)) text = `https://${text}`
  const match = ORIGIN.exec(text)
  if (!match) return null
  const host = match[2].toLowerCase()
  if (!/^(\[[0-9a-f:.]+\]|[a-z0-9.-]+)(:\d{1,5})?$/.test(host) || host.startsWith('.') || host.startsWith(':')) return null
  const path = match[3].replace(/\/+$/, '').replace(/\/v1$/, '')
  return `${match[1].toLowerCase()}://${host}${path}`
}

/** The realtime socket's URL for a base URL. */
export function realtimeUrl(base: string, model?: string) {
  const ws = base.replace(/^https:/i, 'wss:').replace(/^http:/i, 'ws:')
  return `${ws}/v1/realtime${model ? `?model=${encodeURIComponent(model)}` : ''}`
}

/**
 * An output file's URL as the phone can fetch it. The daemon writes `http://<Host>/v1/files/...` from the request's
 * Host header, which behind an HTTPS proxy names the right host with the wrong scheme; the path is what identifies
 * the file, so it is joined to the box's own base URL. A relative path (`/v1/jobs` answers with those) is too.
 */
export function fileUrl(base: string, url: string) {
  const at = url.indexOf('/v1/files/')
  if (at >= 0 && (at === 0 || /^https?:\/\/[^/]+$/i.test(url.slice(0, at)))) return `${base}${url.slice(at)}`
  return url
}

/** A pairing code as `localrouter pair add` prints it: 8 Crockford base32 characters, shown as XXXX-XXXX. */
export function normalizeCode(input: string) {
  const code = input.toUpperCase().replace(/[\s-]/g, '')
  return /^[0-9A-HJKMNP-TV-Z]{8}$/.test(code) ? `${code.slice(0, 4)}-${code.slice(4)}` : null
}

function queryParam(query: string, name: string) {
  for (const pair of query.split('&')) {
    const eq = pair.indexOf('=')
    const key = eq < 0 ? pair : pair.slice(0, eq)
    if (key !== name) continue
    try {
      return decodeURIComponent((eq < 0 ? '' : pair.slice(eq + 1)).replace(/\+/g, ' '))
    } catch {
      return null
    }
  }
  return null
}

/** `localrouter://pair?url=...&code=...`, the link `localrouter pair add NAME --url URL` prints. */
export function parsePairLink(link: string): { baseUrl: string; code: string | null } | null {
  const match = /^localrouter:\/\/\/?pair\/?\?(.*)$/i.exec(link.trim())
  if (!match) return null
  const baseUrl = normalizeBaseUrl(queryParam(match[1], 'url') ?? '')
  if (!baseUrl) return null
  return { baseUrl, code: normalizeCode(queryParam(match[1], 'code') ?? '') }
}
