import { describe, expect, it } from 'vitest'

import { fileUrl, normalizeBaseUrl, normalizeCode, parsePairLink, realtimeUrl } from './address'

describe('box addresses', () => {
  it('normalizes what a user types to an origin', () => {
    expect(normalizeBaseUrl('box.tail.ts.net:10443')).toBe('https://box.tail.ts.net:10443')
    expect(normalizeBaseUrl(' HTTPS://Box.Tail.ts.net:10443/v1/ ')).toBe('https://box.tail.ts.net:10443')
    expect(normalizeBaseUrl('http://10.0.0.5:8190')).toBe('http://10.0.0.5:8190')
    expect(normalizeBaseUrl('ftp://box')).toBeNull()
    expect(normalizeBaseUrl('not a host')).toBeNull()
    expect(normalizeBaseUrl('')).toBeNull()
  })

  it('makes the realtime URL and rebases file URLs onto the box', () => {
    expect(realtimeUrl('https://box:10443')).toBe('wss://box:10443/v1/realtime')
    expect(realtimeUrl('http://box:8190', 'qwen 3')).toBe('ws://box:8190/v1/realtime?model=qwen%203')
    // Behind tailscale serve the daemon names the right host with the wrong scheme.
    expect(fileUrl('https://box:10443', 'http://box:10443/v1/files/j1/0.png')).toBe('https://box:10443/v1/files/j1/0.png')
    expect(fileUrl('https://box:10443', '/v1/files/j1/0.mp4')).toBe('https://box:10443/v1/files/j1/0.mp4')
    expect(fileUrl('https://box:10443', 'https://elsewhere/x.png')).toBe('https://elsewhere/x.png')
  })
})

describe('pairing', () => {
  it('accepts a code in any case, with or without the dash, and refuses lookalike letters', () => {
    expect(normalizeCode('k7qm-3xpd')).toBe('K7QM-3XPD')
    expect(normalizeCode('K7QM3XPD')).toBe('K7QM-3XPD')
    expect(normalizeCode('K7QM-3XPI')).toBeNull() // no I, L, O or U in Crockford base32
    expect(normalizeCode('K7QM')).toBeNull()
  })

  it('reads the link the box prints', () => {
    expect(parsePairLink('localrouter://pair?url=https%3A%2F%2Fbox.tail.ts.net%3A10443&code=K7QM-3XPD')).toEqual({ baseUrl: 'https://box.tail.ts.net:10443', code: 'K7QM-3XPD' })
    expect(parsePairLink('localrouter:///pair?url=box:10443')).toEqual({ baseUrl: 'https://box:10443', code: null })
    expect(parsePairLink('https://box/pair?url=x')).toBeNull()
    expect(parsePairLink('localrouter://pair?code=K7QM-3XPD')).toBeNull()
  })
})
