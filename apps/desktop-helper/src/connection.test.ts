import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DEFAULT_ASSISTANT, startNotice } from './assistant.ts'
import { DISABLED_MESSAGE, loadConfig, saveConfig, setDisabled } from './config.ts'
import { ComputerUse, ComputerUseRuns } from './computer/controller.ts'
import type { Answer, Question } from './computer/decisions.ts'
import { Connection } from './connection.ts'
import type { Dial, SocketEvents } from './connection.ts'
import { fakeMac } from './fake.ts'
import { silentLogger } from './log.ts'
import type { Fields } from './validate.ts'

type FakeSocket = { url: string; token: string; events: SocketEvents; sent: Fields[]; closed: boolean }

/** A transport that records every socket dialed and every frame sent on it. */
function fakeDial() {
  const sockets: FakeSocket[] = []
  const dial: Dial = (url, token, events) => {
    const socket: FakeSocket = { url, token, events, sent: [], closed: false }
    sockets.push(socket)
    return {
      send: (text) => void socket.sent.push(JSON.parse(text) as Fields),
      close: () => {
        socket.closed = true
      },
    }
  }
  return { dial, sockets, last: () => sockets[sockets.length - 1] }
}

const flush = () => vi.advanceTimersByTimeAsync(0)
const c = (choice: string): Answer => ({ type: 'choice', choice, probability: 0.9 })
const WELCOME = { type: 'welcome', helper_id: 'mac_3f2a', limits: { max_steps: 15, timeout_ms: 90_000, approval_timeout_ms: 60_000 } }
const START = { type: 'use.start', id: 'b2', task_id: 'task_1', goal: 'Search my notes for groceries', text: 'groceries', limits: { max_steps: 15, timeout_ms: 90_000 } }

function setup(disabled: () => boolean = () => false, onWelcome?: () => string) {
  const mac = fakeMac()
  const transport = fakeDial()
  // As src/main.ts: the notice names the box's assistant, looked up again at each welcome.
  let assistant = DEFAULT_ASSISTANT
  const runs = new ComputerUseRuns({
    create: (session) => new ComputerUse({ ...session, tabs: null, desktop: mac.desktop, dock: mac.dock, apps: mac.apps, log: silentLogger, settleMs: 0, sleep: async () => undefined }),
    onStart: (task) => void mac.controls.notify('LocalRouter Desktop', startNotice(assistant, task.goal)),
    log: silentLogger,
  })
  const connection = new Connection({
    url: 'https://box.example:10443',
    token: 'secret-token',
    helper: { name: "JJ's MacBook Pro", platform: 'darwin', version: '0.1.0' },
    access: async () => ({ accessibility: true, screen_recording: false }),
    controls: mac.controls,
    runs,
    disabled,
    ...(onWelcome ? { onWelcome: () => void (assistant = onWelcome()) } : {}),
    dial: transport.dial,
    random: () => 0,
    log: silentLogger,
  })
  /** Connects, says hello, and receives the welcome. */
  const welcomed = async () => {
    connection.start()
    const socket = transport.last()
    socket.events.open()
    await flush()
    socket.events.message(JSON.stringify(WELCOME))
    return socket
  }
  return { mac, transport, runs, connection, welcomed }
}

const send = (socket: FakeSocket, frame: Fields) => socket.events.message(JSON.stringify(frame))
const ofType = (socket: FakeSocket, type: string) => socket.sent.filter((frame) => frame.type === type)

/**
 * The box's side of a task on the fake Notes app: everything is allowed, and the decisions search
 * for the task's text, press Enter, and are done once the results show.
 */
function groceries(frame: Fields): Fields {
  if (frame.type === 'authorize') return { ok: true, grant: 'allow' }
  const questions = frame.questions as Record<string, Question>
  if ('surface' in questions) return { ok: true, answers: { surface: c('current') } }
  const next = questions.next
  const criteria = next.type === 'choice' ? next.criteria : {}
  if ('type_text' in criteria) return { ok: true, answers: { next: c('type_text'), type_into: c('c0') } }
  const title = (frame.state as { screen: { title: string } }).screen.title
  if ('press_enter' in criteria && !title.includes('found')) return { ok: true, answers: { next: c('press_enter') } }
  return { ok: true, answers: { next: c('done') } }
}

/** Answers the helper's requests on `socket` with `answer` until `until` holds. */
async function serve(socket: FakeSocket, answer: (frame: Fields) => Fields | null, until: () => boolean) {
  let seen = 0
  for (let round = 0; round < 100 && !until(); round += 1) {
    await flush()
    while (seen < socket.sent.length) {
      const frame = socket.sent[seen]
      seen += 1
      if (frame.type !== 'decide' && frame.type !== 'authorize') continue
      const reply = answer(frame)
      if (reply) send(socket, { type: 'result', id: frame.id, ...reply })
    }
  }
}

describe('Connection', () => {
  let current: ReturnType<typeof setup> | null = null
  const make = (disabled?: () => boolean, onWelcome?: () => string) => {
    current = setup(disabled, onWelcome)
    return current
  }
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    current?.connection.close()
    current = null
    vi.useRealTimers()
  })

  it('dials <url>/v1/desktop over wss with the token, says hello, and is ready after welcome', async () => {
    const { transport, connection, welcomed } = make()
    const socket = await welcomed()
    expect(socket.url).toBe('wss://box.example:10443/v1/desktop')
    expect(socket.token).toBe('secret-token')
    expect(socket.sent[0]).toEqual({
      type: 'hello',
      protocol: 1,
      helper: { name: "JJ's MacBook Pro", platform: 'darwin', version: '0.1.0' },
      capabilities: { controls: ['open_app', 'quit_app', 'volume', 'media', 'dark_mode', 'lock_screen'], computer_use: true },
      access: { accessibility: true, screen_recording: false },
    })
    expect(connection.status).toMatchObject({ connected: true, helperId: 'mac_3f2a', task: null })
    expect(transport.sockets).toHaveLength(1)
    // The token never travels in a frame.
    expect(JSON.stringify(socket.sent)).not.toContain('secret-token')
  })

  it('asks for the assistant’s name at each welcome, and the task-start notice uses it', async () => {
    let welcomes = 0
    const { mac, welcomed } = make(undefined, () => {
      welcomes += 1
      return 'Friday'
    })
    const socket = await welcomed()
    expect(welcomes).toBe(1)
    send(socket, START)
    await flush()
    expect(mac.controls.notices).toEqual([{ title: 'LocalRouter Desktop', text: 'Friday is using your Mac: Search my notes for groceries. Move the pointer to the top-left corner to stop.' }])
  })

  it('ignores requests before the welcome', async () => {
    const { connection, transport } = make()
    connection.start()
    const socket = transport.last()
    socket.events.open()
    await flush()
    send(socket, START)
    await flush()
    expect(ofType(socket, 'result')).toEqual([])
  })

  it('runs a fixed control and answers with a sentence, or with the reason it could not', async () => {
    const { mac, welcomed } = make()
    const socket = await welcomed()
    send(socket, { type: 'control', id: 'b1', task_id: 't1', control: { kind: 'open_app', app: 'Notes' } })
    send(socket, { type: 'control', id: 'b2', task_id: 't2', control: { kind: 'shell', command: 'rm -rf /' } })
    await flush()
    expect(ofType(socket, 'result')).toEqual([
      { type: 'result', id: 'b2', ok: false, error: 'control.kind should be one of open_app, quit_app, volume, media, dark_mode, lock_screen' },
      { type: 'result', id: 'b1', ok: true, done: 'Pretended to open Notes (fake Mac).' },
    ])
    expect(mac.controls.calls).toEqual([{ kind: 'open_app', app: 'Notes' }])
  })

  it('runs a task: decide and authorize by id, progress as it goes, then finished exactly once', async () => {
    const { mac, welcomed } = make()
    const socket = await welcomed()
    send(socket, START)
    await flush()
    expect(ofType(socket, 'result')).toEqual([{ type: 'result', id: 'b2', ok: true }])
    expect(mac.controls.notices).toEqual([{ title: 'LocalRouter Desktop', text: 'Jarvis is using your Mac: Search my notes for groceries. Move the pointer to the top-left corner to stop.' }])
    // A reply to an id the helper never used is ignored; the request waits for its own id.
    const firstDecide = ofType(socket, 'decide')[0]
    expect(firstDecide).toMatchObject({ task_id: 'task_1', state: { goal: 'Search my notes for groceries' } })
    send(socket, { type: 'result', id: 'not-mine', ok: true, answers: { surface: c('app') } })
    await serve(socket, groceries, () => ofType(socket, 'finished').length > 0)
    await flush()
    const finished = ofType(socket, 'finished')
    expect(finished).toHaveLength(1)
    expect(finished[0]).toMatchObject({ task_id: 'task_1', outcome: { status: 'done', surface: 'Notes: Notes – 1 found for “groceries”' } })
    expect(mac.desktop.searched).toBe(true)
    expect(ofType(socket, 'authorize').map((frame) => frame.action)).toEqual([
      { kind: 'type', ref: 'search', role: 'textbox', label: 'Search', text: 'groceries' },
      { kind: 'press', key: 'enter', field: { role: 'textbox', label: 'Search' } },
    ])
    expect(ofType(socket, 'authorize').map((frame) => frame.step)).toEqual([1, 2])
    expect(ofType(socket, 'progress').map((frame) => frame.step)).toEqual(['worked in the app in front', 'typed the text into the textbox “Search”', 'pressed Enter'])
    // Every request id is unique on the socket.
    const ids = socket.sent.filter((frame) => frame.type === 'decide' || frame.type === 'authorize').map((frame) => frame.id)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('refuses a second task while one runs', async () => {
    const { welcomed } = make()
    const socket = await welcomed()
    send(socket, START)
    send(socket, { ...START, id: 'b3', task_id: 'task_2' })
    await flush()
    expect(ofType(socket, 'result')).toEqual([{ type: 'result', id: 'b2', ok: true }, { type: 'result', id: 'b3', ok: false, error: 'busy' }])
  })

  it('ends the task blocked when the box denies an action, and acts on nothing', async () => {
    const { mac, welcomed } = make()
    const socket = await welcomed()
    send(socket, START)
    await serve(socket, (frame) => (frame.type === 'authorize' ? { ok: false, denied: 'You did not approve that.' } : groceries(frame)), () => ofType(socket, 'finished').length > 0)
    expect(ofType(socket, 'finished')).toEqual([{ type: 'finished', task_id: 'task_1', outcome: { status: 'blocked', reason: 'You did not approve that.', steps: ['worked in the app in front'], surface: 'Notes: Notes' } }])
    expect(mac.desktop.actions).toEqual([])
  })

  it('stops the task when the box says stop, and sends finished stopped once', async () => {
    const { welcomed } = make()
    const socket = await welcomed()
    send(socket, START)
    await flush()
    send(socket, { type: 'stop', id: 'b9', task_id: 'task_1' })
    // The decide in flight still gets its answer; the task then ends instead of acting.
    await serve(socket, groceries, () => ofType(socket, 'finished').length > 0)
    expect(ofType(socket, 'result')).toContainEqual({ type: 'result', id: 'b9', ok: true })
    expect(ofType(socket, 'finished')).toEqual([{ type: 'finished', task_id: 'task_1', outcome: expect.objectContaining({ status: 'stopped', reason: 'Stopped from LocalRouter.' }) }])
    send(socket, { type: 'stop', id: 'b10', task_id: 'task_1' })
    await flush()
    expect(ofType(socket, 'result')).toContainEqual({ type: 'result', id: 'b10', ok: false, error: 'That task is not running on this Mac.' })
  })

  it('aborts the task on a drop mid-task, and sends nothing of it after reconnecting', async () => {
    const { mac, runs, transport, welcomed } = make()
    const socket = await welcomed()
    send(socket, START)
    await serve(socket, (frame) => (frame.type === 'authorize' ? null : groceries(frame)), () => ofType(socket, 'authorize').length > 0)
    // The box goes away while the helper waits for a grant.
    socket.events.close('1006')
    await flush()
    expect(runs.current).toBeNull()
    expect(mac.desktop.actions).toEqual([])
    expect(socket.closed).toBe(true)
    await vi.advanceTimersByTimeAsync(1_000)
    expect(transport.sockets).toHaveLength(2)
    const again = transport.last()
    again.events.open()
    await flush()
    again.events.message(JSON.stringify(WELCOME))
    await vi.advanceTimersByTimeAsync(5_000)
    expect(again.sent.map((frame) => frame.type)).toEqual(['hello'])
    expect(ofType(socket, 'finished')).toEqual([])
  })

  it('gives up on an authorize after the approval timeout plus 15 s, and ends the task stuck', async () => {
    const { welcomed } = make()
    const socket = await welcomed()
    send(socket, START)
    await serve(socket, (frame) => (frame.type === 'authorize' ? null : groceries(frame)), () => ofType(socket, 'authorize').length > 0)
    // The box keeps the socket alive while the user has not tapped.
    for (let waited = 0; waited < 70_000; waited += 10_000) {
      await vi.advanceTimersByTimeAsync(10_000)
      send(socket, { type: 'ping' })
    }
    expect(ofType(socket, 'finished')).toEqual([])
    await vi.advanceTimersByTimeAsync(5_000)
    await flush()
    expect(ofType(socket, 'finished')).toEqual([{ type: 'finished', task_id: 'task_1', outcome: expect.objectContaining({ status: 'stuck', reason: 'The box did not allow the next step: The box did not answer authorize in time.' }) }])
    expect(ofType(socket, 'pong').length).toBeGreaterThan(0)
  })

  it('pings every 20 s, treats 60 s of silence as a drop, and reconnects with backoff from 1 s up to 60 s', async () => {
    const { transport, connection, welcomed } = make()
    const socket = await welcomed()
    await vi.advanceTimersByTimeAsync(20_000)
    expect(ofType(socket, 'ping')).toHaveLength(1)
    await vi.advanceTimersByTimeAsync(40_000)
    expect(socket.closed).toBe(false)
    await vi.advanceTimersByTimeAsync(5_000)
    expect(socket.closed).toBe(true)
    expect(connection.status.connected).toBe(false)
    // 1 s, then 2 s, 4 s … while dials keep failing, never more than 60 s.
    await vi.advanceTimersByTimeAsync(999)
    expect(transport.sockets).toHaveLength(1)
    await vi.advanceTimersByTimeAsync(1)
    expect(transport.sockets).toHaveLength(2)
    transport.last().events.close('1006')
    await vi.advanceTimersByTimeAsync(1_999)
    expect(transport.sockets).toHaveLength(2)
    await vi.advanceTimersByTimeAsync(1)
    expect(transport.sockets).toHaveLength(3)
    for (let failures = 0; failures < 8; failures += 1) {
      transport.last().events.close('1006')
      await vi.advanceTimersByTimeAsync(60_000)
    }
    expect(transport.sockets).toHaveLength(11)
    // A welcome resets the backoff to 1 s.
    const fresh = transport.last()
    fresh.events.open()
    await flush()
    fresh.events.message(JSON.stringify(WELCOME))
    fresh.events.close('1006')
    await vi.advanceTimersByTimeAsync(1_000)
    expect(transport.sockets).toHaveLength(12)
  })
})

describe('disabled on this Mac', () => {
  const dirs: string[] = []
  afterEach(async () => {
    vi.useRealTimers()
    await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })))
  })

  it('refuses tasks and controls with "Computer control is switched off on this Mac" after disable, and takes them again after enable', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'lrd-'))
    dirs.push(dir)
    await saveConfig(dir, { url: 'https://box.example:10443', token: 'secret-token', name: 'Mac', disabled: false })
    let config = await setDisabled(dir, true)
    expect((await loadConfig(dir))?.disabled).toBe(true)
    vi.useFakeTimers()
    const { mac, connection, welcomed } = setup(() => config.disabled)
    try {
      const socket = await welcomed()
      send(socket, START)
      send(socket, { type: 'control', id: 'b5', task_id: 't5', control: { kind: 'lock_screen' } })
      await flush()
      expect(ofType(socket, 'result')).toEqual([
        { type: 'result', id: 'b2', ok: false, error: DISABLED_MESSAGE },
        { type: 'result', id: 'b5', ok: false, error: DISABLED_MESSAGE },
      ])
      expect(mac.controls.calls).toEqual([])
      expect(ofType(socket, 'decide')).toEqual([])
      vi.useRealTimers()
      config = await setDisabled(dir, false)
      vi.useFakeTimers()
      send(socket, { ...START, id: 'b6' })
      await flush()
      expect(ofType(socket, 'result').at(-1)).toEqual({ type: 'result', id: 'b6', ok: true })
    } finally {
      connection.close()
    }
  })
})
