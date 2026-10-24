#!/usr/bin/env node
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs'
import { hostname, tmpdir } from 'node:os'
import { join } from 'node:path'
import { parseArgs } from 'node:util'
import type { Card, PresenceGate } from '../executor/broker.ts'
import { to24k } from './audio.ts'
import type { ServerEvent } from './realtime.ts'
import { startRuntime } from './runtime.ts'
import { pair } from './spark.ts'

/**
 * The harness headless, for a live check against a box without a Mac: the same runtime the app runs, on the fake Mac
 * (a scripted Notes app; nothing touches a real screen), with approvals answered by this runner after a delay — which
 * the app never does, and which only the fake Mac allows.
 *
 *   node src/harness/cli.ts pair --url https://box:10443 --code XXXX-XXXX [--name NAME]
 *   LOCALROUTER_TOKEN=… node src/harness/cli.ts turns --url URL --fake-mac --share DIR \
 *       [--speak] [--approve-after-ms 1500] [--expect list_files,computer_use,generate_image] [--expect-approval] --json out.json "turn 1" "turn 2"
 *
 * `--speak` synthesizes each turn with the box's speech model and sends it as microphone audio (push to talk), so the
 * box transcribes it; otherwise the turn is typed. Exit 0 when every turn finished and every expected tool ran OK.
 */
const log = (...a: unknown[]) => process.stderr.write(`${new Date().toISOString().slice(11, 19)} ${a.map((x) => (typeof x === 'string' ? x : JSON.stringify(x))).join(' ')}\n`)

class RunnerGate implements PresenceGate {
  readonly method = 'headless runner (fake Mac only)'
  async verify(_card: Card) {
    return true
  }
}

async function main(argv: string[]): Promise<number> {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: { url: { type: 'string' }, code: { type: 'string' }, name: { type: 'string' }, 'fake-mac': { type: 'boolean' }, share: { type: 'string' }, speak: { type: 'boolean' }, 'approve-after-ms': { type: 'string' }, deny: { type: 'boolean' }, 'expect-approval': { type: 'boolean' }, 'expect-handoff': { type: 'boolean' }, 'expect-no-handoff': { type: 'boolean' }, expect: { type: 'string' }, json: { type: 'string' }, dir: { type: 'string' }, timeout: { type: 'string' } },
  })
  const [command, ...turns] = positionals
  const url = values.url ?? process.env.LOCALROUTER_URL ?? ''
  if (!url) throw new Error('--url is required')
  if (command === 'pair') {
    if (!values.code) throw new Error('pair needs --code')
    const p = await pair(url, values.code, values.name ?? `harness on ${hostname()}`)
    process.stdout.write(`${JSON.stringify({ device: p.device, token: p.token })}\n`)
    return 0
  }
  if (command !== 'turns' || !turns.length) throw new Error('usage: cli.ts turns --url URL --fake-mac "a turn" …')
  if (!values['fake-mac']) throw new Error('the headless runner drives only the fake Mac (--fake-mac); the real one is the app’s')
  const dir = values.dir ?? mkdtempSync(join(tmpdir(), 'lr-harness-'))
  const share = values.share ?? join(dir, 'shared')
  mkdirSync(share, { recursive: true })
  const rt = await startRuntime({ base: url, token: process.env.LOCALROUTER_TOKEN ?? '', dir, name: 'Fake Mac (headless)', machine: 'fake', gate: new RunnerGate(), shared: () => [share] })
  const caps = await rt.spark.capabilities()
  log('box', caps.server, 'contract', caps.contract, 'tool modes', caps.realtime?.tool_execution)
  const approvals: Array<{ card: Card; answered: string; result: unknown; at_ms: number }> = []
  const delay = Number(values['approve-after-ms'] ?? 1500)
  rt.executor.broker.on((e) => {
    if (e.type !== 'approval') return
    log('approval card:', e.card.action, '—', e.card.why)
    setTimeout(() => {
      const decision = values.deny ? 'deny' : 'approve'
      void rt.executor.broker.answer(e.card.id, decision, 'headless runner').then((result) => {
        approvals.push({ card: e.card, answered: decision, result, at_ms: delay })
        log('answered', decision, result)
      })
    }, delay)
  })
  const events: ServerEvent[] = []
  let audioBytes = 0
  let created = 0
  let done = 0
  const waiters: Array<() => void> = []
  const conversation = rt.journal.create()
  const session = rt.session({
    conversation,
    pushToTalk: true,
    onEvent: (ev) => {
      if (ev.type === 'response.audio.delta') audioBytes += Buffer.from(String(ev.delta ?? ''), 'base64').length
      else events.push(ev)
      if (ev.type === 'response.created') created += 1
      if (ev.type === 'response.done') done += 1
      if (ev.type === 'error') log('error event', ev.error)
      if (ev.type === 'response.done' || ev.type.startsWith('localrouter.job')) waiters.splice(0).forEach((w) => w())
    },
  })
  await session.connect()
  log('session', session.sessionId, 'client tool mode; tools', rt.registry.names)
  const timeoutMs = Number(values.timeout ?? 360) * 1000
  const results: Array<Record<string, unknown>> = []
  for (const [n, text] of turns.entries()) {
    const startCalls = session.calls.length
    const startAudio = audioBytes
    const t0 = Date.now()
    if (values.speak) {
      const r = await rt.spark.request('/v1/audio/speech', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ input: text, response_format: 'pcm' }) })
      if (!r.ok) throw new Error(`speech: ${r.status} ${await r.text()}`)
      const rate = Number(r.headers.get('x-sample-rate') ?? 24000)
      const pcm = to24k(Buffer.from(await r.arrayBuffer()), rate)
      log(`turn ${n + 1}: speaking ${(pcm.length / 48000).toFixed(1)} s of audio (${rate} Hz → 24 kHz)`)
      session.speak(pcm)
    } else {
      log(`turn ${n + 1}: typing`, text)
      session.say(text)
    }
    // The turn ends when every response has ended and no tool is running or waiting.
    for (;;) {
      if (Date.now() - t0 > timeoutMs) throw new Error(`turn ${n + 1} did not finish in ${timeoutMs / 1000} s`)
      await new Promise<void>((r) => {
        waiters.push(r)
        setTimeout(r, 500)
      })
      if (created > 0 && created === done && !session.busy) {
        await new Promise((r) => setTimeout(r, 1500)) // a continuation may start right after the outputs
        if (created === done && !session.busy) break
      }
    }
    const calls = session.calls.slice(startCalls)
    const said = events.filter((e) => e.type === 'response.audio_transcript.done' || e.type === 'response.text.done').map((e) => String(e.transcript ?? e.text ?? '')).slice(-3)
    const heard = events.filter((e) => e.type === 'conversation.item.input_audio_transcription.completed').map((e) => String(e.transcript ?? '')).slice(-1)[0] ?? null
    const rec = { turn: n + 1, text, heard: values.speak ? heard : null, calls: calls.map((c) => ({ ...c, output: c.output && c.output.length > 600 ? `${c.output.slice(0, 600)}…` : c.output })), said, audio_s: Number(((audioBytes - startAudio) / 48000).toFixed(2)), ms: Date.now() - t0 }
    log(`turn ${n + 1} done in ${rec.ms} ms:`, calls.map((c) => `${c.name}:${c.ok ? 'ok' : 'error'}`), 'said', said[said.length - 1] ?? '')
    results.push(rec)
  }
  // A generation that finishes after its turn still reports here.
  await new Promise((r) => setTimeout(r, 2000))
  session.close()
  const jobs = events.filter((e) => e.type.startsWith('localrouter.job'))
  const handoffs = events.filter((e) => e.type === 'localrouter.companion.handoff')
  const expected = (values.expect ?? '').split(',').map((s) => s.trim()).filter(Boolean)
  const ran = session.calls.filter((c) => c.ok).map((c) => c.name)
  const checks = {
    client_mode: true,
    every_turn_spoken: results.every((r) => Number(r.audio_s) > 0),
    expected_tools_ran: expected.every((name) => ran.includes(name)),
    approval_raised_here: !values['expect-approval'] || approvals.length > 0,
    handoff_as_expected: values['expect-handoff'] ? handoffs.length === 1 : values['expect-no-handoff'] ? handoffs.length === 0 : true,
    approvals_answered_here: approvals.every((a) => (a.result as { ok?: boolean }).ok === true || values.deny === true),
    no_box_tool_events: !events.some((e) => e.type === 'localrouter.job.created' && !session.calls.some((c) => c.call_id === e.call_id)),
  }
  const report = {
    box: { url: rt.spark.base, contract: caps.contract, server: caps.server },
    session: session.sessionId,
    conversation,
    turns: results,
    approvals,
    jobs,
    handoffs,
    audit: rt.executor.broker.audit(200),
    fake_mac: { actions: rt.executor.built.fake?.desktop.actions, sent: rt.executor.built.fake?.desktop.sent, controls: rt.executor.built.fake?.controls.calls },
    journal: rt.journal.export(conversation),
    checks,
    passed: Object.values(checks).every(Boolean),
  }
  if (values.json) writeFileSync(values.json, `${JSON.stringify(report, null, 1)}\n`)
  log(report.passed ? 'PASSED' : 'FAILED', checks)
  await rt.close()
  return report.passed ? 0 : 1
}

main(process.argv.slice(2)).then(
  (code) => process.exit(code),
  (error: unknown) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`)
    process.exit(2)
  },
)
