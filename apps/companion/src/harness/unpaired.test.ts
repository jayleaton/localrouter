import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { startRuntime, UNPAIRED } from './runtime.ts'
import { startExecutor } from '../executor/executor.ts'
import type { Executor } from '../executor/executor.ts'
import { handleMcp } from '../executor/mcp.ts'
import { executorSocket } from '../executor/paths.ts'
import { ExecutorClient, notRunning } from '../executor/server.ts'
import { TestGate } from '../executor/testing.ts'

/**
 * The app starts its executor at launch, paired or not (Cody's Windows test: with the app open and unpaired, every MCP
 * call said "LocalRouter is not running"). Unpaired, an enrolled harness gets what needs no Spark and no approval
 * window, and every refusal says the real reason.
 */
const NO_WINDOW = 'Approving needs LocalRouter’s window, which opens once LocalRouter is paired with a Spark.'

const dirs: string[] = []
const opened: Executor[] = []
afterEach(async () => {
  for (const e of opened.splice(0)) await e.close()
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
})

async function unpaired(paired: { value: boolean }) {
  // Unix socket paths are short (104 bytes on macOS): a short temp folder, whatever TMPDIR is.
  const dir = mkdtempSync(join(tmpdir().length > 40 && process.platform !== 'win32' ? '/tmp' : tmpdir(), 'lr-unp-'))
  dirs.push(dir)
  const executor = await startExecutor({ dir, name: 'Test PC', machine: 'fake', decider: null, gate: new TestGate(), listen: true, approvalSurface: () => (paired.value ? null : NO_WINDOW) })
  opened.push(executor)
  executor.broker.setDecider(null, UNPAIRED)
  const { token } = executor.enrollment.add('opencode')
  const client = new ExecutorClient(executorSocket(dir), token)
  const tool = async (name: string, args: Record<string, unknown> = {}) => {
    const reply = await handleMcp({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name, arguments: args } }, (op, a) => client.request(op, a), '0')
    const result = reply?.result as { content: Array<{ type: string; text?: string }>; isError: boolean }
    return { text: result.content.find((c) => c.type === 'text')?.text ?? '', isError: result.isError }
  }
  return { dir, executor, client, tool }
}

describe('the executor before pairing', () => {
  it('answers enrolled harnesses: fixed controls and reads work, computer use and approvals say why they wait', async () => {
    const { executor, client, tool } = await unpaired({ value: false })
    const caps = JSON.parse((await tool('computer_capabilities')).text) as { computer_use: boolean; computer_use_unavailable?: string }
    expect(caps).toMatchObject({ computer_use: false, computer_use_unavailable: UNPAIRED })

    expect(await tool('computer_control', { control: { kind: 'open_app', app: 'Notes' } })).toMatchObject({ isError: false })
    expect(executor.built.fake?.controls.calls).toEqual([{ kind: 'open_app', app: 'Notes' }])
    expect((await tool('computer_observe')).isError).toBe(false)

    const use = await tool('computer_use', { goal: 'search my notes for groceries', wait_s: 1 })
    expect(use.isError).toBe(true)
    expect(use.text).toContain(UNPAIRED)

    // A step that needs a tap is refused at once with the reason, and no card is shown that nobody could answer.
    const cards: string[] = []
    executor.broker.on((e) => void (e.type === 'approval' && cards.push(e.card.id)))
    const lock = JSON.parse((await tool('computer_control', { control: { kind: 'lock_screen' } })).text) as { ok: boolean; text: string }
    expect(lock).toMatchObject({ ok: false, text: NO_WINDOW })
    expect(cards).toEqual([])
    expect(executor.built.fake?.controls.calls).toHaveLength(1)
    client.close()
  })

  it('pairing gives it the Spark’s decisions and unpairing takes them back; the executor keeps serving', async () => {
    const paired = { value: false }
    const { dir, executor, client } = await unpaired(paired)
    paired.value = true
    const runtime = await startRuntime({ base: 'http://127.0.0.1:9', token: 't', dir, name: 'Test PC', machine: 'fake', gate: new TestGate(), shared: () => [], executor })
    expect(runtime.executor).toBe(executor)
    expect(((await client.request('capabilities')) as { result: { computer_use: boolean } }).result.computer_use).toBe(true)
    await runtime.close()
    paired.value = false
    const after = (await client.request('capabilities')) as { ok: boolean; result: { computer_use: boolean; computer_use_unavailable?: string } }
    expect(after).toMatchObject({ ok: true, result: { computer_use: false, computer_use_unavailable: UNPAIRED } })
    client.close()
  })

  it('a socket nobody listens on means the app is not running, and says what an older open app means', async () => {
    const dir = mkdtempSync(join(tmpdir().length > 40 && process.platform !== 'win32' ? '/tmp' : tmpdir(), 'lr-none-'))
    dirs.push(dir)
    const reply = await new ExecutorClient(executorSocket(dir), 'x').request('capabilities').catch((e: unknown) => ({ ok: false, error: (e as Error).message }))
    expect(reply).toMatchObject({ ok: false })
    expect((reply as { error: string }).error).toMatch(/^LocalRouter is not running on this computer \(\w+\): open the LocalRouter app\. If it is open, it is an older build/)
    expect(notRunning('ENOENT')).toContain('pair it on its Connect page')
  })
})
