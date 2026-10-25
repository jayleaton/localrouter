import { mkdtempSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { handleMcp, TOOLS } from './mcp.ts'
import { Enrollment, ExecutorClient, ExecutorServer, OPS, serve } from './server.ts'
import { fakeBroker } from './testing.ts'

const dirs: string[] = []
const temp = () => {
  // Unix socket paths are short (104 bytes on macOS): a short temp folder, whatever TMPDIR is.
  const d = mkdtempSync(join(tmpdir().length > 40 ? '/tmp' : tmpdir(), 'lr-exec-'))
  dirs.push(d)
  return d
}
afterEach(() => {
  for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
})

describe('other harnesses reach the executor only through bounded, enrolled operations', () => {
  it('no operation or MCP tool can answer an approval', async () => {
    expect(OPS).not.toContain('approve')
    expect(TOOLS.map((t) => t.name).some((n) => /approv|answer|policy|shell/.test(n))).toBe(false)
    const { broker } = fakeBroker([])
    const enrollment = new Enrollment(join(temp(), 'callers.json'))
    const { token } = enrollment.add('opencode')
    for (const op of ['approve', 'answer', 'decide', 'setDisabled']) expect(await serve(broker, enrollment, { op, token, args: { id: 'appr_x', decision: 'approve' } })).toMatchObject({ ok: false, error: expect.stringMatching(/unknown op/) })
    const call = await handleMcp({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'approve', arguments: {} } }, async () => ({ ok: true, result: {} }), '0')
    expect(call).toMatchObject({ error: { code: -32602 } })
  })

  it('a caller the person did not enroll, or revoked, is refused', async () => {
    const { broker, mac } = fakeBroker([])
    const enrollment = new Enrollment(join(temp(), 'callers.json'))
    expect(await serve(broker, enrollment, { op: 'control', token: 'made-up', args: { control: { kind: 'open_app', app: 'Safari' } } })).toMatchObject({ ok: false, error: expect.stringMatching(/not enrolled/) })
    const { token } = enrollment.add('opencode')
    expect(await serve(broker, enrollment, { op: 'control', token, args: { control: { kind: 'open_app', app: 'Safari' } } })).toMatchObject({ ok: true, result: { ok: true } })
    enrollment.revoke('opencode')
    expect(await serve(broker, enrollment, { op: 'control', token, args: { control: { kind: 'open_app', app: 'Safari' } } })).toMatchObject({ ok: false })
    expect(mac.controls.calls).toHaveLength(1)
  })

  it('MCP over the socket: a computer_use task from another harness still waits for an approval here', async () => {
    const dir = temp()
    const { broker, mac } = fakeBroker(['c4', 'c3', 'done'])
    const enrollment = new Enrollment(join(dir, 'callers.json'))
    const { token } = enrollment.add('claude-code')
    const server = new ExecutorServer(join(dir, 'executor.sock'), broker, enrollment)
    await server.listen()
    expect(statSync(server.path).mode & 0o777).toBe(0o600)
    const client = new ExecutorClient(server.path, token)
    const call = (op: string, args: Record<string, unknown>) => client.request(op, args)
    const listed = await handleMcp({ jsonrpc: '2.0', id: 1, method: 'tools/list' }, call, '0')
    expect((listed?.result as { tools: unknown[] }).tools).toHaveLength(TOOLS.length)
    const approved = new Promise<void>((resolve) => broker.on((e) => { if (e.type === 'approval') { expect(e.card.requester).toBe('MCP: claude-code'); void broker.answer(e.card.id, 'approve', 'app window').then(() => resolve()) } }))
    const result = await handleMcp({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'computer_use', arguments: { goal: 'send the Groceries note', wait_s: 30 } } }, call, '0')
    await approved
    const text = JSON.parse(((result?.result as { content: Array<{ text: string }> }).content[0]).text) as { status: string }
    expect(text.status).toBe('done')
    expect(mac.desktop.sent).toEqual(['Groceries: milk, eggs, bread'])
    client.close()
    await server.close()
  })

  it('reads the front window for an enrolled harness: the tree, and a scaled screenshot as MCP image content', async () => {
    const { broker, audit } = fakeBroker([])
    const enrollment = new Enrollment(join(temp(), 'callers.json'))
    const { token } = enrollment.add('codex')
    const call = (op: string, args: Record<string, unknown>) => serve(broker, enrollment, { op, token, args })
    const tree = await handleMcp({ jsonrpc: '2.0', id: 1, method: 'tools/call', params: { name: 'computer_observe', arguments: {} } }, call, '0')
    const seen = JSON.parse(((tree?.result as { content: Array<{ text: string }> }).content[0]).text) as { app: string; controls: Array<{ label: string }> }
    expect(seen.app).toBe('Notes')
    expect(seen.controls.map((c) => c.label)).toContain('Send')
    const shot = await handleMcp({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: 'computer_screenshot', arguments: { max_edge: 1000 } } }, call, '0')
    const content = (shot?.result as { content: Array<{ type: string; data?: string; mimeType?: string; text?: string }> }).content
    expect(content[0]).toMatchObject({ type: 'image', mimeType: 'image/png' })
    expect(Buffer.from(content[0].data ?? '', 'base64').subarray(0, 4)).toEqual(Buffer.from([0x89, 0x50, 0x4e, 0x47]))
    const meta = JSON.parse(content[1].text ?? '{}') as { width: number; height: number; scale: number; region: { x: number } }
    expect([meta.width, meta.height]).toEqual([1000, 667])
    expect(meta.scale).toBeCloseTo(1000 / 1200)
    expect(audit.tail(10).map((line) => line.event)).toEqual(expect.arrayContaining(['observe', 'screenshot']))
  })

  it('never reads for the box or a voice turn, while switched off, or when LocalRouter itself is in front', async () => {
    const { broker } = fakeBroker([])
    for (const kind of ['box', 'voice'] as const) {
      expect(await broker.observe({ caller: { id: kind, name: kind, kind }, utterance: 'what is on my screen' })).toMatchObject({ ok: false, error: expect.stringMatching(/only for harnesses/) })
    }
    broker.setDisabled(true)
    expect(await broker.screenshot({ caller: { id: 'm', name: 'MCP: x', kind: 'mcp' } })).toMatchObject({ ok: false, error: expect.stringMatching(/switched off/) })
    const own = fakeBroker([], { ownApps: ['Notes'] }).broker
    expect(await own.observe({ caller: { id: 'm', name: 'MCP: x', kind: 'mcp' } })).toMatchObject({ ok: false, error: expect.stringMatching(/never read/) })
    expect(await own.screenshot({ caller: { id: 'm', name: 'MCP: x', kind: 'mcp' } })).toMatchObject({ ok: false, error: expect.stringMatching(/never read/) })
  })
})
