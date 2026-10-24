import type { Reply } from './server.ts'

/**
 * The executor as an MCP server (stdio): any harness on this computer (OpenCode, Claude Code, Codex, our own) can use
 * computer control through it, and the executor's lease, policy and approvals apply to it exactly as to our app. The
 * tools are bounded: a fixed control, a computer-use task with a goal, its status, and stop. There is no tool that
 * approves, configures or reaches a shell. Approvals appear in the LocalRouter app on this computer.
 */
export const PROTOCOL_VERSIONS = ['2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05']

const CONTROL_SCHEMA = {
  type: 'object',
  description: 'One fixed control: {"kind":"open_app","app"}, {"kind":"quit_app","app"}, {"kind":"volume","action":"up|down|mute|unmute|set","level"?,"target":"system|Spotify|Music"}, {"kind":"media","action":"play|pause|next|previous","player":"Spotify|Music"}, {"kind":"dark_mode","action":"on|off|toggle"}, {"kind":"lock_screen"} (asks the person first)',
  properties: {
    kind: { type: 'string', enum: ['open_app', 'quit_app', 'volume', 'media', 'dark_mode', 'lock_screen'] },
    app: { type: 'string' },
    action: { type: 'string' },
    level: { type: 'integer', minimum: 0, maximum: 100 },
    target: { type: 'string', enum: ['system', 'Spotify', 'Music'] },
    player: { type: 'string', enum: ['Spotify', 'Music'] },
  },
  required: ['kind'],
  additionalProperties: false,
}

export const TOOLS = [
  { name: 'computer_capabilities', description: 'What this computer’s executor offers: its controls, whether computer use is available, its limits, how approvals are confirmed, and whether control is switched off.', inputSchema: { type: 'object', properties: {}, additionalProperties: false } },
  { name: 'computer_control', description: 'Run one fixed control on this computer (open or quit an app, volume, media, dark mode, lock the screen). Locking asks the person on this computer first.', inputSchema: { type: 'object', properties: { control: CONTROL_SCHEMA }, required: ['control'], additionalProperties: false } },
  {
    name: 'computer_use',
    description: 'Do a small task on this computer’s screen (at most 15 steps, 90 s). Steps that could send, buy, delete, post or type into a field that is not a search box wait for the person to approve them on this computer. Returns the outcome (done, stuck, blocked, unavailable, stopped, failed) with its steps, or the task while it still runs.',
    inputSchema: { type: 'object', properties: { goal: { type: 'string', maxLength: 1000 }, text: { type: 'string', description: 'the exact text to type, if the task types anything', maxLength: 4000 }, wait_s: { type: 'integer', minimum: 0, maximum: 600, description: 'how long to wait for the outcome (default 120)' } }, required: ['goal'], additionalProperties: false },
  },
  { name: 'computer_status', description: 'A task’s state (and whether it waits for an approval), or this computer’s recent tasks.', inputSchema: { type: 'object', properties: { task_id: { type: 'string' }, wait_s: { type: 'integer', minimum: 0, maximum: 600 } }, additionalProperties: false } },
  { name: 'computer_stop', description: 'Stop the running task (or the one named).', inputSchema: { type: 'object', properties: { task_id: { type: 'string' } }, additionalProperties: false } },
] as const

const OP: Record<string, { op: string; args: (a: Record<string, unknown>) => Record<string, unknown> }> = {
  computer_capabilities: { op: 'capabilities', args: () => ({}) },
  computer_control: { op: 'control', args: (a) => ({ control: a.control }) },
  computer_use: { op: 'use', args: (a) => ({ goal: a.goal, text: a.text ?? null, wait_s: a.wait_s }) },
  computer_status: { op: 'status', args: (a) => ({ task_id: a.task_id, wait_s: a.wait_s }) },
  computer_stop: { op: 'stop', args: (a) => ({ task_id: a.task_id }) },
}

type Rpc = { jsonrpc?: string; id?: string | number | null; method?: string; params?: Record<string, unknown> }

/** Answers one JSON-RPC message; null for a notification. `call` reaches the executor. */
export async function handleMcp(message: Rpc, call: (op: string, args: Record<string, unknown>) => Promise<Reply>, version: string): Promise<Record<string, unknown> | null> {
  const id = message.id ?? null
  const ok = (result: unknown) => ({ jsonrpc: '2.0', id, result })
  const fail = (code: number, msg: string) => ({ jsonrpc: '2.0', id, error: { code, message: msg } })
  if (message.id === undefined) return null // notifications (initialized, cancelled) need no answer
  switch (message.method) {
    case 'initialize': {
      const asked = typeof message.params?.protocolVersion === 'string' ? message.params.protocolVersion : PROTOCOL_VERSIONS[0]
      return ok({ protocolVersion: PROTOCOL_VERSIONS.includes(asked) ? asked : PROTOCOL_VERSIONS[0], capabilities: { tools: {} }, serverInfo: { name: 'localrouter-executor', version } })
    }
    case 'ping':
      return ok({})
    case 'tools/list':
      return ok({ tools: TOOLS })
    case 'tools/call': {
      const name = typeof message.params?.name === 'string' ? message.params.name : ''
      const route = OP[name]
      if (!route) return fail(-32602, `unknown tool ${JSON.stringify(name)}: this server offers ${TOOLS.map((t) => t.name).join(', ')}`)
      const args = (typeof message.params?.arguments === 'object' && message.params.arguments !== null ? message.params.arguments : {}) as Record<string, unknown>
      const reply = await call(route.op, route.args(args))
      const failed = !reply.ok || (typeof reply.result === 'object' && reply.result !== null && (reply.result as { ok?: unknown }).ok === false)
      const text = reply.ok ? JSON.stringify(reply.result) : reply.error
      return ok({ content: [{ type: 'text', text }], isError: failed })
    }
    default:
      return fail(-32601, `method not found: ${String(message.method)}`)
  }
}
