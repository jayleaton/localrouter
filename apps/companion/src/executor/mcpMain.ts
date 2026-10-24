#!/usr/bin/env node
import { createInterface } from 'node:readline'
import { handleMcp } from './mcp.ts'
import { dataDir, executorSocket } from './paths.ts'
import { ExecutorClient } from './server.ts'

/**
 * `localrouter-executor-mcp`: the LocalRouter executor over MCP stdio, for any harness on this computer. It is a thin
 * bridge to the running LocalRouter app (which holds the executor, its policy and its approval cards):
 *
 *   LOCALROUTER_EXECUTOR_TOKEN=<token from LocalRouter › Settings › Harnesses> localrouter-executor-mcp
 *
 * `LOCALROUTER_EXECUTOR_SOCKET` names another socket (default: the app's data folder).
 */
const VERSION = '0.1.0'
const token = process.env.LOCALROUTER_EXECUTOR_TOKEN?.trim() ?? ''
const client = new ExecutorClient(process.env.LOCALROUTER_EXECUTOR_SOCKET?.trim() || executorSocket(dataDir()), token)
const call = async (op: string, args: Record<string, unknown>) => {
  try {
    return await client.request(op, args)
  } catch (error) {
    return { ok: false as const, error: error instanceof Error ? error.message : String(error) }
  }
}

const lines = createInterface({ input: process.stdin })
let chain = Promise.resolve()
lines.on('line', (line) => {
  if (!line.trim()) return
  chain = chain.then(async () => {
    let reply: Record<string, unknown> | null
    try {
      reply = await handleMcp(JSON.parse(line), call, VERSION)
    } catch {
      reply = { jsonrpc: '2.0', id: null, error: { code: -32700, message: 'parse error' } }
    }
    if (reply) process.stdout.write(`${JSON.stringify(reply)}\n`)
  })
})
lines.on('close', () => {
  void chain.then(() => client.close())
})
