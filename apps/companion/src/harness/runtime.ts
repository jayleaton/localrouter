import { join } from 'node:path'
import { chatDecider } from '../executor/decide.ts'
import { startExecutor } from '../executor/executor.ts'
import type { Executor, MachineKind } from '../executor/executor.ts'
import type { Caller, PresenceGate } from '../executor/broker.ts'
import type { Logger } from '../../../desktop-helper/src/log.ts'
import { Journal } from './journal.ts'
import { ClientToolsSession } from './realtime.ts'
import type { SessionOptions } from './realtime.ts'
import { Registry } from './registry.ts'
import { Spark } from './spark.ts'
import { computerTools, fileTools, mediaTools } from './tools.ts'

/**
 * The reference harness put together, the same for the app and the headless runner: the Spark client, this computer's
 * executor (decisions asked of the Spark's chat model, as advice), the tool registry (computer control through the
 * executor, shared folders, the Spark's media), the journal, and realtime sessions in client tool mode.
 */
export const HARNESS_INSTRUCTIONS = [
  'You run inside the LocalRouter app on the user’s own computer, and its tools run there.',
  'computer_control and computer_use act on this computer; anything that could send, buy, delete or post waits for the user to approve it on screen, so say that you are asking them.',
  'list_files and read_file see only the folders the user shared. generate_image and generate_video make media on their LocalRouter box; the user sees it when it is done.',
].join(' ')

/** Thinking off for decisions and voice, as the box's own voice settings (`voice.chat_extra`). */
export const NO_THINKING = { reasoning_effort: 'none', chat_template_kwargs: { enable_thinking: false } }

export type RuntimeOptions = {
  base: string
  token: string
  dir: string
  name: string
  machine: MachineKind
  gate: PresenceGate
  shared: () => string[]
  listen?: boolean
  ownApps?: string[]
  log?: Logger
  fetch?: typeof fetch
}

export type Runtime = {
  spark: Spark
  executor: Executor
  registry: Registry
  journal: Journal
  session(o?: Partial<SessionOptions>): ClientToolsSession
  close(): Promise<void>
}

export const VOICE_CALLER: Caller = { id: 'voice', name: 'Jarvis (voice)', kind: 'voice' }

export async function startRuntime(o: RuntimeOptions): Promise<Runtime> {
  const spark = new Spark(o.base, o.token, o.fetch)
  const executor = await startExecutor({
    dir: o.dir,
    name: o.name,
    machine: o.machine,
    decider: chatDecider((body) => spark.complete(body), NO_THINKING),
    gate: o.gate,
    listen: o.listen ?? false,
    ...(o.ownApps ? { ownApps: o.ownApps } : {}),
    ...(o.log ? { log: o.log } : {}),
  })
  const registry = new Registry().add(...computerTools(executor.broker, VOICE_CALLER), ...fileTools(o.shared), ...mediaTools(spark))
  const journal = new Journal(join(o.dir, 'conversations'))
  return {
    spark,
    executor,
    registry,
    journal,
    session: (s = {}) => new ClientToolsSession({ spark, registry, journal, instructions: HARNESS_INSTRUCTIONS, ...s }),
    close: () => executor.close(),
  }
}
