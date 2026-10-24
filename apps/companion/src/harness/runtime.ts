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
import type { Tool } from './registry.ts'
import { Spark } from './spark.ts'
import { computerTools, fileTools, pictureTools, VideoHandoff } from './tools.ts'

/**
 * The reference harness put together, the same for the app and the headless runner: the Spark client, this computer's
 * executor (decisions asked of the Spark's chat model, as advice), the tool registry (computer control through the
 * executor, shared folders, the Spark's media), the journal, and realtime sessions in client tool mode.
 */
export const HARNESS_INSTRUCTIONS = [
  'You run inside the LocalRouter app on the user’s own computer, and its tools run there.',
  'computer_control and computer_use act on this computer; anything that could send, buy, delete or post waits for the user to approve it on screen, so say that you are asking them.',
  'list_files and read_file see only the folders the user shared. generate_image makes a picture on their LocalRouter box and edit_image changes one; the user sees it when it is done.',
  'A video is never made during a call: generate_video gives you a question to ask first; follow what it returns.',
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
  /** A call's tools (computer, files, pictures; a video only through the handoff). */
  registry: Registry
  /** Typed chat's local tools. */
  chat: Registry
  journal: Journal
  session(o?: Partial<SessionOptions>): ClientToolsSession
  close(): Promise<void>
}

/** The tools a voice call offers. */
export function voiceRegistry(local: Tool[], picture: Tool[], handoff: VideoHandoff): Registry {
  return new Registry().add(...local, ...picture, ...handoff.tools())
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
  const local = () => [...computerTools(executor.broker, VOICE_CALLER), ...fileTools(o.shared)]
  // Typed chat: this computer's tools and picture edits (the page makes its own pictures and videos, with their cards).
  const chat = new Registry().add(...local(), ...pictureTools(spark).filter((t) => t.name === 'edit_image'))
  const journal = new Journal(join(o.dir, 'conversations'))
  return {
    spark,
    executor,
    registry: voiceRegistry(local(), pictureTools(spark), new VideoHandoff()),
    chat,
    journal,
    // Each call gets its own registry, so a video proposal belongs to that call only.
    session: (s = {}) => new ClientToolsSession({ spark, registry: voiceRegistry(local(), pictureTools(spark), new VideoHandoff()), journal, instructions: HARNESS_INSTRUCTIONS, ...s }),
    close: () => executor.close(),
  }
}
