import { ComputerUseRuns } from '../../../desktop-helper/src/computer/controller.ts'
import type { ComputerTask, Limits, Session, StartResult } from '../../../desktop-helper/src/computer/controller.ts'
import { Connection } from '../../../desktop-helper/src/connection.ts'
import type { Dial, HelloAccess } from '../../../desktop-helper/src/connection.ts'
import { silentLogger } from '../../../desktop-helper/src/log.ts'
import type { Logger } from '../../../desktop-helper/src/log.ts'
import type { Broker, Requester } from './broker.ts'

/**
 * The box's legacy computer control, for clients that still use it (the phone app, the `/voice` page in its default
 * `server` tool mode): this computer connects out to the box's `/v1/desktop` like the old desktop helper did, with a
 * `desktop_helper` token. Everything the box asks still goes through this computer's broker: the box's grant is one
 * more input that may refuse a step, never permission, so a step the local rules refuse stays refused and a step they
 * ask about waits for an approval here even when the box already approved it.
 */
export const BOX_CALLER: Requester = { caller: { id: 'box', name: 'LocalRouter box (legacy)', kind: 'box' } }

/** The box's `use.start` and `stop`, run by the broker instead of a bare controller. */
class BrokeredRuns extends ComputerUseRuns {
  private readonly broker: Broker
  private running: { id: string; goal: string } | null = null

  constructor(broker: Broker, log: Logger) {
    super({ create: () => { throw new Error('the broker runs every task') }, log })
    this.broker = broker
  }

  override get current() {
    return this.running
  }

  override start(id: string, task: ComputerTask, session: Session, limits: Limits = {}): StartResult {
    const started = this.broker.startUse(BOX_CALLER, task.goal, task.text, { decider: session.decider, advice: session.authorize, limits, taskId: id, ...(session.onStep ? { onStep: session.onStep } : {}) })
    // The connection replies with `error` as the reason; the broker's reasons are words for the person, not just "busy".
    if (!started.ok) return { ok: false, error: started.error as 'busy' }
    this.running = { id, goal: task.goal }
    void started.finished.finally(() => {
      if (this.running?.id === id) this.running = null
    })
    return { ok: true, finished: started.finished }
  }

  override stop(id?: string, reason = 'You stopped it.'): boolean {
    return this.broker.stop(id || this.running?.id, 'the box', reason) > 0
  }
}

export type LegacyOptions = { url: string; token: string; name: string; version: string; broker: Broker; access: () => Promise<HelloAccess>; dial?: Dial; log?: Logger }

export class LegacyBridge {
  private readonly connection: Connection

  constructor(options: LegacyOptions) {
    const log = options.log ?? silentLogger
    const broker = options.broker
    this.connection = new Connection({
      url: options.url,
      token: options.token,
      helper: { name: options.name, platform: process.platform, version: options.version },
      access: options.access,
      controls: {
        run: async (control) => {
          const r = await broker.control(BOX_CALLER, control)
          if (!r.ok) throw new Error(r.text)
          return { done: r.text }
        },
      },
      runs: new BrokeredRuns(broker, log),
      disabled: () => broker.disabled,
      ...(options.dial ? { dial: options.dial } : {}),
      log,
    })
  }

  get status() {
    return this.connection.status
  }

  start() {
    this.connection.start()
  }

  close() {
    this.connection.close()
  }
}
