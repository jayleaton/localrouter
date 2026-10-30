import { randomBytes } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import type { Decider } from '../../../desktop-helper/src/computer/decisions.ts'
import { silentLogger } from '../../../desktop-helper/src/log.ts'
import type { Logger } from '../../../desktop-helper/src/log.ts'
import { LocalStop } from '../../../desktop-helper/src/stop.ts'
import { createPlatform } from '../platform/index.ts'
import { stopShortcut } from '../platform/shortcut.ts'
import type { Built, PlatformChoice } from '../platform/index.ts'
import { FileAudit } from './audit.ts'
import { Broker } from './broker.ts'
import type { PresenceGate, Standing } from './broker.ts'
import { auditPath, callersPath, executorSocket, identityPath } from './paths.ts'
import { Enrollment, ExecutorServer } from './server.ts'

/**
 * The executor put together: this computer's platform (platform/: macOS, Linux, Windows, or the fake Notes app), its
 * broker with the local stop, the audit file and the caller enrollment in the data folder, and the socket other
 * harnesses reach it on. The app and the headless runner both start it this way. The policy, approvals, budgets,
 * audit and stop are the same on every OS; only the platform differs.
 */
export type MachineKind = PlatformChoice

export type { Built }

/** This installation's stable id and the computer's name, kept in the data folder. */
export function identity(dir: string, name: string): { executor: string; name: string } {
  try {
    const doc = JSON.parse(readFileSync(identityPath(dir), 'utf8')) as { executor?: unknown; name?: unknown }
    if (typeof doc.executor === 'string' && doc.executor) return { executor: doc.executor, name: typeof doc.name === 'string' && doc.name ? doc.name : name }
  } catch {
    // First start.
  }
  const made = { executor: `exec_${randomBytes(8).toString('hex')}`, name }
  mkdirSync(dir, { recursive: true, mode: 0o700 })
  writeFileSync(identityPath(dir), `${JSON.stringify(made)}\n`, { mode: 0o600 })
  return made
}

export type ExecutorOptions = {
  dir: string
  name: string
  machine: MachineKind
  decider: Decider | null
  gate: PresenceGate
  /** The person's standing grant for computer control (BrokerOptions.standing). */
  standing?: Standing
  /** The app's own process and window names, never targets. */
  ownApps?: string[]
  /** Why nobody can approve right now (no window shows approval cards), or null: see BrokerOptions.approvalSurface. */
  approvalSurface?: () => string | null
  /** Listen on the socket for other harnesses (MCP). */
  listen?: boolean
  log?: Logger
}

export type Executor = { broker: Broker; enrollment: Enrollment; server: ExecutorServer | null; built: Built; close(): Promise<void> }

export async function startExecutor(o: ExecutorOptions): Promise<Executor> {
  const id = identity(o.dir, o.name)
  const built = await createPlatform(o.machine, { dataDir: o.dir })
  const platform = built.platform
  const log = o.log ?? silentLogger
  const pointer = platform.pointer
  const displays = platform.displays
  const stop = pointer ? new LocalStop({ pointer, ...(displays ? { displays } : {}), log }) : null
  const broker = new Broker({
    executorId: id.executor,
    name: id.name,
    machine: platform,
    stopHint: pointer ? 'Move the pointer to the top-left corner to stop.' : `Press ${stopShortcut(platform.os).words}, or use LocalRouter’s tray menu, to stop.`,
    decider: o.decider,
    audit: new FileAudit(auditPath(o.dir)),
    gate: o.gate,
    ...(o.standing ? { standing: o.standing } : {}),
    ...(stop ? { watch: (abort: (reason: string) => void) => stop.watch(abort) } : {}),
    ownApps: o.ownApps ?? ['LocalRouter', 'Electron'],
    ...(o.approvalSurface ? { approvalSurface: o.approvalSurface } : {}),
    log,
  })
  const enrollment = new Enrollment(callersPath(o.dir))
  let server: ExecutorServer | null = null
  if (o.listen) {
    server = new ExecutorServer(executorSocket(o.dir), broker, enrollment)
    await server.listen()
  }
  return {
    broker,
    enrollment,
    server,
    built,
    close: async () => {
      broker.stop(undefined, 'shutdown', 'LocalRouter is quitting.')
      await server?.close()
      platform.close?.()
    },
  }
}
