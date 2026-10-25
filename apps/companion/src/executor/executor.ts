import { randomBytes } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { ChromeTabs } from '../../../desktop-helper/src/computer/chromeTabs.ts'
import type { Decider } from '../../../desktop-helper/src/computer/decisions.ts'
import { DesktopBackend, macAccess } from '../../../desktop-helper/src/computer/desktop.ts'
import { DockApps } from '../../../desktop-helper/src/computer/dock.ts'
import { MacControls } from '../../../desktop-helper/src/computer/macControls.ts'
import { fakeMac } from '../../../desktop-helper/src/fake.ts'
import { silentLogger } from '../../../desktop-helper/src/log.ts'
import type { Logger } from '../../../desktop-helper/src/log.ts'
import { LocalStop } from '../../../desktop-helper/src/stop.ts'
import { FileAudit } from './audit.ts'
import { Broker } from './broker.ts'
import type { Machine, PresenceGate } from './broker.ts'
import { auditPath, callersPath, executorSocket, identityPath } from './paths.ts'
import { Enrollment, ExecutorServer } from './server.ts'

/**
 * The executor put together: this computer's machine (the Mac backend ported from the desktop helper, or the fake
 * Mac), its broker with the local stop, the audit file and the caller enrollment in the data folder, and the socket
 * other harnesses reach it on. The app and the headless runner both start it this way.
 */
export type MachineKind = 'mac' | 'fake'

export type Access = { accessibility: boolean; screen_recording: boolean }

export type Built = { machine: Machine; access: () => Promise<Access>; pointer: (() => Promise<{ x: number; y: number }>) | null; fake: ReturnType<typeof fakeMac> | null }

export function buildMachine(kind: MachineKind): Built {
  if (kind === 'fake') {
    const mac = fakeMac()
    return { machine: { desktop: mac.desktop, tabs: null, dock: mac.dock, apps: mac.apps, controls: mac.controls }, access: mac.access, pointer: () => mac.desktop.pointer(), fake: mac }
  }
  const controls = new MacControls()
  const desktop = new DesktopBackend()
  return {
    machine: { desktop, tabs: new ChromeTabs(), dock: new DockApps(), apps: { running: () => controls.running(), activate: (name) => controls.activate(name) }, controls },
    access: async () => {
      const a = await macAccess()
      return { accessibility: a.accessibility, screen_recording: a.screenRecording }
    },
    pointer: () => desktop.pointer(),
    fake: null,
  }
}

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
  /** The app's own process and window names, never targets. */
  ownApps?: string[]
  /** Listen on the socket for other harnesses (MCP). */
  listen?: boolean
  log?: Logger
}

export type Executor = { broker: Broker; enrollment: Enrollment; server: ExecutorServer | null; built: Built; close(): Promise<void> }

export async function startExecutor(o: ExecutorOptions): Promise<Executor> {
  const id = identity(o.dir, o.name)
  const built = buildMachine(o.machine)
  const log = o.log ?? silentLogger
  const desktop = built.machine.desktop
  const stop = built.pointer ? new LocalStop({ pointer: built.pointer, ...(desktop?.displays ? { displays: () => desktop.displays!() } : {}), log }) : null
  const broker = new Broker({
    executorId: id.executor,
    name: id.name,
    machine: built.machine,
    decider: o.decider,
    audit: new FileAudit(auditPath(o.dir)),
    gate: o.gate,
    ...(stop ? { watch: (abort: (reason: string) => void) => stop.watch(abort) } : {}),
    ownApps: o.ownApps ?? ['LocalRouter', 'Electron'],
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
    },
  }
}
