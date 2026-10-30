#!/usr/bin/env node
import { execFile } from 'node:child_process'
import { mkdir, rm, writeFile } from 'node:fs/promises'
import { homedir, hostname, userInfo } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseArgs, promisify } from 'node:util'
import { ChromeTabs } from './computer/chromeTabs.ts'
import { ComputerUse, ComputerUseRuns } from './computer/controller.ts'
import type { Deps } from './computer/controller.ts'
import { DesktopBackend, macAccess } from './computer/desktop.ts'
import { DockApps } from './computer/dock.ts'
import { MacControls } from './computer/macControls.ts'
import type { ComputerControl } from './computer/macControls.ts'
import type { BrowserTabs, Point, ScreenBackend } from './computer/types.ts'
import { DISABLED_MESSAGE, checkUrl, configDir, configPath, controlSocketPath, describeConfig, desktopSocketUrl, loadConfig, saveConfig, setDisabled } from './config.ts'
import type { Config } from './config.ts'
import { assistantName, DEFAULT_ASSISTANT, startNotice } from './assistant.ts'
import { Connection } from './connection.ts'
import type { HelloAccess } from './connection.ts'
import { fakeMac } from './fake.ts'
import { createLogger } from './log.ts'
import { ControlSocket, LocalStop, notRunning, sendControl } from './stop.ts'
import type { ControlCommand, ControlReply } from './stop.ts'

/**
 * localrouter-desktop: LocalRouter's helper on the Mac. `pair` stores the box's address and device
 * token, `run` keeps the connection to the box (the LaunchAgent runs it), `stop`/`disable`/`enable`/
 * `status` talk to a running helper through its local control socket, `check` shows what macOS has
 * granted, and `install-agent`/`uninstall-agent` manage the LaunchAgent.
 */
const VERSION = '0.1.0'
const LABEL = 'com.localrouter.desktop'
const NOTICE_TITLE = 'LocalRouter Desktop'

const USAGE = `Usage: node src/main.ts <command>

  pair --url https://<box>:10443 --token <token> [--name "JJ's MacBook Pro"]
                    Store the box's address and this Mac's device token (scope desktop_helper).
  run               Connect to the box and wait for requests (the LaunchAgent runs this).
  stop              Stop the task running now.
  disable           Refuse every request until enable (persists).
  enable            Accept requests again.
  status            Show whether the helper runs, is connected, and what it is doing.
  check             Show the box address and the Accessibility and Screen Recording grants.
  install-agent     Install and start the LaunchAgent ${LABEL}.
  uninstall-agent   Stop and remove the LaunchAgent.
`

const execFileAsync = promisify(execFile)
const say = (line: string) => process.stdout.write(`${line}\n`)

class UsageError extends Error {}

async function main(argv: string[]): Promise<number> {
  const command: string | undefined = argv[0]
  const rest = argv.slice(1)
  const dir = configDir()
  switch (command) {
    case 'pair':
      return pair(dir, rest)
    case 'run':
      return run(dir)
    case 'stop':
    case 'disable':
    case 'enable':
    case 'status':
      return control(dir, command)
    case 'check':
      return check(dir)
    case 'install-agent':
      return installAgent()
    case 'uninstall-agent':
      return uninstallAgent()
    case undefined:
    case 'help':
    case '--help':
    case '-h':
      say(USAGE)
      return command === undefined ? 1 : 0
    default:
      throw new UsageError(`Unknown command: ${command}`)
  }
}

async function pair(dir: string, args: string[]): Promise<number> {
  const { values } = parseArgs({ args, options: { url: { type: 'string' }, token: { type: 'string' }, name: { type: 'string' } }, strict: true })
  if (!values.url || !values.token) throw new UsageError('pair needs --url and --token')
  const url = checkUrl(values.url)
  const token = values.token.trim()
  if (!token) throw new UsageError('The token is empty')
  const existing = await loadConfig(dir).catch(() => null)
  const name = values.name?.trim() || (await computerName())
  await saveConfig(dir, { url, token, name, disabled: existing?.disabled ?? false })
  say(`Paired with ${url} as “${name}”. The token is kept in ${configPath(dir)}, readable only by you.`)
  say('Next: node src/main.ts install-agent')
  return 0
}

/** The Mac's own name ("JJ's MacBook Pro"), else its host name. */
async function computerName(): Promise<string> {
  if (process.platform === 'darwin') {
    try {
      const { stdout } = await execFileAsync('scutil', ['--get', 'ComputerName'])
      if (stdout.trim()) return stdout.trim()
    } catch {
      // Fall back to the host name.
    }
  }
  return hostname().replace(/\.local$/i, '')
}

/** What the helper drives: the real Mac, or the fake one. */
type Mac = {
  desktop: ScreenBackend
  tabs: BrowserTabs | null
  dock: Deps['dock']
  apps: Deps['apps']
  controls: { run(control: ComputerControl): Promise<{ done: string }>; notify(title: string, text: string): Promise<void> }
  access: () => Promise<HelloAccess>
  pointer: () => Promise<Point>
}

function realMac(): Mac {
  const controls = new MacControls()
  const desktop = new DesktopBackend()
  return {
    desktop,
    tabs: new ChromeTabs(),
    dock: new DockApps(),
    apps: { running: () => controls.running(), activate: (name) => controls.activate(name) },
    controls,
    access: async () => {
      const access = await macAccess()
      return { accessibility: access.accessibility, screen_recording: access.screenRecording }
    },
    pointer: () => desktop.pointer(),
  }
}

function fakeOne(): Mac {
  const mac = fakeMac()
  return { desktop: mac.desktop, tabs: null, dock: mac.dock, apps: mac.apps, controls: mac.controls, access: mac.access, pointer: () => mac.desktop.pointer() }
}

async function run(dir: string): Promise<number> {
  const fake = process.env.LOCALROUTER_DESKTOP_FAKE === '1'
  if (process.platform !== 'darwin' && !fake) throw new Error('LocalRouter Desktop runs on macOS. Set LOCALROUTER_DESKTOP_FAKE=1 to run it against a fake Mac.')
  const paired = await loadConfig(dir)
  if (!paired) throw new Error('This Mac is not paired yet: run `node src/main.ts pair --url … --token …` first.')
  let config: Config = paired
  const log = createLogger()
  const mac = fake ? fakeOne() : realMac()
  const stop = new LocalStop({ pointer: mac.pointer, log })
  // The box's name for its assistant, read again each time it welcomes the helper; the default until it answers.
  let assistant = DEFAULT_ASSISTANT
  const lookUpAssistant = () =>
    void assistantName(config.url, config.token).then((name) => {
      if (name !== assistant) log.info('assistant.name', { name })
      assistant = name
    })
  const runs = new ComputerUseRuns({
    create: (session) => new ComputerUse({ ...session, tabs: mac.tabs, desktop: mac.desktop, dock: mac.dock, apps: mac.apps, log }),
    watch: (abort) => stop.watch(abort),
    onStart: (task) => void mac.controls.notify(NOTICE_TITLE, startNotice(assistant, task.goal)),
    log,
  })
  const connection = new Connection({
    url: config.url,
    token: config.token,
    helper: { name: config.name, platform: process.platform, version: VERSION },
    access: mac.access,
    controls: mac.controls,
    runs,
    disabled: () => config.disabled,
    onWelcome: lookUpAssistant,
    log,
  })
  const handle = async (command: ControlCommand): Promise<ControlReply> => {
    switch (command) {
      case 'stop':
        return runs.stop(undefined, 'You stopped it on the Mac.') ? { ok: true, message: 'Stopped the task.' } : { ok: true, message: 'No task is running.' }
      case 'disable':
        config = await setDisabled(dir, true)
        runs.stop(undefined, `${DISABLED_MESSAGE}.`)
        return { ok: true, message: `${DISABLED_MESSAGE} until \`enable\`.` }
      case 'enable':
        config = await setDisabled(dir, false)
        return { ok: true, message: 'Computer control is switched on again.' }
      case 'status': {
        const status = connection.status
        return { ok: true, message: JSON.stringify({ running: true, fake, connected: status.connected, helper_id: status.helperId, task: status.task, ...describeConfig(config) }) }
      }
    }
  }
  await mkdir(dir, { recursive: true, mode: 0o700 })
  const socket = new ControlSocket(controlSocketPath(dir), handle, log)
  await socket.listen()
  log.info('helper.started', { version: VERSION, fake, url: config.url, name: config.name, disabled: config.disabled })
  connection.start()
  await new Promise<void>((resolve) => {
    const shutdown = (signal: string) => {
      log.info('helper.stopping', { signal })
      connection.close()
      void socket.close().finally(() => resolve())
    }
    process.once('SIGINT', () => shutdown('SIGINT'))
    process.once('SIGTERM', () => shutdown('SIGTERM'))
  })
  return 0
}

async function control(dir: string, command: ControlCommand): Promise<number> {
  try {
    const reply = await sendControl(controlSocketPath(dir), command)
    say(reply.message)
    return reply.ok ? 0 : 1
  } catch (error) {
    if (!notRunning(error)) throw error
  }
  switch (command) {
    case 'disable':
    case 'enable':
      await setDisabled(dir, command === 'disable')
      say(command === 'disable' ? `LocalRouter Desktop is not running. ${DISABLED_MESSAGE} from when it starts.` : 'LocalRouter Desktop is not running. Computer control is switched on for when it starts.')
      return 0
    case 'stop':
      say('LocalRouter Desktop is not running, so nothing is running on this Mac.')
      return 0
    case 'status': {
      const config = await loadConfig(dir)
      say(JSON.stringify({ running: false, ...(config ? describeConfig(config) : { paired: false }) }))
      return 1
    }
  }
}

async function check(dir: string): Promise<number> {
  const config = await loadConfig(dir)
  say(config ? `Box: ${desktopSocketUrl(config.url)} as “${config.name}”${config.disabled ? ' (computer control switched off here)' : ''}` : 'Box: not paired yet (run pair)')
  if (process.platform !== 'darwin') {
    say('Not macOS: there is no Accessibility or Screen Recording access to check.')
    return config ? 0 : 1
  }
  try {
    const access = await macAccess()
    const who = access.host || process.execPath
    say(`Accessibility: ${access.accessibility ? 'granted' : `missing: allow ${who} in System Settings › Privacy & Security › Accessibility`}`)
    say(`Screen Recording: ${access.screenRecording ? 'granted' : `missing (only needed for apps that hide their window from accessibility): allow ${who} in System Settings › Privacy & Security › Screen Recording`}`)
    return config && access.accessibility ? 0 : 1
  } catch (error) {
    say(`Could not check access: ${error instanceof Error ? error.message : String(error)}`)
    return 1
  }
}

const escapeXml = (text: string) => text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

const agentPaths = () => ({
  plist: join(homedir(), 'Library', 'LaunchAgents', `${LABEL}.plist`),
  log: join(homedir(), 'Library', 'Logs', 'LocalRouter Desktop.log'),
  domain: `gui/${userInfo().uid}`,
})

/** The LaunchAgent: this node binary runs this file's `run` at login and again whenever it exits. */
function agentPlist(node: string, main: string, log: string, home?: string): string {
  const string = (value: string) => `<string>${escapeXml(value)}</string>`
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  ${string(LABEL)}
  <key>ProgramArguments</key>
  <array>
    ${string(node)}
    ${string(main)}
    ${string('run')}
  </array>
  <key>WorkingDirectory</key>
  ${string(dirname(dirname(main)))}
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ProcessType</key>
  ${string('Interactive')}
  <key>StandardOutPath</key>
  ${string(log)}
  <key>StandardErrorPath</key>
  ${string(log)}${home ? `
  <key>EnvironmentVariables</key>
  <dict>
    <key>LOCALROUTER_DESKTOP_HOME</key>
    ${string(home)}
  </dict>` : ''}
</dict>
</plist>
`
}

async function installAgent(): Promise<number> {
  if (process.platform !== 'darwin') throw new Error('The LaunchAgent is for macOS.')
  const paths = agentPaths()
  const main = fileURLToPath(import.meta.url)
  await mkdir(dirname(paths.plist), { recursive: true })
  await mkdir(dirname(paths.log), { recursive: true })
  await writeFile(paths.plist, agentPlist(process.execPath, main, paths.log, process.env.LOCALROUTER_DESKTOP_HOME?.trim() || undefined), { mode: 0o644 })
  // A previous copy is unloaded first so the new one takes effect; it is fine when there was none.
  await execFileAsync('launchctl', ['bootout', `${paths.domain}/${LABEL}`]).catch(() => undefined)
  await execFileAsync('launchctl', ['bootstrap', paths.domain, paths.plist])
  say(`Installed ${paths.plist}: ${process.execPath} ${main} run, at login and kept running.`)
  say(`Log: ${paths.log}`)
  say(`Grant Accessibility (and Screen Recording, for apps that hide their window) to ${process.execPath} in System Settings › Privacy & Security, then run: node src/main.ts check`)
  return 0
}

async function uninstallAgent(): Promise<number> {
  if (process.platform !== 'darwin') throw new Error('The LaunchAgent is for macOS.')
  const paths = agentPaths()
  await execFileAsync('launchctl', ['bootout', `${paths.domain}/${LABEL}`]).catch(() => undefined)
  await rm(paths.plist, { force: true })
  say(`Removed ${paths.plist}. The pairing stays in ${configDir()}; delete that folder to forget it.`)
  return 0
}

main(process.argv.slice(2)).then(
  (code) => {
    process.exitCode = code
  },
  (error: unknown) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`)
    if (error instanceof UsageError) process.stderr.write(`\n${USAGE}`)
    process.exitCode = 1
  },
)
