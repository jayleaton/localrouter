import type { CommandOptions, CommandRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import type { Session, Tool } from './session.ts'

/** Fakes for the Linux platform's tests: no test here runs a desktop tool. */

export type Call = { file: string; args: string[]; input?: string; options: CommandOptions }

/** The program's name without its directory ("/usr/bin/xdotool" is "xdotool"). */
export const tool = (call: Call) => call.file.split('/').pop() ?? call.file

/** The op of an AT-SPI script call (`python3 -I -c SCRIPT <op>`), or null for any other call. */
export const atspiOp = (call: Call) => (tool(call) === 'python3' && call.args[1] === '-c' && call.args.length === 4 ? call.args[3] : null)

/** A runner that records every call and answers with `respond` (an Error is thrown). */
export function fakeRunner(respond: (call: Call) => string | Error = () => '') {
  const calls: Call[] = []
  const runner: CommandRunner = {
    async run(file, args, options = {}) {
      const call: Call = { file, args, options, ...(options.input !== undefined ? { input: options.input } : {}) }
      calls.push(call)
      const out = respond(call)
      if (out instanceof Error) throw out
      return out
    },
  }
  /** Each call as "tool arg arg…", scripts shown by their op. */
  const lines = () => calls.map((call) => (atspiOp(call) ? `atspi ${atspiOp(call)}` : [tool(call), ...call.args].join(' ')))
  return { runner, calls, lines }
}

/** A detected session with `tools` installed in /usr/bin. */
export function fakeSession(over: Partial<Omit<Session, 'tools'>> & { tools?: Tool[] } = {}): Session {
  const { tools = [], ...rest } = over
  return {
    server: 'x11',
    compositor: 'other',
    distro: 'debian',
    ydotoold: false,
    atspi: true,
    gio: true,
    pixbuf: true,
    desktop: '',
    env: {},
    ...rest,
    tools: Object.fromEntries(tools.map((name) => [name, `/usr/bin/${name}`])),
  }
}
