import { describe, expect, it } from 'vitest'
import { ComputerUse } from '../../../../desktop-helper/src/computer/controller.ts'
import type { AuthorizeRequest, Grant } from '../../../../desktop-helper/src/computer/controller.ts'
import type { Answers, Question } from '../../../../desktop-helper/src/computer/decisions.ts'
import { silentLogger } from '../../../../desktop-helper/src/log.ts'
import { WindowsScreen } from './screen.ts'
import type { WindowsCall } from './worker.ts'

/**
 * "Go to example.com" in Chrome on Windows, through the real Windows backend (WindowsScreen) and controller, with the
 * helper's own observation shape: Chrome's omnibox is an Edit named "Address and search bar" (ref with its RuntimeId,
 * `focused` from HasKeyboardFocus), and the page's URL is left out while the bar has the focus. Cody's runs on bc910bc:
 * cu_7e3cfcaa1b35 kept clicking the address bar for 15 steps; cu_aeb7e27bccad typed and pressed Enter in 3.
 */
type Chrome = { bar: string; focused: boolean; page: string; enterLost: number }

const BAR = '1311234:0.3.1#42.1311234.7'
const RELOAD = '1311234:0.2#42.1311234.3'

function chrome(start: Partial<Chrome> = {}) {
  const s: Chrome = { bar: 'github.com/jayleaton/localrouter', focused: false, page: 'github.com/jayleaton/localrouter', enterLost: 0, ...start }
  const ops: string[] = []
  const worker: WindowsCall = {
    async call(op, args = {}) {
      ops.push(op)
      switch (op) {
        case 'observe': {
          const loaded = s.page.startsWith('example.com')
          return {
            app: 'Google Chrome',
            title: loaded ? 'Example Domain - Google Chrome' : 'jayleaton/localrouter - Google Chrome',
            text: loaded ? 'Example Domain This domain is for use in documentation examples' : 'LocalRouter',
            // The helper reads the page from the bar only while it is not focused (else it holds what is typed).
            ...(s.focused ? {} : { url: s.page }),
            window: { x: -8, y: -8, w: 1936, h: 1048 },
            truncated: false,
            controls: [
              { ref: RELOAD, role: 'button', label: 'Reload', bounds: { x: 90, y: 50, w: 34, h: 34 } },
              { ref: BAR, role: 'textbox', label: 'Address and search bar', value: s.bar, ...(s.focused ? { focused: true } : {}), bounds: { x: 180, y: 50, w: 1400, h: 34 } },
            ],
          }
        }
        case 'semantic': {
          if (args.kind === 'type' && args.ref === BAR) {
            // ValuePattern.SetValue, after SetFocus (the helper's order since #104).
            s.focused = true
            s.bar = String(args.text)
            return { done: true, how: 'value', focused: true }
          }
          return { done: false, how: '' } // an Edit has no Invoke: a click goes to the pointer
        }
        case 'point':
          return { hwnd: 1311234, x: 880, y: 67 }
        case 'click':
          s.focused = true // a click in the omnibox focuses it
          return { x: 880, y: 67 }
        case 'key':
          if (args.vk === 0x0d) {
            if (s.focused && s.enterLost > 0) {
              // The first Enter after typing went elsewhere: the bar puts the page's address back.
              s.enterLost -= 1
              s.bar = s.page
            } else if (s.focused && /example\.com/.test(s.bar)) {
              s.page = 'example.com'
              s.bar = 'example.com'
              s.focused = false
            }
          }
          return {}
        default:
          return {}
      }
    },
  }
  return { state: s, worker, ops }
}

/**
 * A decider with the preference the Windows runs showed: the focused address bar whenever it is offered, else type,
 * then Enter, then done. Only offered keys are chosen (the controller refuses anything else).
 */
function barLover() {
  const asked: Array<Record<string, Question>> = []
  return {
    asked,
    async ask(_state: unknown, questions: Record<string, Question>): Promise<Answers> {
      asked.push(questions)
      const out: Answers = {}
      const pick = (id: string, prefer: (keys: string[], q: Extract<Question, { type: 'choice' }>) => string | undefined) => {
        const q = questions[id]
        if (q?.type !== 'choice') return
        const keys = Object.keys(q.criteria)
        const key = prefer(keys, q) ?? (keys.includes('stuck') ? 'stuck' : keys.includes('none') ? 'none' : keys[0])
        out[id] = { type: 'choice', choice: key, probability: 0.9 }
      }
      pick('surface', () => 'current')
      pick('next', (keys, q) => keys.find((k) => /Address and search bar/.test(q.criteria[k])) ?? keys.find((k) => k === 'type_text') ?? keys.find((k) => k === 'press_enter') ?? keys.find((k) => k === 'done'))
      pick('type_into', (keys, q) => keys.find((k) => /Address and search bar/.test(q.criteria[k])))
      return out
    },
  }
}

async function run(goal: string, text: string | null, start: Partial<Chrome> = {}) {
  const c = chrome(start)
  const decider = barLover()
  const requests: AuthorizeRequest[] = []
  const computer = new ComputerUse({
    decider,
    authorize: async (r): Promise<Grant> => {
      requests.push(r)
      return { ok: true, grant: 'allow' }
    },
    tabs: null,
    desktop: new WindowsScreen(c.worker),
    dock: null,
    apps: null,
    log: silentLogger,
    settleMs: 0,
    sleep: async () => undefined,
  })
  const outcome = await computer.run({ goal, text })
  return { outcome, requests, chrome: c.state, asked: decider.asked }
}

describe('going to an address in Chrome on Windows, through the Windows backend (bc910bc: 15 steps of clicking the address bar)', () => {
  it.each(['go to example.com', 'open Chrome and go to example.com', 'take me to example.com in Chrome', 'navigate Chrome to https://example.com', 'bring up example.com'])(
    '%s: the address is the text to type, and the focused bar is typed into, not clicked again',
    async (goal) => {
      const { outcome, requests, chrome: state } = await run(goal, null)
      expect(outcome.status).toBe('done')
      expect(state.page).toBe('example.com')
      expect(requests.length).toBeLessThanOrEqual(4)
      expect(requests.find((r) => r.action.kind === 'type')?.action).toMatchObject({ kind: 'type', label: 'Address and search bar', text: expect.stringMatching(/example\.com/) })
    },
  )

  it('an Enter that went elsewhere (the bar put the old address back while it kept the focus): types the address again, no loop', async () => {
    const { outcome, requests, chrome: state } = await run('go to example.com', null, { enterLost: 1 })
    expect(outcome.status).toBe('done')
    expect(state.page).toBe('example.com')
    // Into the bar (it started without the focus), type, the lost Enter, type again, Enter: the page is there.
    expect(requests.map((r) => r.action.kind)).toEqual(['click', 'type', 'press', 'type', 'press'])
  })

  it('a goal with no address to type still ends in a few steps instead of clicking the bar for all 15', async () => {
    const { requests } = await run('make Chrome show the page I was on', null, { focused: true })
    // The focused bar is never offered as a click, so it is not clicked at all.
    expect(requests.filter((r) => r.action.kind === 'click')).toEqual([])
  })
})
