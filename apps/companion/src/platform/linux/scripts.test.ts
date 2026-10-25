import { execFile, execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterAll, describe, expect, it } from 'vitest'
import type { CommandRunner } from '../../../../desktop-helper/src/computer/runner.ts'
import { ATSPI_SCRIPT, LinuxScreen } from './atspi.ts'
import { PORTAL_SCREENSHOT } from './capture.ts'
import { GDK_MONITORS } from './compositor.ts'
import { MULTILINE, PORTAL_KEYS, chooseInjector } from './input.ts'
import { GI_PROBE } from './session.ts'
import { fakeRunner, fakeSession } from './testing.ts'

/**
 * The embedded Python, checked without a desktop: every script compiles, and the AT-SPI script runs against a fake
 * `gi.repository.Atspi` (plain Python objects, no D-Bus) so its walk, role mapping, password handling and
 * semantic-first actions are exercised for real. Skipped where python3 is missing.
 */
const python = (() => {
  // These are Linux scripts with Linux paths: run where Linux runs them (and macOS), not on a Windows runner.
  if (process.platform === 'win32') return false
  try {
    execFileSync('python3', ['--version'], { stdio: 'ignore' })
    return true
  } catch {
    return false
  }
})()

// PYTHONPATH splits on ':', so the fake gi must live in a directory without one (TMPDIR may have one).
const dir = python ? mkdtempSync(join(tmpdir().includes(':') ? '/tmp' : tmpdir(), 'localrouter-py-')) : ''
afterAll(() => {
  if (dir) rmSync(dir, { recursive: true, force: true })
})

const SCRIPTS = { ATSPI_SCRIPT, PORTAL_KEYS, PORTAL_SCREENSHOT, GDK_MONITORS, GI_PROBE }

describe.skipIf(!python)('embedded Python', () => {
  it.each(Object.entries(SCRIPTS))('%s compiles', (name, source) => {
    const file = join(dir, `${name}.py`)
    writeFileSync(file, source)
    execFileSync('python3', ['-m', 'py_compile', file], { stdio: 'pipe' })
  })
})

/** A fake gi: just enough of Atspi for the script, every effect appended to FAKE_ATSPI_LOG as a JSON line. */
const FAKE_ATSPI = String.raw`
import json, os

class _Names:
    def __init__(self, names):
        for name in names.split():
            setattr(self, name, name)

StateType = _Names('ACTIVE SHOWING VISIBLE FOCUSED EDITABLE SINGLE_LINE SELECTED HORIZONTAL VERTICAL')
Role = _Names('APPLICATION FRAME WINDOW DIALOG PUSH_BUTTON TOGGLE_BUTTON LINK ENTRY PASSWORD_TEXT TEXT PAGE_TAB PAGE_TAB_LIST '
              'LIST LIST_ITEM LIST_BOX TABLE_CELL MENU MENU_BAR MENU_ITEM CHECK_BOX RADIO_BUTTON COMBO_BOX LABEL STATIC HEADING '
              'PARAGRAPH SECTION SCROLL_PANE SCROLL_BAR TOOL_BAR DOCUMENT_WEB DESKTOP_FRAME IMAGE')
CoordType = _Names('SCREEN WINDOW')
KeySynthType = _Names('SYM PRESS RELEASE PRESSRELEASE')

def log(entry):
    with open(os.environ['FAKE_ATSPI_LOG'], 'a') as f:
        f.write(json.dumps(entry) + '\n')

class Rect:
    def __init__(self, box):
        self.x, self.y, self.width, self.height = box

class States:
    def __init__(self, states):
        self.states = set(states)
    def contains(self, state):
        return state in self.states

class Node:
    def __init__(self, spec, parent=None, index=0):
        self.spec, self.parent, self.index = spec, parent, index
        self.kids = [Node(kid, self, i) for i, kid in enumerate(spec.get('children', []))]
    def get_child_count(self): return len(self.kids)
    def get_child_at_index(self, i): return self.kids[i]
    def get_role(self): return self.spec.get('role')
    def get_name(self): return self.spec.get('name', '')
    def get_description(self): return ''
    def get_state_set(self): return States(self.spec.get('states', ['SHOWING']))
    def get_extents(self, _coord):
        if 'box' not in self.spec:
            raise RuntimeError('no Component')
        return Rect(self.spec['box'])
    def get_process_id(self): return self.spec.get('pid', 0)
    def get_parent(self): return self.parent
    def get_index_in_parent(self): return self.index
    def grab_focus(self): return False
    def get_text_iface(self): return self if 'text' in self.spec else None
    def get_character_count(self): return len(self.spec['text'])
    def get_text(self, start, end): return self.spec['text'][start:end]
    def get_editable_text_iface(self): return self if self.spec.get('editable') else None
    def set_text_contents(self, text):
        log({'set_text': self.get_name()})
        if self.spec.get('rejects'):
            return True
        self.spec['text'] = text
        return True
    def delete_text(self, start, end):
        return True
    def insert_text(self, position, text, length):
        return True
    def get_action_iface(self): return self if self.spec.get('actions') else None
    def get_n_actions(self): return len(self.spec['actions'])
    def get_action_name(self, i): return self.spec['actions'][i]
    def do_action(self, i):
        log({'action': self.spec['actions'][i], 'on': self.get_name()})
        return True
    def get_selection_iface(self): return self if self.spec.get('selection') else None
    def select_child(self, i):
        log({'select': i, 'in': self.get_role()})
        return True
    def get_value_iface(self): return self if 'value' in self.spec else None
    def get_minimum_value(self): return self.spec['value'][0]
    def get_maximum_value(self): return self.spec['value'][1]
    def get_current_value(self): return self.spec['value'][2]
    def get_minimum_increment(self): return 1.0
    def set_current_value(self, value):
        log({'scroll_to': value})
        self.spec['value'][2] = value
        return True

def get_desktop(_index):
    with open(os.environ['FAKE_ATSPI_TREE']) as f:
        return Node(json.load(f))

def generate_keyboard_event(keyval, _string, _kind):
    log({'key': keyval})
    return True
`

const box = (x: number, y: number, w = 80, h = 30) => [x, y, w, h]

/** Firefox in front (active) and gedit behind it. Refs below are child-index paths from Firefox's frame. */
const TREE = {
  role: 'DESKTOP_FRAME',
  children: [
    {
      role: 'APPLICATION', name: 'Firefox', pid: 50,
      children: [{
        role: 'FRAME', name: 'Inbox — Mozilla Firefox', states: ['ACTIVE', 'SHOWING'], box: [0, 0, 1000, 800],
        children: [
          { role: 'TOOL_BAR', box: [0, 0, 1000, 80], children: [
            { role: 'PAGE_TAB_LIST', selection: true, box: [0, 0, 600, 40], children: [
              { role: 'PAGE_TAB', name: 'Inbox', states: ['SHOWING', 'SELECTED'], box: box(0, 0, 200, 40) },
              { role: 'PAGE_TAB', name: 'News', box: box(200, 0, 200, 40) },
            ] },
            { role: 'ENTRY', name: 'Search or enter address', editable: true, text: 'mail.example.com/inbox', states: ['SHOWING', 'EDITABLE', 'SINGLE_LINE', 'FOCUSED'], box: box(10, 45, 900, 30) },
          ] },
          { role: 'PUSH_BUTTON', name: 'Send', actions: ['press'], box: box(300, 700) },
          { role: 'PUSH_BUTTON', box: box(400, 700, 30, 30), children: [{ role: 'LABEL', name: 'Attach' }] },
          { role: 'PASSWORD_TEXT', name: 'Password', editable: true, text: 'hunter2', states: ['SHOWING', 'EDITABLE', 'SINGLE_LINE'], box: box(10, 100, 300, 30) },
          { role: 'ENTRY', name: 'Subject', editable: true, rejects: true, text: '', states: ['SHOWING', 'EDITABLE', 'SINGLE_LINE'], box: box(10, 140, 600, 30) },
          { role: 'TEXT', name: 'Body', editable: true, text: '', states: ['SHOWING', 'EDITABLE'], box: box(10, 180, 900, 400) },
          { role: 'LABEL', name: 'Hello world', box: box(10, 600, 200, 20) },
          { role: 'PUSH_BUTTON', name: 'Hidden', states: [], box: box(500, 700) },
          { role: 'SCROLL_PANE', box: [600, 100, 380, 500], children: [
            { role: 'SCROLL_BAR', states: ['SHOWING', 'VERTICAL'], value: [0, 1000, 100], box: box(960, 100, 20, 500) },
            { role: 'LIST', selection: true, box: [600, 100, 360, 500], children: [{ role: 'LIST_ITEM', name: 'Row A', box: box(600, 100, 360, 30) }] },
          ] },
        ],
      }],
    },
    { role: 'APPLICATION', name: 'gedit', pid: 60, children: [{ role: 'FRAME', name: 'notes', box: [0, 0, 500, 500], children: [] }] },
  ],
}

const REFS = { send: '1', attach: '2', password: '3', subject: '4', body: '5', row: '8.1.0', news: '0.0.1' }

/**
 * Writes the fake gi and the tree; returns a runner that runs python3 calls for real with -S instead of -I: no
 * site-packages, so the system's real gi cannot load and nothing reaches an accessibility bus, and PYTHONPATH (the
 * fake) is honoured. D-Bus is disabled in the child's environment as well.
 */
function fakeDesktop() {
  const root = mkdtempSync(join(dir, 'gi-'))
  mkdirSync(join(root, 'gi', 'repository'), { recursive: true })
  writeFileSync(join(root, 'gi', '__init__.py'), 'def require_version(name, version):\n    pass\n')
  writeFileSync(join(root, 'gi', 'repository', '__init__.py'), '')
  writeFileSync(join(root, 'gi', 'repository', 'GLib.py'), 'class Error(Exception):\n    message = ""\n')
  writeFileSync(join(root, 'gi', 'repository', 'Atspi.py'), FAKE_ATSPI)
  const tree = join(root, 'tree.json')
  const logFile = join(root, 'log.jsonl')
  writeFileSync(tree, JSON.stringify(TREE))
  writeFileSync(logFile, '')
  const others = fakeRunner()
  const runner: CommandRunner = {
    run(file, args, options = {}) {
      if (!file.endsWith('python3')) return others.runner.run(file, args, options)
      return new Promise((resolve, reject) => {
        const env = { PATH: process.env.PATH, PYTHONPATH: root, DBUS_SESSION_BUS_ADDRESS: 'disabled:', AT_SPI_BUS_ADDRESS: 'disabled:', FAKE_ATSPI_TREE: tree, FAKE_ATSPI_LOG: logFile }
        const child = execFile('python3', ['-S', ...args.filter((arg) => arg !== '-I')], { env }, (error, stdout, stderr) => (error ? reject(new Error(String(stderr).trim() || error.message)) : resolve(String(stdout))))
        child.stdin?.end(options.input ?? '')
      })
    },
  }
  const effects = () => readFileSync(logFile, 'utf8').split('\n').filter(Boolean).map((line) => JSON.parse(line) as Record<string, unknown>)
  const session = fakeSession({ server: 'x11', tools: ['xdotool'] })
  session.tools.python3 = 'python3'
  return { screen: new LinuxScreen({ session, runner, injector: chooseInjector(session, runner) }), effects, others }
}

describe.skipIf(!python)('the AT-SPI script against a fake accessibility tree', () => {
  it('reads the active window: roles, labels, bounds, the address bar, visible text, and never a password’s value', async () => {
    const { screen } = fakeDesktop()
    const seen = await screen.observe()
    expect(seen).toMatchObject({ app: 'Firefox', title: 'Inbox — Mozilla Firefox', url: 'mail.example.com/inbox', text: 'Hello world' })
    const byLabel = Object.fromEntries(seen.controls.map((control) => [control.label, control]))
    expect(byLabel.Inbox).toMatchObject({ role: 'tab', ref: '0.0.0' })
    expect(byLabel['Search or enter address']).toMatchObject({ role: 'textbox', value: 'mail.example.com/inbox', focused: true, bounds: { x: 10, y: 45, w: 900, h: 30 } })
    expect(byLabel.Attach).toMatchObject({ role: 'button', ref: REFS.attach })
    expect(byLabel['Row A']).toMatchObject({ role: 'row', ref: REFS.row })
    expect(byLabel.Password).toEqual({ ref: REFS.password, role: 'textbox', label: 'Password', bounds: { x: 10, y: 100, w: 300, h: 30 } })
    expect(JSON.stringify(seen)).not.toContain('hunter2')
    expect(byLabel.Hidden).toBeUndefined()
    // Reading order: top to bottom, then left to right.
    expect(seen.controls.map((control) => control.label).slice(0, 3)).toEqual(['Inbox', 'News', 'Search or enter address'])
  })

  it('clicks with the element’s own action, selects tabs and rows in their parent, and needs no pointer for either', async () => {
    const { screen, effects, others } = fakeDesktop()
    expect(await screen.act({ kind: 'click', ref: REFS.send })).toEqual({ background: true })
    expect(await screen.act({ kind: 'click', ref: REFS.news })).toEqual({ background: true })
    expect(effects()).toEqual([{ action: 'press', on: 'Send' }, { select: 1, in: 'PAGE_TAB_LIST' }])
    expect(others.calls).toHaveLength(0)
  })

  it('falls back to the pointer at the centre of a control with no action', async () => {
    const { screen, others } = fakeDesktop()
    expect(await screen.act({ kind: 'click', ref: REFS.attach })).toEqual({ pointer: { x: 415, y: 715 } })
    expect(others.lines()).toEqual(['xdotool mousemove --sync 415 715 click 1'])
  })

  it('sets text directly (multi-line where the box allows it), and refuses line breaks in a single-line box', async () => {
    const { screen, others } = fakeDesktop()
    expect(await screen.act({ kind: 'type', ref: REFS.body, text: 'Dear Sam,\nsee you at 5.' })).toEqual({ background: true })
    expect(await screen.act({ kind: 'type', ref: REFS.password, text: 'correct horse' })).toEqual({ background: true })
    await expect(screen.act({ kind: 'type', ref: REFS.subject, text: 'one\ntwo' })).rejects.toThrow(MULTILINE)
    expect(others.calls).toHaveLength(0)
  })

  it('types through the keyboard when the box ignores a direct set, and fails clearly when the text still did not land', async () => {
    const { screen, others } = fakeDesktop()
    await expect(screen.act({ kind: 'type', ref: REFS.subject, text: 'Lunch' })).rejects.toThrow('The text did not land in that box as typed')
    expect(others.lines()).toEqual(['xdotool mousemove --sync 310 155 click 1', 'xdotool key --clearmodifiers ctrl+a', 'xdotool type --clearmodifiers --delay 12 -- Lunch'])
  })

  it('scrolls a list through its scroll bar’s value, found next to it in the scrolled pane', async () => {
    const { screen, effects, others } = fakeDesktop()
    expect(await screen.act({ kind: 'scroll', direction: 'down', ref: REFS.row })).toEqual({ background: true })
    expect(effects()).toEqual([{ scroll_to: 200 }])
    expect(others.calls).toHaveLength(0)
  })

  it('says a control is gone when its path no longer resolves', async () => {
    const { screen } = fakeDesktop()
    await expect(screen.act({ kind: 'click', ref: '42' })).rejects.toThrow('That control is gone; look again.')
    await expect(screen.act({ kind: 'click', ref: '7' })).rejects.toThrow('That control is gone; look again.')
  })
})
