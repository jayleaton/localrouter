import { execFile } from 'node:child_process'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { afterAll, describe, expect, it } from 'vitest'
import { bootstrapLine, bootstrapScript, PS_OPS, REFERENCED_ASSEMBLIES } from './bootstrap.ts'
import { HELPER_CS } from './helper.cs.ts'
import { accessShape } from './index.ts'
import { WindowsScreen } from './screen.ts'
import { powershellPath, PowerShellWorker } from './worker.ts'

const run = promisify(execFile)

/** Runs a PowerShell script file with Windows PowerShell 5.1 and returns its stdout. */
async function powershell(script: string, env: Record<string, string> = {}): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'lr-ps-'))
  try {
    const file = join(dir, 'check.ps1')
    await writeFile(file, script, 'utf8')
    const { stdout } = await run(powershellPath(), ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', file], { env: { ...process.env, ...env }, timeout: 120_000, windowsHide: true, maxBuffer: 8 * 1024 * 1024 })
    return stdout
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
}

/** Windows PowerShell 5.1's own parser over a script: its syntax errors, one a line. */
async function parseErrors(text: string): Promise<string[]> {
  const dir = await mkdtemp(join(tmpdir(), 'lr-parse-'))
  try {
    const file = join(dir, 'input.ps1')
    await writeFile(file, text, 'utf8')
    const out = await powershell(String.raw`$tokens = $null; $errors = $null
$null = [System.Management.Automation.Language.Parser]::ParseInput([IO.File]::ReadAllText($env:LR_INPUT), [ref]$tokens, [ref]$errors)
foreach ($e in $errors) { [Console]::Out.WriteLine(('{0}: {1}' -f $e.Extent.StartLineNumber, $e.Message)) }
[Console]::Out.WriteLine('parsed')`, { LR_INPUT: file })
    expect(out).toContain('parsed')
    return out.split(/\r?\n/).filter((line) => line && line !== 'parsed')
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
}

// Windows PowerShell 5.1, the C# 5 compiler Add-Type uses and a desktop session exist only on Windows.
describe.runIf(process.platform === 'win32')('Windows PowerShell 5.1 worker (real Windows)', () => {
  const worker = new PowerShellWorker({ timeoutMs: 30_000, startTimeoutMs: 120_000 })
  afterAll(() => worker.close())

  it('parses the bootstrap, its first line and every PowerShell operation', { timeout: 180_000 }, async () => {
    expect(await parseErrors(bootstrapScript())).toEqual([])
    expect(await parseErrors(bootstrapLine())).toEqual([])
    for (const [name, body] of Object.entries(PS_OPS)) expect({ name, errors: await parseErrors(body) }).toEqual({ name, errors: [] })
  })

  it('compiles the C# helper with Add-Type (the .NET Framework C# 5 compiler)', { timeout: 180_000 }, async () => {
    const dir = await mkdtemp(join(tmpdir(), 'lr-cs-'))
    try {
      const file = join(dir, 'helper.cs')
      await writeFile(file, HELPER_CS, 'utf8')
      const out = await powershell(String.raw`$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition ([IO.File]::ReadAllText($env:LR_CS)) -Language CSharp -ReferencedAssemblies @(${REFERENCED_ASSEMBLIES.map((name) => `'${name}'`).join(', ')})
[Console]::Out.WriteLine([LocalRouterWin.Json]::Write(@{ ok = $true; text = 'Caf' + [char]0xe9 }))`, { LR_CS: file })
      expect(JSON.parse(out.trim())).toEqual({ ok: true, text: 'Café' })
      expect(out.trim()).toMatch(/^[\x20-\x7e]+$/)
    } finally {
      await rm(dir, { recursive: true, force: true })
    }
  })

  it('starts the worker per-monitor DPI aware and reports the session', { timeout: 180_000 }, async () => {
    const facts = accessShape(await worker.call('access', {}, { timeoutMs: 60_000 }))
    expect(facts.uia).toBe(true)
    expect(facts.dpi).toBe('per_monitor_v2')
    expect(facts.session).toBeGreaterThan(0)
  })

  it('reads displays, the pointer, the window in front and a screenshot, without acting', { timeout: 180_000 }, async () => {
    const screen = new WindowsScreen(worker)
    const displays = await screen.displays()
    expect(displays.length).toBeGreaterThan(0)
    expect(displays[0]).toMatchObject({ x: 0, y: 0 })
    const pointer = await screen.pointer()
    expect(Number.isFinite(pointer.x) && Number.isFinite(pointer.y)).toBe(true)
    // Whatever is in front on the runner (it may be nothing): the shape, or the sentence for no window.
    const seen = await screen.observe().then((o) => o, (error: Error) => error)
    if (seen instanceof Error) expect(seen.message).toMatch(/No app is in front|has no open window/)
    else {
      expect(typeof seen.app).toBe('string')
      expect(seen.controls.length).toBeLessThanOrEqual(150)
      for (const control of seen.controls) expect(control.ref).toMatch(/^\d+:\d+(\.\d+)*(#-?\d+(\.-?\d+)*)?$/)
    }
    const shot = await screen.screenshot({ maxEdge: 400 }).then((s) => s, (error: Error) => error)
    if (shot instanceof Error) expect(shot.message).toMatch(/No app is in front|has no open window/)
    else {
      expect([...shot.png.slice(0, 8)]).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
      expect(Math.max(shot.width, shot.height)).toBeLessThanOrEqual(400)
      expect(shot.scale).toBeLessThanOrEqual(1)
    }
  })

  it('maps a helper failure to its sentence (a ref to a window that does not exist)', { timeout: 60_000 }, async () => {
    await expect(new WindowsScreen(worker).act({ kind: 'click', ref: '1:0' })).rejects.toThrow('That control is gone; look again.')
  })

  it('answers the read-only system operations', { timeout: 180_000 }, async () => {
    expect(Array.isArray(await worker.call('windows'))).toBe(true)
    expect(Array.isArray(await worker.call('start_apps', {}, { timeoutMs: 60_000 }))).toBe(true)
    const hello = (await worker.call('hello_available', {}, { timeoutMs: 60_000 })) as { availability: string }
    expect(typeof hello.availability).toBe('string')
    const tabs = (await worker.call('tabs')) as { tabs: unknown[] }
    expect(Array.isArray(tabs.tabs)).toBe(true)
  })

  it('background input (a posted key) does not move the real pointer', { timeout: 120_000 }, async () => {
    const before = (await worker.call('pointer', {}, { timeoutMs: 20_000 })) as { x: number; y: number }
    // A key posted to the foreground window's focused control: background, no SendInput, so the pointer is untouched.
    await worker.call('postkey', { vk: 0x09 }, { timeoutMs: 20_000 }) // VK_TAB, harmless
    await new Promise((resolve) => setTimeout(resolve, 150))
    const after = (await worker.call('pointer', {}, { timeoutMs: 20_000 })) as { x: number; y: number }
    expect({ x: after.x, y: after.y }).toEqual({ x: before.x, y: before.y })
  })

  it('asks Windows Hello for a window through IUserConsentVerifierInterop, and reads its answer', { timeout: 180_000 }, async () => {
    // A runner has no Hello set up: the request answers at once (no prompt) with a result, which is what this checks.
    const hello = (await worker.call('hello_available', {}, { timeoutMs: 60_000 })) as { availability: string }
    if (hello.availability === 'Available') return
    const start = (await worker.call('hello_start', { prompt: 'LocalRouter CI', hwnd: 0 }, { timeoutMs: 60_000 })) as { front: boolean; seen: boolean }
    expect(typeof start.front).toBe('boolean')
    const done = (await worker.call('hello_wait', { ms: 30_000 }, { timeoutMs: 60_000 })) as { result: string }
    expect(['DeviceNotPresent', 'NotConfiguredForUser', 'DisabledByPolicy', 'DeviceBusy', 'Canceled']).toContain(done.result)
    // Cancelling after the answer (an approval that ended another way) is harmless: nothing is open.
    expect(await worker.call('hello_cancel', {}, { timeoutMs: 60_000 })).toEqual({ cancelled: false })
  })
})
