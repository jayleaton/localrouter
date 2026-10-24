import { lstat, readdir, readFile, realpath, stat } from 'node:fs/promises'
import { isAbsolute, join, relative, resolve, sep } from 'node:path'
import { parseControl } from '../../../desktop-helper/src/computer/macControls.ts'
import type { Broker, Caller } from '../executor/broker.ts'
import type { Tool, ToolContext } from './registry.ts'
import type { McpResult, Spark } from './spark.ts'

/**
 * The reference harness's tools. Computer control goes to this computer's executor (its broker decides and asks the
 * person); local files are read only inside the folders the person shared; media and decisions are the Spark's MCP
 * tools. Each returns plain JSON for the model.
 */

// ------------------------------------------------------------------ computer control (the executor)

export function computerTools(broker: Broker, caller: Caller, waitMs = 150_000): Tool[] {
  return [
    {
      name: 'computer_control',
      description: 'Run one fixed control on the user’s computer: {"kind":"open_app","app"}, {"kind":"quit_app","app"}, {"kind":"volume","action":"up|down|mute|unmute|set","level"?,"target":"system|Spotify|Music"}, {"kind":"media","action":"play|pause|next|previous","player":"Spotify|Music"}, {"kind":"dark_mode","action":"on|off|toggle"}, {"kind":"lock_screen"}. Locking asks the user on screen first.',
      parameters: { type: 'object', properties: { control: { type: 'object', description: 'the control, as above' } }, required: ['control'], additionalProperties: false },
      async run(args, ctx) {
        const r = await broker.control({ caller, utterance: ctx.utterance }, parseControl(args.control))
        return { ok: r.ok, said: r.text }
      },
    },
    {
      name: 'computer_use',
      description: 'Do a small task on the user’s computer screen toward a goal (at most 15 steps). Anything that could send, buy, delete, post or type into a field that is not a search box waits for the user to approve it on their screen: tell them so. Pass any exact text to type as `text`, never invent it.',
      parameters: { type: 'object', properties: { goal: { type: 'string', description: 'what to do, in the user’s words', maxLength: 1000 }, text: { type: 'string', description: 'exact text to type, if any', maxLength: 4000 } }, required: ['goal'], additionalProperties: false },
      async run(args, ctx) {
        const started = broker.startUse({ caller, utterance: ctx.utterance }, String(args.goal), typeof args.text === 'string' ? args.text : null)
        if (!started.ok) return { status: 'refused', reason: started.error }
        const stop = () => broker.stop(started.task_id, 'the conversation', 'The conversation moved on, so it stopped.')
        ctx.signal.addEventListener('abort', stop, { once: true })
        try {
          const outcome = await Promise.race([started.finished, new Promise<null>((r) => setTimeout(() => r(null), waitMs))])
          if (!outcome) return { status: 'running', task_id: started.task_id, note: 'still working on it' }
          return { status: outcome.status, reason: outcome.reason, steps: outcome.steps, where: outcome.surface }
        } finally {
          ctx.signal.removeEventListener('abort', stop)
        }
      },
    },
  ]
}

// ------------------------------------------------------------------ local files, inside shared folders only

export const MAX_READ = 64 * 1024

/** The real path of `path` when it is inside one of `roots` (after links); else an error. */
export async function inside(roots: string[], path: string): Promise<string> {
  if (!roots.length) throw new Error('no folder is shared with Jarvis: add one in LocalRouter › Settings › Files')
  const real = await Promise.all(roots.map((r) => realpath(r).catch(() => null)))
  const base = real.find(Boolean)
  if (!base) throw new Error('the shared folders are gone')
  const wanted = isAbsolute(path) ? path : join(base, path || '.')
  let target: string
  try {
    target = await realpath(resolve(wanted))
  } catch {
    throw new Error(`there is no ${path || 'such file'} in the shared folders`)
  }
  for (const r of real) {
    if (!r) continue
    const rel = relative(r, target)
    if (rel === '' || (!rel.startsWith('..') && !isAbsolute(rel) && !rel.split(sep).includes('..'))) return target
  }
  throw new Error(`${path} is outside the folders shared with Jarvis`)
}

export function fileTools(roots: () => string[]): Tool[] {
  return [
    {
      name: 'list_files',
      description: 'List a folder the user shared with you (or one inside it): names, kinds and sizes. Paths are relative to the first shared folder.',
      parameters: { type: 'object', properties: { path: { type: 'string', description: 'a folder; empty for the shared folder itself' } }, additionalProperties: false },
      async run(args) {
        const dir = await inside(roots(), typeof args.path === 'string' ? args.path : '')
        const names = (await readdir(dir)).filter((n) => !n.startsWith('.')).sort().slice(0, 200)
        const entries = await Promise.all(names.map(async (name) => {
          const s = await lstat(join(dir, name)).catch(() => null)
          return { name, kind: s?.isDirectory() ? 'folder' : s?.isSymbolicLink() ? 'link' : 'file', bytes: s?.isFile() ? s.size : undefined }
        }))
        return { folder: dir, entries }
      },
    },
    {
      name: 'read_file',
      description: 'Read a text file in a folder the user shared with you (at most 64 KB).',
      parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'], additionalProperties: false },
      async run(args) {
        const file = await inside(roots(), String(args.path))
        const s = await stat(file)
        if (!s.isFile()) throw new Error(`${String(args.path)} is not a file`)
        const data = await readFile(file)
        const head = data.subarray(0, MAX_READ)
        if (head.includes(0)) throw new Error(`${String(args.path)} is not a text file`)
        return { path: file, bytes: s.size, truncated: s.size > MAX_READ, text: head.toString('utf8') }
      },
    },
  ]
}

// ------------------------------------------------------------------ media and decisions on the Spark (MCP)

type JobSummary = { id: string; status: string; files?: string[]; error?: string; progress?: number }

export function jobOf(r: McpResult): JobSummary {
  const text = r.content.find((c) => c.type === 'text')?.text ?? ''
  try {
    const j = JSON.parse(text) as JobSummary
    if (typeof j.id === 'string') return j
  } catch {
    // An error text.
  }
  throw new Error(text || 'the Spark did not start it')
}

async function generation(spark: Spark, kind: 'image' | 'video', tool: string, args: Record<string, unknown>, ctx: ToolContext, waitS: number) {
  const started = await spark.mcp(tool, { ...args, wait_s: 0, ...(kind === 'image' ? { inline_images: false } : {}) }, ctx.signal)
  if (started.isError) throw new Error(started.content.map((c) => c.text ?? '').join(' ') || 'the Spark refused it')
  const job = jobOf(started)
  ctx.emit?.({ type: 'job.created', call_id: ctx.callId, job_id: job.id, kind })
  const finish = (j: JobSummary) => {
    ctx.emit?.({ type: 'job.done', call_id: ctx.callId, job_id: j.id, kind, status: j.status, urls: j.files ?? [], error: j.error ?? '' })
  }
  if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
    finish(job)
    return { status: job.status, job_id: job.id, urls: job.files ?? [], ...(job.error ? { error: job.error } : {}) }
  }
  // Wait a while here (a picture is usually seconds); then let it finish in the background and appear by itself.
  const poll = async (wait: number) => jobOf(await spark.mcp('get_job', { id: job.id, wait_s: wait }))
  const first = waitS > 0 ? await poll(waitS) : job
  const ended = (j: JobSummary) => j.status === 'completed' || j.status === 'failed' || j.status === 'cancelled'
  if (ended(first)) {
    finish(first)
    return { status: first.status, job_id: first.id, urls: first.files ?? [], ...(first.error ? { error: first.error } : {}) }
  }
  void (async () => {
    let j = first
    for (let i = 0; i < 20 && !ended(j); i++) j = await poll(300).catch(() => j)
    if (ended(j)) finish(j)
  })()
  return { status: 'in_progress', job_id: job.id, note: `The ${kind === 'image' ? 'picture' : 'video'} is being made and appears for the user when it is done; tell them so.` }
}

export function mediaTools(spark: Spark, opts: { imageWaitS?: number } = {}): Tool[] {
  return [
    {
      name: 'generate_image',
      description: 'Make a picture on the user’s LocalRouter box from a detailed description. The user sees it when it is done.',
      parameters: { type: 'object', properties: { prompt: { type: 'string', description: 'what to draw, in detail', maxLength: 4000 }, size: { type: 'string', description: 'WIDTHxHEIGHT in multiples of 16, e.g. 1024x1024 (default) or 1344x768' } }, required: ['prompt'], additionalProperties: false },
      run: (args, ctx) => generation(spark, 'image', 'generate_image', { prompt: args.prompt, ...(args.size ? { size: args.size } : {}) }, ctx, opts.imageWaitS ?? 45),
    },
    {
      name: 'generate_video',
      description: 'Make a short video clip with sound on the user’s LocalRouter box. It takes minutes; the user sees it when it is done.',
      parameters: { type: 'object', properties: { prompt: { type: 'string', description: 'what happens, in detail', maxLength: 4000 }, seconds: { type: 'integer', minimum: 1, maximum: 20, description: 'length (default 5)' } }, required: ['prompt'], additionalProperties: false },
      run: (args, ctx) => generation(spark, 'video', 'generate_video', { prompt: args.prompt, ...(args.seconds ? { seconds: args.seconds } : {}) }, ctx, 0),
    },
    {
      name: 'decide',
      description: 'Ask the box’s decision model typed questions about a state (choice, yes/no or score), answered with probabilities. Advice only.',
      parameters: { type: 'object', properties: { state: { type: 'object' }, questions: { type: 'object' } }, required: ['state', 'questions'], additionalProperties: false },
      async run(args, ctx) {
        const r = await spark.mcp('decide', { state: args.state, questions: args.questions }, ctx.signal)
        const text = r.content.map((c) => c.text ?? '').join('\n')
        if (r.isError) throw new Error(text)
        return { answers: text }
      },
    },
  ]
}
