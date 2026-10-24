/**
 * The harness's tools: the functions it offers a model (over realtime in client tool mode, or Chat Completions) and
 * runs itself, on this computer. A call's arguments are checked against the tool's schema before it runs, once, with
 * the complete arguments; a malformed call is answered with an error the model can read, never half-run. What a tool
 * may do to the computer is still the executor's decision: the computer tools go through its broker.
 */
export type Schema = {
  type: 'object'
  properties: Record<string, { type: 'string' | 'integer' | 'number' | 'boolean' | 'object' | 'array'; description?: string; enum?: readonly string[]; minimum?: number; maximum?: number; maxLength?: number; properties?: Record<string, unknown>; required?: readonly string[]; items?: unknown; additionalProperties?: boolean }>
  required?: readonly string[]
  additionalProperties?: false
}

export type ToolEvent =
  | { type: 'job.created'; call_id: string; job_id: string; kind: 'image' | 'video' }
  | { type: 'job.done'; call_id: string; job_id: string; kind: 'image' | 'video'; status: string; urls: string[]; error: string }

export type ToolContext = {
  callId: string
  /** The person's own words in this turn (computer control needs them). */
  utterance: string
  signal: AbortSignal
  /** Events for the UI (a generation's job), if one is listening. */
  emit?: (event: ToolEvent) => void
}

export interface Tool {
  readonly name: string
  readonly description: string
  readonly parameters: Schema
  run(args: Record<string, unknown>, ctx: ToolContext): Promise<unknown>
}

export type Ran = { ok: boolean; output: string; ms: number }

export class Registry {
  private readonly tools = new Map<string, Tool>()

  add(...tools: Tool[]) {
    for (const t of tools) {
      if (this.tools.has(t.name)) throw new Error(`two tools are named ${t.name}`)
      this.tools.set(t.name, t)
    }
    return this
  }

  has(name: string) {
    return this.tools.has(name)
  }

  get names() {
    return [...this.tools.keys()]
  }

  /** The realtime session's `tools`. */
  realtime() {
    return [...this.tools.values()].map((t) => ({ type: 'function', name: t.name, description: t.description, parameters: t.parameters }))
  }

  /** Chat Completions' `tools`. */
  chat() {
    return [...this.tools.values()].map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }))
  }

  /** Runs one complete call; the output is the JSON text returned to the model. */
  async run(name: string, argumentsText: string, ctx: ToolContext): Promise<Ran> {
    const started = Date.now()
    const done = (ok: boolean, value: unknown): Ran => ({ ok, output: JSON.stringify(ok ? value : { error: value }), ms: Date.now() - started })
    const tool = this.tools.get(name)
    if (!tool) return done(false, `there is no tool named ${name}`)
    let args: unknown
    try {
      args = argumentsText.trim() ? JSON.parse(argumentsText) : {}
    } catch {
      return done(false, 'the arguments are not JSON')
    }
    const problem = check(tool.parameters, args)
    if (problem) return done(false, problem)
    try {
      return done(true, await tool.run(args as Record<string, unknown>, ctx))
    } catch (error) {
      return done(false, error instanceof Error ? error.message : String(error))
    }
  }
}

/** What is wrong with `args` for `schema`, or null. Shallow: top-level fields, their types, enums and bounds. */
export function check(schema: Schema, args: unknown): string | null {
  if (typeof args !== 'object' || args === null || Array.isArray(args)) return 'the arguments should be an object'
  const a = args as Record<string, unknown>
  for (const key of schema.required ?? []) if (a[key] === undefined || a[key] === null) return `${key} is required`
  for (const [key, value] of Object.entries(a)) {
    const p = schema.properties[key]
    if (!p) {
      if (schema.additionalProperties === false) return `there is no argument ${key}`
      continue
    }
    if (value === null || value === undefined) continue
    const ok =
      p.type === 'integer' ? Number.isInteger(value)
      : p.type === 'number' ? typeof value === 'number' && Number.isFinite(value)
      : p.type === 'array' ? Array.isArray(value)
      : p.type === 'object' ? typeof value === 'object' && !Array.isArray(value)
      : typeof value === p.type
    if (!ok) return `${key} should be ${p.type === 'integer' ? 'a whole number' : `a ${p.type}`}`
    if (p.enum && !p.enum.includes(value as string)) return `${key} should be one of ${p.enum.join(', ')}`
    if (typeof value === 'number' && ((p.minimum !== undefined && value < p.minimum) || (p.maximum !== undefined && value > p.maximum))) return `${key} should be from ${p.minimum} to ${p.maximum}`
    if (typeof value === 'string' && p.maxLength !== undefined && value.length > p.maxLength) return `${key} is longer than ${p.maxLength} characters`
  }
  return null
}
