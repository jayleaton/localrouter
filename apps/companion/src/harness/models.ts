import type { Spark } from './spark.ts'

/**
 * Model management around a video moved from a call to the chat: room is made first (the box's `reserve_for`, which
 * unloads what the video needs and never the chat model), and after the video the voice models are asked back if the
 * box has not brought them back already. The box's scheduler decides; these only ask.
 */
type ToolRow = { id: string; kind: string; state: string; keep_loaded?: boolean; residency?: string }

/** Kept loaded by the box: "always" residency, or `keep_loaded` on a box from before residency. */
const alwaysLoaded = (t: ToolRow) => (t.residency === undefined ? t.keep_loaded === true : t.residency === 'always')

const text = (r: { content: Array<{ text?: string }> }) => r.content.map((c) => c.text ?? '').join(' ').trim()

export async function prepareVideo(spark: Spark): Promise<{ ok: boolean; text: string }> {
  try {
    const r = await spark.mcp('reserve_for', { model: 'video', dry_run: false, wait_s: 600 })
    const t = text(r)
    let summary = t
    try {
      const j = JSON.parse(t) as { unloads?: string[]; message?: string }
      summary = j.unloads && j.unloads.length ? `paused ${j.unloads.join(', ')}` : (j.message ?? 'nothing had to move')
    } catch {
      // Plain text.
    }
    return { ok: !r.isError, text: summary.slice(0, 300) }
  } catch (error) {
    return { ok: false, text: error instanceof Error ? error.message : String(error) }
  }
}

/** The voice models (speech and transcription the box keeps always loaded; one loaded per call comes back with the next call) back: what it asked for, or that they are back. */
export async function restoreVoice(spark: Spark, waitS = 300): Promise<{ ok: boolean; text: string }> {
  try {
    const r = await spark.request('/v1/tools')
    if (!r.ok) return { ok: false, text: `the box said ${r.status}` }
    const rows = ((await r.json()) as { data: ToolRow[] }).data
    const voice = rows.filter((t) => (t.kind === 'speech' || t.kind === 'transcription') && alwaysLoaded(t))
    const away = voice.filter((t) => t.state !== 'ready')
    if (!away.length) return { ok: true, text: `back (${voice.map((t) => t.id).join(', ')})` }
    const results = await Promise.all(away.map(async (t) => {
      const res = await spark.mcp('load_model', { model: t.id, wait_s: waitS }).catch((e: unknown) => ({ content: [{ text: e instanceof Error ? e.message : String(e) }], isError: true }))
      return { id: t.id, ok: !res.isError, text: text(res) }
    }))
    const bad = results.filter((x) => !x.ok)
    return bad.length ? { ok: false, text: bad.map((x) => `${x.id}: ${x.text.slice(0, 120)}`).join('; ') } : { ok: true, text: `loaded again (${results.map((x) => x.id).join(', ')})` }
  } catch (error) {
    return { ok: false, text: error instanceof Error ? error.message : String(error) }
  }
}
