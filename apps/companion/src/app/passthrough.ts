import type { Spark } from '../harness/spark.ts'

/**
 * What the app passes between its window (the bundled /voice page at lr://app) and the Spark: the page's own /v1
 * calls, whatever their method and body (JSON, a voice clone's multipart clips, PATCH, DELETE), with the device token
 * in place of anything the page sends; and the microphone, which the page uses for calls and to record a voice to
 * clone. Nothing here keeps a body: it is handed on and dropped, so no recording or voice is stored on this computer.
 */
export const ORIGIN = 'lr://app'

export const PASS_REQUEST = ['content-type', 'accept', 'range']
export const PASS_RESPONSE = ['content-type', 'content-length', 'content-range', 'accept-ranges', 'cache-control', 'x-sample-rate', 'x-channels', 'x-sample-format', 'last-modified', 'etag']

export const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } })
export const failure = (status: number, message: string) => json({ error: { message, type: 'companion_error' } }, status)

/** The page's request, sent on to the Spark as the device (`spark` is null until the app is paired). */
export async function proxy(spark: Pick<Spark, 'request'> | null, req: Request, pathAndQuery: string): Promise<Response> {
  if (!spark) return failure(503, 'LocalRouter is not paired with a LocalRouter server yet: open Settings')
  const headers: Record<string, string> = {}
  for (const h of PASS_REQUEST) {
    const v = req.headers.get(h)
    if (v) headers[h] = v
  }
  // Whole, so a multipart body keeps its boundary and length; an empty one (DELETE) is none.
  const data = req.method === 'GET' || req.method === 'HEAD' ? null : await req.arrayBuffer()
  try {
    const r = await spark.request(pathAndQuery, { method: req.method, headers, body: data && data.byteLength ? data : null, signal: req.signal })
    const out = new Headers()
    for (const h of PASS_RESPONSE) {
      const v = r.headers.get(h)
      if (v) out.set(h, v)
    }
    return new Response(r.body, { status: r.status, headers: out })
  } catch (error) {
    // fetch's own message is only "fetch failed": the reason (refused, reset, closed, a timeout) is its cause.
    const cause = error instanceof Error && error.cause ? (error.cause as { code?: string; message?: string }) : null
    const why = cause ? ` (${[cause.code, cause.message].filter(Boolean).join(': ')})` : ''
    return failure(502, `the LocalRouter server did not answer: ${error instanceof Error ? error.message : String(error)}${why}`)
  }
}

/** Only this app's own pages, and of the media only the microphone (and copying to the clipboard). */
export function mediaAllowed(origin: string, permission: string, mediaTypes?: string[]): boolean {
  const ours = origin === ORIGIN || origin.startsWith(`${ORIGIN}/`)
  return ours && ((permission === 'media' && (mediaTypes ?? ['audio']).every((t) => t === 'audio')) || permission === 'clipboard-sanitized-write')
}
