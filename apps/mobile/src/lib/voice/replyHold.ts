/*
 * When a reply's audio may start (ported from the box's web client, docs/dev/VOICE.md "Playback start"). Speech
 * usually comes faster than it plays, but not while it shares the GPU with the chat model writing the rest of the
 * reply, or with a picture being made (0.5 to 0.9x real time on DGX-01). A reply's length is not known until it is
 * written, so no fixed buffer covers that. The start waits until:
 * - the text is complete (`localrouter.response.text_done`, with its length) and enough is in hand to play all of
 *   it at the rate it comes in (at least START_MS; GEN_START_MS while a picture or video is being made);
 * - or, while the text is still being written, speech keeps up anyway (KEEP_UP over RATE_MS), then START_MS;
 * - or all of the reply's speech is in (`localrouter.response.speech_done`);
 * - and never more than HOLD_CAP_MS after the first piece, or WRITING_CAP_MS while the text is still being written:
 *   a reply speaks while it is written, at the cost of a pause where speech falls behind.
 * A short reply on a free GPU is written before its first audio and starts as soon as it would have.
 */

export const START_MS = 200
export const GEN_START_MS = 500
export const KEEP_UP = 1.05
export const RATE_MS = 250
export const HOLD_CAP_MS = 10_000
/** While the text is still being written, a reply starts at most this long after its first piece. */
export const WRITING_CAP_MS = 1_000
/** A written reply's audio is about this long per character. */
export const MS_PER_CHAR = 65
const MARGIN = 1.25

type Held<T> = { pieces: T[]; firstAt: number; haveMs: number; firstMs: number }

export class ReplyHold<T> {
  private held = new Map<string, Held<T>>()
  private started = new Set<string>()
  private textDone = new Map<string, number>()
  private speechDone = new Set<string>()
  /** The server's measure of how fast speech comes with the GPU to itself (text_done's speech_rate). */
  private freeRate = new Map<string, number>()
  /** How fast the last fully received reply's speech came. */
  private lastRate = 0
  private got = new Map<string, { first: number; ms: number; firstMs: number }>()

  /** `generating`: a picture or video is being made (speech shares the GPU with it). */
  constructor(private readonly generating: () => boolean = () => false) {}

  /** A new session or a barge-in: everything held is dropped. */
  reset() {
    this.held.clear()
    this.started.clear()
    this.textDone.clear()
    this.speechDone.clear()
    this.freeRate.clear()
    this.got.clear()
  }

  /** Whether some reply's start is still being held (the engine then calls `tick` every 25 ms). */
  get holding() {
    return this.held.size > 0
  }

  /** A piece of `item`'s audio of `ms` arrived: the pieces to schedule now (it alone once the reply has started). */
  piece(item: string, piece: T, ms: number, now: number): T[] {
    const got = this.got.get(item) ?? { first: now, ms: 0, firstMs: ms }
    got.ms += ms
    this.got.set(item, got)
    if (this.started.has(item)) return [piece]
    const held = this.held.get(item) ?? { pieces: [], firstAt: now, haveMs: 0, firstMs: ms }
    held.pieces.push(piece)
    held.haveMs += ms
    this.held.set(item, held)
    return this.check(item, now)
  }

  text(item: string, chars: number, speechRate: number | null, toolCalls: boolean, now: number) {
    this.textDone.set(item, chars)
    if (!toolCalls && speechRate) this.freeRate.set(item, speechRate)
    return this.check(item, now)
  }

  speech(item: string, now: number) {
    this.speechDone.add(item)
    const got = this.got.get(item)
    const span = got ? now - got.first : 0
    if (got && span > 300) this.lastRate = (got.ms - got.firstMs) / span
    this.got.delete(item)
    return this.check(item, now)
  }

  /** Re-checks every held reply (time passing changes the measured rate and the cap). */
  tick(now: number): T[] {
    return [...this.held.keys()].flatMap((item) => this.check(item, now))
  }

  private startMs() {
    return this.generating() ? GEN_START_MS : START_MS
  }

  /** Audio ms per wall ms since the first piece (the first piece not counted), once RATE_MS has passed; else null. */
  private arrivalRate(held: Held<T>, now: number) {
    const since = now - held.firstAt
    return since < RATE_MS ? null : (held.haveMs - held.firstMs) / since
  }

  private need(item: string, rate: number) {
    const chars = this.textDone.get(item)
    if (chars === undefined) return rate >= KEEP_UP ? this.startMs() : Infinity
    return Math.max(this.startMs(), chars * MS_PER_CHAR * (1 - Math.min(rate, 1)) * MARGIN)
  }

  private check(item: string, now: number): T[] {
    const held = this.held.get(item)
    if (!held) return []
    const measured = this.arrivalRate(held, now)
    // A reply already written, with the GPU free, comes from here on as fast as speech does with the GPU to itself.
    const prior = this.textDone.has(item) && !this.generating() ? this.freeRate.get(item) || this.lastRate : 0
    const rate = prior >= KEEP_UP ? Math.max(measured ?? 0, prior) : measured
    const ready = this.speechDone.has(item) || (rate !== null && held.haveMs >= this.need(item, rate))
    const cap = this.textDone.has(item) ? HOLD_CAP_MS : WRITING_CAP_MS
    if (!ready && now - held.firstAt < cap) return []
    this.held.delete(item)
    this.started.add(item)
    return held.pieces
  }
}
