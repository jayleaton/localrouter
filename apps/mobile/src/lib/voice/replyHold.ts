/*
 * When a reply's audio may start (ported from the box's web client, src/voice/voice.html, the comment above START_MS).
 * A reply's audio plays back to back to its end without a pause: rather than pausing, it waits a little longer to
 * start. Speech comes faster than it plays with the GPU to itself (1.4x on DGX-02), slower beside the chat model
 * writing the reply (0.55x; the server's speech first keeps it about level) or a picture being made. So the start
 * waits until the audio in hand plays the rest without running dry at the rate it comes in:
 *   in hand >= remaining x (1 - rate) + margin
 * - remaining: the audio still to come for the text written so far (the whole text once
 *   `localrouter.response.text_done` gives its length), at the voice's ms a character (learned per voice, MS_PER_CHAR
 *   at least);
 * - rate: an EWMA (EWMA_ALPHA) of the arrival rate over RATE_WINDOW_MS windows from the first piece, seeded with the
 *   voice's rate on its last replies, less RATE_HEADROOM of its spread; once the text is complete with the GPU free,
 *   the server's measure of speech with the GPU to itself (text_done's speech_rate, else the last reply's) counts;
 * - margin: MARGIN_MS, and more for the rest of the session after a pause (`gap`).
 * A slow stream starts later, in proportion: at a quarter of real time, three quarters of the rest must be in hand. A
 * short reply written on a free GPU needs START_MS (no margin); all of a reply's speech in (speech_done) starts it.
 * Never more than HOLD_CAP_MS after the first piece.
 */

export const START_MS = 200
export const GEN_START_MS = 500
export const HOLD_CAP_MS = 30_000
/** A reply's audio is at least about this long per character. */
export const MS_PER_CHAR = 65
export const MARGIN_MS = 300
/** A pause adds twice its length to the margin for the rest of the session, up to this. */
export const MARGIN_UP_MAX_MS = 2_000
const RATE_WINDOW_MS = 150
const EWMA_ALPHA = 0.7
const RATE_HEADROOM = 0.5
const PRIOR_RATE = 1.0

/** What the start learns per voice across replies: its arrival rate and its audio a character. */
export type VoicePrior = { rate?: number; mpc?: number }
export type VoicePriors = Map<string, VoicePrior>

type Held<T> = { pieces: T[]; firstAt: number; haveMs: number }
/** An item's arrival rate (EWMA and its spread) since the last window, and its audio a character. */
type Rate = { rate: number; v: number; at: number; from: number; mpc: number }

export class ReplyHold<T> {
  private held = new Map<string, Held<T>>()
  private started = new Set<string>()
  private textDone = new Map<string, number>()
  private speechDone = new Set<string>()
  /** Characters of each reply's spoken text written so far. */
  private writtenChars = new Map<string, number>()
  /** The server's measure of how fast speech comes with the GPU to itself (text_done's speech_rate). */
  private freeRate = new Map<string, number>()
  /** How fast the last fully received reply's speech came. */
  private lastRate = 0
  private got = new Map<string, { first: number; ms: number; firstMs: number }>()
  private rates = new Map<string, Rate>()
  /** Added to the margin after a pause, for the rest of the session. */
  private marginUp = 0

  /**
   * `generating`: a picture or video is being made (speech shares the GPU with it). `voice`: the voice speaking now,
   * the key of its learned prior in `priors` (kept by the caller, across sessions).
   */
  constructor(
    private readonly generating: () => boolean = () => false,
    private readonly voice: () => string = () => '',
    private readonly priors: VoicePriors = new Map(),
  ) {}

  /** A barge-in drops everything held; a new session (`session`) also starts at the usual margin. */
  reset(session = false) {
    this.held.clear()
    this.started.clear()
    this.textDone.clear()
    this.speechDone.clear()
    this.writtenChars.clear()
    this.freeRate.clear()
    this.got.clear()
    this.rates.clear()
    if (session) this.marginUp = 0
  }

  /** Whether some reply's start is still being held (the engine then calls `tick` every 25 ms). */
  get holding() {
    return this.held.size > 0
  }

  /** The margin a reply not yet settled needs on top of what its rate needs. */
  get margin() {
    return MARGIN_MS + this.marginUp
  }

  /** A piece of `item`'s audio of `ms` arrived: the pieces to schedule now (it alone once the reply has started). */
  piece(item: string, piece: T, ms: number, now: number): T[] {
    const got = this.got.get(item) ?? { first: now, ms: 0, firstMs: ms }
    got.ms += ms
    this.got.set(item, got)
    if (this.started.has(item)) return [piece]
    const held = this.held.get(item) ?? { pieces: [], firstAt: now, haveMs: 0 }
    held.pieces.push(piece)
    held.haveMs += ms
    this.held.set(item, held)
    return this.check(item, now)
  }

  /** `chars` more of the reply's spoken text were written (`response.audio_transcript.delta`). */
  written(item: string, chars: number) {
    this.writtenChars.set(item, (this.writtenChars.get(item) ?? 0) + chars)
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
    this.learn(item, now) // this voice's rate and audio a character, for the next replies' start
    this.got.delete(item)
    this.rates.delete(item)
    return this.check(item, now)
  }

  /** Playback ran dry mid-reply for `ms`: the session's next replies start with more in hand. */
  gap(ms: number) {
    this.marginUp = Math.min(MARGIN_UP_MAX_MS, this.marginUp + 2 * ms)
  }

  /** Re-checks every held reply (time passing changes the measured rate and the cap). */
  tick(now: number): T[] {
    return [...this.held.keys()].flatMap((item) => this.check(item, now))
  }

  private startMs() {
    return this.generating() ? GEN_START_MS : START_MS
  }

  private learn(item: string, now: number) {
    const got = this.got.get(item)
    if (!got) return
    const span = now - got.first
    const chars = this.textDone.get(item) ?? 0
    const key = this.voice()
    const prior = this.priors.get(key) ?? {}
    if (span > 300) {
      const rate = (got.ms - got.firstMs) / span
      prior.rate = prior.rate === undefined ? rate : 0.5 * prior.rate + 0.5 * rate
    }
    if (chars > 20) prior.mpc = prior.mpc === undefined ? got.ms / chars : 0.5 * prior.mpc + 0.5 * (got.ms / chars)
    this.priors.set(key, prior)
  }

  /**
   * The item's arrival rate: each RATE_WINDOW_MS window from the first piece (that piece not counted) into an EWMA
   * with its spread, seeded with the voice's rate on its last replies; and its audio a character: the voice's on its
   * last replies with 15% to spare, MS_PER_CHAR at least.
   */
  private rateOf(item: string, held: Held<T>, now: number) {
    let r = this.rates.get(item)
    if (!r) {
      const p = this.priors.get(this.voice()) ?? {}
      r = {
        rate: p.rate === undefined ? PRIOR_RATE : Math.min(1.2, Math.max(0.5, p.rate)),
        v: 0,
        at: held.firstAt,
        from: held.haveMs,
        mpc: Math.max(MS_PER_CHAR, 1.15 * (p.mpc ?? 0)),
      }
      this.rates.set(item, r)
    }
    if (now - r.at >= RATE_WINDOW_MS) {
      const d = (held.haveMs - r.from) / (now - r.at) - r.rate
      r.rate += EWMA_ALPHA * d
      r.v = (1 - EWMA_ALPHA) * (r.v + EWMA_ALPHA * d * d)
      r.at = now
      r.from = held.haveMs
    }
    return r
  }

  /** Audio in hand that plays the rest of the reply to its end without running dry (see the header comment). */
  private need(item: string, held: Held<T>, now: number) {
    const r = this.rateOf(item, held, now)
    const done = this.textDone.has(item)
    const free = done && !this.generating() ? this.freeRate.get(item) || this.lastRate || 0 : 0
    const chars = done ? (this.textDone.get(item) ?? 0) : (this.writtenChars.get(item) ?? 0)
    const rest = Math.max(0, chars * r.mpc - (this.got.get(item)?.ms ?? 0))
    let rate = Math.max(0, r.rate - RATE_HEADROOM * Math.sqrt(r.v))
    if (free > 1) rate = Math.max(rate, 1)
    const settled = done && free >= 1 && rate >= 1 // written, and speech has the GPU to itself: no margin
    return Math.max(this.startMs(), rest * (1 - Math.min(rate, 1)) + (settled ? 0 : this.margin))
  }

  private check(item: string, now: number): T[] {
    const held = this.held.get(item)
    if (!held) return []
    const ready = this.speechDone.has(item) || held.haveMs >= this.need(item, held, now)
    if (!ready && now - held.firstAt < HOLD_CAP_MS) return []
    this.held.delete(item)
    this.rates.delete(item)
    this.started.add(item)
    return held.pieces
  }
}
