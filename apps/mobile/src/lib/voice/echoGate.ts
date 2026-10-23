/**
 * Echo gate. Native echo cancellation leaves some of the assistant's voice
 * in the microphone when it plays through the speaker, most at the start of
 * a call while the canceller is still adapting, and every voice provider
 * hears that as the user talking: it interrupts itself and answers its own
 * words. While the phone is playing, only microphone audio clearly louder
 * than the echo expected from what is playing goes to the computer; the
 * rest becomes silence, so the stream keeps its timing.
 *
 * How much of the playback comes back (the coupling) is learned during the
 * call from chunks that are not the user talking. It starts high (assume a
 * strong echo) and settles within about a second of the assistant speaking.
 * Speech that stays well above the expected echo opens the gate (barge-in),
 * including the chunk held while it was confirmed, so no words are lost.
 *
 * A reply's audio can run dry for a moment and resume (a late chunk). Its last
 * words still echo in that gap, and a speaker or headset that lags the schedule
 * pushes the echo later still, so the gap is treated as the reply playing on
 * (`echoReference`). Without that the echo passed as the user in the gap, and
 * the assistant interrupted itself and started its reply over. Talking over the
 * reply in a gap opens the gate exactly as it does while audio plays.
 */

/** Microphone RMS (full scale 1) below which a chunk is never the user talking over a reply. */
export const MIN_BARGE_IN_RMS = 0.03
/** The user talking over a reply is this many times louder than the echo expected then. */
export const SPEECH_OVER_ECHO = 3
/** The coupling assumed before any is learned: a strong echo, as at the start of a call. */
export const INITIAL_COUPLING = 0.6
const MIN_COUPLING = 0.02
const MAX_COUPLING = 1.5
/** The microphone chunk the gate sees. */
const CHUNK_MS = 40
/** How fast the coupling follows what is measured: 30% per 100 ms, whatever the chunk size. */
const LEARN_RATE = 1 - 0.7 ** (CHUNK_MS / 100)
/** Consecutive loud chunks (40 ms each, so 200 ms) that confirm the user is talking over the reply. */
export const CONFIRM_CHUNKS = 5
/** The gate stays open this long after the last chunk that was clearly the user. */
export const OPEN_HOLD_MS = 700

/** A reply keeps its echo reference this long after its audio last played, while more of it may still come. */
export const REPLY_GAP_BRIDGE_MS = 1_500

/** Audio scheduled to play, in wall-clock ms, with its level. */
export type PlayedSegment = { from: number; to: number; rms: number }

/**
 * The echo to expect for a microphone chunk that ended at `now`: the loudest
 * audio playing during it, or up to `tailMs` before it (the speaker and the
 * canceller lag the schedule). `replyOpen`: the reply is still streaming, so a
 * gap in its audio is bridged with its recent level; `bridged` says so, and
 * the gate learns nothing from a bridged chunk (nothing is actually playing).
 */
export function echoReference(played: PlayedSegment[], now: number, options: { chunkMs: number; tailMs: number; replyOpen: boolean }): { level: number; bridged: boolean } {
  const start = now - options.chunkMs - options.tailMs
  const level = played.reduce((loudest, segment) => (segment.from < now && segment.to > start ? Math.max(loudest, segment.rms) : loudest), 0)
  if (level > 0 || !options.replyOpen) return { level, bridged: false }
  const recent = played.reduce((loudest, segment) => (segment.from < now && segment.to > now - REPLY_GAP_BRIDGE_MS ? Math.max(loudest, segment.rms) : loudest), 0)
  return { level: recent, bridged: recent > 0 }
}

/**
 * `playLevel`: RMS of what the phone played in the echo window for this chunk (0 when nothing).
 * `bridged`: nothing is playing, but the reply is paused mid-stream (see `echoReference`).
 */
export type GateInput = { data: string; rms: number; now: number; playLevel: number; bridged?: boolean }

export class EchoGate {
  private held: string[] = []
  private openUntil = 0
  private learned = INITIAL_COUPLING
  /** How much of what plays comes back through the microphone, as learned so far. */
  get coupling() {
    return this.learned
  }

  /** The last chunk confirmed the user talking over a reply: the phone stops playback (barge-in). */
  bargedIn = false

  /** The chunks to send for this microphone chunk; `silence` stands in for gated audio. */
  process({ data, rms, now, playLevel, bridged = false }: GateInput, silence: string): string[] {
    this.bargedIn = false
    const expectedEcho = this.learned * playLevel
    const talking = rms >= Math.max(MIN_BARGE_IN_RMS, SPEECH_OVER_ECHO * expectedEcho)
    if (playLevel <= 0 || now < this.openUntil) {
      if (talking && playLevel > 0) this.openUntil = now + OPEN_HOLD_MS
      const sent = [...this.held, data]
      this.held = []
      return sent
    }
    if (!talking) {
      // Echo (or quiet): it tells how much of the playback comes back. In a bridged gap nothing plays, so it tells nothing.
      if (!bridged) this.learned = Math.min(MAX_COUPLING, Math.max(MIN_COUPLING, this.learned + LEARN_RATE * (rms / playLevel - this.learned)))
      this.held = []
      return [silence]
    }
    this.held.push(data)
    if (this.held.length < CONFIRM_CHUNKS) return [silence]
    // Confirmed: the user is talking over the reply. Send what was held (it replaces silence already sent).
    this.openUntil = now + OPEN_HOLD_MS
    this.bargedIn = true
    const sent = this.held
    this.held = []
    return sent
  }

  reset() {
    this.held = []
    this.openUntil = 0
    this.learned = INITIAL_COUPLING
  }
}
