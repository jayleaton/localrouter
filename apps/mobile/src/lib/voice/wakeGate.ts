/*
 * The energy gate in front of the wake-phrase recognizer (ADR 0021). The
 * on-device recognizer is the largest standby drain when it hears every
 * chunk, silence included; behind this gate it hears only sound that rises
 * above the room's noise floor, with the chunks just before it (pre-roll)
 * so the start of "hey jarvis" is not clipped. After a quiet spell the gate
 * closes and the recognizer is told to go idle.
 */

/** Level (RMS) the gate always treats as sound, however quiet the room. */
const MIN_OPEN_RMS = 0.006
/**
 * Sound is this many times the learned floor (+3 dB): a voice as loud as the
 * background, whatever the background. A larger ratio let a learned music or
 * conversation floor push the threshold above a voice speaking over it.
 */
const OPEN_RATIO = 1.4
/** The floor follows quieter audio quickly and louder audio over about this long, so a steady fan or hum stops opening the gate. */
const FLOOR_RISE_MS = 10_000
const FLOOR_FALL = 0.3
/** A quiet room until the gate has heard this one, so speech already under way when standby starts is heard. */
const INITIAL_FLOOR = MIN_OPEN_RMS / OPEN_RATIO

export type WakeGateOptions = {
  chunkMs: number
  /** Kept from before the gate opens (the start of the phrase). */
  preRollMs?: number
  /** Quiet this long after sound closes the gate (the recognizer needs the end of the phrase). */
  hangMs?: number
}

export type WakeGateStep<T> = {
  /** Chunks for the recognizer now, oldest first. */
  feed: T[]
  /** The gate just closed: the recognizer can stop until it opens again. */
  closed: boolean
}

export class WakeGate<T> {
  private readonly preRollChunks: number
  private readonly hangChunks: number
  private readonly floorRise: number
  private floor = INITIAL_FLOOR
  private preRoll: T[] = []
  private open = false
  private quietChunks = 0

  constructor(options: WakeGateOptions) {
    this.preRollChunks = Math.max(1, Math.round((options.preRollMs ?? 800) / options.chunkMs))
    this.hangChunks = Math.max(1, Math.round((options.hangMs ?? 1_500) / options.chunkMs))
    this.floorRise = Math.min(1, options.chunkMs / FLOOR_RISE_MS)
  }

  /** Whether the recognizer is being fed. */
  get isOpen() {
    return this.open
  }

  /** One microphone chunk and its level. */
  process(chunk: T, rms: number): WakeGateStep<T> {
    const floor = this.floor
    const loud = rms >= Math.max(MIN_OPEN_RMS, floor * OPEN_RATIO)
    this.floor = floor + (rms - floor) * (rms < floor ? FLOOR_FALL : this.floorRise)
    if (this.open) {
      this.quietChunks = loud ? 0 : this.quietChunks + 1
      if (this.quietChunks < this.hangChunks) return { feed: [chunk], closed: false }
      this.open = false
      return { feed: [chunk], closed: true }
    }
    if (!loud) {
      this.preRoll.push(chunk)
      if (this.preRoll.length > this.preRollChunks) this.preRoll.shift()
      return { feed: [], closed: false }
    }
    this.open = true
    this.quietChunks = 0
    const feed = [...this.preRoll, chunk]
    this.preRoll = []
    return { feed, closed: false }
  }

  /** Forgets the room (a session ended, or standby restarted). */
  reset() {
    this.floor = INITIAL_FLOOR
    this.preRoll = []
    this.open = false
    this.quietChunks = 0
  }
}
