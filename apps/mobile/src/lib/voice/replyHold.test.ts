import { describe, expect, it } from 'vitest'

import { HOLD_CAP_MS, MARGIN_MS, ReplyHold, START_MS, type VoicePriors } from './replyHold'

/** Feeds `item` a first piece of `firstMs` at 0, then `ms` every 150 ms; returns when (and with what) it started. */
function stream(hold: ReplyHold<number>, firstMs: number, ms: number, until = 200) {
  let have = firstMs
  if (hold.piece('a', 0, firstMs, 0).length) return { k: 0, have }
  for (let k = 1; k < until; k++) {
    have += ms
    if (hold.piece('a', k, ms, 150 * k).length) return { k, have }
  }
  return null
}

describe('the reply start hold', () => {
  it('a short reply written on a free GPU starts with START_MS in hand, no margin', () => {
    const hold = new ReplyHold<number>()
    expect(hold.text('a', 30, 1.4, false, 0)).toEqual([])
    expect(hold.piece('a', 1, 100, 0)).toEqual([])
    expect(hold.piece('a', 2, 100, 10)).toEqual([1, 2])
    expect(hold.piece('a', 3, 100, 20)).toEqual([3]) // started: later pieces go straight out
  })

  it('a slow stream of a long written text waits for the rest x (1 - rate) + margin, and starts once that is in hand', () => {
    // 40 chars (2600 ms) at 0.25x: in hand >= (2600 - in hand) x 0.75 + 300, about 1286 ms.
    const hold = new ReplyHold<number>()
    hold.text('a', 40, null, false, 0)
    const started = stream(hold, 50, 37.5)
    expect(started).not.toBeNull()
    expect(started!.have).toBeGreaterThanOrEqual((2_600 - started!.have) * 0.75 + MARGIN_MS)
    // Not before: one piece earlier was short of it.
    const before = started!.have - 37.5
    expect(before).toBeLessThan((2_600 - before) * 0.75 + MARGIN_MS)
    expect(started!.have).toBe(1_287.5)
  })

  it('a stream at 1x or more still being written starts once the margin is in hand', () => {
    const hold = new ReplyHold<number>()
    hold.written('a', 2_000)
    hold.piece('a', 0, 46, 0)
    expect(hold.piece('a', 1, 210, 150)).toEqual([]) // 1.4x, but 256 ms is short of the 300 ms margin
    expect(hold.piece('a', 2, 210, 300)).toEqual([0, 1, 2]) // 466 ms in hand

    // While a picture is being made, GEN_START_MS: 466 ms is not enough, 676 is.
    const busy = new ReplyHold<number>(() => true)
    busy.written('a', 2_000)
    busy.piece('a', 0, 46, 0)
    busy.piece('a', 1, 210, 150)
    expect(busy.piece('a', 2, 210, 300)).toEqual([])
    expect(busy.piece('a', 3, 210, 450)).toEqual([0, 1, 2, 3])
  })

  it('all of the reply speech in starts it, and it never holds past the cap', () => {
    const hold = new ReplyHold<number>()
    hold.text('a', 400, null, false, 0)
    hold.piece('a', 0, 46, 0)
    expect(hold.piece('a', 1, 100, 400)).toEqual([])
    expect(hold.speech('a', 500)).toEqual([0, 1])

    const capped = new ReplyHold<number>()
    capped.text('a', 400, null, false, 0)
    capped.piece('a', 0, 46, 0)
    expect(capped.tick(HOLD_CAP_MS - 1)).toEqual([])
    expect(capped.tick(HOLD_CAP_MS)).toEqual([0])
  })

  it('after a pause, the session next replies need more in hand, until a new session', () => {
    const hold = new ReplyHold<number>()
    hold.gap(100) // the margin grows by 200 ms: 500 ms
    hold.written('a', 2_000)
    hold.piece('a', 0, 46, 0)
    hold.piece('a', 1, 210, 150)
    expect(hold.piece('a', 2, 210, 300)).toEqual([]) // 466 ms is not enough now
    expect(hold.piece('a', 3, 210, 450)).toEqual([0, 1, 2, 3])
    expect(hold.margin).toBe(MARGIN_MS + 200)

    hold.reset() // a barge-in keeps it
    expect(hold.margin).toBe(MARGIN_MS + 200)
    hold.gap(5_000)
    expect(hold.margin).toBe(MARGIN_MS + 2_000) // at most 2 s more
    hold.reset(true)
    expect(hold.margin).toBe(MARGIN_MS)

    // A short reply written on a free GPU still needs START_MS only.
    hold.gap(500)
    hold.text('b', 30, 1.4, false, 0)
    expect(hold.piece('b', 0, START_MS, 0)).toEqual([0])
  })

  it('a voice that learned a longer audio a character needs more in hand', () => {
    // 20 chars at 0.5x, seeded with a 0.5 rate: 65 ms a char starts at 650 ms in hand, 130 (x 1.15 to spare) at 1250.
    const plain = new ReplyHold<number>(() => false, () => 'v', new Map([['v', { rate: 0.5 }]]))
    plain.text('a', 20, null, false, 0)
    expect(stream(plain, 50, 75)?.have).toBe(650)

    const slow = new ReplyHold<number>(() => false, () => 'v', new Map([['v', { rate: 0.5, mpc: 130 }]]))
    slow.text('a', 20, null, false, 0)
    expect(stream(slow, 50, 75)?.have).toBe(1_250)
  })

  it('learns the voice rate and audio a character when all of a reply speech is in', () => {
    const priors: VoicePriors = new Map()
    const hold = new ReplyHold<number>(() => false, () => 'm:v', priors)
    hold.text('a', 40, null, false, 0)
    hold.piece('a', 0, 200, 0)
    hold.piece('a', 1, 4_000, 1_000)
    hold.speech('a', 1_000)
    expect(priors.get('m:v')).toEqual({ rate: 4, mpc: 105 })
    hold.text('b', 40, null, false, 2_000)
    hold.piece('b', 0, 200, 2_000)
    hold.piece('b', 1, 2_000, 3_000)
    hold.speech('b', 3_000)
    expect(priors.get('m:v')).toEqual({ rate: 3, mpc: 80 }) // half the last, half this one
  })
})
