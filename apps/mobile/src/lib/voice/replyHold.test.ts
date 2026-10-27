import { describe, expect, it } from 'vitest'

import { HOLD_CAP_MS, ReplyHold, START_MS, WRITING_CAP_MS } from './replyHold'

describe('the reply start hold', () => {
  it('a written reply on a free GPU starts once 200 ms is in hand', () => {
    const hold = new ReplyHold<number>()
    expect(hold.text('a', 30, 1.4, false, 0)).toEqual([])
    expect(hold.piece('a', 1, 100, 0)).toEqual([])
    expect(hold.piece('a', 2, 100, 10)).toEqual([1, 2])
    expect(hold.piece('a', 3, 100, 20)).toEqual([3]) // started: later pieces go straight out
  })

  it('while the text is still written, a stream that keeps up starts; a slow one waits for the text at most 1 s', () => {
    const fast = new ReplyHold<number>()
    fast.piece('a', 0, 46, 0)
    expect(fast.piece('a', 1, 186, 125)).toEqual([])
    expect(fast.piece('a', 2, 186, 250)).toEqual([0, 1, 2]) // 372 ms of audio in 250 ms: 1.49x

    const slow = new ReplyHold<number>()
    slow.piece('a', 0, 46, 0)
    slow.piece('a', 1, 100, 200)
    expect(slow.piece('a', 2, 100, 300)).toEqual([]) // 0.67x and length unknown: wait
    // Written now, 400 chars at 65 ms each: needs 26 s x (1 - 0.67) x 1.25 in hand, so it still waits...
    expect(slow.text('a', 400, null, false, 310)).toEqual([])
    // ...until all of its speech is in.
    expect(slow.speech('a', 400)).toEqual([0, 1, 2])

    const long = new ReplyHold<number>()
    long.piece('a', 0, 46, 0)
    expect(long.piece('a', 1, 100, 200)).toEqual([]) // 0.5x, still being written
    expect(long.tick(WRITING_CAP_MS - 1)).toEqual([])
    expect(long.tick(WRITING_CAP_MS)).toEqual([0, 1]) // it speaks while it is written
  })

  it('never holds past the cap, and a picture being made needs more in hand', () => {
    const hold = new ReplyHold<number>()
    hold.text('a', 400, null, false, 0)
    hold.piece('a', 0, 46, 0)
    expect(hold.tick(HOLD_CAP_MS - 1)).toEqual([])
    expect(hold.tick(HOLD_CAP_MS)).toEqual([0])

    const busy = new ReplyHold<number>(() => true)
    busy.text('a', 10, 1.4, false, 0)
    busy.piece('a', 0, START_MS, 0)
    // The free rate is not trusted while the GPU is shared, and nothing more came: wait.
    expect(busy.tick(300)).toEqual([])
    // 1.33x since the first piece, so the length no longer matters, but 500 ms (not 200) must be in hand: 600 is.
    expect(busy.piece('a', 1, 400, 300)).toEqual([0, 1])
  })
})
