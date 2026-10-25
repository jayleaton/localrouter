import { describe, expect, it } from 'vitest'
import { chatDecider, decisionRequest, parseDecision, transient } from './decide.ts'

const questions = { next: { type: 'choice' as const, instructions: 'Which?', criteria: { c0: 'Click Send', stuck: 'none' } }, consequential: { type: 'noul' as const, instructions: 'Could it send?' } }

describe('decisions from the Spark chat model, as the box asks them', () => {
  it('constrains every choice to its offered keys', () => {
    const body = decisionRequest({ goal: 'x' }, questions, { reasoning_effort: 'none' })
    const schema = (body.response_format as { json_schema: { schema: { properties: Record<string, { properties: Record<string, { enum: string[] }> }> } } }).json_schema.schema
    expect(schema.properties.next.properties.choice.enum).toEqual(['c0', 'stuck'])
    expect(schema.properties.consequential.properties.answer.enum).toEqual(['yes', 'no'])
    expect(body.temperature).toBe(0)
    expect(body.reasoning_effort).toBe('none')
  })
  it('maps confidence to probability and drops anything not offered', () => {
    expect(parseDecision('```json\n{"next":{"choice":"c0","confidence":"high"},"consequential":{"answer":"no","confidence":"medium"}}```', questions)).toEqual({ next: { type: 'choice', choice: 'c0', probability: 0.9 }, consequential: { type: 'noul', probability: 0.4 } })
    expect(parseDecision('{"next":{"choice":"c9","confidence":"high"}}', questions)).toEqual({})
    expect(() => parseDecision('I think c0', questions)).toThrow(/JSON/)
  })
  it('asks once per question set', async () => {
    const bodies: unknown[] = []
    const d = chatDecider(async (body) => { bodies.push(body); return '{"next":{"choice":"stuck","confidence":"low"}}' })
    expect(await d.ask({}, { next: questions.next })).toEqual({ next: { type: 'choice', choice: 'stuck', probability: 0.3 } })
    expect(bodies).toHaveLength(1)
  })
})

describe('decisions that go missing without anything being wrong with the task (Windows run, 2026-10-25)', () => {
  const good = JSON.stringify({ next: { choice: 'c0', confidence: 'high' }, consequential: { answer: 'no', confidence: 'high' } })
  const gone = Object.assign(new Error("the model's server is gone (Closed); log tail:"), { status: 500 })

  it('reads the JSON object past a thinking block or fences and before anything written after it', () => {
    expect(parseDecision(`<think>\n</think>\n\`\`\`json\n${good}\n\`\`\`\nThat is my answer {really}.`, questions).next).toMatchObject({ choice: 'c0' })
    expect(parseDecision('{"next":{"choice":"c0","confidence":"high","note":"a } in a string"}}', questions).next).toMatchObject({ choice: 'c0' })
    expect(() => parseDecision('{"next":{"choice":"c0"', questions)).toThrow('the decision was not the JSON asked for')
  })

  it('asks once more when the reply is not the JSON asked for, then gives up', async () => {
    const replies = ['Sure! I would click Send.', good]
    const asked: number[] = []
    const decider = chatDecider(async () => { asked.push(1); return replies.shift() ?? 'still not JSON' })
    expect((await decider.ask({}, questions)).next).toMatchObject({ choice: 'c0' })
    expect(asked).toHaveLength(2)
    const never = chatDecider(async () => 'not JSON')
    await expect(never.ask({}, questions)).rejects.toThrow('the decision was not the JSON asked for')
  })

  it('waits for a decision service that restarts (a deploy), and stops waiting after 30 s', async () => {
    const slept: number[] = []
    const sleep = async (ms: number) => void slept.push(ms)
    const outcomes: Array<Error | string> = [gone, Object.assign(new Error('fetch failed'), {}), Object.assign(new Error('Service Unavailable'), { status: 503 }), good]
    const decider = chatDecider(async () => { const o = outcomes.shift()!; if (o instanceof Error) throw o; return o }, {}, { sleep })
    expect((await decider.ask({}, questions)).next).toMatchObject({ choice: 'c0' })
    expect(slept).toEqual([1000, 2000, 4000])
    const down = chatDecider(async () => { throw gone }, {}, { sleep: async () => undefined })
    await expect(down.ask({}, questions)).rejects.toThrow(/did not come back within 30 s/)
  })

  it('does not wait for what is not a restart: a refused token or a bad request fails at once', async () => {
    const slept: number[] = []
    const refused = chatDecider(async () => { throw Object.assign(new Error('invalid_api_key'), { status: 401 }) }, {}, { sleep: async (ms) => void slept.push(ms) })
    await expect(refused.ask({}, questions)).rejects.toThrow('invalid_api_key')
    expect(slept).toEqual([])
    expect(transient(Object.assign(new Error('bad schema'), { status: 400 }))).toBe(false)
    expect(transient(Object.assign(new Error('model loading'), { status: 500 }))).toBe(true)
    expect(transient(Object.assign(new Error('internal error'), { status: 500 }))).toBe(false)
  })
})

