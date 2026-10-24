import { describe, expect, it } from 'vitest'
import { chatDecider, decisionRequest, parseDecision } from './decide.ts'

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
