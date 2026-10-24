import { fakeMac } from '../../../desktop-helper/src/fake.ts'
import type { Answers, Decider, Question } from '../../../desktop-helper/src/computer/decisions.ts'
import { MemoryAudit } from './audit.ts'
import { Broker } from './broker.ts'
import type { BrokerOptions, Card, PresenceGate } from './broker.ts'

/**
 * Test and headless doubles for the executor: a decision provider that answers from a script of next steps, and a
 * presence gate that says yes or no. The app never uses these; the headless runner (`--fake-mac`) does.
 */
export function scriptedDecider(next: string[], judge = 0.9): Decider & { asked: Array<Record<string, Question>> } {
  const asked: Array<Record<string, Question>> = []
  return {
    asked,
    async ask(_state: unknown, questions: Record<string, Question>): Promise<Answers> {
      asked.push(questions)
      const out: Answers = {}
      if (questions.surface) out.surface = { type: 'choice', choice: 'current', probability: 0.9 }
      if (questions.consequential) out.consequential = { type: 'noul', probability: judge }
      if (questions.type_into) out.type_into = { type: 'choice', choice: 'c0', probability: 0.9 }
      if (questions.next) {
        const step = next.shift()
        if (!step) throw new Error('no step scripted')
        out.next = { type: 'choice', choice: step, probability: 0.9 }
      }
      return out
    },
  }
}

export class TestGate implements PresenceGate {
  readonly method = 'test presence'
  readonly asked: Card[] = []
  answer = true

  async verify(card: Card) {
    this.asked.push(card)
    return this.answer
  }
}

export function fakeBroker(next: string[], extra: Partial<BrokerOptions> = {}) {
  const mac = fakeMac()
  const audit = new MemoryAudit()
  const gate = new TestGate()
  const decider = scriptedDecider(next)
  const broker = new Broker({ executorId: 'exec_test', name: 'Test Mac', machine: { desktop: mac.desktop, tabs: null, dock: mac.dock, apps: mac.apps, controls: mac.controls }, decider, audit, gate, minDwellMs: 0, settleMs: 5, ...extra })
  return { broker, mac, audit, gate, decider }
}
