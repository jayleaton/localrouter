import { fakeMac } from '../../../desktop-helper/src/fake.ts'
import type { Answers, Decider, Question } from '../../../desktop-helper/src/computer/decisions.ts'
import { MemoryAudit } from './audit.ts'
import { Broker } from './broker.ts'
import type { BrokerOptions, Card, PresenceGate, Standing } from './broker.ts'

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
  readonly signals: AbortSignal[] = []
  answer = true

  async verify(card: Card, _waiting?: (text: string) => void, signal?: AbortSignal) {
    this.asked.push(card)
    if (signal) this.signals.push(signal)
    return this.answer
  }
}

/** A standing grant in `kept`: a test hands the same object to a second broker to stand for a restart or an update. */
export class MemoryStanding implements Standing {
  readonly kept: { granted: boolean; askEach: boolean }

  constructor(kept: { granted: boolean; askEach: boolean } = { granted: false, askEach: false }) {
    this.kept = kept
  }

  granted() {
    return this.kept.granted
  }

  set(granted: boolean) {
    this.kept.granted = granted
  }

  askEachAction() {
    return this.kept.askEach
  }
}

export function fakeBroker(next: string[], extra: Partial<BrokerOptions> = {}) {
  const mac = fakeMac()
  const audit = new MemoryAudit()
  const gate = new TestGate()
  const decider = scriptedDecider(next)
  const broker = new Broker({ executorId: 'exec_test', name: 'Test Mac', machine: { screen: mac.desktop, tabs: null, launcher: mac.dock, apps: mac.apps, controls: mac.controls }, decider, audit, gate, minDwellMs: 0, settleMs: 5, refocusMs: 0, ...extra })
  return { broker, mac, audit, gate, decider }
}
