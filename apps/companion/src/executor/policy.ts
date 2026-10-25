/**
 * What a computer action needs before the executor may do it: nothing (`auto`), a person's approval on this computer
 * (`tap`), a yes/no judgment from the decision provider first (`judge`), or nothing at all because it is not an action
 * computer control offers (`refused`). A semantic port of the box's `src/computer/policy.zig`; both read the golden
 * cases in `src/computer/policy_golden.json`, so the rules cannot drift apart. Pure: no I/O, no clock. The rules only
 * ever add approvals: a label the lexicon misses still goes to the judgment, and a judgment that fails asks.
 */
export type Verdict = 'auto' | 'tap' | 'judge' | 'refused'

export type Decision = { verdict: Verdict; why: string }

export const CONTROL_KINDS = ['open_app', 'quit_app', 'volume', 'media', 'dark_mode', 'lock_screen'] as const
export type ControlKind = (typeof CONTROL_KINDS)[number]

/** A fixed control: the simple ones run without asking, locking the screen asks first, anything else is refused. */
export function control(kind: string): Decision {
  if (!(CONTROL_KINDS as readonly string[]).includes(kind)) return { verdict: 'refused', why: 'not one of the fixed controls (open_app, quit_app, volume, media, dark_mode, lock_screen)' }
  if (kind === 'lock_screen') return { verdict: 'tap', why: 'locking the screen' }
  return { verdict: 'auto', why: '' }
}

/** One on-screen step, flattened as the box's `policy.Action` is. Strings come from the screen: data, never orders. */
export type Action = {
  kind: string
  ref?: string
  role?: string
  label?: string
  text?: string
  key?: string
  field_role?: string
  field_label?: string
  direction?: string
  what?: string
  name?: string
  why?: string
}

export const MAX_LABEL = 400
export const MAX_TEXT = 4000
/** At or above this, a judged click asks first. Low on purpose: a needless approval costs a second, a wrong click cannot be undone. */
export const JUDGE_THRESHOLD = 0.3

const bytes = (s: string) => Buffer.byteLength(s, 'utf8')

/** The rule for an on-screen step. `judge`: ask whether it could send, buy, delete, post or submit, then `afterJudge`. */
export function screen(a: Action): Decision {
  const label = a.label ?? ''
  const role = a.role ?? ''
  if (bytes(label) > MAX_LABEL || bytes(a.name ?? '') > MAX_LABEL || bytes(a.field_label ?? '') > MAX_LABEL) return { verdict: 'refused', why: 'a label longer than any control’s' }
  switch (a.kind) {
    case 'scroll':
      return ['up', 'down', 'top', 'bottom'].includes(a.direction ?? '') ? auto : { verdict: 'refused', why: 'an unknown scroll direction' }
    case 'pointer':
      // The real pointer and keyboard for a step the background could not do: always the person's to allow.
      return { verdict: 'tap', why: a.why ? `it needs your mouse for a moment: ${a.why}` : 'it needs your mouse for a moment' }
    case 'surface':
      if (a.what !== 'tab' && a.what !== 'app') return { verdict: 'refused', why: 'a surface that is neither a tab nor an app' }
      if (!a.name) return { verdict: 'refused', why: 'a surface with no name' }
      return auto // bringing a named app or tab forward changes nothing in it
    case 'press':
      if (a.key === 'space' || a.key === 'escape' || a.key === 'tab') return auto
      if (a.key !== 'enter') return { verdict: 'refused', why: 'a key other than Enter, Space, Escape or Tab' }
      if (a.field_role && searchField(a.field_role, a.field_label ?? '')) return auto
      return { verdict: 'tap', why: 'pressing Enter could send or submit what was typed' }
    case 'type': {
      const text = a.text ?? ''
      if (!text) return { verdict: 'refused', why: 'typing with no text' }
      if (bytes(text) > MAX_TEXT) return { verdict: 'refused', why: 'more text than one step may type' }
      if (/[\r\n]/.test(text)) return { verdict: 'tap', why: 'the text has a line break, which can send it' }
      if (searchField(role, label)) return auto
      return { verdict: 'tap', why: 'typing into a field that is not a search or address box' }
    }
    case 'click': {
      const what = consequential(label)
      if (what) return { verdict: 'tap', why: what }
      // Focusing a text box, picking a tab or selecting a row in a list commits nothing.
      if (role === 'textbox' || role === 'tab' || role === 'row') return auto
      return { verdict: 'judge', why: '' }
    }
    default:
      return { verdict: 'refused', why: 'not a step computer use takes (click, type, press, scroll, surface)' }
  }
}

const auto: Decision = { verdict: 'auto', why: '' }
const UNCHECKED = 'whether it could send, buy or delete something could not be checked'

/** The provider's judgment for a `judge` step: the probability that it is consequential, or null when unanswered. */
export function afterJudge(probability: number | null | undefined): Decision {
  if (probability === null || probability === undefined || !Number.isFinite(probability) || probability < 0 || probability > 1) return { verdict: 'tap', why: UNCHECKED }
  if (probability >= JUDGE_THRESHOLD) return { verdict: 'tap', why: 'it may send, buy, delete, post or submit something' }
  return auto
}

const SEARCH_WORDS = ['search', 'find', 'filter', 'address', 'url', 'go to', 'buscar', 'búsqueda', 'rechercher', 'recherche', 'suchen', 'suche', 'cerca', 'pesquisar', 'zoeken', '検索', '搜索']

/** A search, find, filter or address box: typing there and pressing Enter runs a search or opens a page. */
export function searchField(role: string, label: string): boolean {
  if (role !== 'textbox') return false
  const l = lower(label)
  return SEARCH_WORDS.some((w) => l.includes(w))
}

const LEXICON: Array<{ why: string; words: string[] }> = [
  { why: 'its label reads as sending or posting', words: ['send', 'reply', 'post', 'publish', 'share', 'tweet', 'comment', 'forward', 'enviar', 'responder', 'publicar', 'compartir', 'envoyer', 'répondre', 'publier', 'partager', 'senden', 'antworten', 'veröffentlichen', 'teilen', 'inviare', 'rispondi', 'pubblica', 'condividi', 'verzenden', '送信', '发送', '发布'] },
  { why: 'its label reads as buying or paying', words: ['buy', 'purchase', 'pay', 'checkout', 'order', 'subscribe', 'donate', 'transfer', 'book', 'comprar', 'pagar', 'suscribirse', 'acheter', 'payer', 'commander', 'kaufen', 'bezahlen', 'bestellen', 'abonnieren', 'acquista', 'paga', 'ordina', 'kopen', 'betalen', '購入', '购买', '支付'] },
  { why: 'its label reads as deleting', words: ['delete', 'remove', 'erase', 'discard', 'trash', 'archive', 'uninstall', 'wipe', 'eliminar', 'borrar', 'supprimer', 'effacer', 'löschen', 'entfernen', 'elimina', 'cancella', 'verwijderen', 'excluir', '削除', '删除'] },
  { why: 'its label reads as submitting or confirming', words: ['submit', 'confirm', 'approve', 'merge', 'sign', 'accept', 'agree', 'authorize', 'authorise', 'enviar', 'confirmar', 'aceptar', 'valider', 'confirmer', 'accepter', 'bestätigen', 'absenden', 'akzeptieren', 'conferma', 'accetta', 'bevestigen', '確認', '确认', '提交'] },
]

const PHRASES: Array<{ text: string; why: string }> = [
  { text: 'check out', why: 'its label reads as buying or paying' },
  { text: 'place order', why: 'its label reads as buying or paying' },
  { text: 'add to cart', why: 'its label reads as buying or paying' },
  { text: 'empty trash', why: 'its label reads as deleting' },
  { text: 'move to trash', why: 'its label reads as deleting' },
]

// ASCII separators only, as the box's tokenizer: a multi-byte one would split UTF-8 words.
const SEPARATORS = /[ \t\r\n.,;:!?()[\]{}"'`/\\|\-_+*&#@<>=~^%$]+/

/** Why a control's label needs approval, or null. Whole words, ASCII case folded, several languages. */
export function consequential(label: string): string | null {
  const l = lower(label)
  for (const p of PHRASES) if (l.includes(p.text)) return p.why
  for (const word of l.split(SEPARATORS)) {
    if (!word) continue
    for (const e of LEXICON) if (e.words.includes(word)) return e.why
  }
  return null
}

/** ASCII lower case only, as the box folds: other characters are kept, so UTF-8 words still match. */
function lower(s: string): string {
  return s.replace(/[A-Z]/g, (c) => c.toLowerCase())
}
