/**
 * The app's settings.json as text, and the language a call's speech to text hears. A file from an older build (no
 * `language`), an unknown value or an unreadable file loads with the defaults, so a restart or an update keeps what
 * was chosen and never breaks on what was not.
 */

/**
 * Language names, code to name: the same list as the `/voice` page's `LANGUAGES` (src/voice/voice.html). The choices
 * are the box's transcription model's own `languages` (`heardLanguages`) when it lists them; these (Whisper large-v3's)
 * otherwise, and while the box has not been read.
 */
export const LANGUAGES: Readonly<Record<string, string>> = {
  af: 'Afrikaans', sq: 'Albanian', am: 'Amharic', ar: 'Arabic', hy: 'Armenian', as: 'Assamese', az: 'Azerbaijani',
  ba: 'Bashkir', eu: 'Basque', be: 'Belarusian', bn: 'Bengali', bs: 'Bosnian', br: 'Breton', bg: 'Bulgarian',
  my: 'Burmese', yue: 'Cantonese', ca: 'Catalan', zh: 'Chinese', hr: 'Croatian', cs: 'Czech', da: 'Danish',
  nl: 'Dutch', en: 'English', et: 'Estonian', fo: 'Faroese', fi: 'Finnish', fr: 'French', gl: 'Galician',
  ka: 'Georgian', de: 'German', el: 'Greek', gu: 'Gujarati', ht: 'Haitian Creole', ha: 'Hausa', haw: 'Hawaiian',
  he: 'Hebrew', hi: 'Hindi', hu: 'Hungarian', is: 'Icelandic', id: 'Indonesian', it: 'Italian', ja: 'Japanese',
  jw: 'Javanese', kn: 'Kannada', kk: 'Kazakh', km: 'Khmer', ko: 'Korean', lo: 'Lao', la: 'Latin', lv: 'Latvian',
  ln: 'Lingala', lt: 'Lithuanian', lb: 'Luxembourgish', mk: 'Macedonian', mg: 'Malagasy', ms: 'Malay',
  ml: 'Malayalam', mt: 'Maltese', mi: 'Maori', mr: 'Marathi', mn: 'Mongolian', ne: 'Nepali', no: 'Norwegian',
  nn: 'Norwegian Nynorsk', oc: 'Occitan', ps: 'Pashto', fa: 'Persian', pl: 'Polish', pt: 'Portuguese',
  pa: 'Punjabi', ro: 'Romanian', ru: 'Russian', sa: 'Sanskrit', sr: 'Serbian', sn: 'Shona', sd: 'Sindhi',
  si: 'Sinhala', sk: 'Slovak', sl: 'Slovenian', so: 'Somali', es: 'Spanish', su: 'Sundanese', sw: 'Swahili',
  sv: 'Swedish', tl: 'Tagalog', tg: 'Tajik', ta: 'Tamil', tt: 'Tatar', te: 'Telugu', th: 'Thai', bo: 'Tibetan',
  tr: 'Turkish', tk: 'Turkmen', uk: 'Ukrainian', ur: 'Urdu', uz: 'Uzbek', vi: 'Vietnamese', cy: 'Welsh',
  yi: 'Yiddish', yo: 'Yoruba',
}

export const DEFAULT_LANGUAGE = 'en'

/** The languages the box hears: null for none listed (any of `LANGUAGES`). */
export type Heard = readonly string[] | null

/**
 * The transcription model's declared `languages` from `/v1/models`' `data` (the one whose default_for lists
 * speech_to_text, else the first transcription model), or null when it lists none.
 */
export function heardLanguages(models: unknown): Heard {
  if (!Array.isArray(models)) return null
  const stt = models.filter((m): m is Record<string, unknown> => !!m && typeof m === 'object' && (m as { kind?: unknown }).kind === 'transcription')
  const m = stt.find((x) => Array.isArray(x.default_for) && x.default_for.includes('speech_to_text')) ?? stt[0]
  const list = m?.languages
  return Array.isArray(list) ? list.filter((c): c is string => typeof c === 'string' && c.length > 0) : null
}

/** A code the box hears, or "auto" (detect it on each turn); anything else is English, or "auto" if it hears none. */
export function sttLanguage(v: unknown, heard: Heard = null): string {
  if (v === 'auto' || (typeof v === 'string' && (heard ? heard.includes(v) : Object.hasOwn(LANGUAGES, v)))) return v as string
  return !heard || heard.includes(DEFAULT_LANGUAGE) ? DEFAULT_LANGUAGE : 'auto'
}

/** The picker's choices: English (when heard), Auto-detect, then the others by name. */
export function languageChoices(heard: Heard = null): Array<{ code: string; name: string }> {
  const all = heard ? heard.map((c): [string, string] => [c, LANGUAGES[c] ?? c]) : Object.entries(LANGUAGES)
  const rest = all.filter(([c]) => c !== 'en').sort((a, b) => a[1].localeCompare(b[1], 'en'))
  const en = !heard || heard.includes('en') ? [{ code: 'en', name: 'English' }] : []
  return [...en, { code: 'auto', name: 'Auto-detect' }, ...rest.map(([code, name]) => ({ code, name }))]
}

/** settings.json's text (null: no file yet) over the defaults. */
export function parseSettings<T extends { language: string }>(text: string | null, dflt: T): T {
  let saved: unknown = null
  try {
    saved = text == null ? null : JSON.parse(text)
  } catch {
    return dflt
  }
  if (!saved || typeof saved !== 'object' || Array.isArray(saved)) return dflt
  return { ...dflt, ...(saved as Partial<T>), language: sttLanguage((saved as { language?: unknown }).language) }
}

export function settingsText(s: object): string {
  return `${JSON.stringify(s, null, 1)}\n`
}
