/**
 * The app's settings.json as text, and the language a call's speech to text hears. A file from an older build (no
 * `language`), an unknown value or an unreadable file loads with the defaults, so a restart or an update keeps what
 * was chosen and never breaks on what was not.
 */

/** Whisper large-v3's languages, code to name: the same list as the `/voice` page's `LANGUAGES` (src/voice/voice.html). */
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

/** A Whisper code or "auto" (detect it on each turn); anything else is English. */
export function sttLanguage(v: unknown): string {
  return v === 'auto' || (typeof v === 'string' && Object.hasOwn(LANGUAGES, v)) ? v : DEFAULT_LANGUAGE
}

/** The picker's choices: English, Auto-detect, then the others by name. */
export function languageChoices(): Array<{ code: string; name: string }> {
  const rest = Object.entries(LANGUAGES).filter(([c]) => c !== 'en').sort((a, b) => a[1].localeCompare(b[1], 'en'))
  return [{ code: 'en', name: 'English' }, { code: 'auto', name: 'Auto-detect' }, ...rest.map(([code, name]) => ({ code, name }))]
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
