import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { heardLanguages, languageChoices, LANGUAGES, parseSettings, settingsText, sttLanguage } from './settings.ts'

const dflt = { base: '', name: 'Mac', shared: [] as string[], legacy: false, language: 'en' }

describe('settings.json and the language a call hears', () => {
  const dirs: string[] = []
  afterEach(() => {
    for (const d of dirs.splice(0)) rmSync(d, { recursive: true, force: true })
  })
  /** The app's load and save on a real file: each `load` is a new start of the app. */
  function file() {
    const d = mkdtempSync(join(tmpdir(), 'lr-settings-'))
    dirs.push(d)
    const path = join(d, 'settings.json')
    const load = () => {
      let text: string | null = null
      try {
        text = readFileSync(path, 'utf8')
      } catch {
        // no file yet
      }
      return parseSettings(text, dflt)
    }
    return { path, load, save: (s: object) => writeFileSync(path, settingsText(s)) }
  }

  it('a fresh install hears English', () => {
    expect(file().load().language).toBe('en')
    expect(parseSettings(null, dflt)).toEqual(dflt)
  })

  it('a settings file from an older build (no language) loads as English and keeps the rest', () => {
    const f = file()
    writeFileSync(f.path, JSON.stringify({ base: 'https://spark:10443', name: 'JJ', shared: ['/notes'], legacy: true, askEachAction: true }))
    expect(f.load()).toEqual({ base: 'https://spark:10443', name: 'JJ', shared: ['/notes'], legacy: true, askEachAction: true, language: 'en' })
  })

  it('the language survives a restart, auto included', () => {
    const f = file()
    f.save({ ...f.load(), language: 'de' })
    expect(f.load().language).toBe('de')
    f.save({ ...f.load(), language: 'auto' })
    expect(f.load().language).toBe('auto')
  })

  it('an unknown or broken value falls back to English; a broken file to the defaults', () => {
    expect(parseSettings(JSON.stringify({ language: 'klingon' }), dflt).language).toBe('en')
    expect(parseSettings(JSON.stringify({ language: 7 }), dflt).language).toBe('en')
    expect(parseSettings(JSON.stringify({ language: 'toString' }), dflt).language).toBe('en')
    expect(parseSettings('{not json', dflt)).toEqual(dflt)
    expect(parseSettings('[]', dflt)).toEqual(dflt)
    expect(sttLanguage('yue')).toBe('yue')
  })

  it('the choices: English, Auto-detect, then the others by name', () => {
    const c = languageChoices()
    expect(c.slice(0, 3)).toEqual([{ code: 'en', name: 'English' }, { code: 'auto', name: 'Auto-detect' }, { code: 'af', name: 'Afrikaans' }])
    expect(c.length).toBe(Object.keys(LANGUAGES).length + 1)
    const names = c.slice(2).map((l) => l.name)
    expect(names).toEqual([...names].sort((a, b) => a.localeCompare(b, 'en')))
  })

  it('the box’s declared languages narrow the choices; without them (or offline) the built-in list', () => {
    const models = [
      { id: 'other', kind: 'transcription', default_for: [], languages: ['fr'] },
      { id: 'ears', kind: 'transcription', default_for: ['speech_to_text'], languages: ['de', 'en', 'xx'] },
      { id: 'voice', kind: 'speech' },
    ]
    const heard = heardLanguages(models)
    expect(heard).toEqual(['de', 'en', 'xx'])
    expect(languageChoices(heard)).toEqual([{ code: 'en', name: 'English' }, { code: 'auto', name: 'Auto-detect' }, { code: 'de', name: 'German' }, { code: 'xx', name: 'xx' }])
    expect(sttLanguage('de', heard)).toBe('de')
    expect(sttLanguage('ja', heard)).toBe('en')
    expect(sttLanguage('ja', ['de'])).toBe('auto')
    expect(languageChoices(['de'])[0]).toEqual({ code: 'auto', name: 'Auto-detect' })
    expect(heardLanguages([{ id: 'ears', kind: 'transcription' }])).toBeNull()
    expect(heardLanguages('nope')).toBeNull()
    expect(languageChoices(null)).toEqual(languageChoices())
  })

  it('the app’s list is the voice page’s list', () => {
    const page = readFileSync(join(import.meta.dirname, '../../../../src/voice/voice.html'), 'utf8')
    const body = /const LANGUAGES = \{([^}]*)\};/.exec(page)?.[1] ?? ''
    const listed = Object.fromEntries([...body.matchAll(/(\w+): '([^']+)'/g)].map((m) => [m[1], m[2]]))
    expect(listed).toEqual(LANGUAGES)
  })
})
