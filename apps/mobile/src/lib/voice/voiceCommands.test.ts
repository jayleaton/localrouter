import { DEFAULT_VOICE_COMMAND_PHRASES, lastCommand, voiceCommandPhrases } from './voiceCommands'
import { expect, it } from 'vitest'

it('recognizes the default phrases as whole words, latest wins', () => {
  const phrases = DEFAULT_VOICE_COMMAND_PHRASES
  expect(lastCommand('Mute audio', phrases)).toBe('mute')
  expect(lastCommand('okay mute the audio please', phrases)).toBe('mute')
  expect(lastCommand('Hey, Jarvis.', phrases)).toBe('unmute')
  expect(lastCommand('mute audio and then hey jarvis', phrases)).toBe('unmute')
  expect(lastCommand('hey jarvis mute audio', phrases)).toBe('mute')
  expect(lastCommand('the audio is muted', phrases)).toBeNull()
  expect(lastCommand('commute audio', phrases)).toBeNull()
  expect(lastCommand('jarvis', phrases)).toBeNull()
})

it('uses the phrases set on the computer, including ones that contain each other', () => {
  const phrases = { mute: ['mute audio'], muteMic: [], muteVoice: [], unmute: ['unmute audio', "let's talk"], endCall: [] }
  expect(lastCommand('Unmute audio', phrases)).toBe('unmute')
  expect(lastCommand('ok mute audio', phrases)).toBe('mute')
  expect(lastCommand("Let's talk", phrases)).toBe('unmute')
  expect(lastCommand('hey jarvis', phrases)).toBeNull()
})

it('tells muting the mic apart from muting the voice', () => {
  const phrases = DEFAULT_VOICE_COMMAND_PHRASES
  expect(lastCommand('okay mute my mic', phrases)).toBe('muteMic')
  expect(lastCommand('Mute microphone.', phrases)).toBe('muteMic')
  expect(lastCommand('mute voice please', phrases)).toBe('muteVoice')
  expect(lastCommand('mute speaker', phrases)).toBe('muteVoice')
  expect(lastCommand('mute audio', phrases)).toBe('mute')
  expect(lastCommand('the microphone is fine', phrases)).toBeNull()
  expect(lastCommand('okay thanks, end call', phrases)).toBe('endCall')
  expect(lastCommand('when does the call end', phrases)).toBeNull()
})

it('wakes with the name the assistant was given, and only with a greeting', () => {
  const phrases = voiceCommandPhrases('Friday')
  expect(phrases.unmute).toEqual(['hey Friday'])
  expect(lastCommand('Hey, Friday!', phrases)).toBe('unmute')
  expect(lastCommand('okay   FRIDAY', phrases)).toBe('unmute')
  expect(lastCommand('hi friday', phrases)).toBe('unmute')
  expect(lastCommand('hey friday mute audio', phrases)).toBe('mute')
  expect(lastCommand('mute audio then hey friday', phrases)).toBe('unmute')
  expect(lastCommand('hey jarvis', phrases)).toBeNull()
  expect(lastCommand('friday', phrases)).toBeNull()
  expect(lastCommand('hey', phrases)).toBeNull()
  expect(lastCommand('see you friday', phrases)).toBeNull()
  expect(lastCommand('hey there friday', phrases)).toBeNull()
})

it('hears a name a letter off when it is four letters or longer', () => {
  const friday = voiceCommandPhrases('Friday')
  expect(lastCommand('hey fryday', friday)).toBe('unmute')
  expect(lastCommand('hay friday', friday)).toBe('unmute')
  expect(lastCommand('hey fridays', friday)).toBe('unmute')
  expect(lastCommand('hey frida', friday)).toBe('unmute')
  expect(lastCommand('hey fry day', friday)).toBe('unmute')
  expect(lastCommand('hey freddy', friday)).toBeNull()
  expect(lastCommand('hey friend', friday)).toBeNull()
  // Jarvis too, the default.
  expect(lastCommand('hey jarvi', DEFAULT_VOICE_COMMAND_PHRASES)).toBe('unmute')
  expect(lastCommand('hey travis', DEFAULT_VOICE_COMMAND_PHRASES)).toBeNull()
  // A short name must be heard exactly.
  const ava = voiceCommandPhrases('Ava')
  expect(lastCommand('hey ava', ava)).toBe('unmute')
  expect(lastCommand('hey eva', ava)).toBeNull()
  expect(lastCommand('hey a', ava)).toBeNull()
})

it('wakes with a name of several words, however the recognizer splits them', () => {
  const phrases = voiceCommandPhrases('Mary-Jane Watson')
  expect(lastCommand('Hey Mary Jane Watson.', phrases)).toBe('unmute')
  expect(lastCommand('hey maryjane watson', phrases)).toBe('unmute')
  expect(lastCommand('hey mary jane watsen', phrases)).toBe('unmute')
  expect(lastCommand('hey mary jane', phrases)).toBeNull()
  expect(lastCommand('hey mary', phrases)).toBeNull()
  const accented = voiceCommandPhrases('Zoë')
  expect(lastCommand('Hey Zoe', accented)).toBe('unmute')
  expect(lastCommand('hey zoë', accented)).toBe('unmute')
  const apostrophe = voiceCommandPhrases("O'Brien")
  expect(lastCommand('hey obrien', apostrophe)).toBe('unmute')
  expect(lastCommand("hey o'brien", apostrophe)).toBe('unmute')
})
