import { DEFAULT_VOICE_COMMAND_PHRASES, lastCommand } from './voiceCommands'
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
