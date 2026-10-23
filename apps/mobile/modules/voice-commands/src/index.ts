import { Platform } from 'react-native'
import { requireOptionalNativeModule } from 'expo'

type Subscription = { remove(): void }

export type SpeechPermission = 'granted' | 'denied' | 'restricted' | 'notDetermined'

type Events = {
  /** The on-device transcript of the current listening window, updated as the user speaks. */
  onTranscript: (event: { text: string; isFinal: boolean }) => void
  /** Recognition failed; the listening window is renewed after a pause. For diagnostics. */
  onError: (event: { message: string; domain: string; code: number }) => void
}

type NativeVoiceCommands = {
  requestPermission(): Promise<SpeechPermission>
  /**
   * Starts listening to microphone audio passed in with `append`. Resolves
   * false when on-device recognition is unavailable: audio is never sent to
   * a speech server.
   */
  start(sampleRate: number, phrases: string[]): Promise<boolean>
  /** Base64 PCM16 mono at the started sample rate. */
  append(pcm16Base64: string): void
  /** Clears the transcript (after a command) so it cannot fire twice. */
  reset(): void
  /**
   * Nothing to hear for now (standby's energy gate closed): the current
   * listening window finishes the audio it has, and no new one starts until
   * the next `append`.
   */
  idle(): void
  stop(): void
  addListener<K extends keyof Events>(event: K, listener: Events[K]): Subscription
}

/**
 * Spoken app commands, recognized on the phone from the call's own
 * (echo-cancelled) microphone stream. It keeps listening while the mic is
 * muted, and that audio never leaves the phone. iOS, and Android 13 or newer
 * (`start` resolves false on older Android); null on web.
 */
export const VoiceCommands = Platform.OS === 'web' ? null : requireOptionalNativeModule<NativeVoiceCommands>('VoiceCommands')
