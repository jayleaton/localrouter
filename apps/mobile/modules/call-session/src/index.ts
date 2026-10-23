import { Platform } from 'react-native'
import { requireOptionalNativeModule } from 'expo'

type Subscription = { remove(): void }

type Events = {
  /** The system activated the call's audio session; start capture and playback now. */
  onAudioActivated: () => void
  onAudioDeactivated: () => void
  /** Mute changed from the system call UI, CarPlay, Android Auto, or a headset. */
  onMuteChanged: (event: { muted: boolean }) => void
  /** The user ended the call from the system UI, car, or headset. */
  onEndRequested: () => void
  /** iOS: the phone joined or left a CarPlay car. */
  onCarChanged: (event: { connected: boolean }) => void
  /** iOS 17+: the AirPods or headset mute gesture, outside a system call. A toggle: the app keeps its own mute state. */
  onHeadsetMuteGesture: () => void
}

type NativeCallSession = {
  startCall(displayName: string): Promise<string>
  reportConnected(): Promise<void>
  setMuted(muted: boolean): Promise<void>
  endCall(): Promise<void>
  isActive(): boolean
  /** iOS: whether the phone is in a CarPlay car. Missing on builds before it was added and on Android. */
  isCarConnected?(): boolean
  /** Android: keeps local voice standby listening with the screen off (a microphone foreground service). Start only while the app is on screen. Missing on iOS and older builds. */
  startStandby?(): Promise<void>
  stopStandby?(): Promise<void>
  addListener<K extends keyof Events>(event: K, listener: Events[K]): Subscription
}

/**
 * The voice session as a system phone call (CallKit / Android Core-Telecom).
 * Null in Expo Go or builds without the module; the app then runs voice
 * without system call integration.
 */
export const CallSession = Platform.OS === 'web' ? null : requireOptionalNativeModule<NativeCallSession>('CallSession')
