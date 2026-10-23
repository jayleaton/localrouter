import { AppState, Platform, type AppStateStatus } from 'react-native'

import { getBox, subscribeBox, updateSettings, useBox } from '@/lib/app/box'
import type { Settings } from '@/lib/app/storage'
import { getVoiceState, loadVoiceEngine, subscribeVoice } from './voiceState'

/*
 * When this phone listens for the wake phrase: voice is turned on, a box is paired, and the app is on screen, or
 * background listening is allowed. Standby only starts while the app is on screen, because neither platform lets
 * recording start from the background. A live call carries on in the background; standby stops once it ends.
 */

let appState: AppStateStatus = AppState.currentState
/** A start that failed (permission denied) is not retried until the app returns or a setting changes. */
let blocked = false
let starting = false

export type VoiceSettings = Pick<Settings, 'enabled' | 'background'>

export const useVoiceSettings = () => useBox((box) => box.settings)

/** Changes a voice setting on this phone and applies it at once. */
export function updateVoiceSettings(patch: Partial<Settings>) {
  const before = getBox().settings
  updateSettings(patch)
  blocked = false
  // Android keeps background listening in a foreground service that only starts with standby; the recognizer
  // reads the wake-phrase setting when it starts. Either change restarts standby.
  const restart = (patch.background !== undefined && patch.background !== before.background && Platform.OS === 'android') || (patch.wakeWord !== undefined && patch.wakeWord !== before.wakeWord)
  if (restart && getVoiceState().status === 'standby') {
    void loadVoiceEngine().then(async (engine) => {
      await engine?.stopStandby()
      reconcile()
    })
    return
  }
  reconcile()
}

function wantsListening() {
  const box = getBox()
  return box.settings.enabled && box.status === 'paired' && (appState === 'active' || box.settings.background)
}

function reconcile() {
  const voice = getVoiceState()
  if (wantsListening()) {
    if (voice.status === 'off' && appState === 'active' && !blocked && !starting) {
      starting = true
      void loadVoiceEngine().then(async (engine) => {
        await engine?.startStandby({ background: getBox().settings.background })
        starting = false
        if (getVoiceState().status === 'off') blocked = true
      })
    }
  } else if (voice.status === 'standby') {
    void loadVoiceEngine().then((engine) => engine?.stopStandby())
  }
}

/** Starts deciding when to listen. Returns a function that stops it. */
export function startStandbyController() {
  appState = AppState.currentState
  blocked = false
  const stopVoice = subscribeVoice(reconcile)
  let paired = getBox().status
  let enabled = getBox().settings.enabled
  const stopBox = subscribeBox(() => {
    // Only the pairing and the voice switch decide listening.
    const box = getBox()
    if (box.status === paired && box.settings.enabled === enabled) return
    paired = box.status
    enabled = box.settings.enabled
    reconcile()
  })
  const stopAppState = AppState.addEventListener('change', (next) => {
    appState = next
    if (next === 'active') blocked = false
    reconcile()
  })
  reconcile()
  return () => {
    stopVoice()
    stopBox()
    stopAppState.remove()
  }
}
