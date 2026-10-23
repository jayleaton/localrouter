import { useEffect } from 'react'
import { StyleSheet, View } from 'react-native'

import { EndCallButton, MicButton, SpeakerButton, TalkButton } from '@/components/conversation/voice-control'
import { voiceHint } from '@/components/home/voice-text'
import { ThemedText } from '@/components/themed-text'
import { Spacing } from '@/constants/theme'
import { useBox } from '@/lib/app/box'
import { useVoiceSettings } from '@/lib/voice/standbyController'
import { loadVoiceEngine, useVoice } from '@/lib/voice/voiceState'

/**
 * The bottom of home: Talk outside a call; End, Mic and Speaker in one. Outside a call a hint says what voice is
 * doing (inside one the call view above says it).
 */
export function VoiceDock() {
  const call = useVoice()
  const { enabled } = useVoiceSettings()
  useEndWhenFailed()

  return (
    <View style={styles.dock}>
      {call.status === 'live' ? (
        <View style={styles.actions}>
          <EndCallButton />
          <View style={styles.action}>
            <MicButton />
          </View>
          <SpeakerButton />
        </View>
      ) : (
        <>
          {call.status === 'connecting' ? null : (
            <ThemedText type="small" themeColor="textSecondary" style={styles.hint}>
              {voiceHint(call, enabled)}
            </ThemedText>
          )}
          <TalkButton />
        </>
      )}
    </View>
  )
}

/**
 * A connection that failed for good (not a reconnect in progress) ends the call on the phone: the box has dropped
 * it, and standby carries on.
 */
function useEndWhenFailed() {
  const failed = useBox((box) => box.connection === 'failed')
  const voice = useVoice()
  const active = voice.status === 'live'
  useEffect(() => {
    if (failed && active) void loadVoiceEngine().then((engine) => engine?.endSession())
  }, [failed, active])
}

const styles = StyleSheet.create({
  dock: { gap: Spacing.two, paddingHorizontal: Spacing.three, paddingTop: Spacing.two, paddingBottom: Spacing.two },
  actions: { flexDirection: 'row', alignItems: 'center', gap: 12 },
  action: { flex: 1 },
  hint: { textAlign: 'center' },
})
