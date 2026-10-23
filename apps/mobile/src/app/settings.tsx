import { useState } from 'react'
import { useRouter } from 'expo-router'
import { Alert, Pressable, ScrollView, StyleSheet, View } from 'react-native'

import { AppButton } from '@/components/app-button'
import { ChevronIcon } from '@/components/icons'
import { Screen } from '@/components/screen'
import { SettingsCard, SwitchRow } from '@/components/settings/settings-card'
import { SpeechPickers } from '@/components/settings/speech-pickers'
import { WaitStepper } from '@/components/settings/wait-stepper'
import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'
import { forget, newConversation, reconnect, useBox, type BoxState } from '@/lib/app/box'
import { updateVoiceSettings, useVoiceSettings } from '@/lib/voice/standbyController'

const CONNECTION_TEXT: Record<BoxState['connection'], string> = {
  connecting: 'Connecting…',
  open: 'Connected',
  reconnecting: 'Reconnecting…',
  failed: 'Can’t reach the box',
  closed: 'Not connected',
}

/** The paired box, voice on this phone, the conversation, and about. */
export default function SettingsScreen() {
  const router = useRouter()
  const pairing = useBox((box) => box.pairing)
  const connection = useBox((box) => box.connection)
  const paused = useBox((box) => box.paused)
  const voice = useVoiceSettings()
  const [leaving, setLeaving] = useState(false)

  const back = () => (router.canGoBack() ? router.back() : router.replace('/home'))

  const confirmForget = () =>
    Alert.alert('Forget this box?', 'This phone deletes its pairing. You can pair again with a new code.', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Forget',
        style: 'destructive',
        onPress: async () => {
          setLeaving(true)
          try {
            await forget()
            router.replace('/pair')
          } catch (error) {
            Alert.alert('Could not forget the box', error instanceof Error ? error.message : 'Try again.')
          } finally {
            setLeaving(false)
          }
        },
      },
    ])

  const confirmNew = () =>
    Alert.alert('Start a new conversation?', 'The messages, pictures and videos on this phone are cleared.', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'New conversation', style: 'destructive', onPress: newConversation },
    ])

  return (
    <Screen>
      <ScrollView contentContainerStyle={styles.content}>
        <Pressable accessibilityRole="button" accessibilityLabel="Back to home" onPress={back} style={styles.back}>
          <View style={styles.flip}>
            <ChevronIcon color={Palette.amber} size={14} />
          </View>
          <ThemedText type="linkPrimary">Home</ThemedText>
        </Pressable>
        <ThemedText type="title" accessibilityRole="header">
          Settings
        </ThemedText>

        <SettingsCard title="Box">
          {pairing ? (
            <>
              <ThemedText selectable>{pairing.baseUrl}</ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                {pairing.token ? `This phone: ${pairing.deviceName || 'paired'}` : 'Not paired: this box does not require pairing yet.'}
              </ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                {CONNECTION_TEXT[connection]}
                {paused ? ' · voice paused for a job' : ''}
              </ThemedText>
              <AppButton label="Reconnect" variant="secondary" accessibilityLabel="Reconnect to the box" onPress={reconnect} />
              <AppButton label="Forget this box" variant="danger" accessibilityLabel="Forget this box" onPress={confirmForget} loading={leaving} />
              <ThemedText type="small" themeColor="textSecondary">
                Forgetting deletes the pairing on this phone. The box’s operator revokes the device with{' '}
                <ThemedText type="code">localrouter pair revoke</ThemedText>.
              </ThemedText>
            </>
          ) : (
            <AppButton label="Pair a box" accessibilityLabel="Pair a box" onPress={() => router.replace('/pair')} />
          )}
        </SettingsCard>

        <SettingsCard title="Voice">
          <SwitchRow
            label="Voice on"
            value={voice.enabled}
            onChange={(enabled) => updateVoiceSettings({ enabled })}
            testID="voice-enabled"
            note="This phone listens for the wake phrase itself; nothing leaves the phone until you say it or tap Talk."
          />
          <SwitchRow label="Wake phrase “Hey Jarvis”" value={voice.wakeWord} onChange={(wakeWord) => updateVoiceSettings({ wakeWord })} disabled={!voice.enabled} testID="voice-wake-word" note="Off: only Talk opens a call." />
          <SwitchRow
            label="Keep listening in the background"
            value={voice.background}
            onChange={(background) => updateVoiceSettings({ background })}
            disabled={!voice.enabled}
            testID="voice-background"
            note="Off: listening pauses when you leave the app. On: the wake phrase works with the app in the background and the phone locked, at a battery cost still being measured."
          />
          <WaitStepper />
          <SpeechPickers />
        </SettingsCard>

        <SettingsCard title="Conversation">
          <ThemedText type="small" themeColor="textSecondary">
            The conversation is kept on this phone; the box keeps it only while a session is open.
          </ThemedText>
          <AppButton label="New conversation" variant="secondary" accessibilityLabel="Start a new conversation" onPress={confirmNew} />
        </SettingsCard>

        <SettingsCard title="About">
          <ThemedText>LocalRouter mobile</ThemedText>
          <ThemedText type="small" themeColor="textSecondary">
            Chat, voice, pictures and videos with Jarvis on your own LocalRouter box. The app’s design is adapted from SideBud.
          </ThemedText>
          <ThemedText type="small" themeColor="textSecondary" selectable>
            github.com/jayleaton/localrouter
          </ThemedText>
        </SettingsCard>
      </ScrollView>
    </Screen>
  )
}

const styles = StyleSheet.create({
  content: { gap: Spacing.three, padding: Spacing.four },
  back: { alignItems: 'center', alignSelf: 'flex-start', flexDirection: 'row', gap: 2, minHeight: 44 },
  flip: { transform: [{ scaleX: -1 }] },
})
