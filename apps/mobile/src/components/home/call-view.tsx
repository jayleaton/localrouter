import { useMemo } from 'react'
import { StyleSheet, View } from 'react-native'

import { callLine, canInterrupt } from '@/components/home/voice-text'
import { VoiceWave } from '@/components/home/voice-wave'
import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'
import { useConversation } from '@/lib/app/box'
import type { AssistantMessage, Message, UserMessage } from '@/lib/localrouter/conversation'
import { loadVoiceEngine, useVoice } from '@/lib/voice/voiceState'

/** The latest thing the user said and the reply to it (still streaming, or finished), for the call view. */
function lastExchange(messages: Message[]) {
  let user: UserMessage | null = null
  let assistant: AssistantMessage | null = null
  for (let i = messages.length - 1; i >= 0 && !user; i--) {
    const m = messages[i]
    if (m.role === 'user') user = m
    else if (m.role === 'assistant' && !assistant) assistant = m
  }
  return { user, assistant }
}

/** Shown while a call is open or opening: the wave, what the call is doing, and its last lines. */
export function CallView() {
  const voice = useVoice()
  const messages = useConversation()
  const { user, assistant } = useMemo(() => lastExchange(messages), [messages])
  if (voice.status !== 'live' && voice.status !== 'connecting') return null

  const interruptible = canInterrupt(voice)
  return (
    <View style={styles.view}>
      <VoiceWave
        state={voice.status === 'live' && voice.activity !== 'thinking' ? 'live' : 'busy'}
        caption={callLine(voice)}
        onInterrupt={interruptible ? () => void loadVoiceEngine().then((engine) => engine?.interrupt()) : undefined}
      />
      {user?.text ? (
        <ThemedText type="small" numberOfLines={2} style={styles.user}>
          {user.text}
        </ThemedText>
      ) : null}
      {assistant?.text ? (
        <ThemedText type="small" themeColor="textSecondary" numberOfLines={3}>
          {assistant.text}
          {assistant.streaming ? ' …' : ''}
        </ThemedText>
      ) : null}
    </View>
  )
}

const styles = StyleSheet.create({
  view: { gap: Spacing.one, paddingHorizontal: Spacing.three, paddingTop: Spacing.two },
  user: { color: Palette.amber },
})
