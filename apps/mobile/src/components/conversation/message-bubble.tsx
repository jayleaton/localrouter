import { StyleSheet, View } from 'react-native'

import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'
import { useTheme } from '@/hooks/use-theme'
import type { AssistantMessage, UserMessage } from '@/lib/localrouter/conversation'

/** The line under a bubble: still being heard or written, cut short, failed, or spoken. */
function meta(message: UserMessage | AssistantMessage) {
  if (message.role === 'user') {
    if (message.error) return { text: `Not understood: ${message.error}`, tone: Palette.red }
    if (message.partial) return { text: 'Hearing you…', tone: null }
    return message.via === 'voice' ? { text: 'Spoken', tone: null } : null
  }
  if (message.failed) return { text: 'The reply failed', tone: Palette.red }
  if (message.interrupted) return { text: 'Interrupted', tone: Palette.warning }
  if (message.streaming) return { text: 'Writing…', tone: null }
  return null
}

/** One conversation line: the user's on the right in amber, the assistant's on the left. */
export function MessageBubble({ message }: { message: UserMessage | AssistantMessage }) {
  const theme = useTheme()
  const mine = message.role === 'user'
  const line = meta(message)
  const faded = (message.role === 'user' && message.partial) || (message.role === 'assistant' && message.streaming && !message.text)
  return (
    <View style={[styles.row, mine ? styles.mine : styles.theirs]}>
      <View style={[styles.bubble, { backgroundColor: mine ? Palette.amber : theme.backgroundElement, opacity: faded ? 0.65 : 1 }]}>
        <ThemedText selectable style={{ color: mine ? Palette.amberInk : theme.text }}>
          {message.text || '…'}
        </ThemedText>
      </View>
      {line ? (
        <ThemedText type="small" themeColor="textSecondary" style={[styles.meta, line.tone ? { color: line.tone } : null]}>
          {line.text}
        </ThemedText>
      ) : null}
    </View>
  )
}

const styles = StyleSheet.create({
  row: { marginVertical: Spacing.one, maxWidth: '85%' },
  mine: { alignSelf: 'flex-end', alignItems: 'flex-end' },
  theirs: { alignSelf: 'flex-start', alignItems: 'flex-start' },
  bubble: { borderRadius: 17, paddingHorizontal: Spacing.three, paddingVertical: Spacing.two },
  meta: { marginTop: Spacing.half, fontSize: 12 },
})
