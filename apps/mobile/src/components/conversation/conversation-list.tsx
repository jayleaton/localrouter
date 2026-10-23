import { useCallback, useRef } from 'react'
import { FlatList, Pressable, RefreshControl, StyleSheet, View, type ListRenderItemInfo } from 'react-native'

import { MessageBubble } from '@/components/conversation/message-bubble'
import { GenerationCard } from '@/components/media/generation-card'
import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'
import { isGeneration, type Message } from '@/lib/localrouter/conversation'

const SUGGESTIONS = ['Make a picture of a fox in the snow', 'What can you do?', 'Make a short video of waves on a beach']

type Props = {
  messages: Message[]
  onSend: (text: string) => void
  onOpen: (id: string) => void
  onRetry: (id: string) => void
  /** Pull to retry: reconnects to the box (the lost-connection notice asks for it). */
  onRefresh: () => void
}

/**
 * The conversation, oldest first, following the newest line as it streams. Not inverted, so pulling down at the top
 * is the usual pull to refresh.
 */
export function ConversationList({ messages, onSend, onOpen, onRetry, onRefresh }: Props) {
  const list = useRef<FlatList<Message>>(null)
  const renderItem = useCallback(
    ({ item }: ListRenderItemInfo<Message>) => (isGeneration(item) ? <GenerationCard generation={item} onOpen={onOpen} onRetry={onRetry} /> : <MessageBubble message={item} />),
    [onOpen, onRetry]
  )

  return (
    <FlatList
      ref={list}
      style={styles.flex}
      contentContainerStyle={[styles.content, messages.length === 0 ? styles.emptyContent : null]}
      data={messages}
      keyExtractor={(m) => m.id}
      renderItem={renderItem}
      keyboardShouldPersistTaps="handled"
      onContentSizeChange={() => list.current?.scrollToEnd({ animated: true })}
      refreshControl={<RefreshControl refreshing={false} onRefresh={onRefresh} tintColor={Palette.muted} />}
      ListEmptyComponent={<Suggestions onSend={onSend} />}
    />
  )
}

function Suggestions({ onSend }: { onSend: (text: string) => void }) {
  return (
    <View style={styles.empty}>
      <ThemedText themeColor="textSecondary" style={styles.center}>
        Talk or type to Jarvis. Ask a question, or ask for a picture or a video.
      </ThemedText>
      {SUGGESTIONS.map((text) => (
        <Pressable
          key={text}
          accessibilityRole="button"
          accessibilityLabel={`Send: ${text}`}
          onPress={() => onSend(text)}
          style={({ pressed }) => [styles.suggestion, pressed && styles.pressed]}
        >
          <ThemedText type="small">{text}</ThemedText>
        </Pressable>
      ))}
    </View>
  )
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: { paddingHorizontal: Spacing.three, paddingVertical: Spacing.two },
  emptyContent: { flexGrow: 1, justifyContent: 'center' },
  empty: { alignItems: 'center', gap: Spacing.two, padding: Spacing.three },
  center: { marginBottom: Spacing.two, textAlign: 'center' },
  suggestion: { backgroundColor: Palette.surface, borderColor: Palette.border, borderRadius: 18, borderWidth: 1, justifyContent: 'center', minHeight: 44, paddingHorizontal: Spacing.three, paddingVertical: Spacing.two },
  pressed: { backgroundColor: Palette.raised },
})
