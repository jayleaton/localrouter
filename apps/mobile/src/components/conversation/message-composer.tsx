import { useState } from 'react'
import { Pressable, StyleSheet, TextInput, View } from 'react-native'

import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'
import { useTheme } from '@/hooks/use-theme'
import { useAssistantName } from '@/lib/app/box'

/** Longer messages are cut at the input: a phone message this long is a paste gone wrong, not a question. */
const MAX_TEXT_LENGTH = 4000

/** The typed message field. In a call a typed message gets a spoken reply, like a spoken one. */
export function MessageComposer({ onSend }: { onSend: (text: string) => void }) {
  const theme = useTheme()
  const name = useAssistantName()
  const [text, setText] = useState('')
  const ready = text.trim().length > 0
  const send = () => {
    if (!ready) return
    onSend(text)
    setText('')
  }
  return (
    <View style={styles.row}>
      <TextInput
        value={text}
        onChangeText={setText}
        maxLength={MAX_TEXT_LENGTH}
        placeholder={`Message ${name}…`}
        placeholderTextColor={theme.textSecondary}
        accessibilityLabel="Message"
        testID="message-input"
        returnKeyType="send"
        submitBehavior="submit"
        onSubmitEditing={send}
        style={[styles.input, { color: theme.text, backgroundColor: theme.backgroundElement }]}
      />
      <Pressable
        accessibilityRole="button"
        accessibilityLabel="Send message"
        accessibilityState={{ disabled: !ready }}
        onPress={send}
        disabled={!ready}
        style={[styles.send, { opacity: ready ? 1 : 0.4 }]}
      >
        <ThemedText style={styles.arrow}>↑</ThemedText>
      </Pressable>
    </View>
  )
}

const styles = StyleSheet.create({
  row: { alignItems: 'center', flexDirection: 'row', gap: Spacing.two, paddingHorizontal: Spacing.three, paddingTop: Spacing.two },
  input: { borderRadius: 22, flex: 1, fontSize: 16, height: 44, paddingHorizontal: Spacing.three },
  send: { alignItems: 'center', backgroundColor: Palette.amber, borderRadius: 22, height: 44, justifyContent: 'center', width: 44 },
  arrow: { color: Palette.amberInk, fontSize: 24 },
})
