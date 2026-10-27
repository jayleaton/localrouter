import { useState } from 'react'
import { StyleSheet, View } from 'react-native'

import { AppButton } from '@/components/app-button'
import { AppTextInput } from '@/components/app-text-input'
import { ThemedText } from '@/components/themed-text'
import { Spacing } from '@/constants/theme'
import { renameAssistant, useBox } from '@/lib/app/box'
import { ASSISTANT_NAME_MAX, assistantNameError, normalizeAssistantName } from '@/lib/localrouter/assistant'

/**
 * The assistant's name, kept on the box: every app and the voice loop use it. Saving needs the `manage` permission;
 * an older box has no such setting and keeps the default. Under it, the box's default voice as a hint.
 */
export function AssistantNameField() {
  const assistant = useBox((box) => box.assistant)
  const engines = useBox((box) => box.engines)
  const [draft, setDraft] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const [busy, setBusy] = useState(false)

  const value = draft ?? assistant.name
  const changed = normalizeAssistantName(value) !== assistant.name

  const save = async () => {
    // An empty name restores the box's default.
    const invalid = normalizeAssistantName(value) ? assistantNameError(value) : null
    if (invalid) return setError(invalid)
    setBusy(true)
    setError(null)
    try {
      await renameAssistant(value)
      setDraft(null)
      setSaved(true)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Could not save the name. Try again.')
    } finally {
      setBusy(false)
    }
  }

  const engine = assistant.voice ? engines.find((e) => e.id === assistant.voice?.model) : undefined
  const voiceName = assistant.voice ? (engine?.voices.find((v) => v.id === assistant.voice?.voice)?.name ?? assistant.voice.voice) : null

  return (
    <View style={styles.group}>
      <AppTextInput
        label="Assistant name"
        value={value}
        onChangeText={(text) => {
          setDraft(text)
          setError(null)
          setSaved(false)
        }}
        editable={assistant.supported && !busy}
        maxLength={ASSISTANT_NAME_MAX + 8}
        autoCapitalize="words"
        autoCorrect={false}
        returnKeyType="done"
        onSubmitEditing={() => {
          if (changed) void save()
        }}
        error={error}
        testID="assistant-name"
      />
      {assistant.supported ? (
        <AppButton label="Save name" variant="secondary" accessibilityLabel="Save the assistant’s name" onPress={() => void save()} loading={busy} disabled={!changed} testID="assistant-name-save" />
      ) : null}
      <ThemedText type="small" themeColor="textSecondary">
        {assistant.supported
          ? saved
            ? `Saved. Say “Hey ${assistant.name}” to start a call.`
            : `Kept on the box: every app and the wake phrase use it.${assistant.custom ? ` Clear it and save to go back to “${assistant.defaultName}”.` : ''}`
          : 'This box is too old to rename the assistant. Update LocalRouter on it to choose a name.'}
      </ThemedText>
      {assistant.voice ? (
        <ThemedText type="small" themeColor="textSecondary" testID="assistant-voice">
          Speaks as {voiceName} ({engine?.name ?? assistant.voice.model}) by default on the box.
        </ThemedText>
      ) : null}
    </View>
  )
}

const styles = StyleSheet.create({
  group: { gap: Spacing.one },
})
