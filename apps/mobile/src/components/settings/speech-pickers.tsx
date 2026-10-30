import { Pressable, StyleSheet, View } from 'react-native'

import { Dropdown } from '@/components/dropdown'
import { ThemedText } from '@/components/themed-text'
import { Spacing } from '@/constants/theme'
import { updateSettings, useBox } from '@/lib/app/box'
import type { SpeechEngine } from '@/lib/localrouter/api'

/** The engine the session speaks with: the chosen one if the box still lists it, else the box's default. */
export function chosenEngine(engines: SpeechEngine[], speechModel: string) {
  return engines.find((e) => e.id === speechModel) ?? engines.find((e) => e.isDefault) ?? engines[0] ?? null
}

/** The speech engine (only when the box has more than one) and that engine's voices, exactly as the box lists them. */
export function SpeechPickers() {
  const engines = useBox((box) => box.engines)
  const settings = useBox((box) => box.settings)
  const engine = chosenEngine(engines, settings.speechModel)

  if (!engine) {
    return (
      <ThemedText type="small" themeColor="textSecondary">
        The box has not listed its voices yet. They appear once it is connected.
      </ThemedText>
    )
  }
  const voiceId = settings.voices[engine.id]
  const voice = engine.voices.find((v) => v.id === voiceId) ?? null
  const defaultLabel = `Default${engine.defaultVoice ? ` (${engine.voices.find((v) => v.id === engine.defaultVoice)?.name ?? engine.defaultVoice})` : ''}`

  const chooseVoice = (id: string | null) => {
    const voices = { ...settings.voices }
    if (id) voices[engine.id] = id
    else delete voices[engine.id]
    updateSettings({ voices })
  }

  return (
    <>
      {engines.length > 1 ? (
        <View style={styles.group}>
          <ThemedText type="small" themeColor="textSecondary">
            Speech engine
          </ThemedText>
          <Dropdown accessibilityLabel={`Speech engine, ${engine.name}`} testID="engine-dropdown" summary={<ThemedText>{engine.name}</ThemedText>}>
            {(close) =>
              engines.map((e) => (
                <Option
                  key={e.id}
                  label={e.isDefault ? `${e.name} (box default)` : e.name}
                  checked={e.id === engine.id}
                  onPress={() => {
                    close()
                    updateSettings({ speechModel: e.id })
                  }}
                />
              ))
            }
          </Dropdown>
        </View>
      ) : null}
      <View style={styles.group}>
        <ThemedText type="small" themeColor="textSecondary">
          Voice
        </ThemedText>
        <Dropdown accessibilityLabel={`Voice, ${voice?.name ?? defaultLabel}`} testID="voice-dropdown" summary={<ThemedText>{voice?.name ?? defaultLabel}</ThemedText>}>
          {(close) => (
            <>
              <Option
                label={defaultLabel}
                checked={!voice}
                onPress={() => {
                  close()
                  chooseVoice(null)
                }}
              />
              {engine.voices.map((v) => (
                <Option
                  key={v.id}
                  label={v.name}
                  detail={v.description}
                  checked={v.id === voice?.id}
                  onPress={() => {
                    close()
                    chooseVoice(v.id)
                  }}
                />
              ))}
            </>
          )}
        </Dropdown>
      </View>
    </>
  )
}

function Option({ label, detail, checked, onPress }: { label: string; detail?: string; checked: boolean; onPress: () => void }) {
  return (
    <Pressable accessibilityRole="radio" accessibilityLabel={label} accessibilityState={{ checked }} onPress={onPress} style={styles.option}>
      <ThemedText style={styles.radio}>{checked ? '◉' : '○'}</ThemedText>
      <View style={styles.optionText}>
        <ThemedText>{label}</ThemedText>
        {detail ? (
          <ThemedText type="small" themeColor="textSecondary">
            {detail}
          </ThemedText>
        ) : null}
      </View>
    </Pressable>
  )
}

const styles = StyleSheet.create({
  group: { gap: Spacing.one },
  option: { alignItems: 'center', flexDirection: 'row', gap: Spacing.two, minHeight: 44, paddingVertical: Spacing.one },
  optionText: { flex: 1 },
  radio: { fontSize: 18, width: 22 },
})
