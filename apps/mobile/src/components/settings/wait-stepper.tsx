import { Pressable, StyleSheet, View } from 'react-native'

import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'
import { updateSettings, useBox } from '@/lib/app/box'
import { clampWait, WAIT_MAX_MS, WAIT_MIN_MS } from '@/lib/localrouter/protocol'

const STEP_MS = 100

/** How long the user can pause before Jarvis answers, in 0.1 s steps (the app has no slider module). */
export function WaitStepper() {
  const waitMs = useBox((box) => box.settings.waitMs)
  // Steps land on tenths even if a saved value does not.
  const step = (direction: -1 | 1) => updateSettings({ waitMs: clampWait(Math.round(waitMs / STEP_MS) * STEP_MS + direction * STEP_MS) })
  const seconds = `${(waitMs / 1000).toFixed(1)} s`
  return (
    <View style={styles.group}>
      <View style={styles.row}>
        <ThemedText style={styles.label}>Wait before replying</ThemedText>
        <StepButton label="−" a11y="Shorter wait" disabled={waitMs <= WAIT_MIN_MS} onPress={() => step(-1)} />
        <ThemedText type="smallBold" style={styles.value} accessibilityLiveRegion="polite" accessibilityLabel={`Wait ${seconds}`}>
          {seconds}
        </ThemedText>
        <StepButton label="+" a11y="Longer wait" disabled={waitMs >= WAIT_MAX_MS} onPress={() => step(1)} />
      </View>
      <ThemedText type="small" themeColor="textSecondary">
        How long you can pause before Jarvis answers; talking again restarts the wait.
      </ThemedText>
    </View>
  )
}

function StepButton({ label, a11y, disabled, onPress }: { label: string; a11y: string; disabled: boolean; onPress: () => void }) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={a11y}
      accessibilityState={{ disabled }}
      disabled={disabled}
      onPress={onPress}
      style={({ pressed }) => [styles.step, { opacity: disabled ? 0.4 : pressed ? 0.7 : 1 }]}
    >
      <ThemedText type="subtitle" style={styles.stepLabel}>
        {label}
      </ThemedText>
    </Pressable>
  )
}

const styles = StyleSheet.create({
  group: { gap: Spacing.one },
  row: { alignItems: 'center', flexDirection: 'row', gap: Spacing.two },
  label: { flex: 1 },
  value: { minWidth: 48, textAlign: 'center' },
  step: { alignItems: 'center', backgroundColor: Palette.raised, borderRadius: 22, height: 44, justifyContent: 'center', width: 44 },
  stepLabel: { color: Palette.amber },
})
