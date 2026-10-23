import type { ReactNode } from 'react'
import { StyleSheet, Switch, View } from 'react-native'

import { ThemedText } from '@/components/themed-text'
import { ThemedView } from '@/components/themed-view'
import { Palette, Spacing } from '@/constants/theme'

/** One settings section: a titled card. */
export function SettingsCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <ThemedView type="backgroundElement" style={styles.card}>
      <ThemedText type="smallBold" accessibilityRole="header">
        {title}
      </ThemedText>
      {children}
    </ThemedView>
  )
}

type SwitchRowProps = {
  label: string
  value: boolean
  onChange: (value: boolean) => void
  disabled?: boolean
  testID?: string
  /** What the switch does, under it. */
  note?: string
}

export function SwitchRow({ label, value, onChange, disabled = false, testID, note }: SwitchRowProps) {
  return (
    <View style={styles.group}>
      <View style={styles.row}>
        <ThemedText style={[styles.label, disabled ? styles.disabled : null]}>{label}</ThemedText>
        <Switch
          accessibilityLabel={label}
          testID={testID}
          disabled={disabled}
          value={value}
          onValueChange={onChange}
          trackColor={{ true: Palette.amber, false: Palette.border }}
        />
      </View>
      {note ? (
        <ThemedText type="small" themeColor="textSecondary">
          {note}
        </ThemedText>
      ) : null}
    </View>
  )
}

const styles = StyleSheet.create({
  card: { borderColor: Palette.border, borderRadius: 20, borderWidth: 1, gap: Spacing.two, padding: Spacing.three },
  group: { gap: Spacing.one },
  row: { alignItems: 'center', flexDirection: 'row', gap: 12, justifyContent: 'space-between', minHeight: 44 },
  label: { flex: 1 },
  disabled: { opacity: 0.5 },
})
