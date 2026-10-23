import { useState, type ReactNode } from 'react'
import { Pressable, StyleSheet, View } from 'react-native'

import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'

type DropdownProps = {
  /** The collapsed row: usually the current choice or a count. */
  summary: ReactNode
  /** Receives `close`, so choosing an option can collapse the list. */
  children: (close: () => void) => ReactNode
  accessibilityLabel: string
  testID?: string
}

/** A row that expands in place to show its options, so long lists stay collapsed by default. */
export function Dropdown({ summary, children, accessibilityLabel, testID }: DropdownProps) {
  const [open, setOpen] = useState(false)
  return (
    <View style={styles.container}>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={accessibilityLabel}
        accessibilityState={{ expanded: open }}
        onPress={() => setOpen((value) => !value)}
        style={styles.trigger}
        testID={testID}
      >
        <View style={styles.summary}>{summary}</View>
        <ThemedText themeColor="textSecondary">{open ? '▴' : '▾'}</ThemedText>
      </Pressable>
      {open ? <View style={styles.options}>{children(() => setOpen(false))}</View> : null}
    </View>
  )
}

const styles = StyleSheet.create({
  container: { borderColor: Palette.border, borderRadius: 14, borderWidth: 1 },
  trigger: { alignItems: 'center', flexDirection: 'row', gap: Spacing.two, paddingHorizontal: Spacing.three, paddingVertical: Spacing.two },
  summary: { flex: 1 },
  options: { borderColor: Palette.border, borderTopWidth: 1, paddingHorizontal: Spacing.three, paddingVertical: Spacing.one },
})
