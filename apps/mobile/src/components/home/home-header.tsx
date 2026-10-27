import { useRouter } from 'expo-router'
import { Pressable, StyleSheet, View } from 'react-native'

import { BrandMark } from '@/components/brand-mark'
import { GearIcon } from '@/components/icons'
import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'
import { reconnect, useAssistantName, useBox, type BoxState } from '@/lib/app/box'

/** The pill's words and colour for the connection; a paused voice model is the one non-connection state. */
function pill(connection: BoxState['connection'], paused: boolean) {
  if (connection === 'open') return paused ? { text: 'Voice paused', tone: Palette.warning } : { text: 'Connected', tone: Palette.green }
  if (connection === 'connecting') return { text: 'Connecting', tone: Palette.amber }
  if (connection === 'reconnecting') return { text: 'Reconnecting', tone: Palette.amber }
  if (connection === 'failed') return { text: 'Offline', tone: Palette.red }
  return { text: 'Not connected', tone: Palette.dim }
}

/** Home's header: the mark, the connection pill (tap to reconnect), and Settings. */
export function HomeHeader() {
  const router = useRouter()
  const connection = useBox((box) => box.connection)
  const paused = useBox((box) => box.paused)
  const name = useAssistantName()
  const { text, tone } = pill(connection, paused)
  return (
    <View style={styles.header}>
      <View style={styles.identity}>
        <BrandMark size={28} style={styles.mark} />
        <View style={styles.identityText}>
          <ThemedText type="smallBold" numberOfLines={1} style={styles.brand}>
            LocalRouter
          </ThemedText>
          <ThemedText type="small" themeColor="textSecondary" numberOfLines={1} style={styles.tagline}>
            {name} on your own box
          </ThemedText>
        </View>
      </View>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={`Box ${text.toLowerCase()}. Tap to reconnect`}
        onPress={reconnect}
        hitSlop={8}
        style={styles.pill}
        testID="connection-pill"
      >
        <View style={[styles.dot, { backgroundColor: tone }]} />
        <ThemedText type="small" numberOfLines={1} style={[styles.pillText, { color: tone }]}>
          {text}
        </ThemedText>
      </Pressable>
      <Pressable accessibilityRole="button" accessibilityLabel="Settings" onPress={() => router.push('/settings')} style={styles.settings} testID="open-settings">
        <GearIcon size={22} />
      </Pressable>
    </View>
  )
}

const styles = StyleSheet.create({
  header: { alignItems: 'center', flexDirection: 'row', gap: 6, paddingLeft: 20, paddingRight: Spacing.two, paddingVertical: Spacing.one },
  identity: { alignItems: 'center', flexDirection: 'row', flexShrink: 1, gap: 4 },
  identityText: { flexShrink: 1 },
  mark: { marginLeft: -4 },
  brand: { fontSize: 15, lineHeight: 20 },
  tagline: { fontSize: 11, lineHeight: 15 },
  pill: { alignItems: 'center', backgroundColor: Palette.surface, borderColor: Palette.border, borderRadius: 16, borderWidth: 1, flexDirection: 'row', gap: 6, marginLeft: 'auto', minHeight: 32, paddingHorizontal: 10 },
  dot: { borderRadius: 4, height: 8, width: 8 },
  pillText: { fontSize: 12, lineHeight: 16 },
  settings: { alignItems: 'center', height: 44, justifyContent: 'center', width: 44 },
})
