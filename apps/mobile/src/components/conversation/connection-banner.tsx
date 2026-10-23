import { Pressable, StyleSheet, View } from 'react-native'

import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'
import { reconnect, useBox } from '@/lib/app/box'

/**
 * Why the box is not answering right now. Connected (or connecting for the first time) shows nothing; a lost
 * connection can be retried with a tap. `paused`: a voice model made room for a picture or video job.
 */
export function ConnectionBanner() {
  const connection = useBox((box) => box.connection)
  const paused = useBox((box) => box.paused)

  if (connection === 'failed') {
    return (
      <Pressable accessibilityRole="button" accessibilityLabel="Can’t reach your box. Tap to retry" onPress={reconnect} style={[styles.banner, { backgroundColor: `${Palette.red}22` }]}>
        <ThemedText type="small" style={{ color: Palette.red }}>
          Can’t reach your box. Check that Tailscale is on and the box is running. Tap to retry.
        </ThemedText>
      </Pressable>
    )
  }
  if (connection === 'reconnecting') {
    return (
      <Pressable accessibilityRole="button" accessibilityLabel="Reconnecting to your box. Tap to retry now" onPress={reconnect} style={[styles.banner, { backgroundColor: `${Palette.dim}22` }]}>
        <ThemedText type="small" style={{ color: Palette.muted }}>
          Reconnecting to your box…
        </ThemedText>
      </Pressable>
    )
  }
  if (paused) {
    return (
      <View accessible accessibilityRole="text" style={[styles.banner, { backgroundColor: `${Palette.warning}22` }]}>
        <ThemedText type="small" style={{ color: Palette.warning }}>
          Voice is paused while the box makes a picture or video. It comes back by itself.
        </ThemedText>
      </View>
    )
  }
  return null
}

const styles = StyleSheet.create({
  banner: { justifyContent: 'center', minHeight: 44, paddingHorizontal: Spacing.three, paddingVertical: Spacing.two },
})
