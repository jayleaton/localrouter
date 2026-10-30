import { useEffect, useState } from 'react'
import { Animated, Easing, Pressable, StyleSheet, View } from 'react-native'

import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'

const BARS = 44
const HEIGHT = 64

/** Bar heights (0–1): tallest in the middle, tapering to the edges, with a fixed ripple. */
const SHAPE = Array.from({ length: BARS }, (_, index) => {
  const x = index / (BARS - 1)
  const envelope = Math.pow(Math.sin(Math.PI * x), 1.6)
  const ripple = 0.55 + 0.45 * Math.abs(Math.sin(index * 1.7) * Math.cos(index * 0.6))
  return Math.max(0.06, envelope * ripple)
})

type Props = {
  state: 'idle' | 'busy' | 'live'
  caption: string
  /** Set while a reply can be stopped: the wave becomes a button that interrupts it. */
  onInterrupt?: () => void
}

/**
 * The call's voice surface: a waveform that breathes while idle and moves while a call is live, captioned with what
 * the call is doing.
 */
export function VoiceWave({ state, caption, onInterrupt }: Props) {
  const [phase] = useState(() => new Animated.Value(0))
  const live = state === 'live'

  useEffect(() => {
    phase.setValue(0)
    const loop = Animated.loop(
      Animated.timing(phase, { toValue: 1, duration: live ? 900 : state === 'busy' ? 1400 : 3200, easing: Easing.linear, useNativeDriver: true })
    )
    loop.start()
    return () => loop.stop()
  }, [live, phase, state])

  // Energy per bar at four points of the cycle; neighbours are offset so the wave travels.
  const swing = live ? 0.55 : state === 'busy' ? 0.3 : 0.14
  const bars = (
    <View style={styles.wave}>
      {SHAPE.map((height, index) => {
        const steps = [0, 1, 2, 3].map((step) => 1 - swing + swing * Math.abs(Math.sin(index * 0.45 + step * 0.8)))
        const scaleY = phase.interpolate({ inputRange: [0, 0.25, 0.5, 0.75, 1], outputRange: [...steps, steps[0]] })
        const center = 1 - Math.abs(index / (BARS - 1) - 0.5) * 2
        return (
          <Animated.View
            key={index}
            style={[
              styles.bar,
              {
                height: Math.max(3, height * HEIGHT),
                backgroundColor: live ? Palette.amber : Palette.muted,
                opacity: 0.35 + 0.6 * center,
                transform: [{ scaleY }],
              },
            ]}
          />
        )
      })}
    </View>
  )
  return (
    <View style={styles.card}>
      {onInterrupt ? (
        <Pressable accessibilityRole="button" accessibilityLabel={`${caption}. Tap to interrupt`} onPress={onInterrupt} testID="voice-interrupt">
          {bars}
        </Pressable>
      ) : (
        <View accessible accessibilityRole="text" accessibilityLabel={caption}>
          {bars}
        </View>
      )}
      <ThemedText type="small" themeColor="textSecondary" numberOfLines={2} style={styles.caption}>
        {onInterrupt ? `${caption} · tap to interrupt` : caption}
      </ThemedText>
    </View>
  )
}

const styles = StyleSheet.create({
  card: { alignItems: 'center', backgroundColor: Palette.surface, borderColor: Palette.border, borderRadius: 24, borderWidth: 1, gap: Spacing.two, paddingHorizontal: Spacing.three, paddingBottom: Spacing.two, paddingTop: Spacing.three },
  wave: { alignItems: 'center', flexDirection: 'row', gap: 3, height: HEIGHT, justifyContent: 'center' },
  bar: { borderRadius: 2, width: 2.5 },
  caption: { textAlign: 'center' },
})
