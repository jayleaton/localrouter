import { useEffect, useState, type ReactNode } from 'react'
import { ActivityIndicator, Animated, Easing, Image, Pressable, StyleSheet, View } from 'react-native'

import { useFileSource } from '@/components/media/file-source'
import { PlayIcon } from '@/components/icons'
import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'
import type { Generation } from '@/lib/localrouter/conversation'

type Props = {
  generation: Generation
  /** `large` in the conversation, `small` in the generations reel. */
  size?: 'large' | 'small'
  onOpen: (id: string) => void
  onRetry: (id: string) => void
}

const SIDE = { large: 220, small: 96 } as const

export const workingLabel = (kind: Generation['kind']) => (kind === 'video' ? 'Making a video…' : 'Creating a picture…')

/** A picture or video: working, done (tap to open), or failed with its reason and Retry. */
export function GenerationCard({ generation, size = 'large', onOpen, onRetry }: Props) {
  const side = SIDE[size]
  const small = size === 'small'
  const what = generation.kind === 'video' ? 'video' : 'picture'
  return (
    <View style={[styles.card, { width: side }, small ? null : styles.inline]}>
      {generation.status === 'working' ? (
        <Shimmer side={side}>
          <ActivityIndicator color={Palette.amber} />
          <ThemedText type="small" themeColor="textSecondary" numberOfLines={1} style={styles.center}>
            {small ? (generation.kind === 'video' ? 'Video…' : 'Picture…') : workingLabel(generation.kind)}
          </ThemedText>
        </Shimmer>
      ) : generation.status === 'completed' ? (
        <Pressable
          accessibilityRole="imagebutton"
          accessibilityLabel={`Open the ${what}${generation.prompt ? `: ${generation.prompt}` : ''}`}
          onPress={() => onOpen(generation.id)}
          style={({ pressed }) => [styles.tile, { height: side, opacity: pressed ? 0.85 : 1 }]}
          testID={`generation-${generation.id}`}
        >
          {generation.kind === 'video' ? (
            <View style={styles.play}>
              <PlayIcon size={small ? 26 : 40} color={Palette.amber} />
            </View>
          ) : (
            <Thumbnail uri={generation.urls[0]} />
          )}
        </Pressable>
      ) : (
        <View style={[styles.tile, styles.failed, { height: side }]}>
          <ThemedText type="small" numberOfLines={small ? 2 : 4} style={[styles.center, styles.error]}>
            {small ? 'Failed' : `The ${what} failed: ${generation.error ?? 'unknown reason'}`}
          </ThemedText>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel={`Retry the ${what}`}
            onPress={() => onRetry(generation.id)}
            style={({ pressed }) => [styles.retry, { opacity: pressed ? 0.8 : 1 }]}
            testID={`retry-${generation.id}`}
          >
            <ThemedText type="smallBold" style={styles.retryLabel}>
              Retry
            </ThemedText>
          </Pressable>
        </View>
      )}
      {generation.prompt && !small ? (
        <ThemedText type="small" themeColor="textSecondary" numberOfLines={2}>
          {generation.prompt}
        </ThemedText>
      ) : null}
    </View>
  )
}

function Thumbnail({ uri }: { uri: string | undefined }) {
  const source = useFileSource(uri)
  if (!source) return null
  return <Image source={source} style={StyleSheet.absoluteFill} resizeMode="cover" accessibilityIgnoresInvertColors />
}

/** The working tile: its background pulses while the box makes the picture or video. */
function Shimmer({ side, children }: { side: number; children: ReactNode }) {
  const [pulse] = useState(() => new Animated.Value(0))
  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(pulse, { toValue: 1, duration: 900, easing: Easing.inOut(Easing.quad), useNativeDriver: true }),
        Animated.timing(pulse, { toValue: 0, duration: 900, easing: Easing.inOut(Easing.quad), useNativeDriver: true }),
      ])
    )
    loop.start()
    return () => loop.stop()
  }, [pulse])
  return (
    <View style={[styles.tile, { height: side }]} accessible accessibilityRole="progressbar" accessibilityLabel="Working">
      <Animated.View style={[StyleSheet.absoluteFill, styles.glow, { opacity: pulse.interpolate({ inputRange: [0, 1], outputRange: [0.15, 0.5] }) }]} />
      {children}
    </View>
  )
}

const styles = StyleSheet.create({
  card: { gap: Spacing.one },
  inline: { alignSelf: 'flex-start', marginVertical: Spacing.one },
  tile: { alignItems: 'center', backgroundColor: Palette.surface, borderColor: Palette.border, borderRadius: 16, borderWidth: 1, gap: Spacing.two, justifyContent: 'center', overflow: 'hidden', padding: Spacing.two },
  glow: { backgroundColor: Palette.raised },
  play: { alignItems: 'center', backgroundColor: Palette.raised, borderRadius: 999, height: '50%', justifyContent: 'center', aspectRatio: 1 },
  failed: { borderColor: `${Palette.red}66` },
  center: { textAlign: 'center' },
  error: { color: Palette.red },
  retry: { alignItems: 'center', backgroundColor: Palette.raised, borderRadius: 12, justifyContent: 'center', minHeight: 44, minWidth: 64, paddingHorizontal: Spacing.three },
  retryLabel: { color: Palette.amber },
})
