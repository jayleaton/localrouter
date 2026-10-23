import { useVideoPlayer, VideoView } from 'expo-video'
import { Image, Modal, Pressable, ScrollView, StyleSheet, View } from 'react-native'
import { useSafeAreaInsets } from 'react-native-safe-area-context'

import { callLine, canInterrupt } from '@/components/home/voice-text'
import { CloseIcon } from '@/components/icons'
import { useFileSource } from '@/components/media/file-source'
import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'
import type { Generation } from '@/lib/localrouter/conversation'
import { loadVoiceEngine, useVoice } from '@/lib/voice/voiceState'

type Props = {
  /** The generation to show; null hides the lightbox. */
  generation: Generation | null
  onClose: () => void
}

/**
 * A finished picture full size, or a video playing, over everything. A call carries on behind it; the pill at the
 * bottom shows it and can interrupt or end it. The prompt is selectable text (the app has no clipboard module).
 */
export function Lightbox({ generation, onClose }: Props) {
  const insets = useSafeAreaInsets()
  const url = generation?.urls[0]
  return (
    <Modal visible={generation !== null} animationType="fade" presentationStyle="fullScreen" onRequestClose={onClose} statusBarTranslucent>
      <View style={[styles.root, { paddingTop: insets.top, paddingBottom: insets.bottom + Spacing.two }]}>
        <View style={styles.bar}>
          <Pressable accessibilityRole="button" accessibilityLabel="Close" onPress={onClose} style={styles.close} testID="lightbox-close">
            <CloseIcon size={18} />
          </Pressable>
        </View>
        <View style={styles.media}>
          {generation && url ? generation.kind === 'video' ? <VideoPane uri={url} /> : <ImagePane uri={url} prompt={generation.prompt} /> : null}
        </View>
        {generation?.prompt ? (
          <ScrollView style={styles.prompt} contentContainerStyle={styles.promptContent}>
            <ThemedText type="small" themeColor="textSecondary" selectable>
              {generation.prompt}
            </ThemedText>
          </ScrollView>
        ) : null}
        <CallPill />
      </View>
    </Modal>
  )
}

function ImagePane({ uri, prompt }: { uri: string; prompt: string }) {
  const source = useFileSource(uri)
  if (!source) return null
  return <Image source={source} style={styles.fill} resizeMode="contain" accessibilityIgnoresInvertColors accessibilityLabel={prompt || 'Picture'} />
}

function VideoPane({ uri }: { uri: string }) {
  const source = useFileSource(uri)
  const player = useVideoPlayer(source, (p) => {
    p.loop = true
    p.play()
  })
  return <VideoView player={player} style={styles.fill} nativeControls contentFit="contain" accessibilityLabel="Video" />
}

/** The call behind the lightbox: what it is doing, Interrupt while a reply can be stopped, and End. */
function CallPill() {
  const voice = useVoice()
  if (voice.status !== 'live' && voice.status !== 'connecting') return null
  return (
    <View style={styles.pill}>
      <View style={[styles.dot, { backgroundColor: voice.status === 'live' ? Palette.amber : Palette.muted }]} />
      <ThemedText type="smallBold" style={styles.pillText} numberOfLines={1}>
        {callLine(voice)}
      </ThemedText>
      {canInterrupt(voice) ? (
        <Pressable accessibilityRole="button" accessibilityLabel="Interrupt the reply" onPress={() => void loadVoiceEngine().then((engine) => engine?.interrupt())} style={styles.pillButton}>
          <ThemedText type="smallBold" style={styles.interrupt}>
            Interrupt
          </ThemedText>
        </Pressable>
      ) : null}
      <Pressable accessibilityRole="button" accessibilityLabel="End the call" onPress={() => void loadVoiceEngine().then((engine) => engine?.endSession())} style={[styles.pillButton, styles.end]}>
        <ThemedText type="smallBold" style={styles.endLabel}>
          End
        </ThemedText>
      </Pressable>
    </View>
  )
}

const styles = StyleSheet.create({
  root: { backgroundColor: '#000', flex: 1, gap: Spacing.two },
  bar: { alignItems: 'flex-end', paddingHorizontal: Spacing.two },
  close: { alignItems: 'center', backgroundColor: Palette.surface, borderRadius: 22, height: 44, justifyContent: 'center', width: 44 },
  media: { flex: 1 },
  fill: { flex: 1, width: '100%' },
  prompt: { flexGrow: 0, maxHeight: 96 },
  promptContent: { paddingHorizontal: Spacing.three },
  pill: { alignItems: 'center', alignSelf: 'center', backgroundColor: Palette.surface, borderColor: Palette.border, borderRadius: 28, borderWidth: 1, flexDirection: 'row', gap: Spacing.two, maxWidth: '92%', paddingLeft: Spacing.three, paddingRight: Spacing.one, paddingVertical: Spacing.one },
  dot: { borderRadius: 5, height: 10, width: 10 },
  pillText: { flexShrink: 1 },
  pillButton: { alignItems: 'center', borderRadius: 22, justifyContent: 'center', minHeight: 44, paddingHorizontal: Spacing.three },
  interrupt: { color: Palette.amber },
  end: { backgroundColor: Palette.red },
  endLabel: { color: Palette.text },
})
