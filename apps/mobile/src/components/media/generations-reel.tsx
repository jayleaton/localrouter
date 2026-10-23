import { useState } from 'react'
import { FlatList, Pressable, StyleSheet, View } from 'react-native'

import { ChevronIcon } from '@/components/icons'
import { GenerationCard } from '@/components/media/generation-card'
import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'
import type { Generation } from '@/lib/localrouter/conversation'

type Props = {
  /** Newest first. */
  generations: Generation[]
  onOpen: (id: string) => void
  onRetry: (id: string) => void
}

/** Every picture and video of the conversation in one strip above the dock, so none is lost up the chat. */
export function GenerationsReel({ generations, onOpen, onRetry }: Props) {
  const [open, setOpen] = useState(true)
  if (generations.length === 0) return null
  const working = generations.filter((g) => g.status === 'working').length
  const summary = `${generations.length} ${generations.length === 1 ? 'picture or video' : 'pictures and videos'}${working ? ` · ${working} in progress` : ''}`
  return (
    <View style={styles.reel}>
      <Pressable
        accessibilityRole="button"
        accessibilityLabel={`${summary}. ${open ? 'Tap to hide' : 'Tap to show'}`}
        accessibilityState={{ expanded: open }}
        onPress={() => setOpen((value) => !value)}
        style={styles.toggle}
        testID="reel-toggle"
      >
        <ThemedText type="smallBold" themeColor="textSecondary" style={styles.summary}>
          {summary}
        </ThemedText>
        <ChevronIcon direction={open ? 'down' : 'right'} size={10} />
      </Pressable>
      {open ? (
        <FlatList
          horizontal
          data={generations}
          keyExtractor={(g) => g.id}
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.strip}
          renderItem={({ item }) => <GenerationCard generation={item} size="small" onOpen={onOpen} onRetry={onRetry} />}
        />
      ) : null}
    </View>
  )
}

const styles = StyleSheet.create({
  reel: { borderColor: Palette.border, borderTopWidth: 1 },
  toggle: { alignItems: 'center', flexDirection: 'row', gap: Spacing.two, minHeight: 44, paddingHorizontal: Spacing.three },
  summary: { flex: 1 },
  strip: { gap: Spacing.two, paddingBottom: Spacing.two, paddingHorizontal: Spacing.three },
})
