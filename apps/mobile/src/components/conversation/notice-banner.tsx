import { Pressable, StyleSheet, View } from 'react-native'

import { CloseIcon } from '@/components/icons'
import { ThemedText } from '@/components/themed-text'
import { Palette, Spacing } from '@/constants/theme'
import { dismissNotice, useBox } from '@/lib/app/box'

/** The box's latest message to the user (an error, a refused voice), until dismissed. */
export function NoticeBanner() {
  const notice = useBox((box) => box.notice)
  if (!notice) return null
  return (
    <View style={styles.banner}>
      <ThemedText type="small" selectable style={styles.text}>
        {notice}
      </ThemedText>
      <Pressable accessibilityRole="button" accessibilityLabel="Dismiss" onPress={dismissNotice} style={styles.dismiss} testID="notice-dismiss">
        <CloseIcon size={14} color={Palette.warning} />
      </Pressable>
    </View>
  )
}

const styles = StyleSheet.create({
  banner: { alignItems: 'center', backgroundColor: `${Palette.warning}22`, flexDirection: 'row', gap: Spacing.two, paddingLeft: Spacing.three },
  text: { color: Palette.warning, flex: 1, paddingVertical: Spacing.two },
  dismiss: { alignItems: 'center', height: 44, justifyContent: 'center', width: 44 },
})
