import { ActivityIndicator, Pressable, StyleSheet, Text, type PressableProps } from 'react-native'

import { Palette, Spacing } from '@/constants/theme'
import { useTheme } from '@/hooks/use-theme'

type AppButtonProps = PressableProps & {
  label: string
  loading?: boolean
  /** `danger` is for destructive, confirmed actions (for example deleting the account). */
  variant?: 'primary' | 'secondary' | 'danger'
}

export function AppButton({ label, loading = false, variant = 'primary', disabled, ...rest }: AppButtonProps) {
  const theme = useTheme()
  const isSecondary = variant === 'secondary'

  return (
    <Pressable
      accessibilityRole="button"
      disabled={disabled || loading}
      style={({ pressed }) => [
        styles.base,
        {
          backgroundColor: isSecondary ? theme.backgroundSelected : variant === 'danger' ? Palette.red : Palette.amber,
          opacity: pressed || (disabled ?? false) ? 0.7 : 1,
        },
      ]}
      {...rest}
    >
      {loading ? (
        <ActivityIndicator color={isSecondary ? theme.text : Palette.amberInk} />
      ) : (
        <Text style={[styles.label, { color: isSecondary ? theme.text : Palette.amberInk }]}>{label}</Text>
      )}
    </Pressable>
  )
}

const styles = StyleSheet.create({
  base: {
    alignItems: 'center',
    borderRadius: 16,
    justifyContent: 'center',
    // A fixed height with this padding left 16pt for 16pt text, clipping descenders.
    minHeight: 48,
    paddingHorizontal: Spacing.four,
    paddingVertical: Spacing.two,
  },
  label: {
    fontSize: 16,
    fontWeight: '600',
  },
})
