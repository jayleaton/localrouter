import { StyleSheet, View } from 'react-native'

import { Palette } from '@/constants/theme'

/**
 * Line icons drawn with views, so the app needs no icon font or native
 * symbol module. Each fits a `size` square and strokes in `color`.
 */
type IconProps = { size?: number; color?: string }

export function MicIcon({ size = 22, color = Palette.text }: IconProps) {
  const stroke = Math.max(1.5, size / 11)
  return (
    <View style={{ width: size, height: size, alignItems: 'center' }}>
      <View style={{ width: size * 0.4, height: size * 0.6, borderRadius: size * 0.2, backgroundColor: color }} />
      <View
        style={{
          position: 'absolute',
          top: size * 0.28,
          width: size * 0.66,
          height: size * 0.5,
          borderColor: color,
          borderWidth: stroke,
          borderTopWidth: 0,
          borderBottomLeftRadius: size * 0.33,
          borderBottomRightRadius: size * 0.33,
        }}
      />
      <View style={{ position: 'absolute', bottom: 0, width: stroke, height: size * 0.22, backgroundColor: color }} />
    </View>
  )
}

/** A speaker; `off` strikes it through (the assistant's voice is muted). */
export function SpeakerIcon({ size = 22, color = Palette.text, off = false }: IconProps & { off?: boolean }) {
  const stroke = Math.max(1.5, size / 11)
  return (
    <View style={{ width: size, height: size, justifyContent: 'center' }}>
      <View style={{ position: 'absolute', left: size * 0.08, width: size * 0.22, height: size * 0.36, backgroundColor: color, borderRadius: 1 }} />
      <View
        style={{
          position: 'absolute', left: size * 0.22, width: 0, height: 0,
          borderTopWidth: size * 0.32, borderBottomWidth: size * 0.32, borderRightWidth: size * 0.3,
          borderTopColor: 'transparent', borderBottomColor: 'transparent', borderRightColor: color,
        }}
      />
      {off ? (
        <View style={{ position: 'absolute', left: size * 0.6, width: size * 0.34, height: stroke, backgroundColor: color, transform: [{ rotate: '45deg' }] }} />
      ) : (
        <View style={{ position: 'absolute', left: size * 0.58, width: size * 0.3, height: size * 0.5, borderColor: color, borderWidth: stroke, borderLeftWidth: 0, borderTopRightRadius: size * 0.3, borderBottomRightRadius: size * 0.3 }} />
      )}
    </View>
  )
}

/** A chevron pointing right (disclosure) or down (menu). */
export function ChevronIcon({ size = 12, color = Palette.muted, direction = 'right' }: IconProps & { direction?: 'right' | 'down' }) {
  const arm = size * 0.62
  return (
    <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
      <View
        style={{
          width: arm,
          height: arm,
          borderColor: color,
          borderRightWidth: 2,
          borderTopWidth: 2,
          transform: [{ translateX: direction === 'right' ? -arm * 0.2 : 0 }, { translateY: direction === 'down' ? -arm * 0.2 : 0 }, { rotate: direction === 'right' ? '45deg' : '135deg' }],
        }}
      />
    </View>
  )
}

export function GearIcon({ size = 22, color = Palette.muted }: IconProps) {
  const tooth = { width: size * 0.2, height: size, borderRadius: 2, backgroundColor: color }
  return (
    <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
      {[0, 45, 90, 135].map((angle) => (
        <View key={angle} style={[StyleSheet.absoluteFill, { alignItems: 'center', transform: [{ rotate: `${angle}deg` }] }]}>
          <View style={tooth} />
        </View>
      ))}
      <View style={{ width: size * 0.74, height: size * 0.74, borderRadius: size, backgroundColor: color, alignItems: 'center', justifyContent: 'center' }}>
        <View style={{ width: size * 0.32, height: size * 0.32, borderRadius: size, backgroundColor: Palette.background }} />
      </View>
    </View>
  )
}

/** A play triangle: a finished video (no poster frame is available, so its tile shows this). */
export function PlayIcon({ size = 22, color = Palette.text }: IconProps) {
  return (
    <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
      <View
        style={{
          width: 0, height: 0, marginLeft: size * 0.12,
          borderTopWidth: size * 0.36, borderBottomWidth: size * 0.36, borderLeftWidth: size * 0.62,
          borderTopColor: 'transparent', borderBottomColor: 'transparent', borderLeftColor: color,
        }}
      />
    </View>
  )
}

/** A cross, for closing a modal or dismissing a notice. */
export function CloseIcon({ size = 16, color = Palette.text }: IconProps) {
  const stroke = Math.max(1.5, size / 8)
  return (
    <View style={{ width: size, height: size, alignItems: 'center', justifyContent: 'center' }}>
      <View style={{ position: 'absolute', width: size, height: stroke, borderRadius: stroke, backgroundColor: color, transform: [{ rotate: '45deg' }] }} />
      <View style={{ position: 'absolute', width: size, height: stroke, borderRadius: stroke, backgroundColor: color, transform: [{ rotate: '-45deg' }] }} />
    </View>
  )
}
