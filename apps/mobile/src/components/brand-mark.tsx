import { Image, type ImageSourcePropType, type ImageStyle, type StyleProp } from 'react-native'

const MARK: ImageSourcePropType = require('@/assets/images/sidebud-mark.png')

/** The app's mark, kept from SideBud (docs/brand/sidebud-mark-master.png at 144 px). */
export function BrandMark({ size = 34, style }: { size?: number; style?: StyleProp<ImageStyle> }) {
  return <Image source={MARK} style={[{ height: size, width: size }, style]} accessibilityIgnoresInvertColors accessibilityLabel="LocalRouter" />
}
