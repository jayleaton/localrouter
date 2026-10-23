/**
 * Sidebud's palette (docs/brand.md): charcoal ground, ivory text, amber
 * accent. Surfaces step up from the charcoal; status tones are warm-tuned
 * to sit beside the amber. One dark palette keeps native navigation consistent.
 */

import { Platform } from 'react-native';

export const Palette = {
  /** Brand charcoal. */
  background: '#151718',
  surface: '#1f2223',
  raised: '#292d2e',
  border: '#353a3b',
  /** Brand ivory. */
  text: '#fef3e1',
  muted: '#aba496',
  /** Brand amber. */
  amber: '#feb93e',
  /** Text on amber: brand charcoal. */
  amberInk: '#151718',
  /** Statuses: done, attention, failure, and quiet/neutral. */
  green: '#7fd0a0',
  warning: '#ffa84a',
  red: '#ff7a68',
  dim: '#8a857a',
} as const;

export const Colors = {
  light: {
    text: Palette.text,
    background: Palette.background,
    backgroundElement: Palette.surface,
    backgroundSelected: Palette.raised,
    textSecondary: Palette.muted,
  },
  dark: {
    text: Palette.text,
    background: Palette.background,
    backgroundElement: Palette.surface,
    backgroundSelected: Palette.raised,
    textSecondary: Palette.muted,
  },
} as const;

export type ThemeColor = keyof typeof Colors.light & keyof typeof Colors.dark;

export const Fonts = Platform.select({
  ios: {
    /** iOS `UIFontDescriptorSystemDesignDefault` */
    sans: 'system-ui',
    /** iOS `UIFontDescriptorSystemDesignSerif` */
    serif: 'ui-serif',
    /** iOS `UIFontDescriptorSystemDesignRounded` */
    rounded: 'ui-rounded',
    /** iOS `UIFontDescriptorSystemDesignMonospaced` */
    mono: 'ui-monospace',
  },
  default: {
    sans: 'normal',
    serif: 'serif',
    rounded: 'normal',
    mono: 'monospace',
  },
});

export const Spacing = {
  half: 2,
  one: 4,
  two: 8,
  three: 16,
  four: 24,
  five: 32,
  six: 64,
} as const;
