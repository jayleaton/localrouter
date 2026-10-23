import { defineConfig, globalIgnores } from 'eslint/config'
import expoConfig from 'eslint-config-expo/flat.js'

export default defineConfig([
  ...expoConfig,
  globalIgnores(['.expo/**', 'dist/**', 'web-build/**', 'expo-env.d.ts', 'android/**', 'ios/**']),
  {
    // The protocol layer runs under Node too (e2e/): it must not import React Native.
    files: ['src/lib/localrouter/**/*.ts', 'e2e/**/*.ts'],
    rules: {
      'no-restricted-imports': ['error', { patterns: [{ group: ['react-native', 'react-native/*', 'expo', 'expo-*', '@/lib/app/*', '@/lib/voice/*'], message: 'The protocol layer stays free of React Native so the headless check can run it.' }] }],
    },
  },
])
