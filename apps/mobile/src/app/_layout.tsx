import { useEffect, useRef } from 'react'
import { useLinkingURL } from 'expo-linking'
import { DarkTheme, Stack, ThemeProvider, useRouter } from 'expo-router'
import * as SplashScreen from 'expo-splash-screen'
import { StatusBar } from 'expo-status-bar'
import { StyleSheet } from 'react-native'
import { GestureHandlerRootView } from 'react-native-gesture-handler'

import { Palette } from '@/constants/theme'
import { initialize, useBox } from '@/lib/app/box'
import { parsePairLink } from '@/lib/localrouter/address'
import { startStandbyController } from '@/lib/voice/standbyController'

void SplashScreen.preventAutoHideAsync()

const theme = { ...DarkTheme, colors: { ...DarkTheme.colors, background: Palette.background, card: Palette.background, text: Palette.text, primary: Palette.amber, border: Palette.border } }

/**
 * Root layout: restores the pairing, settings and conversation before the splash screen goes, and starts the
 * controller that decides when this phone listens for the wake phrase.
 */
export default function RootLayout() {
  const status = useBox((box) => box.status)

  useEffect(() => {
    void initialize()
  }, [])

  useEffect(() => startStandbyController(), [])

  useEffect(() => {
    if (status !== 'loading') void SplashScreen.hideAsync()
  }, [status])

  if (status === 'loading') return null

  return (
    <GestureHandlerRootView style={styles.root}>
      <ThemeProvider value={theme}>
        <StatusBar style="light" />
        <Stack screenOptions={{ headerShown: false, contentStyle: { backgroundColor: Palette.background } }} />
        <PairLinks />
      </ThemeProvider>
    </GestureHandlerRootView>
  )
}

/**
 * `localrouter://pair?url=…&code=…` (printed by `localrouter pair add NAME --url URL`) opens the pair screen filled
 * in, also when a box is already paired. Rendered after the Stack, so its effect runs once the Stack is mounted.
 */
function PairLinks() {
  const url = useLinkingURL()
  const router = useRouter()
  const handled = useRef<string | null>(null)
  useEffect(() => {
    if (!url || handled.current === url) return
    handled.current = url
    const link = parsePairLink(url)
    if (link) router.replace({ pathname: '/pair', params: { url: link.baseUrl, code: link.code ?? '' } })
  }, [url, router])
  return null
}

const styles = StyleSheet.create({ root: { flex: 1 } })
