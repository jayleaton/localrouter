import { useState } from 'react'
import { useLocalSearchParams, useRouter } from 'expo-router'
import { KeyboardAvoidingView, Platform, ScrollView, StyleSheet, View } from 'react-native'

import { AppButton } from '@/components/app-button'
import { AppTextInput } from '@/components/app-text-input'
import { Screen } from '@/components/screen'
import { ThemedText } from '@/components/themed-text'
import { ThemedView } from '@/components/themed-view'
import { Palette, Spacing } from '@/constants/theme'
import { pair, useBox } from '@/lib/app/box'

const DEFAULT_DEVICE_NAME = Platform.OS === 'ios' ? 'iPhone' : 'Android phone'

/** A route param as one string (a repeated query key arrives as an array). */
const param = (value: string | string[] | undefined) => (Array.isArray(value) ? value[0] : value) ?? ''

/**
 * Pairs this phone with a box: its address and the code `localrouter pair add` prints there. A pairing link
 * (`localrouter://pair?url=…&code=…`) opens this screen filled in.
 */
export default function PairScreen() {
  const params = useLocalSearchParams<{ url?: string; code?: string }>()
  const url = param(params.url)
  const code = param(params.code)
  // A new link while the screen is open starts the form again with its values.
  return <PairForm key={`${url}\n${code}`} initialUrl={url} initialCode={code} />
}

function PairForm({ initialUrl, initialCode }: { initialUrl: string; initialCode: string }) {
  const router = useRouter()
  const paired = useBox((box) => box.status === 'paired')
  const [address, setAddress] = useState(initialUrl)
  const [code, setCode] = useState(initialCode)
  const [deviceName, setDeviceName] = useState(DEFAULT_DEVICE_NAME)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const submit = async () => {
    setBusy(true)
    setError(null)
    try {
      // An empty code is passed on: the box itself says whether it requires pairing.
      await pair(address, code, deviceName.trim() || DEFAULT_DEVICE_NAME)
      router.replace('/home')
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Pairing failed. Try again.')
      setBusy(false)
    }
  }

  return (
    <Screen>
      <KeyboardAvoidingView style={styles.flex} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled">
          <ThemedText type="title">Pair your box</ThemedText>
          <ThemedText themeColor="textSecondary">
            On the box, run <ThemedText type="code">localrouter pair add &quot;My phone&quot; --url &lt;address&gt;</ThemedText> (or{' '}
            <ThemedText type="code">deploy/jarvis/pair.sh add …</ThemedText>) and enter the code it prints. This phone must be on the same tailnet, with the Tailscale app on.
          </ThemedText>

          <ThemedView type="backgroundElement" style={styles.card}>
            <View style={styles.field}>
              <AppTextInput
                label="Box address"
                value={address}
                onChangeText={setAddress}
                placeholder="https://box.your-tailnet.ts.net:10443"
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="url"
                textContentType="URL"
                testID="pair-address"
              />
            </View>
            <View style={styles.field}>
              <AppTextInput
                label="Pairing code"
                value={code}
                onChangeText={setCode}
                placeholder="XXXX-XXXX"
                autoCapitalize="characters"
                autoCorrect={false}
                maxLength={9}
                testID="pair-code"
              />
            </View>
            <View style={styles.field}>
              <AppTextInput label="This phone’s name" value={deviceName} onChangeText={setDeviceName} autoCapitalize="words" testID="pair-name" />
            </View>
            {error ? (
              <ThemedText type="small" style={styles.error} selectable>
                {error}
              </ThemedText>
            ) : null}
            <AppButton label="Pair" accessibilityLabel="Pair with the box" onPress={() => void submit()} loading={busy} disabled={!address.trim()} testID="pair-submit" />
          </ThemedView>

          {paired ? <AppButton label="Back to my box" variant="secondary" accessibilityLabel="Back to my box" onPress={() => router.replace('/home')} disabled={busy} /> : null}
        </ScrollView>
      </KeyboardAvoidingView>
    </Screen>
  )
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: { gap: Spacing.three, padding: Spacing.four },
  card: { borderColor: Palette.border, borderRadius: 20, borderWidth: 1, gap: Spacing.three, padding: Spacing.three },
  field: { gap: Spacing.half },
  error: { color: Palette.red },
})
