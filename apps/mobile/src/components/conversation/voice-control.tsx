import { useRef } from 'react'
import { Alert, Pressable, StyleSheet, View } from 'react-native'

import { MicIcon, SpeakerIcon } from '@/components/icons'
import { ThemedText } from '@/components/themed-text'
import { Palette } from '@/constants/theme'
import { updateVoiceSettings, useVoiceSettings } from '@/lib/voice/standbyController'
import { loadVoiceEngine, useVoice } from '@/lib/voice/voiceState'

/**
 * Voice's main button: turns voice on the first time (the mic permission is asked then), opens a call from standby
 * like the wake phrase, or cancels a call that is still opening.
 */
export function TalkButton({ height = 64 }: { height?: number }) {
  const voice = useVoice()
  const settings = useVoiceSettings()

  const action =
    voice.status === 'connecting' ? { label: 'Cancel', a11y: 'Cancel the call', run: async () => (await loadVoiceEngine())?.endSession() }
    : voice.status === 'standby' ? { label: 'Talk', a11y: 'Talk to Jarvis', run: async () => (await loadVoiceEngine())?.wake() }
    : { label: settings.enabled && voice.status === 'off' && voice.error ? 'Try again' : 'Turn on voice', a11y: 'Turn on voice', run: async () => updateVoiceSettings({ enabled: true }) }
  const disabled = voice.status === 'live'
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={action.a11y}
      accessibilityHint="Hold for how calls work"
      disabled={disabled}
      testID="voice-talk"
      onPress={() => void action.run()}
      onLongPress={() =>
        Alert.alert(
          'Voice',
          'This phone listens for the wake phrase itself; nothing leaves the phone until you say it or tap Talk. In a call, pause and Jarvis answers. Say “mute audio” for quiet until the wake phrase, or “end call” to finish.'
        )
      }
      style={({ pressed }) => [styles.button, { height, backgroundColor: Palette.amber, opacity: disabled ? 0.45 : pressed ? 0.85 : 1 }]}
    >
      {voice.status === 'connecting' ? <ThemedText style={styles.busy}>…</ThemedText> : <MicIcon size={22} color={Palette.amberInk} />}
      <ThemedText type="smallBold" style={styles.label}>
        {action.label}
      </ThemedText>
    </Pressable>
  )
}

/** Ends the call; the phone goes back to listening for the wake phrase. */
export function EndCallButton({ size = 64 }: { size?: number }) {
  const voice = useVoice()
  const active = voice.status === 'live' || voice.status === 'connecting'
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel="End the call"
      disabled={!active}
      testID="voice-end"
      onPress={() => void loadVoiceEngine().then((engine) => engine?.endSession())}
      style={({ pressed }) => [styles.end, { width: size, height: size, opacity: pressed ? 0.8 : 1 }]}
    >
      <View style={styles.endMark} />
    </Pressable>
  )
}

const HOLD_MS = 350

/** The call's microphone: tap to mute or unmute; hold to talk while muted (the turn ends on release). */
export function MicButton({ height = 64 }: { height?: number }) {
  const call = useVoice()
  const press = useRef<{ at: number; wasOpen: boolean } | null>(null)
  if (call.status !== 'live') return null
  const open = call.micOpen

  const set = async (next: boolean) => (await loadVoiceEngine())?.setMicOpen(next)
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={open ? 'Microphone on. Tap to mute' : 'Muted. Tap to unmute or hold to talk'}
      accessibilityState={{ checked: open }}
      testID="voice-mic"
      onPressIn={() => {
        press.current = { at: Date.now(), wasOpen: open }
        if (!open) void set(true)
      }}
      onPressOut={() => {
        const started = press.current
        press.current = null
        if (!started) return
        const held = Date.now() - started.at >= HOLD_MS
        // Held while muted: hold-to-talk ends on release. A tap while open mutes; a tap while muted leaves it open.
        if ((!started.wasOpen && held) || (started.wasOpen && !held)) void set(false)
      }}
      style={({ pressed }) => [styles.button, styles.mic, { height, backgroundColor: open ? Palette.amber : Palette.surface, borderColor: open ? Palette.amber : Palette.border, opacity: pressed ? 0.9 : 1 }]}
    >
      <MicIcon size={22} color={open ? Palette.amberInk : Palette.text} />
      <View>
        <ThemedText type="smallBold" style={[styles.label, { color: open ? Palette.amberInk : Palette.text }]}>
          {open ? 'Mic on' : 'Muted'}
        </ThemedText>
        <ThemedText type="small" style={[styles.sublabel, { color: open ? Palette.amberInk : Palette.muted }]}>
          {open ? 'Tap to mute' : 'Tap or hold to talk'}
        </ThemedText>
      </View>
    </Pressable>
  )
}

/**
 * The assistant's voice on this phone: tap to silence it (replies still arrive as text) or bring it back. Separate
 * from the microphone, so the user can stay muted and still hear replies.
 */
export function SpeakerButton({ size = 64 }: { size?: number }) {
  const call = useVoice()
  if (call.status !== 'live') return null
  const muted = call.voiceMuted
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={muted ? 'Jarvis’s voice muted. Tap to hear it' : 'Jarvis’s voice on. Tap to mute it'}
      accessibilityState={{ checked: !muted }}
      testID="voice-speaker"
      onPress={() => void loadVoiceEngine().then((engine) => engine?.setVoiceMuted(!muted))}
      style={({ pressed }) => [styles.speaker, { width: size, height: size, borderColor: muted ? Palette.border : Palette.amber, opacity: pressed ? 0.9 : 1 }]}
    >
      <SpeakerIcon size={22} color={muted ? Palette.muted : Palette.text} off={muted} />
    </Pressable>
  )
}

const styles = StyleSheet.create({
  button: { alignItems: 'center', borderRadius: 22, flexDirection: 'row', gap: 12, justifyContent: 'center', width: '100%' },
  mic: { borderWidth: 1 },
  speaker: { alignItems: 'center', backgroundColor: Palette.surface, borderRadius: 22, borderWidth: 1, justifyContent: 'center' },
  busy: { color: Palette.amberInk, fontSize: 20, fontWeight: '700', lineHeight: 24 },
  label: { color: Palette.amberInk, fontSize: 17, lineHeight: 21 },
  sublabel: { fontSize: 12, lineHeight: 15 },
  end: { alignItems: 'center', backgroundColor: Palette.red, borderRadius: 22, justifyContent: 'center' },
  endMark: { backgroundColor: Palette.text, borderRadius: 3, height: 16, width: 16 },
})
