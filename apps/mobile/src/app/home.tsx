import { useCallback, useMemo, useState } from 'react'
import { Redirect } from 'expo-router'
import { KeyboardAvoidingView, Platform, StyleSheet } from 'react-native'

import { ConnectionBanner } from '@/components/conversation/connection-banner'
import { ConversationList } from '@/components/conversation/conversation-list'
import { MessageComposer } from '@/components/conversation/message-composer'
import { NoticeBanner } from '@/components/conversation/notice-banner'
import { CallView } from '@/components/home/call-view'
import { HomeHeader } from '@/components/home/home-header'
import { VoiceDock } from '@/components/home/voice-dock'
import { Lightbox } from '@/components/lightbox'
import { GenerationsReel } from '@/components/media/generations-reel'
import { Screen } from '@/components/screen'
import { reconnect, retryGeneration, sendText, useBox, useConversation } from '@/lib/app/box'
import { isGeneration } from '@/lib/localrouter/conversation'

/**
 * Home is the whole app: the conversation (typed and spoken turns, pictures and videos inline), the generations reel,
 * the composer, and the voice dock with the live call above it.
 */
export default function HomeScreen() {
  const status = useBox((box) => box.status)
  const messages = useConversation()
  // The reel is the same items as the chat's cards, newest first (Conversation.generations()).
  const generations = useMemo(() => messages.filter(isGeneration).reverse(), [messages])
  const [openId, setOpenId] = useState<string | null>(null)
  // Looked up on each change, so an open video that is retried or replaced shows its latest state.
  const open = useMemo(() => generations.find((g) => g.id === openId && g.status === 'completed') ?? null, [generations, openId])

  const retry = useCallback((id: string) => void retryGeneration(id), [])

  if (status !== 'paired') return <Redirect href="/pair" />

  return (
    <Screen>
      <HomeHeader />
      <ConnectionBanner />
      <NoticeBanner />
      <KeyboardAvoidingView style={styles.flex} behavior={Platform.OS === 'ios' ? 'padding' : undefined}>
        <ConversationList messages={messages} onSend={sendText} onOpen={setOpenId} onRetry={retry} onRefresh={reconnect} />
        <CallView />
        <GenerationsReel generations={generations} onOpen={setOpenId} onRetry={retry} />
        <MessageComposer onSend={sendText} />
        <VoiceDock />
      </KeyboardAvoidingView>
      <Lightbox generation={open} onClose={() => setOpenId(null)} />
    </Screen>
  )
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
})
