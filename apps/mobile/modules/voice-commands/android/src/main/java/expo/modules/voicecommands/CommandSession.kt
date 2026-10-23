package expo.modules.voicecommands

import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.speech.RecognitionListener
import android.speech.RecognitionSupport
import android.speech.RecognitionSupportCallback
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.system.ErrnoException
import android.system.Os
import android.system.OsConstants
import android.util.Base64
import androidx.annotation.RequiresApi

/**
 * One call's command listening: an on-device recognizer fed the call's PCM16
 * audio through a pipe (`EXTRA_AUDIO_SOURCE`), in listening windows renewed
 * as each ends. Everything except `append` runs on the main thread, where
 * SpeechRecognizer must be used.
 */
@RequiresApi(Build.VERSION_CODES.TIRAMISU)
internal class CommandSession(
  private val context: Context,
  private val sampleRate: Int,
  private val phrases: ArrayList<String>,
  private val onTranscript: (text: String, isFinal: Boolean) -> Unit,
  private val onError: (message: String, code: Int) -> Unit,
) {
  private companion object {
    const val LANGUAGE = "en-US"
    /** Recognition sessions end on their own after long speech or silence; each window is renewed. */
    const val WINDOW_MS = 50_000L
    /** A window that ended on silence or no match is renewed almost at once. */
    const val RENEW_MS = 100L
    /** Failed windows back off from here, doubling up to the cap, so a failing recognizer does not spin. */
    const val RETRY_MS = 500L
    const val MAX_RETRY_MS = 10_000L
    /** Pipe writes up to PIPE_BUF are atomic: a chunk is written whole or dropped, never split mid-sample. */
    const val PIPE_ATOMIC_BYTES = 4_096
    /** Audio kept while a window opens or its pipe is full: the gate's pre-roll and about 4 s more. */
    const val MAX_BACKLOG_CHUNKS = 32
    /** While audio is still waiting, the pipe is retried this often. */
    const val DRAIN_RETRY_MS = 40L
  }

  private val main = Handler(Looper.getMainLooper())
  private val recognizer = SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
  /** Identifies the current window, so a replaced window's late callbacks and timers are ignored. */
  private var generation = 0
  @Volatile private var running = false
  private var stopped = false
  /**
   * Nothing to recognize (standby's energy gate closed, or no audio yet): no
   * window runs or renews until the next `append`. Renewing windows over
   * silence kept the recognizer running all the time.
   */
  @Volatile private var waitingForAudio = true
  private var failures = 0
  /** The pipe's read end, handed to the recognizer; kept open until its window ends. */
  private var source: ParcelFileDescriptor? = null

  /** The pipe's non-blocking write end, written from the JavaScript thread. */
  private val sinkLock = Any()
  private var sink: ParcelFileDescriptor? = null
  /** Audio not yet in the pipe, in order: written once the window's reader runs (guarded by `sinkLock`). */
  private val backlog = PcmBacklog(MAX_BACKLOG_CHUNKS)

  /** Reports false when English is not on the phone; a download is requested for a later call. */
  fun start(done: (listening: Boolean) -> Unit) {
    val probe = recognitionIntent(null)
    val begin = { supported: Boolean ->
      if (!supported || stopped) {
        stop()
        done(false)
      } else {
        // The first window opens with the first audio.
        running = true
        waitingForAudio = true
        done(true)
      }
    }
    recognizer.checkRecognitionSupport(probe, context.mainExecutor, object : RecognitionSupportCallback {
      override fun onSupportResult(support: RecognitionSupport) {
        val installed = support.installedOnDeviceLanguages.any(::isLanguage)
        val downloadable = (support.pendingOnDeviceLanguages + support.supportedOnDeviceLanguages).any(::isLanguage)
        if (!installed && downloadable) recognizer.triggerModelDownload(probe)
        // Some recognizers report no languages at all; those are tried.
        begin(installed || !downloadable)
      }

      // Not every recognizer can say what it supports; try listening anyway.
      override fun onError(error: Int) = begin(true)
    })
  }

  /** Clears the transcript (after a command) so it cannot fire twice. */
  fun reset() {
    if (running && !waitingForAudio) beginWindow()
  }

  /** Ends the current window and renews nothing until more audio arrives. */
  fun idle() {
    if (!running || waitingForAudio) return
    waitingForAudio = true
    generation += 1
    closeWindow()
    synchronized(sinkLock) { backlog.clear() }
  }

  fun stop() {
    if (stopped) return
    stopped = true
    running = false
    generation += 1
    closeWindow()
    synchronized(sinkLock) { backlog.clear() }
    recognizer.destroy()
  }

  /**
   * Writes a chunk without ever blocking the JavaScript thread: when the
   * recognizer falls behind, or between windows, the chunk is dropped.
   */
  fun append(pcm16Base64: String) {
    val bytes = try {
      Base64.decode(pcm16Base64, Base64.DEFAULT)
    } catch (error: IllegalArgumentException) {
      return
    }
    if (!running) return
    synchronized(sinkLock) {
      // Idle: the next window opens on the main thread, and this audio waits for its pipe.
      if (sink == null && waitingForAudio && backlog.isEmpty) main.post(::wake)
      backlog.add(bytes)
      drainLocked()
    }
  }

  private fun wake() {
    if (!waitingForAudio || !running) return
    failures = 0
    beginWindow()
  }

  /**
   * Writes waiting audio into the pipe without blocking: what it refuses
   * (EAGAIN: full, until the recognizer reads; EPIPE: the window closed)
   * stays for the next drain. Caller holds `sinkLock`.
   */
  private fun drainLocked(): Boolean {
    val fd = sink?.fileDescriptor ?: return false
    return backlog.drain(PIPE_ATOMIC_BYTES) { bytes, offset, count ->
      try {
        Os.write(fd, bytes, offset, count)
      } catch (error: Exception) {
        0
      }
    }
  }

  /**
   * Keeps draining on the main thread while audio waits and no new chunk
   * arrives to drain it; after about 2 s the next chunk carries on.
   */
  private fun drainLater(current: Int, attempts: Int = 50) {
    main.postDelayed({
      if (current != generation) return@postDelayed
      val done = synchronized(sinkLock) { drainLocked() }
      if (!done && attempts > 1) drainLater(current, attempts - 1)
    }, DRAIN_RETRY_MS)
  }

  private fun beginWindow() {
    closeWindow()
    if (!running) return
    waitingForAudio = false
    generation += 1
    val current = generation
    val (read, write) = ParcelFileDescriptor.createPipe()
    try {
      Os.fcntlInt(write.fileDescriptor, OsConstants.F_SETFL, OsConstants.O_NONBLOCK)
    } catch (error: ErrnoException) {
      read.close()
      write.close()
      onError("Could not open the audio pipe: ${error.message}", -1)
      stop()
      return
    }
    source = read
    // The recognizer reads first, so the waiting audio has somewhere to go.
    recognizer.setRecognitionListener(listener(current))
    recognizer.startListening(recognitionIntent(read))
    val drained = synchronized(sinkLock) {
      sink = write
      drainLocked()
    }
    if (!drained) drainLater(current)
    main.postDelayed({ if (current == generation && !waitingForAudio) beginWindow() }, WINDOW_MS)
  }

  /** Ends the current window now and starts the next after a pause. */
  private fun renewAfter(delayMs: Long) {
    // A new generation, so the ended window's timer cannot renew it too.
    generation += 1
    val next = generation
    closeWindow()
    main.postDelayed({ if (next == generation && !waitingForAudio) beginWindow() }, delayMs)
  }

  private fun closeWindow() {
    recognizer.cancel()
    synchronized(sinkLock) {
      sink?.close()
      sink = null
    }
    source?.close()
    source = null
  }

  private fun listener(current: Int) = object : RecognitionListener {
    private val stale get() = current != generation || !running

    override fun onPartialResults(partialResults: Bundle) {
      if (!stale) transcript(partialResults)?.let { onTranscript(it, false) }
    }

    override fun onResults(results: Bundle) {
      if (stale) return
      failures = 0
      transcript(results)?.let { onTranscript(it, true) }
      renewAfter(0)
    }

    override fun onError(error: Int) {
      if (stale) return
      when (error) {
        // Silence or unclear speech ends a window; that is ordinary, not a failure.
        SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> renewAfter(RENEW_MS)
        // Retrying cannot fix these.
        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS,
        SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED,
        SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE -> {
          onError(describe(error), error)
          stop()
        }
        else -> {
          onError(describe(error), error)
          renewAfter(minOf(RETRY_MS shl minOf(failures, 5), MAX_RETRY_MS))
          failures += 1
        }
      }
    }

    override fun onReadyForSpeech(params: Bundle?) {}
    override fun onBeginningOfSpeech() {}
    override fun onRmsChanged(rmsdB: Float) {}
    override fun onBufferReceived(buffer: ByteArray?) {}
    override fun onEndOfSpeech() {}
    override fun onEvent(eventType: Int, params: Bundle?) {}
  }

  private fun recognitionIntent(audio: ParcelFileDescriptor?) = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
    putExtra(RecognizerIntent.EXTRA_LANGUAGE, LANGUAGE)
    putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
    putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
    putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
    putStringArrayListExtra(RecognizerIntent.EXTRA_BIASING_STRINGS, phrases)
    if (audio != null) {
      putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, audio)
      putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_CHANNEL_COUNT, 1)
      putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_ENCODING, AudioFormat.ENCODING_PCM_16BIT)
      putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE_SAMPLING_RATE, sampleRate)
    }
  }

  private fun transcript(bundle: Bundle) =
    bundle.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.takeIf { it.isNotBlank() }

  private fun isLanguage(tag: String) = tag.replace('_', '-').equals(LANGUAGE, ignoreCase = true)

  private fun describe(error: Int) = when (error) {
    SpeechRecognizer.ERROR_AUDIO -> "Audio error"
    SpeechRecognizer.ERROR_CLIENT -> "Client error"
    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Missing microphone permission"
    SpeechRecognizer.ERROR_NETWORK, SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Network error"
    SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognizer busy"
    SpeechRecognizer.ERROR_SERVER, SpeechRecognizer.ERROR_SERVER_DISCONNECTED -> "Recognition service error"
    SpeechRecognizer.ERROR_TOO_MANY_REQUESTS -> "Too many requests"
    SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED -> "English is not supported on this phone"
    SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE -> "English recognition is not downloaded yet"
    else -> "Recognition error $error"
  }
}
