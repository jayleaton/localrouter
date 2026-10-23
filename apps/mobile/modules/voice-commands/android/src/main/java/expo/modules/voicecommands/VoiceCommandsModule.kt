package expo.modules.voicecommands

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.speech.SpeechRecognizer
import expo.modules.kotlin.Promise
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

/**
 * On-device speech recognition over microphone audio the app already captures
 * (passed in with `append`), for spoken app commands. Android's counterpart of
 * the iOS module: the recognizer reads the call's audio from a pipe, so it never
 * opens the microphone itself, and it only runs on the on-device recognizer, so
 * muted audio never leaves the phone. That needs Android 13; older phones
 * resolve `start` false and keep the buttons. JavaScript decides what counts as
 * a command.
 */
class VoiceCommandsModule : Module() {
  private val main = Handler(Looper.getMainLooper())
  private val context get() = requireNotNull(appContext.reactContext) { "No React context" }

  /** Replaced on the main thread; `append` reads it from the JavaScript thread. */
  @Volatile private var session: CommandSession? = null

  override fun definition() = ModuleDefinition {
    Name("VoiceCommands")
    Events("onTranscript", "onError")

    // The recognizer hears the call's audio, not the microphone, so there is no
    // speech-specific permission; the call has already asked for the microphone.
    AsyncFunction("requestPermission") { promise: Promise ->
      promise.resolve(if (hasMicrophone()) "granted" else "denied")
    }

    AsyncFunction("start") { sampleRate: Double, phrases: List<String>, promise: Promise ->
      main.post {
        stopListening()
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU || !hasMicrophone() || !SpeechRecognizer.isOnDeviceRecognitionAvailable(context)) {
          promise.resolve(false)
          return@post
        }
        val started = CommandSession(
          context,
          sampleRate.toInt(),
          ArrayList(phrases),
          onTranscript = { text, isFinal -> sendEvent("onTranscript", mapOf("text" to text, "isFinal" to isFinal)) },
          onError = { message, code -> sendEvent("onError", mapOf("message" to message, "domain" to "SpeechRecognizer", "code" to code)) },
        )
        session = started
        started.start { listening ->
          if (!listening && session === started) session = null
          promise.resolve(listening)
        }
      }
    }

    Function("append") { pcm16Base64: String ->
      if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) session?.append(pcm16Base64)
    }

    Function("reset") {
      main.post { if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) session?.reset() }
    }

    Function("idle") {
      main.post { if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) session?.idle() }
    }

    Function("stop") {
      main.post { stopListening() }
    }

    OnDestroy {
      main.post { stopListening() }
    }
  }

  private fun stopListening() {
    // A session only exists on Android 13 or newer.
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) session?.stop()
    session = null
  }

  private fun hasMicrophone() = context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
}
