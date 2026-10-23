package expo.modules.callsession

import android.content.Intent
import android.net.Uri
import android.telecom.DisconnectCause
import androidx.core.telecom.CallAttributesCompat
import androidx.core.telecom.CallControlScope
import androidx.core.telecom.CallsManager
import expo.modules.kotlin.Promise
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * Registers the voice session as a self-managed call through Jetpack
 * Core-Telecom, so it shows in the system call UI and Android Auto, with a
 * phone-call foreground service keeping the process alive while locked.
 * Audio stays in react-native-audio-api; there is no session activation step
 * on Android, so onAudioActivated fires once the call is active.
 */
class CallSessionModule : Module() {
  private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)
  private var callJob: Job? = null
  private var control: CallControlScope? = null
  private var endingFromApp = false
  private val context get() = requireNotNull(appContext.reactContext) { "No React context" }
  private val callsManager by lazy {
    CallsManager(context).also { it.registerAppWithTelecom(CallsManager.CAPABILITY_BASELINE) }
  }

  override fun definition() = ModuleDefinition {
    Name("CallSession")
    Events("onAudioActivated", "onAudioDeactivated", "onMuteChanged", "onEndRequested")

    AsyncFunction("startCall") { displayName: String, promise: Promise ->
      if (callJob != null) {
        promise.resolve("active")
        return@AsyncFunction
      }
      CallSessionService.onHangUp = { endFromSystem() }
      context.startForegroundService(Intent(context, CallSessionService::class.java).putExtra(CallSessionService.EXTRA_NAME, displayName))
      val attributes = CallAttributesCompat(
        displayName,
        Uri.parse("rva:voice"),
        CallAttributesCompat.DIRECTION_OUTGOING,
        CallAttributesCompat.CALL_TYPE_AUDIO_CALL,
        CallAttributesCompat.SUPPORTS_SET_INACTIVE
      )
      callJob = scope.launch {
        try {
          callsManager.addCall(
            attributes,
            onAnswer = { },
            onDisconnect = { _ -> finish(fromSystem = !endingFromApp) },
            onSetActive = { },
            onSetInactive = { }
          ) {
            control = this
            launch {
              setActive()
              sendEvent("onAudioActivated", mapOf<String, Any>())
              promise.resolve("active")
            }
            launch { isMuted.collect { muted -> sendEvent("onMuteChanged", mapOf("muted" to muted)) } }
          }
        } catch (error: Exception) {
          finish(fromSystem = false)
          promise.reject("ERR_CALL_START", error.message ?: "Could not start the call", error)
        }
      }
    }

    AsyncFunction("reportConnected") { }

    // Core-Telecom exposes the system mute state but not a setter; the app gates its own microphone.
    AsyncFunction("setMuted") { _: Boolean -> }

    AsyncFunction("endCall") { promise: Promise ->
      val current = control
      if (current == null) {
        finish(fromSystem = false)
        promise.resolve(null)
        return@AsyncFunction
      }
      endingFromApp = true
      scope.launch {
        current.disconnect(DisconnectCause(DisconnectCause.LOCAL))
        finish(fromSystem = false)
        promise.resolve(null)
      }
    }

    Function("isActive") { callJob != null }

    // Local voice standby with the screen off (ADR 0021): call only while the app is on screen.
    AsyncFunction("startStandby") { context.startForegroundService(Intent(context, StandbyService::class.java)); Unit }
    AsyncFunction("stopStandby") { context.stopService(Intent(context, StandbyService::class.java)); Unit }

    OnDestroy { scope.cancel() }
  }

  private fun endFromSystem() {
    val current = control ?: return finish(fromSystem = true)
    scope.launch {
      current.disconnect(DisconnectCause(DisconnectCause.LOCAL))
      finish(fromSystem = true)
    }
  }

  private fun finish(fromSystem: Boolean) {
    val wasActive = callJob != null
    control = null
    callJob?.cancel()
    callJob = null
    endingFromApp = false
    CallSessionService.onHangUp = null
    context.stopService(Intent(context, CallSessionService::class.java))
    if (wasActive) sendEvent("onAudioDeactivated", mapOf<String, Any>())
    if (wasActive && fromSystem) sendEvent("onEndRequested", mapOf<String, Any>())
  }
}
