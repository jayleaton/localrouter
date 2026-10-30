package expo.modules.callsession

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.Person
import androidx.core.app.ServiceCompat

/** Keeps a call's process alive with the ongoing CallStyle notification Android requires. */
class CallSessionService : Service() {
  companion object {
    const val EXTRA_NAME = "name"
    private const val ACTION_HANG_UP = "expo.modules.callsession.HANG_UP"
    private const val CHANNEL = "voice-call"
    private const val NOTIFICATION_ID = 7421
    /** Set by the module while a call is active. */
    var onHangUp: (() -> Unit)? = null
  }

  override fun onBind(intent: Intent?): IBinder? = null

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    if (intent?.action == ACTION_HANG_UP) {
      onHangUp?.invoke()
      return START_NOT_STICKY
    }
    val manager = getSystemService(NotificationManager::class.java)
    manager.createNotificationChannel(NotificationChannel(CHANNEL, "Voice calls", NotificationManager.IMPORTANCE_DEFAULT))
    val name = intent?.getStringExtra(EXTRA_NAME) ?: "Voice assistant"
    val hangUp = PendingIntent.getService(this, 0, Intent(this, CallSessionService::class.java).setAction(ACTION_HANG_UP), PendingIntent.FLAG_IMMUTABLE)
    val open = packageManager.getLaunchIntentForPackage(packageName)?.let { PendingIntent.getActivity(this, 1, it, PendingIntent.FLAG_IMMUTABLE) }
    val notification = NotificationCompat.Builder(this, CHANNEL)
      .setSmallIcon(android.R.drawable.ic_btn_speak_now)
      .setOngoing(true)
      .setContentIntent(open)
      .setStyle(NotificationCompat.CallStyle.forOngoingCall(Person.Builder().setName(name).setImportant(true).build(), hangUp))
      .build()
    val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) ServiceInfo.FOREGROUND_SERVICE_TYPE_PHONE_CALL or ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE else 0
    ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, type)
    return START_NOT_STICKY
  }
}
