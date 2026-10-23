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
import androidx.core.app.ServiceCompat

/**
 * Keeps local voice standby (ADR 0021) listening with the screen off: a
 * microphone-only foreground service with a persistent notification. It is
 * started while the app is on screen (Android 14 refuses to start a
 * microphone service from the background) and runs until voice is turned off,
 * across sessions ending and "mute audio". It holds no wake lock; recording
 * keeps the CPU awake as long as capture needs it.
 */
class StandbyService : Service() {
  companion object {
    private const val CHANNEL = "voice-standby"
    private const val NOTIFICATION_ID = 7422
  }

  override fun onBind(intent: Intent?): IBinder? = null

  override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
    val manager = getSystemService(NotificationManager::class.java)
    manager.createNotificationChannel(NotificationChannel(CHANNEL, "Listening for the wake phrase", NotificationManager.IMPORTANCE_LOW))
    val open = packageManager.getLaunchIntentForPackage(packageName)?.let { PendingIntent.getActivity(this, 2, it, PendingIntent.FLAG_IMMUTABLE) }
    val notification = NotificationCompat.Builder(this, CHANNEL)
      .setSmallIcon(android.R.drawable.ic_btn_speak_now)
      .setContentTitle("Sidebud is listening")
      .setContentText("Say the wake phrase to talk. Nothing leaves your phone until you do.")
      .setOngoing(true)
      .setContentIntent(open)
      .build()
    val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE else 0
    ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, type)
    // Not restarted by the system: a microphone service may not start from the background.
    return START_NOT_STICKY
  }
}
