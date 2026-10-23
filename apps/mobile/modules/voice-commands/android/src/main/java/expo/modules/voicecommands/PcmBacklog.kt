package expo.modules.voicecommands

/**
 * PCM waiting for the recognizer's pipe, oldest first. A pipe holds a
 * limited amount (often 64 KB, sometimes less) and nothing drains it until
 * the recognizer reads, so a write it refuses is kept, at the byte it
 * stopped, for the next drain instead of being dropped. Bounded: when the
 * reader never catches up, the oldest chunks go first.
 */
internal class PcmBacklog(private val maxChunks: Int) {
  private val chunks = ArrayDeque<ByteArray>()
  /** Bytes of the first chunk already written. */
  private var offset = 0

  val isEmpty get() = chunks.isEmpty()

  fun add(bytes: ByteArray) {
    if (chunks.size >= maxChunks) {
      chunks.removeFirst()
      offset = 0
    }
    chunks.addLast(bytes)
  }

  fun clear() {
    chunks.clear()
    offset = 0
  }

  /**
   * Writes in order until the pipe refuses. `write(bytes, offset, count)`
   * returns the bytes written, or 0 or less when the pipe is full or closed.
   * Returns true once everything is written.
   */
  fun drain(maxWrite: Int, write: (ByteArray, Int, Int) -> Int): Boolean {
    while (chunks.isNotEmpty()) {
      val chunk = chunks.first()
      while (offset < chunk.size) {
        val written = write(chunk, offset, minOf(maxWrite, chunk.size - offset))
        if (written <= 0) return false
        offset += written
      }
      chunks.removeFirst()
      offset = 0
    }
    return true
  }
}
