package expo.modules.voicecommands

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayOutputStream

/** A non-blocking pipe with a small capacity: writes up to `atomic` bytes land whole or not at all (EAGAIN). */
private class SmallPipe(private val capacity: Int, private val atomic: Int) {
  private val buffered = ArrayDeque<Byte>()
  val delivered = ByteArrayOutputStream()

  fun write(bytes: ByteArray, offset: Int, count: Int): Int {
    if (count <= atomic && capacity - buffered.size < count) return 0
    val accepted = minOf(count, capacity - buffered.size)
    for (index in offset until offset + accepted) buffered.addLast(bytes[index])
    return accepted
  }

  /** The recognizer reads what it can. */
  fun read(max: Int) {
    repeat(minOf(max, buffered.size)) { delivered.write(buffered.removeFirst().toInt()) }
  }
}

class PcmBacklogTest {
  private fun chunk(index: Int, size: Int = 5_120) = ByteArray(size) { (index * 31 + it).toByte() }

  @Test
  fun deliversPreRollAndPhraseByteForByteThroughASmallPipeWithALateReader() {
    val pipe = SmallPipe(capacity = 16_384, atomic = 4_096)
    val backlog = PcmBacklog(maxChunks = 32)
    // The gate's pre-roll and the opening chunks arrive before the recognizer reads anything.
    val sent = (0 until 12).map { chunk(it) }
    sent.forEach(backlog::add)
    assertFalse(backlog.drain(4_096, pipe::write))
    // The reader starts late and drains a little at a time; each new chunk drains the backlog first.
    var next = 12
    while (!backlog.isEmpty || next < 16) {
      pipe.read(3_000)
      if (next < 16) {
        val more = chunk(next++)
        backlog.add(more)
      }
      backlog.drain(4_096, pipe::write)
    }
    pipe.read(Int.MAX_VALUE)
    val expected = ByteArrayOutputStream().apply { (0 until 16).forEach { write(chunk(it)) } }.toByteArray()
    assertArrayEquals(expected, pipe.delivered.toByteArray())
  }

  @Test
  fun keepsTheNewestAudioWhenTheReaderNeverCatchesUp() {
    val backlog = PcmBacklog(maxChunks = 2)
    (0 until 3).forEach { backlog.add(chunk(it, 4)) }
    val out = ByteArrayOutputStream()
    assertTrue(backlog.drain(4_096) { bytes, offset, count -> out.write(bytes, offset, count); count })
    assertArrayEquals(chunk(1, 4) + chunk(2, 4), out.toByteArray())
  }
}
