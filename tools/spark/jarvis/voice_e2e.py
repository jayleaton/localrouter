"""The voice loop end to end on a real daemon: recorded speech in over /v1/realtime, measured per turn.

Python 3 standard library only (it runs on the GPU host with `python3 -I`). Steps:
  1. The prompts are synthesised by the daemon's own TTS (POST /v1/audio/speech, WAV) and resampled to 24 kHz pcm16:
     they are the microphone.
  2. A session streams the microphone in real time (100 ms chunks on the wall clock, silence between prompts, as a
     browser would), and every server event is logged with its arrival time.
  3. Per turn: speech end (the last voiced sample of the prompt, in wall time) to speech_stopped, to the final
     transcript, to the first reply text, and to the first reply audio; the transcript and the reply text.
  4. Barge-in: a long answer is interrupted by a second prompt. Checks: speech_started for it, the answer's
     response.done is cancelled (turn_detected), no audio of that answer after speech_started, the new turn answered.
  5. A tool call: "make me a picture" must produce generate_image, a localrouter.job.done with a PNG URL (fetched and
     checked), and a spoken follow-up.
  6. Recovery: a short video job runs (POST /v1/videos); then chat, speech to text and text to speech must be ready
     again, and a new session's turn is answered (its latency recorded).
Writes OUT/results.json, OUT/events.jsonl, OUT/prompts/*.wav, OUT/replies/*.wav, OUT/image.png. Exit 0 only when every
check passed (fail closed: a missing measurement is a failed check).
"""

import argparse
import array
import base64
import json
import os
import re
import select
import socket
import struct
import subprocess
import sys
import time
import urllib.request
import wave

RATE = 24000
CHUNK = RATE // 10  # 100 ms of samples
PROMPTS = {
    "warmup": "Hello, can you hear me?",
    "capital": "What is the capital of France? Answer in one short sentence.",
    "tips": "Give me two quick tips for sleeping better.",
    "story": "Tell me a long story about a lighthouse keeper and a winter storm, with plenty of detail.",
    "interrupt": "Sorry to interrupt. Which time zone is Tokyo in?",
    "picture": "Please make me a picture of a red fox sitting in fresh snow.",
    "after_video": "Are you still there and listening to me?",  # one sentence: Drama 3 dropped the second of two here
    "after_eviction": "What is two plus two?",
}


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def http(url, body=None, timeout=900, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                                 headers={"content-type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, dict(r.headers), r.read()


# ---------------------------------------------------------------- audio


def wav_pcm(raw):
    """Mono 16-bit samples and the rate of a WAV file's bytes (stereo is averaged)."""
    import io
    with wave.open(io.BytesIO(raw), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("not 16-bit PCM")
        rate, ch, frames = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    s = array.array("h")
    s.frombytes(frames)
    if sys.byteorder == "big":
        s.byteswap()
    if ch > 1:
        s = array.array("h", (sum(s[i:i + ch]) // ch for i in range(0, len(s), ch)))
    return s, rate


def resample(s, src, dst):
    """Linear interpolation (adequate for a microphone stand-in)."""
    if src == dst:
        return array.array("h", s)
    n = int(len(s) * dst / src)
    out = array.array("h", bytes(2 * n))
    step = src / dst
    last = len(s) - 1
    for i in range(n):
        x = i * step
        j = int(x)
        f = x - j
        a = s[min(j, last)]
        b = s[min(j + 1, last)]
        out[i] = int(a + (b - a) * f)
    return out


def last_voiced(s, threshold=500):
    """The index after the last sample louder than `threshold` (0 for silence)."""
    for i in range(len(s) - 1, -1, -1):
        if abs(s[i]) > threshold:
            return i + 1
    return 0


def squeeze(s, threshold=500, max_gap=0.3, edge=0.1):
    """Speech as a person says one request: pauses longer than `max_gap` s shortened to it, edges trimmed to `edge` s.
    The TTS that records the prompts can leave second-long pauses between sentences, which VAD rightly takes as the
    end of the turn."""
    loud = [i for i in range(0, len(s), 240) if max(abs(x) for x in s[i:i + 240]) > threshold]  # 10 ms blocks
    if not loud:
        return array.array("h", s)
    out, gap, keep = array.array("h"), int(RATE * max_gap) // 240, int(RATE * edge) // 240
    blocks = set(loud)
    first, last = loud[0] // 240, loud[-1] // 240
    quiet = 0
    for b in range(max(first - keep, 0), min(last + keep + 1, (len(s) + 239) // 240)):
        if b * 240 in blocks:
            quiet = 0
        else:
            quiet += 1
            if quiet > gap and first <= b <= last:
                continue
        out.extend(s[b * 240:(b + 1) * 240])
    return out


def write_wav(path, s, rate=RATE):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        b = array.array("h", s)
        if sys.byteorder == "big":
            b.byteswap()
        w.writeframes(b.tobytes())


# ---------------------------------------------------------------- WebSocket (RFC 6455, client side)


class Ws:
    def __init__(self, host, port, path):
        self.sock = socket.create_connection((host, port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                           "Sec-WebSocket-Protocol: realtime\r\n\r\n").encode())
        self.buf = b""
        while b"\r\n\r\n" not in self.buf:
            more = self.sock.recv(4096)
            if not more:
                raise ConnectionError("closed during the handshake")
            self.buf += more
        head, self.buf = self.buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"no upgrade: {head[:200]!r}")
        self.sock.setblocking(False)

    def send(self, obj):
        data = json.dumps(obj).encode()
        mask = os.urandom(4)
        n = len(data)
        head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else
                                bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536 else
                                bytes([0x80 | 127]) + struct.pack(">Q", n))
        body = bytes(b ^ mask[i & 3] for i, b in enumerate(data))
        self.sock.setblocking(True)
        try:
            self.sock.sendall(head + mask + body)
        finally:
            self.sock.setblocking(False)

    def poll(self, timeout):
        """The text messages that arrive within `timeout` seconds (those already buffered at once)."""
        out = []
        deadline = time.monotonic() + max(timeout, 0)
        while True:
            msg = self._frame()
            while msg is not None:
                out.append(msg)
                msg = self._frame()
            if out:
                return out
            r, _, _ = select.select([self.sock], [], [], max(deadline - time.monotonic(), 0))
            if not r:
                return out
            more = self.sock.recv(1 << 20)  # readable: does not block
            if not more:
                raise ConnectionError("the server closed the connection")
            self.buf += more

    def _frame(self):
        """The next complete text message in the buffer, None when it is not all here yet."""
        while True:
            b = self.buf
            if len(b) < 2:
                return None
            op, n, at = b[0] & 0x0F, b[1] & 0x7F, 2
            if n == 126:
                if len(b) < 4:
                    return None
                n, at = struct.unpack(">H", b[2:4])[0], 4
            elif n == 127:
                if len(b) < 10:
                    return None
                n, at = struct.unpack(">Q", b[2:10])[0], 10
            if len(b) < at + n:
                return None
            payload, self.buf = b[at:at + n], b[at + n:]
            if op == 8:
                raise ConnectionError("close frame")
            if op == 1:
                return payload.decode()
            # control or binary frames: skipped

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


# ---------------------------------------------------------------- a session with a real-time microphone


class Session:
    """Streams the microphone on the wall clock; every event is kept with its arrival time."""

    def __init__(self, url, events_file):
        host, port = re.match(r"http://([^:/]+):(\d+)", url).groups()
        self.ws = Ws(host, int(port), "/v1/realtime")
        self.events = []
        self.file = events_file
        self.queue = array.array("h")  # what the microphone says next; silence when empty
        self.next_send = time.monotonic()
        self.sent_samples = 0
        self.audio = {}  # response id -> samples received
        self.first_delta = {}  # response id -> arrival of its first audio delta
        self.voiced = {}  # response id -> (arrival of the delta with its first non-silent sample, that sample's index)
        self.wait_for("session.created", 30)
        self.ws.send({"type": "session.update", "session": {
            "modalities": ["audio", "text"],
            "turn_detection": {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
                               "silence_duration_ms": 500, "create_response": True, "interrupt_response": True}}})
        self.wait_for("session.updated", 30)

    def say(self, samples):
        """Queues speech; returns the wall time its last voiced sample will be sent (speech end)."""
        voiced = last_voiced(samples)
        queued_ahead = len(self.queue)
        start = max(self.next_send, time.monotonic())
        self.queue.extend(samples)
        return start + (queued_ahead + voiced) / RATE

    def pump(self, seconds):
        """Runs the microphone and reads events for `seconds`; returns the new events."""
        end = time.monotonic() + seconds
        new = []
        while True:
            now = time.monotonic()
            while self.next_send <= now:
                chunk = self.queue[:CHUNK]
                del self.queue[:CHUNK]
                if len(chunk) < CHUNK:
                    chunk.extend(array.array("h", [0] * (CHUNK - len(chunk))))
                b = array.array("h", chunk)
                if sys.byteorder == "big":
                    b.byteswap()
                self.ws.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(b.tobytes()).decode()})
                self.sent_samples += CHUNK
                self.next_send += CHUNK / RATE
            if now >= end:
                return new
            for text in self.ws.poll(min(self.next_send, end) - time.monotonic()):
                ev = json.loads(text)
                ev["_t"] = time.monotonic()
                self.events.append(ev)
                new.append(ev)
                if ev.get("type") == "response.audio.delta":
                    pcm = array.array("h")
                    pcm.frombytes(base64.b64decode(ev["delta"]))
                    if sys.byteorder == "big":
                        pcm.byteswap()
                    rid = ev["response_id"]
                    buf = self.audio.setdefault(rid, array.array("h"))
                    before = len(buf)
                    buf.extend(pcm)
                    self.first_delta.setdefault(rid, ev["_t"])
                    if rid not in self.voiced:
                        k = next((i for i, x in enumerate(pcm) if abs(x) > 250), None)  # about -42 dBFS, as the daemon's timing
                        if k is not None:
                            self.voiced[rid] = (ev["_t"], before + k)
                    ev = dict(ev, delta=f"<{len(pcm)} samples>")
                self.file.write(json.dumps(ev) + "\n")

    def wait_for(self, typ, timeout, after=0, pred=lambda e: True):
        """The first event of `typ` at index >= `after` that satisfies `pred`, pumping up to `timeout` s."""
        deadline = time.monotonic() + timeout
        while True:
            for e in self.events[after:]:
                if e.get("type") == typ and pred(e):
                    return e
            if time.monotonic() > deadline:
                raise TimeoutError(f"no {typ} within {timeout} s")
            self.pump(0.05)

    def index(self, ev):
        return next(i for i, e in enumerate(self.events) if e is ev)

    def close(self):
        self.ws.close()


def words(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def overlap(said, heard):
    s, h = set(words(said)), set(words(heard))
    return len(s & h) / max(len(s), 1)


STAGES = ("vad", "stt_wait", "stt", "stt_worker", "to_chat", "chat_ttft", "to_tts_text", "tts_wait", "tts_first_pcm",
          "to_first_audio", "leading_silence")


def stages(r, tm):
    """Where a turn's time went, in ms: VAD's end-of-turn delay (client), then the server's stages from the daemon's
    `localrouter.response.timing` (measured from speech_stopped): transcription (queue, total, the worker's own),
    chat time to first token, the lead hold and chunker before speech gets text, speech queue and time to first
    samples, delivery, and the silence at the head of the audio. With a draft transcript (made at the pause) the
    transcription can start before speech_stopped: stt_wait is then negative."""
    if not tm:
        return None
    m = tm["ms"]
    d = lambda a, b: None if m.get(a) is None or m.get(b) is None else m[a] - m[b]
    return {"vad": r["speech_end_to_vad_stop_ms"], "stt_wait": m.get("stt_start"), "stt": d("stt_end", "stt_start"),
            "stt_worker": tm.get("stt_worker_ms"), "to_chat": d("chat_start", "stt_end"), "chat_ttft": d("first_token", "chat_start"),
            "to_tts_text": d("first_chunk", "first_token"), "tts_wait": d("tts_start", "first_chunk"),
            "tts_first_pcm": d("tts_first_pcm", "tts_start"), "to_first_audio": d("first_audio", "tts_first_pcm"),
            "leading_silence": tm.get("voiced_audio_ms"), "server_first_voiced": m.get("first_voiced"),
            # how the turn ended: on its draft's sentence end, on silence with the draft reused, or on silence
            "turn_end": "early" if tm.get("early_end") else "draft" if tm.get("stt_draft") else "silence"}


def turn(sess, name, samples, results, checks, timeout=120, need_audio=True):
    """One spoken turn; its measurements go to results[name]."""
    mark = len(sess.events)
    t_end = sess.say(samples)
    stopped = sess.wait_for("input_audio_buffer.speech_stopped", timeout, mark)
    item = stopped["item_id"]
    heard = sess.wait_for("conversation.item.input_audio_transcription.completed", timeout, mark, lambda e: e["item_id"] == item)
    created = sess.wait_for("response.created", timeout, sess.index(heard))
    rid = created["response"]["id"]
    done = sess.wait_for("response.done", timeout, sess.index(created), lambda e: e["response"]["id"] == rid)
    span = sess.events[sess.index(created):sess.index(done) + 1]
    text = "".join(e.get("delta", "") for e in span if e["type"] in ("response.audio_transcript.delta", "response.text.delta") and e.get("response_id") == rid)
    first_text = next((e["_t"] for e in span if e["type"] == "response.audio_transcript.delta" and e["response_id"] == rid), None)
    first_audio = next((e["_t"] for e in span if e["type"] == "response.audio.delta" and e["response_id"] == rid), None)
    ms = lambda t: None if t is None else round((t - t_end) * 1000)
    r = {"prompt": PROMPTS[name], "transcript": heard["transcript"], "reply": text, "status": done["response"]["status"],
         "speech_end_to_vad_stop_ms": ms(stopped["_t"]), "speech_end_to_transcript_ms": ms(heard["_t"]),
         "speech_end_to_first_text_ms": ms(first_text), "speech_end_to_first_audio_ms": ms(first_audio),
         "reply_audio_s": round(len(sess.audio.get(rid, [])) / RATE, 2), "response_id": rid}
    tm = next((e for e in span if e["type"] == "localrouter.response.timing" and e["response_id"] == rid), None)
    voiced = None
    if rid in sess.voiced:  # when its first non-silent sample plays: audio plays from its first delta on
        t_arr, pos = sess.voiced[rid]
        voiced = max(t_arr, sess.first_delta[rid] + pos / RATE)
    r["speech_end_to_first_voiced_ms"] = ms(voiced)
    r["stages_ms"] = stages(r, tm)
    results[name] = r
    log(f"{name}: heard {r['transcript']!r}; first audio {r['speech_end_to_first_audio_ms']} ms; reply {text[:100]!r}")
    checks.append({"check": f"{name}: transcript matches the prompt", "ok": overlap(PROMPTS[name], heard["transcript"]) >= 0.6,
                   "detail": heard["transcript"]})
    answered = r["status"] == "completed" and (not need_audio or (bool(text.strip()) and first_audio is not None))
    checks.append({"check": f"{name}: answered" + (", with audio" if need_audio else ""), "ok": answered,
                   "detail": r["status"]})
    return r


BALLAST = r"""
import sys, threading, os, time
b = bytearray(int(sys.argv[1]) << 30)
for i in range(0, len(b), 4096):
    b[i] = 1  # touch every page: the memory is really in use
print("ready", flush=True)
def guard():  # never push the host toward its OOM killer: let go below the floor
    floor = int(sys.argv[2]) << 20
    while True:
        with open("/proc/meminfo") as f:
            avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable:"))
        if avail < floor:
            print("guard", flush=True)
            os._exit(0)
        time.sleep(0.2)
threading.Thread(target=guard, daemon=True).start()
sys.stdin.read()
"""


def mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1048576, 1)
    return 0.0


def ballast(gib, floor_gib):
    """`gib` GiB held by another process, in its own bounded user scope, until `release` (or until MemAvailable falls
    below `floor_gib`, when it lets go at once: the host must never near its OOM killer)."""
    p = subprocess.Popen(["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={gib + 2}G", "timeout", "1800",
                          "python3", "-I", "-c", BALLAST, str(gib), str(floor_gib)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if p.stdout.readline().strip() != "ready":
        p.kill()
        p.wait()
        raise RuntimeError("the ballast did not start")
    return p


def release(p):
    p.stdin.close()
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def video_phase(args, name, results, checks, squeeze):
    """A 1 s video; which of chat, STT and TTS it unloaded; how long each takes to be ready again after it. With
    `squeeze`, idle tools outside the voice set are unloaded first and a ballast leaves only `--eviction-free-gib`
    available while the video runs and while the voice set reloads after it, so the video can be admitted only by
    unloading part of the voice set and the reload must make room by LocalRouter's own admission (evicting the idle
    video), beside the other tenant."""
    voice = (args.chat, args.stt, args.tts)
    held, ballast_gib, guard_tripped, held_through = None, 0, False, False
    if squeeze:
        _, _, body = http(f"{args.url}/v1/tools", timeout=30)
        for t in json.loads(body)["data"]:
            if t["id"] not in voice and t["state"] == "ready":
                http(f"{args.url}/v1/tools/{t['id']}/unload", {}, timeout=120)
        time.sleep(2)
        ballast_gib = max(0, int(mem_available_gib() - args.eviction_free_gib))
        held = ballast(ballast_gib, args.ballast_floor_gib) if ballast_gib else None
    down, back = set(), {}
    try:
        mem_before = mem_available_gib()
        t0 = time.monotonic()
        _, _, body = http(f"{args.url}/v1/videos", {"model": args.video, "prompt": "waves rolling onto a beach at dusk", "size": args.video_size, "seconds": "1"})
        vid = json.loads(body)["id"]
        while True:
            _, _, body = http(f"{args.url}/v1/videos/{vid}", timeout=30)
            v = json.loads(body)
            _, _, body = http(f"{args.url}/v1/tools", timeout=30)
            down |= {t["id"] for t in json.loads(body)["data"] if t["id"] in voice and t["state"] != "ready"}
            if v["status"] in ("completed", "failed", "cancelled") or time.monotonic() - t0 > 1200:
                break
            time.sleep(0.5)
        video_s = time.monotonic() - t0
        t1 = time.monotonic()
        while time.monotonic() - t1 < 600 and len(back) < 3:
            _, _, body = http(f"{args.url}/v1/tools", timeout=30)
            for t in json.loads(body)["data"]:
                if t["id"] in voice and t["state"] == "ready" and t["id"] not in back:
                    back[t["id"]] = round(time.monotonic() - t1, 1)
                elif t["id"] in voice and t["state"] != "ready":
                    down.add(t["id"])
            time.sleep(0.5)
        mem_low = mem_available_gib()
        held_through = held is not None and held.poll() is None  # the other tenant still holds its memory
    finally:
        if held:
            guard_tripped = held.poll() is not None
            release(held)
    results[name] = {"video_status": v["status"], "video_s": round(video_s, 1), "unloaded_by_video": sorted(down),
                     "ready_after_video_s": back, "ballast_gib": ballast_gib, "mem_available_before_gib": mem_before,
                     "mem_available_after_recovery_gib": mem_low, "ballast_guard_tripped": guard_tripped, "ballast_held_through_recovery": held_through, "video_error": v.get("error")}
    log(f"{name}:", results[name])
    checks.append({"check": f"{name}: the clip completed", "ok": v["status"] == "completed", "detail": v.get("error")})
    checks.append({"check": f"{name}: chat, speech to text and text to speech are ready again", "ok": len(back) == 3, "detail": back})
    if squeeze:
        checks.append({"check": f"{name}: the video had to unload part of the voice set", "ok": bool(down), "detail": sorted(down)})
        checks.append({"check": f"{name}: memory never neared the OOM killer (the ballast's guard held)", "ok": not guard_tripped, "detail": mem_low})
        checks.append({"check": f"{name}: every model came back with the ballast still held (admission alone governs the reload)", "ok": held_through and len(back) == 3, "detail": back})


VOICE_INSTRUCTIONS = ("You are a helpful voice assistant running on the user's own machine. Answer in short, natural spoken "
                      "sentences without markdown, lists or emoji.")  # the daemon's default `voice.instructions`


def repeats_first(text):
    """Whether a reply says its own first sentence again (leaked reasoning followed by the answer)."""
    m = re.match(r"\s*(.+?[.!?])(\s|$)", text, re.S)
    if not m:
        return False
    first = " ".join(words(m.group(1)))
    return len(first) >= 8 and first in " ".join(words(text[m.end():]))


def leak_probe(args, extra, n):
    """`n` streamed chat requests as the voice loop sends them, with `extra` in the body: how many replies carry a
    `think>` tag in their content (reasoning leaked into the answer)."""
    prompts = [PROMPTS[k] for k in ("capital", "tips", "after_video", "after_eviction")]
    leaks, samples = 0, []
    for i in range(n):
        body = {"model": args.chat, "stream": True, "messages": [{"role": "system", "content": VOICE_INSTRUCTIONS},
                                                                   {"role": "user", "content": prompts[i % len(prompts)]}]}
        body.update(extra)
        req = urllib.request.Request(f"{args.url}/v1/chat/completions", data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        parts = []
        with urllib.request.urlopen(req, timeout=180) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                for ch in json.loads(data).get("choices", []):
                    t = (ch.get("delta") or {}).get("content")
                    if t:
                        parts.append(t)
        text = "".join(parts)
        if "think>" in text:
            leaks += 1
            samples.append(text[:200])
    return {"requests": n, "leaks": leaks, "samples": samples[:3], "extra": extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tts", required=True, help="the speech model that records the prompts")
    ap.add_argument("--stt", required=True)
    ap.add_argument("--chat", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-size", default="768x448")
    ap.add_argument("--eviction-free-gib", type=int, default=30, help="MemAvailable left by the ballast in the eviction case")
    ap.add_argument("--ballast-floor-gib", type=int, default=6, help="the ballast lets go if MemAvailable falls below this")
    ap.add_argument("--max-median-ms", type=int, default=2500)
    ap.add_argument("--max-barge-ms", type=int, default=4000)
    ap.add_argument("--max-voiced-median-ms", type=int, default=1000, help="speech end to first non-silent reply audio, median")
    ap.add_argument("--max-voiced-p90-ms", type=int, default=1500, help="the same, p90 over the turns without a tool call")
    ap.add_argument("--leak-probe", type=int, default=20, help="streamed chat requests per variant in the reasoning-leak probe (0: none)")
    args = ap.parse_args()
    os.makedirs(f"{args.out}/prompts", exist_ok=True)
    os.makedirs(f"{args.out}/replies", exist_ok=True)
    results, checks = {"turns": {}}, []
    turns = results["turns"]

    # 1. The microphone: each prompt spoken by the daemon's TTS.
    mic = {}
    for name, text in PROMPTS.items():
        t0 = time.monotonic()
        status, headers, raw = http(f"{args.url}/v1/audio/speech", {"model": args.tts, "input": text, "response_format": "wav"})
        s, rate = wav_pcm(raw)
        raw_s = len(s) / rate
        mic[name] = squeeze(resample(s, rate, RATE))
        write_wav(f"{args.out}/prompts/{name}.wav", mic[name])
        log(f"prompt {name}: {raw_s:.2f} s of audio from {args.tts} ({rate} Hz) in {time.monotonic() - t0:.2f} s, {len(mic[name]) / RATE:.2f} s with long pauses shortened")
    silence = lambda s: array.array("h", [0] * int(RATE * s))

    events = open(f"{args.out}/events.jsonl", "w")
    sess = Session(args.url, events)
    try:
        sess.pump(1.0)
        # 2. Plain turns (the first warms every path and is reported apart).
        for name in ("warmup", "capital", "tips"):
            turn(sess, name, mic[name], turns, checks)
            sess.pump(1.0)

        # 4. Barge-in: talk over a long answer once a second of it has arrived.
        mark = len(sess.events)
        sess.say(mic["story"])
        created = sess.wait_for("response.created", 120, mark)
        rid = created["response"]["id"]
        deadline = time.monotonic() + 120
        while len(sess.audio.get(rid, [])) < RATE and time.monotonic() < deadline:
            sess.pump(0.05)
            if any(e["type"] == "response.done" and e["response"]["id"] == rid for e in sess.events[mark:]):
                break  # the answer ended before a second of it arrived: the barge-in check below fails
        barge_mark = len(sess.events)
        t_barge = max(sess.next_send, time.monotonic())
        r = turn(sess, "interrupt", mic["interrupt"], turns, checks)
        started = sess.wait_for("input_audio_buffer.speech_started", 30, barge_mark)
        cut = sess.wait_for("response.done", 30, mark, lambda e: e["response"]["id"] == rid)
        after = sess.events[sess.index(started):]
        late = [e for e in after if e["type"] == "response.audio.delta" and e["response_id"] == rid]
        details = cut["response"].get("status_details") or {}
        results["barge_in"] = {"interrupted_response": rid, "cut_status": cut["response"]["status"], "reason": details.get("reason"),
                               "speech_start_after_barge_ms": round((started["_t"] - t_barge) * 1000),
                               "cancel_after_speech_started_ms": round((cut["_t"] - started["_t"]) * 1000),
                               "audio_deltas_after_speech_started": len(late),
                               "story_audio_heard_s": round(len(sess.audio.get(rid, [])) / RATE, 2)}
        log("barge-in:", results["barge_in"])
        checks.append({"check": "barge-in: the answer is cancelled for the new turn", "ok": cut["response"]["status"] == "cancelled" and details.get("reason") == "turn_detected", "detail": details})
        checks.append({"check": "barge-in: no audio of it after speech_started", "ok": not late, "detail": len(late)})
        checks.append({"check": "barge-in: speech detected within 1 s", "ok": results["barge_in"]["speech_start_after_barge_ms"] <= 1000, "detail": results["barge_in"]["speech_start_after_barge_ms"]})
        sess.pump(1.0)

        # 5. A tool call that makes an image.
        mark = len(sess.events)
        t0 = time.monotonic()
        r = turn(sess, "picture", mic["picture"], turns, checks, timeout=180, need_audio=False)  # it may be only the tool call
        call = next((e for e in sess.events[mark:] if e["type"] == "response.function_call_arguments.done"), None)
        job = sess.wait_for("localrouter.job.done", 900, mark)
        confirm_created = sess.wait_for("response.created", 120, sess.index(job))
        crid = confirm_created["response"]["id"]
        confirm = sess.wait_for("response.done", 180, sess.index(confirm_created), lambda e: e["response"]["id"] == crid)
        spoken = "".join(e.get("delta", "") for e in sess.events[sess.index(confirm_created):] if e["type"] == "response.audio_transcript.delta" and e["response_id"] == crid)
        png_ok = False
        if job.get("status") == "completed" and job.get("urls"):
            _, _, png = http(job["urls"][0], timeout=60)
            png_ok = png[:8] == b"\x89PNG\r\n\x1a\n"
            with open(f"{args.out}/image.png", "wb") as f:
                f.write(png)
        results["tool"] = {"call": call and {"name": call["name"], "arguments": call["arguments"]}, "job": {k: job.get(k) for k in ("job_id", "status", "urls", "error")},
                           "speech_end_to_job_done_s": round(job["_t"] - t0, 1), "confirmation": spoken, "confirmation_status": confirm["response"]["status"]}
        log("tool:", results["tool"])
        checks.append({"check": "tool: the model called generate_image", "ok": bool(call) and call["name"] == "generate_image", "detail": call and call["name"]})
        checks.append({"check": "tool: the job made a PNG served at its URL", "ok": png_ok, "detail": job.get("status")})
        checks.append({"check": "tool: the assistant confirmed aloud", "ok": confirm["response"]["status"] == "completed" and bool(spoken.strip()), "detail": spoken[:120]})
    finally:
        for rid, s in sess.audio.items():
            write_wav(f"{args.out}/replies/{rid}.wav", s)
        sess.close()

    # 6. A short video job, then the voice models must be back and a new session answered. Then the same under memory
    #    pressure (a ballast, as another tenant), so the video must unload part of the voice set and the scheduler must
    #    bring it back.
    for name, prompt, pressure in (("video", "after_video", False), ("eviction", "after_eviction", True)):
        video_phase(args, name, results, checks, pressure)
        sess = Session(args.url, events)
        try:
            sess.pump(1.0)
            turn(sess, prompt, mic[prompt], turns, checks)
        finally:
            sess.close()
    events.close()

    lat = [t["speech_end_to_first_audio_ms"] for n, t in turns.items() if n != "warmup" and t["speech_end_to_first_audio_ms"] is not None]
    pct = lambda xs, p: sorted(xs)[min(len(xs) - 1, max(0, -(-p * len(xs) // 100) - 1))]
    results["latency_ms"] = {k: {"median": pct(v, 50), "p90": pct(v, 90), "min": min(v), "max": max(v)} for k in ("speech_end_to_transcript_ms", "speech_end_to_first_text_ms", "speech_end_to_first_audio_ms")
                             if (v := [t[k] for n, t in turns.items() if n != "warmup" and t[k] is not None])}
    results["first_audio_ms"] = results["latency_ms"].get("speech_end_to_first_audio_ms")
    if lat:
        checks.append({"check": f"latency: median speech end to first audio <= {args.max_median_ms} ms", "ok": pct(lat, 50) <= args.max_median_ms, "detail": pct(lat, 50)})
    barge = turns.get("interrupt", {}).get("speech_end_to_first_audio_ms")
    checks.append({"check": f"latency: the barge-in turn's first audio <= {args.max_barge_ms} ms", "ok": barge is not None and barge <= args.max_barge_ms, "detail": barge})
    voiced = {n: t["speech_end_to_first_voiced_ms"] for n, t in turns.items() if n != "warmup" and t.get("speech_end_to_first_voiced_ms") is not None}
    plain = [v for n, v in voiced.items() if n != "picture"]  # the tool turn speaks after its call
    results["first_voiced_ms"] = {"median": pct(list(voiced.values()), 50), "p90_non_tool": pct(plain, 90) if plain else None} if voiced else None
    checks.append({"check": f"latency: median speech end to first non-silent audio <= {args.max_voiced_median_ms} ms", "ok": bool(voiced) and pct(list(voiced.values()), 50) <= args.max_voiced_median_ms, "detail": results["first_voiced_ms"]})
    checks.append({"check": f"latency: p90 (non-tool turns) speech end to first non-silent audio <= {args.max_voiced_p90_ms} ms", "ok": bool(plain) and pct(plain, 90) <= args.max_voiced_p90_ms, "detail": results["first_voiced_ms"]})
    rows = {n: t["stages_ms"] for n, t in turns.items() if t.get("stages_ms")}
    results["stage_median_ms"] = {k: pct(v, 50) for k in STAGES if (v := [s[k] for n, s in rows.items() if n != "warmup" and s.get(k) is not None])}
    replies = {n: t["reply"] for n, t in turns.items()}
    if "tool" in results:
        replies["picture confirmation"] = results["tool"].get("confirmation", "")
    tagged = {n: r[:160] for n, r in replies.items() if "think>" in r}
    checks.append({"check": "replies: none contains a think> tag", "ok": not tagged, "detail": tagged})
    again = {n: r[:160] for n, r in replies.items() if repeats_first(r)}
    checks.append({"check": "replies: none repeats its own first sentence", "ok": not again, "detail": again})
    if args.leak_probe:  # does the request alone keep reasoning out of the answer? (informational)
        results["leak_probe"] = {
            "thinking_off_only": leak_probe(args, {"chat_template_kwargs": {"enable_thinking": False}}, args.leak_probe),
            "with_reasoning_effort_none": leak_probe(args, {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}, args.leak_probe),
        }
        log("leak probe:", {k: (v["leaks"], v["requests"]) for k, v in results["leak_probe"].items()})
    results["checks"] = checks
    results["passed"] = all(c["ok"] for c in checks) and bool(lat)
    with open(f"{args.out}/results.json", "w") as f:
        json.dump(results, f, indent=1)
    print()
    print(f"{'turn':<12} {'vad stop':>9} {'transcript':>11} {'first text':>11} {'first audio':>12}  transcript / reply")
    for n, t in turns.items():
        print(f"{n:<12} {t['speech_end_to_vad_stop_ms']!s:>9} {t['speech_end_to_transcript_ms']!s:>11} {t['speech_end_to_first_text_ms']!s:>11} "
              f"{t['speech_end_to_first_audio_ms']!s:>12}  {t['transcript'][:50]!r} / {t['reply'][:60]!r}")
    print()
    print(f"{'turn':<15}" + "".join(f"{k[:11]:>12}" for k in STAGES) + f"{'1st voiced':>12}" + f"{'end':>9}")
    for n, t in turns.items():
        s = t.get("stages_ms") or {}
        print(f"{n:<15}" + "".join(f"{s.get(k)!s:>12}" for k in STAGES) + f"{t.get('speech_end_to_first_voiced_ms')!s:>12}" + f"{s.get('turn_end')!s:>9}")
    print(f"{'median':<15}" + "".join(f"{results['stage_median_ms'].get(k)!s:>12}" for k in STAGES) + f"{(results['first_voiced_ms'] or {}).get('median')!s:>12}")
    for c in checks:
        print(("PASS " if c["ok"] else "FAIL ") + c["check"] + ("" if c["ok"] else f"  ({c['detail']})"))
    print("voice e2e:", "PASSED" if results["passed"] else "FAILED")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
