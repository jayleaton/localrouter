"""A stand-in for drama-tensorfold's launcher (recipe/pilot.py) and the server it starts, for the integration tests of
tools/speech/drama3/drama3_worker.py. Plain Python, no dependencies, no GPU: it takes the launcher's arguments (with
LocalRouter's patched --external-admission), takes the lock it is given without blocking (refusing as the launcher
does when it is held), refuses as the launcher's own admission does when MemAvailable is under 88 GiB unless
--external-admission is given (MemAvailable is DRAMA3_FAKE_MEMAVAILABLE_GIB, default 83: what DGX-01 had free with
chat, Whisper and Kitten warm), prints the session line and
DRAMA_PILOT_READY, and serves the native server's speech contract on loopback until SIGTERM.

The contract follows plotarmordev/drama-tensorfold at 1a53b8a (Apache-2.0): POST /v1/audio/speech with model, input,
voice, references, response_format (wav or pcm) and stream (pcm only); a whole reply with Content-Length and
X-Finish-Reason; a stream of chunked 16-bit mono 44.1 kHz PCM whose trailers carry X-Finish-Reason when the audio was
cut short, and X-Long-Form-Segments too for a long-form reply; GET /v1/voices; 400, 503 and 500 errors in OpenAI's shape.

The audio is a constant level, 441 samples (10 ms) a character of input: 1000 for the default voice, 2000 for a named
one, plus 300 a reference. Test hooks in the input: [length] and [silence] end the audio with that finish reason,
[busy] is a 503, [fault] a 500, [slow] streams a chunk every 0.1 s, [break] a stream (or reply) that stops after its
first chunk, [crash] ends the launcher and its server at once. Every speech request's body is written, before the
reply, to RUNTIME_DIR/last_speech_request.json (the launcher's --runtime-dir), so a test can see what the worker sent.

Run as `drama3_pilot.py voice encode CLIP.wav --model M --text T --name N --out N.json` (the test's native_bin a
shell script that execs it so), it stands in for the engine's `tensorfold-native voice encode`: it writes a
drama3-voice/1 file for a PCM WAV, frames = ceil(samples / 2048), a 10 x frames grid of codes derived from the samples
(row 0 in [0, 4096), the rest in [0, 1024)), the trimmed text, codes_sha256, audio_sha256 (the clip file's) and
encoder {"name": "test-stand-in"}, and prints its summary. Like the engine, it refuses (exit 1, "tensorfold: …" on
stderr) a clip over 60 s or one that is not a PCM WAV, and its usage errors exit 2. Test hooks in the text: REFUSE-ME
is refused as a transcript the voice file loader would not take, CRASH-ME aborts, SLOW-ME takes 5 s first.
"""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import struct
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RATE = 44100
MODEL = "fishaudio/drama-3"
FIELDS = {"model", "input", "voice", "speaker", "response_format", "stream", "stream_format", "references"}
REFERENCE_FIELDS = {"text", "vq_codes", "audio", "audio_path", "audio_base64"}
LAST_REQUEST = "last_speech_request.json"  # in the launcher's --runtime-dir
FRAME = 2048  # samples a codec frame at 44.1 kHz
MAX_CLIP_S = 60
VOICE_USAGE = "usage: tensorfold-native voice encode CLIP.wav --model MODEL --text TRANSCRIPT --name NAME --out NAME.json\n"


def voice_encode(argv):
    """`tensorfold-native voice encode`'s stand-in (see the module docstring); returns the exit code."""
    def usage(why):
        sys.stderr.write(f"{VOICE_USAGE}tensorfold-native voice: error: {why}\n")
        return 2

    def fail(why):
        sys.stderr.write(f"tensorfold: {why}\n")
        return 1

    if not argv or argv[0] != "encode":
        return usage("the voice command is encode")
    clip, flags, it = None, {}, iter(argv[1:])
    for arg in it:
        if not arg.startswith("--"):
            if clip is not None:
                return usage("one clip at a time")
            clip = arg
            continue
        name, eq, value = arg.partition("=")
        if name not in ("--model", "--text", "--name", "--out"):
            return usage("unknown flag; the flags are --model, --text, --name and --out")
        if not eq:
            value = next(it, None)
            if value is None:
                return usage("a flag without its value")
        flags[name] = value
    for need, why in ((clip, "no clip"), (flags.get("--model"), "no --model (the Drama 3 checkpoint holding codec.pth)"),
                      (flags.get("--text"), "no --text (the clip's exact transcript)"), (flags.get("--name"), "no --name"),
                      (flags.get("--out"), "no --out")):
        if need is None:
            return usage(why)
    text, name, out = flags["--text"].strip(), flags["--name"], flags["--out"]
    if not text:
        return usage("--text is empty")
    if os.path.basename(out) != name + ".json":
        return usage("--out must be NAME.json: --voices names a voice by its file's stem")
    if not os.path.isfile(os.path.join(flags["--model"], "codec.pth")):
        return fail(f"{os.path.join(flags['--model'], 'codec.pth')}: FileNotFound")
    if "SLOW-ME" in text:
        time.sleep(5)
    if "CRASH-ME" in text:
        sys.stderr.write("CUDA error: an illegal memory access was encountered\n")
        sys.stderr.flush()
        os.abort()
    try:
        audio = open(clip, "rb").read()
    except OSError as e:
        return fail(f"{clip}: {type(e).__name__}")
    try:
        with wave.open(clip, "rb") as w:
            rate, channels, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            pcm = w.readframes(n)
    except (wave.Error, EOFError):
        return fail("the clip is not a RIFF/WAVE file")
    if n == 0:
        return fail("the clip holds no samples")
    seconds = n / rate
    if seconds > MAX_CLIP_S:
        return fail(f"reference audio lasts {seconds:.1f} s, longer than the {MAX_CLIP_S} s limit; shorten it")
    if "REFUSE-ME" in text:
        return fail("the voice file would not load: Drama 3 request: text: REFUSE-ME is refused by the test stand-in")
    frames = -(-(n * 44100 // rate) // FRAME)
    seed = hashlib.sha256(pcm + bytes([channels, width])).digest()
    codes = []
    for k in range(10):
        row = []
        for j in range(frames):
            v = int.from_bytes(hashlib.sha256(seed + bytes([k]) + j.to_bytes(4, "little")).digest()[:4], "little")
            row.append(v % (4096 if k == 0 else 1024))
        codes.append(row)
    doc = {"format": "drama3-voice/1", "text": text,
           "codes_sha256": hashlib.sha256(json.dumps(codes).encode()).hexdigest(),
           "audio_sha256": hashlib.sha256(audio).hexdigest(), "frames": frames,
           "encoder": {"name": "test-stand-in"}, "vq_codes": codes}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
        f.write("\n")
    summary = dict(doc)
    del summary["vq_codes"]
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    sys.stderr.write(f"tensorfold-native voice: {frames} frames encoded in 0.001 s\n")
    return 0


def main():
    if sys.argv[1:2] == ["voice"]:
        raise SystemExit(voice_encode(sys.argv[2:]))
    p = argparse.ArgumentParser()
    p.add_argument("--native-bin", required=True)
    p.add_argument("--runtime-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gpu-lock", required=True)
    p.add_argument("--evidence-dir", required=True)
    p.add_argument("--port", type=int, default=18793)
    p.add_argument("--ready-timeout", type=float, default=900)
    p.add_argument("--external-admission", action="store_true")
    p.add_argument("serve_args", nargs="*")
    args = p.parse_args()
    os.makedirs(args.evidence_dir, exist_ok=False)
    if not (os.path.isfile(args.native_bin) and os.access(args.native_bin, os.X_OK)):
        p.error(f"Missing executable server: {args.native_bin}")
    if not os.path.isfile(args.gpu_lock):
        p.error("The operator-specified shared GPU lock must already exist")
    serve = Serve(args.serve_args)
    with open(args.gpu_lock, "r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error("Another workload owns the GPU slot; no model was started")
        available = float(os.environ.get("DRAMA3_FAKE_MEMAVAILABLE_GIB", "83"))
        if not args.external_admission and available < 88:
            p.error(f"Need 88.00 GiB available, have {available:.2f}; no model was started")
        unit = f"drama3-pilot-{os.getpid()}-{time.time_ns()}.service"
        with open(os.path.join(args.evidence_dir, "session.json"), "w") as f:
            json.dump({"argv": sys.argv[1:], "unit": unit}, f)
        print(json.dumps({"backend": "native", "unit": unit}), flush=True)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        server.serve = serve
        server.last_request = os.path.join(args.runtime_dir, LAST_REQUEST)
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print("DRAMA_PILOT_READY", flush=True)
        while not stop.wait(0.2):
            pass
        server.shutdown()
        server.server_close()
        with open(os.path.join(args.evidence_dir, "completion.json"), "w") as f:
            json.dump({"ready": True, "failures": []}, f)
    print(json.dumps({"ready": True, "failures": []}), flush=True)


class Serve:
    """The server flags the launcher passes on after --."""

    def __init__(self, argv):
        self.voices, self.default_voice, self.long_form, self.long_form_chars = {}, None, False, 500
        it = iter(argv)
        for flag in it:
            if flag == "--voices":
                d = next(it)
                for name in sorted(os.listdir(d)):
                    if name.endswith(".json") and not name.startswith("."):
                        self.voices[name[:-5]] = json.load(open(os.path.join(d, name)))
            elif flag == "--default-voice":
                self.default_voice = next(it)
            elif flag == "--long-form":
                self.long_form = True
            elif flag == "--long-form-chars":
                self.long_form_chars = int(next(it))
            elif flag in ("--max-frames", "--long-form-history"):
                next(it)
            else:
                raise SystemExit(f"unknown serve flag {flag}")
        if self.default_voice is not None and self.default_voice not in self.voices:
            raise SystemExit(f"--default-voice {self.default_voice} is not a loaded voice")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("fake drama3: " + (fmt % args) + "\n")

    def reply(self, status, body, ctype="application/json", headers=()):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def error(self, status, message):
        typ = {400: "invalid_request_error", 503: "server_error"}.get(status, "server_error")
        self.reply(status, json.dumps({"error": {"message": message, "type": typ, "param": None, "code": None}}).encode())

    def do_GET(self):
        serve = self.server.serve
        if self.path == "/health":
            return self.reply(200, b'{"status":"ok"}')
        if self.path == "/v1/voices":
            data = [{"object": "voice", "name": n, "references": 1} for n in serve.voices]
            return self.reply(200, json.dumps({"object": "list", "data": data}).encode())
        self.error(404, "no such route")

    def do_POST(self):
        serve = self.server.serve
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path != "/v1/audio/speech":
            return self.error(404, "no such route")
        tmp = self.server.last_request + ".tmp"
        with open(tmp, "wb") as f:
            f.write(raw)
        os.replace(tmp, self.server.last_request)
        try:
            b = json.loads(raw)
        except ValueError:
            return self.error(400, "speech request body must be a JSON object")
        for k in b:
            if k not in FIELDS:
                return self.error(400, f"unknown speech request field: {k}")
        if b.get("model") != MODEL:
            return self.error(400, f"model {b.get('model')!r} is not served here: send {MODEL}")
        text = b.get("input")
        if not isinstance(text, str) or not text.strip():
            return self.error(400, "input must be a non-empty string")
        fmt = b.get("response_format", "wav")
        if fmt not in ("wav", "pcm"):
            return self.error(400, "response_format must be one of: pcm, wav")
        stream = b.get("stream") is True
        if stream and fmt != "pcm":
            return self.error(400, "stream=true requires response_format='pcm'")
        voice = b.get("voice") or "default"
        level = 1000
        if voice == "default" and serve.default_voice:
            voice = serve.default_voice
        if voice != "default":
            if voice not in serve.voices:
                hint = "only the default voice is supported" if not serve.voices else "use default or a voice the server loaded with --voices"
                return self.error(400, f"Drama 3 request: unknown voice {voice!r}; {hint}")
            level = 2000
        refs = b.get("references") or []
        for i, ref in enumerate(refs):
            for k in (ref if isinstance(ref, dict) else {}):
                if k not in REFERENCE_FIELDS:
                    return self.error(400, f"unknown speech reference field: references.{i}.{k}")
            codes = ref.get("vq_codes") if isinstance(ref, dict) else None
            if not isinstance(codes, list) or len(codes) != 10 or len({len(row) for row in codes}) != 1:
                return self.error(400, f"Drama 3 request: references[{i}].vq_codes must be 10 rows of equal length")
        level += 300 * len(refs)
        if "[busy]" in text:
            return self.error(503, "the engine is busy; retry shortly")
        if "[fault]" in text:
            return self.error(500, "speech synthesis failed")
        if "[crash]" in text:
            os._exit(3)
        finish = "length" if "[length]" in text else "silence" if "[silence]" in text else "stop"
        n = 441 * len(text)
        pcm = struct.pack(f"<{n}h", *([level] * n))
        segments = ""
        if serve.long_form and len(text) > serve.long_form_chars:
            segments = f"2;{441 * (len(text) // 2)}"
        if not stream:
            if "[break]" in text:
                return self.error(500, "the engine finished without complete audio")
            headers = [("X-Finish-Reason", finish)] + ([("X-Long-Form-Segments", segments)] if segments else [])
            if fmt == "pcm":
                return self.reply(200, pcm, "audio/pcm", headers)
            head = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 1, RATE, RATE * 2,
                               2, 16, b"data", len(pcm))
            return self.reply(200, head + pcm, "audio/wav", headers)
        self.send_response(200)
        self.send_header("Content-Type", "audio/pcm")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Sample-Rate", str(RATE))
        self.send_header("X-Channels", "1")
        self.send_header("X-Bit-Depth", "16")
        self.send_header("Trailer", "X-Finish-Reason, X-Long-Form-Segments" if segments else "X-Finish-Reason")
        self.end_headers()
        step = 2 * 4410
        for at in range(0, len(pcm), step):
            part = pcm[at:at + step]
            self.wfile.write(b"%x\r\n" % len(part) + part + b"\r\n")
            self.wfile.flush()
            if "[break]" in text:  # held a moment, so the daemon has begun its own stream before this one breaks
                time.sleep(0.5)
                self.close_connection = True
                return
            time.sleep(0.1 if "[slow]" in text else 0.005)
        trailers = ""
        if segments:
            trailers = f"X-Finish-Reason: {finish}\r\nX-Long-Form-Segments: {segments}\r\n"
        elif finish != "stop":
            trailers = f"X-Finish-Reason: {finish}\r\n"
        self.wfile.write(f"0\r\n{trailers}\r\n".encode())
        self.wfile.flush()


if __name__ == "__main__":
    main()
