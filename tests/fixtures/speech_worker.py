"""A minimal external worker for the integration tests: the `cmd` side of the worker protocol for the speech,
transcription and asset3d kinds, in plain Python with no dependencies. It shows what an adapter (Whisper, Kitten, ...) does:
reply to load with its resident bytes, write speech.wav or transcript.json into the job directory, report progress,
refuse bad requests with invalid_request_error, and release everything on exit.

Test hooks: speech input "crash" exits the process mid-job. The tool's `options` (from LOCALROUTER_TOOL) may set
`phrases`, a list of {"hz", "text"}: the transcript is then the phrase whose hz is within 20% of the input's pitch
(zero crossings of the samples above 200 in magnitude), else "" with no segments, so a test picks what was "said" by
the tone it sends. `speak_ms` and `hear_ms` sleep that long in a speech or transcription job (slow models). It offers
the soft cancel: SIGUSR1 ends the job in progress with {"ok": false, "type": "cancelled"} and keeps it loaded; each
load logs "speech worker loaded" to stderr (the tool's log), so a test can tell a cancel from a reload.

An asset3d request writes what an image_to_3d tool does: `asset.ply` (a binary 3DGS PLY of `num_gaussians` rows, 1024
by default), `input.png` and the `asset.json` manifest. `make_ms` sleeps that long (in 10 ms steps) first; the image
name "broken.png" writes a manifest that names a file it did not write, for the daemon's check.
"""

import json
import os
import zlib
import signal
import struct
import sys
import time
import wave


def tool_options():
    opts = json.loads(os.environ.get("LOCALROUTER_TOOL") or "{}").get("options")
    return opts if isinstance(opts, dict) else {}


OPTIONS = tool_options()


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def pause(key):
    ms = OPTIONS.get(key) or 0
    if ms > 0:
        time.sleep(ms / 1000)


def speak(job_dir, r):
    if r["input"] == "crash":
        os._exit(3)
    voice = r.get("voice") or "plain"
    if voice not in ("plain", "bright"):
        return {"ok": False, "type": "invalid_request_error", "error": f"no voice '{voice}'"}
    pause("speak_ms")
    rate, n = 22050, int(22050 * 0.05 * len(r["input"]) / r.get("speed", 1.0))
    with wave.open(os.path.join(job_dir, "speech.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n}h", *([1000 if voice == "bright" else 500] * n)))
    return {"ok": True, "files": ["speech.wav"], "seed": r.get("seed", 0), "ms": 1}


def phrase(pcm, phrases):
    voiced = [s for s in struct.unpack(f"<{len(pcm) // 2}h", pcm) if abs(s) > 200]
    if not voiced:
        return ""
    crossings = sum(1 for a, b in zip(voiced, voiced[1:]) if (a < 0) != (b < 0))
    hz = crossings / 2 / (len(voiced) / 16000)
    for p in phrases:
        if abs(hz - p["hz"]) <= 0.2 * p["hz"]:
            return p["text"]
    return ""


def transcribe(job_dir, r):
    with wave.open(os.path.join(job_dir, r["audio"]), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            return {"ok": False, "type": "invalid_request_error", "error": "audio.wav is not 16 kHz mono 16-bit"}
        frames = w.getnframes()
        pcm = w.readframes(frames)
    pause("hear_ms")
    send({"progress": {"phase": "transcribe", "step": 1, "of": 1}})
    seconds = frames / 16000
    phrases = OPTIONS.get("phrases")
    text = phrase(pcm, phrases) if phrases is not None else f"external worker heard {frames} samples"
    segments = [{"id": 0, "start": 0.0, "end": seconds, "text": " " + text, "avg_logprob": -0.1}] if text else []
    out = {"text": text, "language": r.get("language") or "en", "duration": seconds, "segments": segments}
    if r.get("timestamps") == "word":
        out["words"] = [{"word": "external", "start": 0.0, "end": seconds / 2}]
    with open(os.path.join(job_dir, "transcript.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)
    return {"ok": True, "files": ["transcript.json"], "seed": r.get("seed", 0), "ms": 1}


def png(w, h, rgb):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def make3d(job_dir, r):
    with open(os.path.join(job_dir, r["image"]), "rb") as f:
        picture = f.read()
    steps = int((OPTIONS.get("make_ms") or 0) / 10)
    for i in range(steps):
        time.sleep(0.01)
        send({"progress": {"phase": "denoise", "step": i + 1, "of": steps}})
    n = r.get("num_gaussians") or 1024
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    head = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n%send_header\n" % (n, "".join("property float %s\n" % p for p in props))
    row = struct.pack("<17f", 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5, 2, -4.6, -4.6, -4.6, 1, 0, 0, 0)
    with open(os.path.join(job_dir, "asset.ply"), "wb") as f:
        f.write(head.encode() + row * n)
    with open(os.path.join(job_dir, "input.png"), "wb") as f:
        f.write(png(8, 8, [len(picture) % 256, 128, 64]))
    manifest = {"object": "asset3d", "version": 1, "representation": "gaussian_splat", "format": "ply",
                "file": "missing.ply" if r["image"].startswith("broken") else "asset.ply", "input": "input.png",
                "gaussians": {"count": n, "sh_degree": 0}, "model": {"name": "external worker"},
                "settings": {"input_mode": r.get("input_mode"), "seed": r.get("seed", 0)}}
    with open(os.path.join(job_dir, "asset.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return {"ok": True, "files": ["asset.ply", "asset.json", "input.png"], "seed": r.get("seed", 0), "ms": 1}


class Cancelled(Exception):
    pass


GENERATING = False


def on_cancel(signum, frame):
    """SIGUSR1: the daemon cancels the job in progress; the worker stays loaded (it said "cancel": "signal")."""
    if GENERATING:
        raise Cancelled()


def generate(cmd):
    global GENERATING
    req = cmd["request"]
    try:
        GENERATING = True  # inside the try: a signal right after it is still caught
        if "speech" in req:
            out = speak(cmd["dir"], req["speech"])
        elif "transcription" in req:
            out = transcribe(cmd["dir"], req["transcription"])
        elif "asset3d" in req:
            out = make3d(cmd["dir"], req["asset3d"])
        else:
            out = {"ok": False, "type": "invalid_request_error", "error": "not a speech request"}
    except Cancelled:
        out = {"ok": False, "type": "cancelled", "error": "cancelled"}
    finally:
        GENERATING = False
    return out


def main():
    signal.signal(signal.SIGUSR1, on_cancel)
    held = None
    for line in sys.stdin:
        cmd = json.loads(line)
        op = cmd["op"]
        if op == "load":
            held = bytearray(8 << 20)
            print("speech worker loaded", file=sys.stderr, flush=True)
            send({"ok": True, "resident": len(held), "cancel": "signal"})
        elif op == "generate":
            send(generate(cmd))
        elif op == "exit":
            held = None
            return
    del held


if __name__ == "__main__":
    main()
