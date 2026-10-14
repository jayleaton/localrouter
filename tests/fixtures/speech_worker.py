"""A minimal external speech worker for the integration tests: the `cmd` side of the worker protocol for the speech
and transcription kinds, in plain Python with no dependencies. It shows what an adapter (Whisper, Kitten, ...) does:
reply to load with its resident bytes, write speech.wav or transcript.json into the job directory, report progress,
refuse bad requests with invalid_request_error, and release everything on exit.

Test hook: speech input "crash" exits the process mid-job.
"""

import json
import os
import struct
import sys
import wave


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def speak(job_dir, r):
    if r["input"] == "crash":
        os._exit(3)
    voice = r.get("voice") or "plain"
    if voice not in ("plain", "bright"):
        return {"ok": False, "type": "invalid_request_error", "error": f"no voice '{voice}'"}
    rate, n = 22050, int(22050 * 0.05 * len(r["input"]) / r.get("speed", 1.0))
    with wave.open(os.path.join(job_dir, "speech.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{n}h", *([1000 if voice == "bright" else 500] * n)))
    return {"ok": True, "files": ["speech.wav"], "seed": r.get("seed", 0), "ms": 1}


def transcribe(job_dir, r):
    with wave.open(os.path.join(job_dir, r["audio"]), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (16000, 1, 2):
            return {"ok": False, "type": "invalid_request_error", "error": "audio.wav is not 16 kHz mono 16-bit"}
        frames = w.getnframes()
    send({"progress": {"phase": "transcribe", "step": 1, "of": 1}})
    seconds = frames / 16000
    text = f"external worker heard {frames} samples"
    out = {"text": text, "language": r.get("language") or "en", "duration": seconds,
           "segments": [{"id": 0, "start": 0.0, "end": seconds, "text": " " + text, "avg_logprob": -0.1}]}
    if r.get("timestamps") == "word":
        out["words"] = [{"word": "external", "start": 0.0, "end": seconds / 2}]
    with open(os.path.join(job_dir, "transcript.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)
    return {"ok": True, "files": ["transcript.json"], "seed": r.get("seed", 0), "ms": 1}


def main():
    held = None
    for line in sys.stdin:
        cmd = json.loads(line)
        op = cmd["op"]
        if op == "load":
            held = bytearray(8 << 20)
            send({"ok": True, "resident": len(held)})
        elif op == "generate":
            req = cmd["request"]
            if "speech" in req:
                send(speak(cmd["dir"], req["speech"]))
            elif "transcription" in req:
                send(transcribe(cmd["dir"], req["transcription"]))
            else:
                send({"ok": False, "type": "invalid_request_error", "error": "not a speech request"})
        elif op == "exit":
            held = None
            return
    del held


if __name__ == "__main__":
    main()
