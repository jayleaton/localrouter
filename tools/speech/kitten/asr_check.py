#!/usr/bin/env python3
"""An objective check of graphs-decode audio before its baseline is accepted, for when nobody can listen.

    python3 -I tools/speech/kitten/asr_check.py URL MODEL EAGER_GATE_DIR GRAPHS_GATE_DIR OUT.json

URL is a LocalRouter daemon serving a speech-to-text MODEL (Whisper). For every fixture it transcribes the eager
worker's speech.wav (EAGER_GATE_DIR/worker/<F>/) and the graphs worker's (GRAPHS_GATE_DIR/graphs-a/<F>/), and scores
each transcript against the fixture's input text by word error rate (WER), both normalized the same way: markup
removed, lower case, punctuation dropped, numbers and currency left to the text normalizer's spelling where they match
exactly and otherwise counted as errors for both. A fixture passes when the graphs WER is at most the eager WER + 0.05
or at most 0.10, its audio is 0.75 to 1.33 times the eager length, and its share of silence (-45 dBFS, 20 ms frames)
is at most the eager share + 0.10. Exit 0 only when every fixture passes. Stdlib only.
"""

import array
import http.client
import json
import math
import os
import re
import sys
import urllib.parse
import uuid
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qualify import FIXTURES  # noqa: E402

UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def spell(n):
    """An integer below a million in English words (enough for the fixtures' numbers)."""
    if n < 20:
        return UNITS[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + UNITS[n % 10])
    if n < 1000:
        return UNITS[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + spell(n % 100))
    return spell(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + spell(n % 1000))


def normalize(text):
    text = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", text)  # emotion and event markup
    text = text.replace("(", " ").replace(")", " ").lower()
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", text)
    text = re.sub(r"\$(\d+)\.(\d\d)", lambda m: f"{m[1]} dollars {int(m[2])}", text)
    text = re.sub(r"(\d+)%", r"\1 percent", text)
    text = re.sub(r"\d+", lambda m: spell(int(m[0])) if int(m[0]) < 10**6 else m[0], text)
    text = re.sub(r"[^\w\s']", " ", text)
    return text.split()


def wer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(1, len(ref))


def audio(path):
    with wave.open(path) as w:
        rate, n = w.getframerate(), w.getnframes()
        samples = array.array("h", w.readframes(n))
    frame = int(rate * 0.02)
    floor = 32767 * 10 ** (-45 / 20)
    frames = [samples[i:i + frame] for i in range(0, len(samples) - frame + 1, frame)]
    silent = sum(1 for f in frames if math.sqrt(sum(s * s for s in f) / len(f)) < floor)
    return n / rate, silent / max(1, len(frames))


def transcribe(url, model, path):
    u = urllib.parse.urlsplit(url)
    b = uuid.uuid4().hex
    body = b"".join([
        f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'.encode(),
        f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        f'Content-Type: audio/wav\r\n\r\n'.encode()
        + open(path, "rb").read() + f"\r\n--{b}--\r\n".encode()])
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=600)
    c.request("POST", "/v1/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    r = c.getresponse()
    data = r.read()
    if r.status != 200:
        raise SystemExit(f"transcription of {path}: HTTP {r.status} {data[:300]!r}")
    return json.loads(data)["text"]


def main():
    url, model, eager_dir, graphs_dir, out = sys.argv[1:6]
    results = []
    for name, f in FIXTURES.items():
        ref = normalize(f["input"])
        row = {"fixture": name, "input": f["input"]}
        for label, path in (("eager", os.path.join(eager_dir, "worker", name, "speech.wav")),
                            ("graphs", os.path.join(graphs_dir, "graphs-a", name, "speech.wav"))):
            text = transcribe(url, model, path)
            seconds, silence = audio(path)
            row[label] = {"transcript": text, "wer": round(wer(ref, normalize(text)), 3), "seconds": round(seconds, 2),
                          "silence_share": round(silence, 3)}
        e, g = row["eager"], row["graphs"]
        row["duration_ratio"] = round(g["seconds"] / e["seconds"], 3)
        row["pass"] = (g["wer"] <= max(e["wer"] + 0.05, 0.10) and 0.75 <= row["duration_ratio"] <= 1.33
                       and g["silence_share"] <= e["silence_share"] + 0.10)
        results.append(row)
        print(json.dumps(row), flush=True)
    report = {"pass": all(r["pass"] for r in results), "model": model, "fixtures": results}
    with open(out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({"pass": report["pass"], "failed": [r["fixture"] for r in results if not r["pass"]]}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
