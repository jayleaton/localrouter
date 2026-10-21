"""Speech timings against /v1/audio/speech and /v1/audio/transcriptions (stdlib only; python3 -I on the Spark).

  speech_bench.py URL MODEL stream RUNS RATE OUTDIR TEXT   streamed raw PCM (s16le mono at RATE): first-audio latency, RTF
  speech_bench.py URL MODEL whole  RUNS OUTDIR TEXT        whole WAV replies: wall time, audio seconds, RTF; keeps one WAV
  speech_bench.py URL MODEL stt    RUNS WAV                transcription of WAV: wall time, RTF, text

RTF is wall time / audio seconds (lower is faster). One JSON line per run on stdout.
"""
import http.client, json, os, struct, sys, time, urllib.parse, uuid

URL, MODEL, CMD = sys.argv[1], sys.argv[2], sys.argv[3]
u = urllib.parse.urlsplit(URL)


def req(path, body, ctype):
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
    c.request("POST", path, body, {"Content-Type": ctype, "Host": u.netloc})
    return c, c.getresponse()


def wav_seconds(b):
    # walk the RIFF chunks for fmt and data
    i, rate, ch, bits, n = 12, None, None, None, None
    while i + 8 <= len(b):
        cid, size = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        if cid == b"fmt ":
            ch, rate = struct.unpack("<HI", b[i + 10:i + 16]); bits = struct.unpack("<H", b[i + 22:i + 24])[0]
        elif cid == b"data":
            n = min(size, len(b) - i - 8)
            break
        i += 8 + size + (size & 1)
    return n / (rate * ch * bits // 8), rate


def out(**kw):
    print(json.dumps(kw), flush=True)


if CMD == "stream":
    runs, rate, outdir, text = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6], sys.argv[7]
    os.makedirs(outdir, exist_ok=True)
    for i in range(runs):
        body = json.dumps({"model": MODEL, "input": text, "response_format": "pcm", "stream": True})
        t0 = time.perf_counter()
        c, r = req("/v1/audio/speech", body, "application/json")
        if r.status != 200:
            raise SystemExit(f"HTTP {r.status}: {r.read()[:500]!r}")
        first, n, pcm = None, 0, bytearray()
        while True:
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            first = first or time.perf_counter()
            pcm += chunk
        wall = time.perf_counter() - t0
        audio_s = len(pcm) / (2 * rate)
        hdr = {k.lower(): v for k, v in r.getheaders() if k.lower().startswith("x-")}
        if i == 0:
            open(os.path.join(outdir, "stream0.pcm"), "wb").write(pcm)
        out(cmd="stream", run=i, first_audio_s=first - t0 if first else None, wall_s=wall, audio_s=audio_s,
            rtf=wall / audio_s if audio_s else None, headers=hdr)
elif CMD == "whole":
    runs, outdir, text = int(sys.argv[4]), sys.argv[5], sys.argv[6]
    os.makedirs(outdir, exist_ok=True)
    for i in range(runs):
        body = json.dumps({"model": MODEL, "input": text, "response_format": "wav"})
        t0 = time.perf_counter()
        c, r = req("/v1/audio/speech", body, "application/json")
        data = r.read()
        wall = time.perf_counter() - t0
        if r.status != 200:
            raise SystemExit(f"HTTP {r.status}: {data[:500]!r}")
        audio_s, rate = wav_seconds(data)
        if i == 0:
            open(os.path.join(outdir, "whole0.wav"), "wb").write(data)
        out(cmd="whole", run=i, wall_s=wall, audio_s=audio_s, rate=rate, rtf=wall / audio_s, bytes=len(data))
elif CMD == "stt":
    runs, path = int(sys.argv[4]), sys.argv[5]
    audio = open(path, "rb").read()
    audio_s, _ = wav_seconds(audio)
    for i in range(runs):
        b = uuid.uuid4().hex
        parts = [f'--{b}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{MODEL}\r\n'.encode(),
                 f'--{b}\r\nContent-Disposition: form-data; name="response_format"\r\n\r\njson\r\n'.encode(),
                 f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
                 + audio + f"\r\n--{b}--\r\n".encode()]
        t0 = time.perf_counter()
        c, r = req("/v1/audio/transcriptions", b"".join(parts), f"multipart/form-data; boundary={b}")
        data = r.read()
        wall = time.perf_counter() - t0
        if r.status != 200:
            raise SystemExit(f"HTTP {r.status}: {data[:500]!r}")
        out(cmd="stt", run=i, wall_s=wall, audio_s=audio_s, rtf=wall / audio_s, text=json.loads(data).get("text"))
else:
    raise SystemExit(__doc__)
