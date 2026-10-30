"""Feature qualification of a speech_to_text tool through a running LocalRouter daemon: real audio over HTTP and MCP,
transcripts against the human references, the limits (silence, noise, long, malformed, refusals), cold and warm
timings with RTF, resident memory, cancellation and unload reclaiming memory. Standard library only.

    python3 qualify.py --url http://127.0.0.1:8190 --model whisper-large-v3-turbo \\
        --fixtures ~/.cache/localrouter-speech/fixtures --report out/qualify.json

The fixtures directory is tools/speech/fixtures' output (refs.json plus the files). MemAvailable is read from this
machine's /proc/meminfo, so run it on the daemon's machine. Exits 1 when a hard check fails; WER/CER are reported
against the thresholds given (--max-en-wer, --max-cer), which are acceptance targets, not numerical proof: the
bit-level proof is twin.py.
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid


def mem_available():
    """MemAvailable on Linux; on macOS (no /proc) vm_stat's free, inactive, speculative and purgeable pages, the
    memory the VM can hand out without paging (file cache counted as free, as MemAvailable does)."""
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        return 0
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = {k.strip(): int(v.strip().rstrip(".")) for k, v in (l.split(":", 1) for l in out.splitlines()[1:] if ":" in l)}
    return page * sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))


class Daemon:
    def __init__(self, url, timeout):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def request(self, method, path, body=None, ctype="application/json"):
        req = urllib.request.Request(self.url + path, data=body, method=method)
        if body is not None:
            req.add_header("content-type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def json(self, method, path, obj=None):
        status, _, body = self.request(method, path, None if obj is None else json.dumps(obj).encode())
        return status, json.loads(body or b"null")

    def transcribe(self, data, filename, fields):
        boundary = uuid.uuid4().hex
        parts = []
        for k, v in fields:
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                     "Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        return self.request("POST", "/v1/audio/transcriptions", b"".join(parts), f"multipart/form-data; boundary={boundary}")

    def mcp(self, tool, args):
        status, body = self.json("POST", "/mcp", {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                                 "params": {"name": tool, "arguments": args}})
        if status != 200 or "result" not in body:
            raise RuntimeError(f"MCP {tool}: {status} {body}")
        return body["result"]

    def tool(self, model):
        _, body = self.json("GET", "/v1/tools")
        return next(t for t in body["data"] if t["id"] == model)


# ---------------------------------------------------------------- scoring

def normalize(text, lang):
    """Lowercase, strip punctuation and symbols, collapse spaces; NFKC. A plain normalizer (not Whisper's English
    spelling normalizer), so WER is slightly pessimistic on numbers and contractions."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return " ".join(text.split())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def scores(hyp, ref, lang):
    h, r = normalize(hyp, lang), normalize(ref, lang)
    wer = edit_distance(h.split(), r.split()) / max(1, len(r.split()))
    hc, rc = h.replace(" ", ""), r.replace(" ", "")
    cer = edit_distance(list(hc), list(rc)) / max(1, len(rc))
    return round(wer, 4), round(cer, 4)


# ---------------------------------------------------------------- checks

class Report:
    def __init__(self):
        self.checks, self.fixtures, self.failures = [], [], 0

    def check(self, name, ok, **detail):
        self.checks.append({"check": name, "ok": bool(ok), **detail})
        if not ok:
            self.failures += 1
        print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, ensure_ascii=False)[:300]}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8190")
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--only", help="comma list of fixture names (default: all)")
    ap.add_argument("--max-en-wer", type=float, default=0.10)
    ap.add_argument("--max-cer", type=float, default=0.15, help="for languages without spaces (ja, zh)")
    ap.add_argument("--max-ml-wer", type=float, default=0.20, help="other languages")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--skip-cancel", action="store_true")
    a = ap.parse_args()
    d = Daemon(a.url, a.timeout)
    rep = Report()
    refs = json.load(open(os.path.join(a.fixtures, "refs.json")))
    only = set(a.only.split(",")) if a.only else None

    def path_of(name, ref):
        for k in ("file", "path", "wav"):
            if k in ref:
                p = ref[k]
                return p if os.path.isabs(p) else os.path.join(a.fixtures, p)
        stem, _, ext = name.rpartition("_")  # en_44k1_stereo_mp3 -> en_44k1_stereo.mp3
        for p in (name + ".wav", name, f"{stem}.{ext}"):
            p = os.path.join(a.fixtures, p)
            if os.path.exists(p):
                return p
        return os.path.join(a.fixtures, name)

    # Cold: everything released, then the first request loads the model.
    d.json("POST", "/v1/tools/release")
    time.sleep(2)
    mem_idle = mem_available()
    rep.check("released before start", d.tool(a.model)["state"] == "unloaded", mem_available=mem_idle)

    status, models = d.json("GET", "/v1/models")
    entry = next((m for m in models["data"] if m["id"] == a.model), None)
    rep.check("model listed as transcription with speech_to_text", entry and entry["kind"] == "transcription"
              and "speech_to_text" in entry["capabilities"], entry=entry)

    first = True
    loaded_mem = None
    for name, ref in refs.items():
        if only and name not in only:
            continue
        p = path_of(name, ref)
        if not os.path.exists(p):
            rep.check(f"fixture present: {name}", False, path=p)
            continue
        data = open(p, "rb").read()
        if ref.get("malformed"):  # every one refused with a reason, a WAV cut short included: nothing is half heard
            status, _, body = d.transcribe(data, os.path.basename(p), [("model", a.model)])
            reason = json.loads(body).get("error", {}).get("message", "") if status == 400 else ""
            rep.check(f"malformed refused: {name}", status == 400 and bool(reason), status=status,
                      body=body[:200].decode("utf-8", "replace"))
            continue
        lang = ref.get("language") or ""
        fields = [("model", a.model), ("response_format", "verbose_json"), ("timestamp_granularities[]", "segment"), ("seed", "0")]
        t0 = time.monotonic()
        status, _, body = d.transcribe(data, os.path.basename(p), fields)
        wall = time.monotonic() - t0
        if status != 200:
            rep.check(f"transcribed: {name}", False, status=status, body=body[:300].decode("utf-8", "replace"))
            continue
        out = json.loads(body)
        dur = out.get("duration") or ref.get("duration_s") or 0
        row = {"fixture": name, "language_ref": lang, "language": out.get("language"), "duration_s": dur,
               "wall_s": round(wall, 3), "rtf": round(wall / dur, 4) if dur else None, "cold": first,
               "text": out["text"], "segments": len(out.get("segments", []))}
        if first:
            loaded_mem = mem_available()
            tool = d.tool(a.model)
            row["resident_bytes"] = tool["resident_bytes"]
            rep.check("cold request loaded the model", tool["state"] in ("ready", "busy"), wall_s=round(wall, 2),
                      resident_bytes=tool["resident_bytes"], mem_drop_bytes=mem_idle - loaded_mem)
            first = False
        segs = out.get("segments", [])
        mono = all(s["start"] <= s["end"] for s in segs) and all(x["start"] <= y["start"] + 1e-6 for x, y in zip(segs, segs[1:]))
        rep.check(f"segments ordered: {name}", mono, segments=len(segs))
        ref_text = ref.get("text", "")
        if ref_text:
            wer, cer = scores(out["text"], ref_text, lang)
            row.update(wer=wer, cer=cer, reference=ref_text)
            if lang in ("ja", "zh"):
                rep.check(f"CER {name}", cer <= a.max_cer, cer=cer, limit=a.max_cer)
            else:
                limit = a.max_en_wer if lang == "en" else a.max_ml_wer
                rep.check(f"WER {name}", wer <= limit, wer=wer, limit=limit)
            if lang:
                rep.check(f"language detected: {name}", out.get("language") == lang, got=out.get("language"), want=lang)
        else:  # silence and noise: no speech invented, not a word
            rep.check(f"no speech invented: {name}", out["text"].strip() == "" and not out.get("segments"),
                      text=out["text"], segments=len(out.get("segments") or []))
        rep.fixtures.append(row)

    # Warm: the same English file again (the model stays loaded).
    en = next((n for n, r in refs.items() if r.get("language") == "en" and not r.get("malformed") and n.startswith("en_librispeech")
               and (not only or n in only)), None)
    if en:
        data = open(path_of(en, refs[en]), "rb").read()
        walls, texts = [], []
        for _ in range(3):
            t0 = time.monotonic()
            status, _, body = d.transcribe(data, "warm.wav", [("model", a.model), ("seed", "0")])
            walls.append(round(time.monotonic() - t0, 3))
            texts.append(json.loads(body)["text"] if status == 200 else None)
        rep.check("warm repeats identical", len(set(texts)) == 1 and texts[0] is not None, walls_s=walls,
                  duration_s=refs[en].get("duration_s"), rtf=[round(w / refs[en]["duration_s"], 4) for w in walls] if refs[en].get("duration_s") else None)
        # MCP and HTTP agree on the same audio.
        res = d.mcp("transcribe_audio", {"audio": base64.b64encode(data).decode(), "model": a.model, "seed": 0, "timestamps": "segment"})
        rep.check("MCP transcribe_audio equals HTTP", not res["isError"] and res["content"][0]["text"] == texts[0],
                  mcp=res["content"][0].get("text", "")[:200])
        # Word timestamps: supported (ordered words) or refused with a reason, never silently missing.
        status, _, body = d.transcribe(data, "w.wav", [("model", a.model), ("response_format", "verbose_json"), ("timestamp_granularities[]", "word"), ("seed", "0")])
        if status == 200:
            words = json.loads(body).get("words") or []
            ok = len(words) > 0 and all(w["start"] <= w["end"] for w in words)
            rep.check("word timestamps", ok, words=len(words), first=words[:3])
        else:
            rep.check("word timestamps refused with a reason", status == 400, status=status, body=body[:200].decode())
        for fmt, needle in (("text", None), ("srt", "-->"), ("vtt", "WEBVTT")):
            status, _, body = d.transcribe(data, "f.wav", [("model", a.model), ("response_format", fmt), ("seed", "0")])
            rep.check(f"response_format {fmt}", status == 200 and (needle is None or needle in body.decode()), status=status)

    # What is not offered is refused with a reason.
    status, _ = d.json("POST", "/v1/audio/translations", {})
    rep.check("translations not offered", status == 404, status=status)
    if en:
        status, _, body = d.transcribe(data, "s.wav", [("model", a.model), ("stream", "true")])
        rep.check("stream refused", status == 400, status=status)
        status, _, body = d.transcribe(data, "l.wav", [("model", a.model), ("language", "klingon")])
        rep.check("bad language refused", status == 400, status=status)

    # Cancel: a long transcription over MCP, cancelled while running. A worker without soft cancel is killed and its
    # memory comes back; one that offers it ("cancel": "signal", the macOS worker) ends the job and stays loaded.
    longest = max(((n, r) for n, r in refs.items() if not r.get("malformed") and (r.get("duration_s") or 0) > 0),
                  key=lambda x: x[1]["duration_s"], default=None)
    if longest and not a.skip_cancel:
        data = open(path_of(*longest), "rb").read()
        res = d.mcp("transcribe_audio", {"audio": base64.b64encode(data).decode(), "model": a.model, "wait_s": 0})
        job = json.loads(next(c["text"] for c in res["content"] if c["type"] == "text"))
        # cancel as soon as it runs: a fast box finishes the longest fixture in under two seconds
        deadline = time.monotonic() + 60
        while job.get("status") in ("queued", None) and time.monotonic() < deadline:
            time.sleep(0.05)
            job = json.loads(d.mcp("get_job", {"id": job["id"], "wait_s": 0})["content"][0]["text"])
        time.sleep(0.3)
        busy_mem = mem_available()
        t0 = time.monotonic()
        res = d.mcp("cancel_job", {"id": job["id"]})
        cancel_s = round(time.monotonic() - t0, 3)  # the cancel call alone (it returns when the job has ended)
        cancelled = json.loads(res["content"][0]["text"])
        time.sleep(2)
        after = mem_available()
        tool = d.tool(a.model)
        rep.check("cancel ends the running transcription", cancelled["status"] == "cancelled"
                  and tool["state"] in ("unloaded", "ready"),
                  status=cancelled["status"], state=tool["state"], cancel_s=cancel_s,
                  mem_busy=busy_mem, mem_after=after, reclaimed_bytes=after - busy_mem, idle_before_load=mem_idle)
        status, _, body = d.transcribe(open(path_of(en, refs[en]), "rb").read(), "again.wav", [("model", a.model), ("seed", "0")]) if en else (200, {}, b"")
        rep.check("next request served after cancel", status == 200, status=status)

    # Unload: release returns the memory.
    before = mem_available()
    d.json("POST", "/v1/tools/release")
    time.sleep(3)
    after = mem_available()
    rep.check("release unloads the model", d.tool(a.model)["state"] == "unloaded", mem_before=before, mem_after=after,
              reclaimed_bytes=after - before, idle_before_load=mem_idle, loaded=loaded_mem)

    out = {"model": a.model, "url": a.url, "failures": rep.failures, "checks": rep.checks, "fixtures": rep.fixtures}
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    json.dump(out, open(a.report, "w"), indent=1, ensure_ascii=False)
    print(f"{len(rep.checks) - rep.failures}/{len(rep.checks)} checks passed; report {a.report}")
    sys.exit(1 if rep.failures else 0)


if __name__ == "__main__":
    main()
