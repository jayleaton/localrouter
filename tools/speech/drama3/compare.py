#!/usr/bin/env python3
"""Side-by-side comparison of speech models a running LocalRouter daemon serves: speed, accuracy and memory.

    python3 -I tools/speech/drama3/compare.py --url http://127.0.0.1:8190 --models A B [--voices V ...] \
        --texts FILE --out DIR [--stt MODEL] [--chat MODEL] [--chat-prompt TEXT] [--load-texts N] \
        [--server-match SUBSTR ...] [--key KEY]

It speaks to the daemon over HTTP only, on the same host. For each model in order it unloads the other listed models
(POST /v1/tools/<id>/unload; 404 and 409 are fine), warms this one with a short request (its wall time is load_s),
then streams every voice x text as raw PCM (/v1/audio/speech, response_format "pcm", stream true), timing the first
PCM byte from the request's send and the whole reply. With --chat it repeats the first --load-texts texts while a
background thread streams chat completions back to back. Once every model has spoken, each alone take is transcribed
(/v1/audio/transcriptions) and scored by word error rate against its text. With --server-match a sampler reads, every
0.5 s, the processes whose /proc/PID/cmdline holds every substring: nvidia-smi's per-process used_memory plus RssAnon
and RssShmem (the GB10 per-process memory method), keeping the peaks per model phase.

The first model is the baseline. DIR receives takes.jsonl (one record a request), report.json, report.md,
listen.html (a page to listen side by side) and the audio as wav/<voice>/<NN>-<model>.wav. Percentiles are
nearest-rank over the takes that have the value, as in the qualification's stats.
"""

import argparse
import collections
import html
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
import unicodedata
import urllib.parse
import wave

DEFAULT_RATE = 44100  # Drama 3's rate; a reply's x-sample-rate header wins
WARM_TEXT = "Hello there."
CHAT_PROMPT = ("Write a long, detailed short story about a lighthouse keeper who finds a message in a bottle. "
               "Describe the weather, the sea and the keeper's daily routine in full sentences.")
# A voice of a later model is flagged against the baseline when its WER mean is at least this much worse and more than
# one of its takes is worse, when any of its takes failed or came back empty, or when its RTF p50 is above real time.
THRESHOLDS = {"wer_mean_worse_by": 0.03, "takes_worse_min": 2, "rtf_p50_max": 1.0}


class Http:
    """One HTTP/1.1 exchange, by hand: http.client drops a stream's trailers, which carry x-finish-reason."""

    def __init__(self, args, method, path, body=None, ctype="application/json", timeout=None):
        u = args.base
        self.sock = socket.create_connection((u.hostname, u.port or 80), timeout=timeout or args.timeout)
        head = f"{method} {args.prefix}{path} HTTP/1.1\r\nHost: {u.netloc}\r\nConnection: close\r\n"
        if args.key:
            head += f"Authorization: Bearer {args.key}\r\n"
        if body is not None:
            head += f"Content-Type: {ctype}\r\nContent-Length: {len(body)}\r\n"
        self.sent = time.monotonic()
        self.sock.sendall(head.encode() + b"\r\n" + (body or b""))
        self.r = self.sock.makefile("rb")
        parts = self.r.readline().split(None, 2)
        if len(parts) < 2 or not parts[0].startswith(b"HTTP/1."):
            raise OSError("the daemon closed the connection without a reply")
        self.status = int(parts[1])
        self.headers = self._fields()
        self.trailers = {}

    def _fields(self):
        fields = {}
        while True:
            line = self.r.readline()
            if line in (b"\r\n", b"\n", b""):
                return fields
            name, _, value = line.decode("latin-1").partition(":")
            fields[name.strip().lower()] = value.strip()

    def chunks(self):
        """The body's pieces as they arrive; a chunked body that ends before its last chunk raises EOFError."""
        if self.headers.get("transfer-encoding", "").lower() == "chunked":
            while True:
                line = self.r.readline()
                if not line.endswith(b"\n"):
                    raise EOFError("the stream ended before its last chunk")
                size = int(line.split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.trailers = self._fields()
                    return
                data = self.r.read(size)
                if len(data) < size or self.r.read(2) != b"\r\n":
                    raise EOFError("the stream ended inside a chunk")
                yield data
        elif "content-length" in self.headers:
            want = int(self.headers["content-length"])
            data = self.r.read(want)
            if len(data) < want:
                raise EOFError(f"the reply ended after {len(data)} of {want} bytes")
            yield data
        else:
            yield self.r.read()

    def body(self):
        return b"".join(self.chunks())

    def close(self):
        try:
            self.r.close()
            self.sock.close()
        except OSError:
            pass


def call(args, method, path, body=None, ctype="application/json"):
    """(status, body) of a whole exchange; status None when it failed on the way."""
    h = None
    try:
        h = Http(args, method, path, body, ctype)
        return h.status, h.body()
    except (OSError, EOFError, ValueError) as e:
        return None, f"{type(e).__name__}: {e}".encode()
    finally:
        if h:
            h.close()


def slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "x"


def speak(args, model, voice, text, rate, wav_path=None):
    """One streamed speech request, measured; the PCM goes to wav_path when it is a good take."""
    rec = {"model": model, "voice": voice, "input": text, "status": None, "ok": False, "complete": False,
           "first_audio_s": None, "total_s": None, "samples": 0, "sample_rate": rate, "audio_s": None, "rtf": None,
           "finish_reason": None, "finish_from": None, "error": None, "wav": None}
    body = json.dumps({"model": model, "input": text, "voice": voice, "response_format": "pcm",
                       "stream": True}).encode()
    pcm, h, start = bytearray(), None, time.monotonic()
    try:
        h = Http(args, "POST", "/v1/audio/speech", body)
        start, rec["status"] = h.sent, h.status
        if h.status != 200:
            rec["error"] = h.body()[:300].decode("utf-8", "replace")
        else:
            rec["sample_rate"] = rate = int(h.headers.get("x-sample-rate") or rate)
            if h.headers.get("x-channels", "1") != "1":
                rec["error"] = f"x-channels {h.headers['x-channels']}"
            for data in h.chunks():
                if data and rec["first_audio_s"] is None:
                    rec["first_audio_s"] = time.monotonic() - start
                pcm += data
                if time.monotonic() - start > args.timeout:
                    raise TimeoutError(f"no last chunk within {args.timeout} s")
            rec["complete"] = True
            for where, fields in (("trailer", h.trailers), ("header", h.headers)):
                if fields.get("x-finish-reason"):
                    rec["finish_reason"], rec["finish_from"] = fields["x-finish-reason"], where
                    break
            else:  # AGENTS.md: a plain last chunk means the audio ended by itself
                rec["finish_reason"], rec["finish_from"] = "stop", "plain last chunk"
    except (OSError, EOFError, ValueError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        if h:
            h.close()
    rec["total_s"] = time.monotonic() - start
    rec["samples"] = len(pcm) // 2
    if rec["samples"]:
        rec["audio_s"] = rec["samples"] / rate
        rec["rtf"] = rec["total_s"] / rec["audio_s"]
    rec["ok"] = rec["status"] == 200 and rec["complete"] and rec["samples"] > 0 and not rec["error"]
    if rec["samples"] and wav_path:
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(bytes(pcm[:rec["samples"] * 2]))
        rec["wav"] = os.path.relpath(wav_path, args.out)
    return rec


def transcribe(args, path):
    """(text, error) from /v1/audio/transcriptions, multipart as OpenAI's API takes it."""
    b = "compare" + os.urandom(8).hex()
    fields = [("model", args.stt), ("response_format", "json")] + ([("language", args.language)] if args.language else [])
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields)
    with open(path, "rb") as f:
        body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(path)}"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n").encode() + f.read() + f"\r\n--{b}--\r\n".encode()
    status, reply = call(args, "POST", "/v1/audio/transcriptions", body, f"multipart/form-data; boundary={b}")
    if status != 200:
        return None, f"{status}: {reply[:300].decode('utf-8', 'replace')}"
    try:
        return json.loads(reply).get("text", ""), None
    except ValueError as e:
        return None, f"not JSON: {e}"


def words(text):
    """Lowercased NFKC words: bracketed tags like [sighs] and punctuation dropped, intra-word apostrophes kept."""
    text = unicodedata.normalize("NFKC", re.sub(r"\[[^\]]*\]", " ", text)).lower()
    text = text.replace("’", "'").replace("‘", "'")
    out = []
    for i, c in enumerate(text):
        if c == "'" and 0 < i < len(text) - 1 and text[i - 1].isalnum() and text[i + 1].isalnum():
            out.append(c)
        else:
            out.append(" " if unicodedata.category(c)[0] in "PS" else c)
    return "".join(out).split()


def edits(ref, hyp):
    """Levenshtein distance over words."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[max(0, math.ceil(p * len(xs)) - 1)] if xs else None


def stats(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "max": max(xs) if xs else None}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


class ChatLoad:
    """Streams chat completions back to back until stopped; counts tokens and their time."""

    def __init__(self, args):
        self.args, self.stop = args, threading.Event()
        self.requests, self.errors = [], []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def one(self):
        body = json.dumps({"model": self.args.chat, "stream": True, "max_tokens": 600,
                           "stream_options": {"include_usage": True},
                           "messages": [{"role": "user", "content": self.args.chat_prompt}]}).encode()
        h = Http(self.args, "POST", "/v1/chat/completions", body, timeout=60)
        try:
            if h.status != 200:
                self.errors.append(f"{h.status}: {h.body()[:200].decode('utf-8', 'replace')}")
                self.stop.wait(2)
                return
            deltas, usage, first, buf = 0, None, None, b""
            for data in h.chunks():
                buf += data
                *lines, buf = buf.split(b"\n")
                for line in lines:
                    if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                        continue
                    ev = json.loads(line[5:])
                    usage = (ev.get("usage") or {}).get("completion_tokens") or usage
                    d = ((ev.get("choices") or [{}])[0].get("delta")) or {}
                    if d.get("content") or d.get("reasoning_content"):
                        deltas += 1
                        first = first or time.monotonic()
                if self.stop.is_set():
                    break
            if first:  # usage when the stream finished, else one token a content delta
                self.requests.append({"tokens": usage or deltas, "decode_s": time.monotonic() - first})
        finally:
            h.close()

    def run(self):
        while not self.stop.is_set():
            try:
                self.one()
            except (OSError, EOFError, ValueError) as e:
                self.errors.append(f"{type(e).__name__}: {e}")
                self.stop.wait(2)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=70)
        tokens, secs = sum(r["tokens"] for r in self.requests), sum(r["decode_s"] for r in self.requests)
        return {"model": self.args.chat, "requests": len(self.requests), "tokens": tokens,
                "tok_s": tokens / secs if secs else None, "errors": self.errors[:5],
                "stopped": not self.thread.is_alive()}


def status_kb(pid, name):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith(name + ":"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


class MemorySampler:
    """Every 0.5 s: the matching processes' nvidia-smi used_memory, RssAnon and RssShmem; peaks per phase."""

    def __init__(self, match):
        self.match, self.phase, self.peaks = match, None, {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def pids(self):
        found = []
        for p in os.listdir("/proc"):
            if not p.isdigit() or int(p) == os.getpid():
                continue
            try:
                with open(f"/proc/{p}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if all(s in cmd for s in self.match):
                found.append(int(p))
        return found

    def sample(self):
        gpu = {}
        try:
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            for line in r.stdout.splitlines():
                pid, _, mib = line.partition(",")
                if pid.strip().isdigit() and mib.strip().isdigit():
                    gpu[int(pid)] = gpu.get(int(pid), 0) + int(mib) * (1 << 20)
        except (OSError, subprocess.TimeoutExpired):
            pass
        pids = self.pids()
        s = {"gpu_bytes": sum(gpu.get(p, 0) for p in pids), "rss_anon_bytes": sum(status_kb(p, "RssAnon") for p in pids),
             "rss_shmem_bytes": sum(status_kb(p, "RssShmem") for p in pids)}
        s["total_bytes"] = sum(s.values())
        peak = self.peaks.setdefault(self.phase, {"samples": 0, "pids_max": 0})
        peak["samples"] += 1
        peak["pids_max"] = max(peak["pids_max"], len(pids))
        for k, v in s.items():
            peak[k] = max(peak.get(k, 0), v)

    def run(self):
        while not self.stop.is_set():
            if self.phase:
                self.sample()
            self.stop.wait(0.5)

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=15)
        return self.peaks


def summarize(model, takes, load, chat, memory):
    mine = [t for t in takes if t["model"] == model]
    alone = [t for t in mine if t["pass"] == "alone"]
    loaded = [t for t in mine if t["pass"] == "chat"]
    ok = lambda ts: [t for t in ts if t["ok"]]
    scored = [t for t in alone if t.get("edits") is not None]
    refs = sum(t["ref_words"] for t in scored)
    voices = {}
    for v in dict.fromkeys(t["voice"] for t in alone):
        vt = [t for t in alone if t["voice"] == v]
        voices[v] = {"first_audio_p50_s": pct([t["first_audio_s"] for t in ok(vt)], 0.5),
                     "rtf_p50": pct([t["rtf"] for t in ok(vt)], 0.5), "wer_mean": mean([t.get("wer") for t in vt]),
                     "failed": sum(not t["ok"] for t in mine if t["voice"] == v)}
    return {"load_s": load.get("total_s"), "warm_ok": load.get("ok"),
            "first_audio_s": {"alone": stats([t["first_audio_s"] for t in ok(alone)]),
                              "under_chat": stats([t["first_audio_s"] for t in ok(loaded)])},
            "rtf": {"alone": stats([t["rtf"] for t in ok(alone)]), "under_chat": stats([t["rtf"] for t in ok(loaded)])},
            "wer": {"pooled": sum(t["edits"] for t in scored) / refs if refs else None,
                    "mean": mean([t["wer"] for t in scored]), "scored": len(scored),
                    "per_voice": {v: s["wer_mean"] for v, s in voices.items()}},
            "failed_takes": [{k: t.get(k) for k in ("voice", "text_index", "pass", "status", "error", "samples")}
                             for t in mine if not t["ok"]],
            "finish_reasons": dict(collections.Counter(t["finish_reason"] or "none" for t in mine if t["pass"] != "warm")),
            "chat_under_load": chat, "memory": memory, "voices": voices}


def flags(models, takes, report):
    out, base = [], models[0]
    for m in models:
        for v, s in report[m]["voices"].items():
            why = []
            if s["failed"]:
                why.append(f"{s['failed']} take(s) failed or empty")
            if s["rtf_p50"] is not None and s["rtf_p50"] > THRESHOLDS["rtf_p50_max"]:
                why.append(f"RTF p50 {s['rtf_p50']:.3f}")
            a = report[base]["voices"].get(v, {}).get("wer_mean")
            if m != base and a is not None and s["wer_mean"] is not None:
                wer = {(t["model"], t["text_index"]): t.get("wer") for t in takes if t["pass"] == "alone" and t["voice"] == v}
                worse = sum(1 for (mm, i), w in wer.items() if mm == m and w is not None
                            and wer.get((base, i)) is not None and w > wer[(base, i)])
                if s["wer_mean"] - a >= THRESHOLDS["wer_mean_worse_by"] and worse >= THRESHOLDS["takes_worse_min"]:
                    why.append(f"WER mean {100 * s['wer_mean']:.1f}% vs {100 * a:.1f}% ({worse} takes worse)")
            if why:
                out.append({"model": m, "voice": v, "baseline": base, "reasons": why})
    return out


def fmt(v, scale=1, nd=3, unit=""):
    return "-" if v is None else f"{v * scale:.{nd}f}{unit}"


def write_md(path, models, rep, voices):
    rows = [("load s", lambda r: fmt(r["load_s"], nd=2))]
    for p in ("alone", "under_chat"):
        for k in ("p50", "p90", "max"):
            rows.append((f"first audio {k} s ({p})", lambda r, p=p, k=k: fmt(r["first_audio_s"][p][k])))
        for k in ("p50", "p90"):
            rows.append((f"RTF {k} ({p})", lambda r, p=p, k=k: fmt(r["rtf"][p][k])))
    rows += [("WER pooled", lambda r: fmt(r["wer"]["pooled"], 100, 2, "%")),
             ("WER mean", lambda r: fmt(r["wer"]["mean"], 100, 2, "%")),
             ("failed takes", lambda r: str(len(r["failed_takes"]))),
             ("finish reasons", lambda r: ", ".join(f"{k} {v}" for k, v in r["finish_reasons"].items())),
             ("chat tok/s under load", lambda r: fmt((r["chat_under_load"] or {}).get("tok_s"), nd=1)),
             ("memory peak GiB", lambda r: fmt(max([p.get("total_bytes", 0) for p in r["memory"].values()] or [None]),
                                                1 / (1 << 30), 2))]
    lines = ["# Speech model comparison", "", f"Baseline: `{models[0]}`. Percentiles are nearest-rank. "
             + ("Interleaved: each text spoken by every model in turn." if rep.get("interleave") else
                "Sequential: each model's passes in turn."), "",
             "| | " + " | ".join(f"`{m}`" for m in models) + " |", "|---" * (len(models) + 1) + "|"]
    lines += [f"| {name} | " + " | ".join(f(rep["models"][m]) for m in models) + " |" for name, f in rows]
    lines += ["", "Per voice: first audio p50 s / RTF p50 / WER mean", "",
              "| voice | " + " | ".join(f"`{m}`" for m in models) + " |", "|---" * (len(models) + 1) + "|"]
    for v in voices:
        cells = []
        for m in models:
            s = rep["models"][m]["voices"].get(v, {})
            cells.append(f"{fmt(s.get('first_audio_p50_s'))} / {fmt(s.get('rtf_p50'))} / "
                         f"{fmt(s.get('wer_mean'), 100, 1, '%')}")
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    lines += ["", f"Thresholds: {json.dumps(THRESHOLDS)}", "", "Flags:", ""]
    lines += [f"- `{f['model']}` {f['voice']}: {'; '.join(f['reasons'])}" for f in rep["flags"]] or ["- none"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_html(path, models, voices, takes):
    wav = {(t["model"], t["voice"], t["text_index"]): t["wav"] for t in takes if t["pass"] == "alone"}
    said = {t["text_index"]: t["input"] for t in takes if t["pass"] == "alone"}
    texts = [said[i] for i in sorted(said)]
    e = html.escape
    out = ["<!doctype html><meta charset=utf-8><title>Speech comparison</title>",
           "<style>body{font-family:sans-serif;margin:1em}td,th{border:1px solid #ccc;padding:4px;vertical-align:top}"
           "table{border-collapse:collapse}</style>",
           "<table><tr><th>voice</th><th>#</th><th>text</th>" + "".join(f"<th>{e(m)}</th>" for m in models) + "</tr>"]
    for v in voices:
        for i, text in enumerate(texts, 1):
            cells = "".join(f'<td><audio controls preload=none src="{e(urllib.parse.quote(wav[(m, v, i)]))}"></audio></td>'
                            if wav.get((m, v, i)) else "<td>failed</td>" for m in models)
            out.append(f"<tr><td>{e(v)}</td><td>{i}</td><td>{e(text)}</td>{cells}</tr>")
    with open(path, "w") as fh:
        fh.write("\n".join(out + ["</table>"]) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Compare speech models a running LocalRouter daemon serves.")
    ap.add_argument("--url", required=True, help="the daemon, e.g. http://127.0.0.1:8190")
    ap.add_argument("--models", nargs="+", help="speech models; the first is the baseline")
    ap.add_argument("--voices", nargs="+", help="default: the voices every model lists")
    ap.add_argument("--texts", help="one sentence a line; # comments and blank lines ignored")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stt", default="whisper-large-v3-turbo")
    ap.add_argument("--language", default="en", help="the transcription language ('' to detect)")
    ap.add_argument("--chat", help="a chat model to keep streaming during the under-load pass")
    ap.add_argument("--chat-prompt", default=CHAT_PROMPT)
    ap.add_argument("--load-texts", type=int, default=6)
    ap.add_argument("--server-match", nargs="+", help="substrings every server process's cmdline holds")
    ap.add_argument("--key")
    ap.add_argument("--timeout", type=float, default=300, help="seconds a request may take")
    ap.add_argument("--interleave", action="store_true",
                    help="every model loaded at once, each text spoken by each in turn: a slowdown from anything "
                         "else on the machine falls on all of them alike (memory is then not sampled per model)")
    ap.add_argument("--rescore", action="store_true",
                    help="score OUT's finished run again: transcribe its saved WAVs and rewrite its reports (its "
                         "speech, load, chat and memory figures are kept as measured)")
    args = ap.parse_args()
    if not args.rescore and not (args.models and args.texts):
        ap.error("--models and --texts are required (unless --rescore)")
    args.base = urllib.parse.urlsplit(args.url)
    if args.base.scheme != "http":
        raise SystemExit("compare.py: --url must be http:// (the daemon on this host)")
    args.prefix = re.sub(r"/v1$", "", args.base.path.rstrip("/"))
    args.out = os.path.abspath(args.out)
    os.makedirs(args.out, exist_ok=True)
    if args.rescore:
        return rescore(args)
    with open(args.texts) as f:
        texts = [s.strip() for s in f if s.strip() and not s.strip().startswith("#")]
    status, body = call(args, "GET", "/v1/models")
    if status != 200:
        raise SystemExit(f"compare.py: GET /v1/models: {status} {body[:200]!r}")
    entries = {m.get("id"): m for m in json.loads(body).get("data", [])}
    missing = [m for m in args.models if m not in entries]
    if missing or not texts:
        raise SystemExit(f"compare.py: unknown models {missing}" if missing else f"compare.py: no texts in {args.texts}")
    voices = args.voices or [v for v in entries[args.models[0]].get("voices") or []
                             if all(v in (entries[m].get("voices") or []) for m in args.models)]
    if not voices:
        raise SystemExit("compare.py: the models share no voice; name them with --voices")
    sampler = MemorySampler(args.server_match) if args.server_match else None
    if sampler:
        sampler.thread.start()
    takes, loads, chats, unloads = [], {}, {}, []
    log = open(os.path.join(args.out, "takes.jsonl"), "w")

    def keep(rec, **extra):
        rec.update(extra)
        takes.append(rec)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"{rec['model']} {rec['pass']} {rec['voice']} #{rec['text_index']}: ok={rec['ok']} "
              f"first={fmt(rec['first_audio_s'])} rtf={fmt(rec['rtf'])} {rec['error'] or ''}", flush=True)

    if args.interleave:
        interleaved(args, entries, voices, texts, keep, loads, chats)
        return score(args, args.models, voices, len(texts), takes, loads, chats, {}, unloads)
    for model in args.models:
        rate = int(entries[model].get("sample_rate") or DEFAULT_RATE)
        for other in args.models:
            if other != model:
                s, b = call(args, "POST", f"/v1/tools/{other}/unload")
                if s not in (200, 204, 404, 409):
                    unloads.append({"model": other, "status": s, "body": b[:200].decode("utf-8", "replace")})
        if sampler:
            sampler.phase = model
        loads[model] = speak(args, model, voices[0], WARM_TEXT, rate)
        keep(loads[model], **{"pass": "warm", "text_index": 0, "under_chat": False})
        for v in voices:
            for i, text in enumerate(texts, 1):
                path = os.path.join(args.out, "wav", slug(v), f"{i:02d}-{slug(model)}.wav")
                keep(speak(args, model, v, text, rate, path), **{"pass": "alone", "text_index": i, "under_chat": False})
        chats[model] = None
        if args.chat:
            load = ChatLoad(args)
            load.thread.start()
            time.sleep(2)  # let the chat stream reach decoding first
            if sampler:
                sampler.phase = f"{model} under chat"
            for v in voices:
                for i, text in enumerate(texts[:args.load_texts], 1):
                    path = os.path.join(args.out, "wav", slug(v), f"{i:02d}-{slug(model)}-chat.wav")
                    keep(speak(args, model, v, text, rate, path), **{"pass": "chat", "text_index": i, "under_chat": True})
            chats[model] = load.finish()
        if sampler:
            sampler.phase = None
    memory = sampler.finish() if sampler else {}
    log.close()
    return score(args, args.models, voices, len(texts), takes, loads, chats, memory, unloads)


def interleaved(args, entries, voices, texts, keep, loads, chats):
    """The passes with every model warm, each text spoken by each model in turn (then the same under chat load)."""
    rates = {m: int(entries[m].get("sample_rate") or DEFAULT_RATE) for m in args.models}
    for m in args.models:
        loads[m] = speak(args, m, voices[0], WARM_TEXT, rates[m])
        keep(loads[m], **{"pass": "warm", "text_index": 0, "under_chat": False})
    for v in voices:
        for i, text in enumerate(texts, 1):
            for m in args.models:
                path = os.path.join(args.out, "wav", slug(v), f"{i:02d}-{slug(m)}.wav")
                keep(speak(args, m, v, text, rates[m], path), **{"pass": "alone", "text_index": i, "under_chat": False})
    load = None
    if args.chat:
        load = ChatLoad(args)
        load.thread.start()
        time.sleep(2)  # let the chat stream reach decoding first
        for v in voices:
            for i, text in enumerate(texts[:args.load_texts], 1):
                for m in args.models:
                    path = os.path.join(args.out, "wav", slug(v), f"{i:02d}-{slug(m)}-chat.wav")
                    keep(speak(args, m, v, text, rates[m], path), **{"pass": "chat", "text_index": i, "under_chat": True})
        load = load.finish()
    for m in args.models:
        chats[m] = load


def rescore(args):
    """A finished run again: its takes and the figures its report kept, scored with a fresh transcription."""
    with open(os.path.join(args.out, "report.json")) as f:
        old = json.load(f)
    with open(os.path.join(args.out, "takes.jsonl")) as f:
        takes = [json.loads(line) for line in f if line.strip()]
    models = old["models_order"]
    args.chat, args.load_texts, args.server_match = old.get("chat"), old.get("load_texts", 0), old.get("server_match")
    args.interleave = old.get("interleave", False)
    loads = {m: next((t for t in takes if t["model"] == m and t["pass"] == "warm"), {}) for m in models}
    chats = {m: old["models"][m].get("chat_under_load") for m in models}
    memory = {}
    for m in models:
        memory.update(old["models"][m].get("memory") or {})
    return score(args, models, old["voices"], old["texts"], takes, loads, chats, memory, old.get("unload_failures", []))


def score(args, models, voices, texts, takes, loads, chats, memory, unloads):
    """Every alone take transcribed and scored, then report.json, report.md and listen.html."""
    for t in takes:
        if t["pass"] != "alone" or not t["wav"]:
            continue
        hyp, err = transcribe(args, os.path.join(args.out, t["wav"]))
        ref = words(t["input"])
        t.update(hypothesis=hyp, stt_error=err, ref_words=len(ref))
        if hyp is not None and ref:
            t["edits"] = edits(ref, words(hyp))
            t["wer"] = t["edits"] / len(ref)
        print(f"WER {t['model']} {t['voice']} #{t['text_index']}: {fmt(t.get('wer'), 100, 1, '%')} {err or ''}",
              flush=True)
    with open(os.path.join(args.out, "takes.jsonl"), "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in takes)

    per = {m: summarize(m, takes, loads[m], chats[m],
                        {k: v for k, v in memory.items() if k in (m, f"{m} under chat")}) for m in models}
    rep = {"url": args.url, "baseline": models[0], "models_order": models, "voices": voices,
           "interleave": bool(args.interleave), "rescored": bool(args.rescore),
           "texts": texts, "load_texts": min(args.load_texts, texts) if args.chat else 0, "stt": args.stt,
           "chat": args.chat, "server_match": args.server_match, "unload_failures": unloads,
           "percentiles": "nearest-rank over the good takes", "wer": "word Levenshtein; NFKC, lowercase, "
           "punctuation and [tags] dropped, intra-word apostrophes kept; pooled = edits / reference words",
           "memory_method": "per matching PID: nvidia-smi used_memory + RssAnon + RssShmem, every 0.5 s; peaks",
           "thresholds": THRESHOLDS, "models": per}
    rep["flags"] = flags(models, takes, per)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    write_md(os.path.join(args.out, "report.md"), models, rep, voices)
    write_html(os.path.join(args.out, "listen.html"), models, voices, takes)
    print(f"report: {os.path.join(args.out, 'report.md')}; flags: {len(rep['flags'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
