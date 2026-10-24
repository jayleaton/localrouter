"""Turns a record_proxy.py recording into a harness fixture (tests/fixtures/harness/*.jsonl).

Client requests and events are kept whole (already redacted); audio is a size, never samples. Server events keep
their type (and an error's code), HTTP responses their status. A run of identical lines on one connection becomes one
line with "repeat": N. Timestamps are dropped, so re-recording the same flow gives a readable diff.

Machine identifiers a client puts in its requests are replaced before anything is written: `--replace OLD=NEW`
(repeatable, applied to every string in order: home directories, working directories, a timezone), and
`--anonymize-uuids`, which numbers each distinct UUID by first appearance (00000000-0000-0000-0000-000000000001...),
so ids that matched still match. Numbers named in `--zero KEY` (repeatable) become 0, inside JSON-in-a-string too.

    python3 -I compact_fixture.py raw/web.jsonl > tests/fixtures/harness/web.jsonl
    python3 -I compact_fixture.py raw/codex.jsonl --replace /home/me/work=/work --replace /home/me=/home/user \
        --anonymize-uuids --zero turn_started_at_unix_ms > tests/fixtures/harness/codex.jsonl
"""
import argparse
import json
import re

UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def compact(rec):
    out = {"conn": rec.get("conn")}
    if "proxy_error" in rec:
        return None
    if "method" in rec:
        out.update({k: rec[k] for k in ("dir", "method", "path", "headers", "body") if k in rec})
        return out
    if "status" in rec:
        out.update({"dir": rec["dir"], "path": rec["path"], "status": rec["status"]})
        return out
    if "ws" in rec:
        ev = rec["ws"]
        out["dir"] = rec["dir"]
        if rec["dir"] == "c2s":
            if ev.get("type") == "input_audio_buffer.append":
                ev = {k: v for k, v in ev.items() if k != "event_id"} | {"audio": "<pcm16>"}
            out["ws"] = ev
        else:
            short = {"type": ev.get("type")}
            if ev.get("type") == "error" and isinstance(ev.get("error"), dict):
                short["code"] = ev["error"].get("code")
            out["ws"] = short
        return out
    if "ws_close" in rec or "ws_binary" in rec or "ws_text" in rec:
        out.update({k: v for k, v in rec.items() if k not in ("client", "ms", "conn")})
        return out
    return None


def redactor(replace, anonymize, zero):
    uuids = {}

    def number(m):
        return uuids.setdefault(m.group(0), f"00000000-0000-0000-0000-{len(uuids) + 1:012d}")

    def text(s):
        for old, new in replace:
            s = s.replace(old, new)
        for key in zero:
            s = re.sub(r'(\\*"' + re.escape(key) + r'\\*":)\d+', r"\g<1>0", s)
        return UUID.sub(number, s) if anonymize else s

    def walk(v):
        if isinstance(v, str):
            return text(v)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {text(k): (0 if k in zero and isinstance(x, int) else walk(x)) for k, x in v.items()}
        return v

    return walk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--replace", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--anonymize-uuids", action="store_true")
    ap.add_argument("--zero", action="append", default=[], metavar="KEY")
    o = ap.parse_args()
    redact = redactor([tuple(r.split("=", 1)) for r in o.replace], o.anonymize_uuids, set(o.zero))
    prev, n = None, 0
    lines = []
    with open(o.path) as f:
        for raw in f:
            c = compact(json.loads(raw))
            c = redact(c) if c is not None else None
            if c is None:
                continue
            if c == prev:
                n += 1
                continue
            if prev is not None:
                lines.append({**prev, "repeat": n} if n > 1 else prev)
            prev, n = c, 1
    if prev is not None:
        lines.append({**prev, "repeat": n} if n > 1 else prev)
    for line in lines:
        print(json.dumps(line, separators=(",", ":")))


if __name__ == "__main__":
    main()
