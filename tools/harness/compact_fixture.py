"""Turns a record_proxy.py recording into a harness fixture (tests/fixtures/harness/*.jsonl).

Client requests and events are kept whole (already redacted); audio is a size, never samples. Server events keep
their type (and an error's code), HTTP responses their status. A run of identical lines on one connection becomes one
line with "repeat": N. Timestamps are dropped, so re-recording the same flow gives a readable diff.

    python3 -I compact_fixture.py raw/web.jsonl > tests/fixtures/harness/web.jsonl
"""
import json
import sys


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


def main(path):
    prev, n = None, 0
    lines = []
    with open(path) as f:
        for raw in f:
            c = compact(json.loads(raw))
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
    main(sys.argv[1])
