"""A harness that runs its own tool, over LocalRouter's /v1/realtime with client tool execution.

Python 3 standard library only. It registers one local function (get_device_time: this machine's clock), asks a
question that needs it, runs the call when the server hands it over, returns the output and hears the continuation:

  session.update {localrouter: {tool_execution: "client"}, tools: [get_device_time]}
  conversation.item.create (the question) + response.create
  <- response.function_call_arguments.done {call_id, name, arguments}     (complete: run it once)
  -> conversation.item.create {type: function_call_output, call_id, output} + response.create
  <- response.audio.delta ... response.done                               (the answer, spoken)

Then it checks that the server refuses a second output for the same call and an output for an unknown call.

  python3 -I client.py [--url http://127.0.0.1:8190] [--turns 3] [--wav answer.wav] [--json results.json]

With pairing on, $LOCALROUTER_TOKEN is sent as a Bearer token. Exit 0 only when every turn made exactly one call,
the continuation was spoken, and both refusals came back.
"""

import argparse
import base64
import datetime
import json
import os
import re
import select
import socket
import struct
import sys
import time
import wave

TOOL = {
    "type": "function",
    "name": "get_device_time",
    "description": "The current local date and time on the user's own device, and its time zone.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}
INSTRUCTIONS = ("You are a voice assistant on the user's device. Whenever the user asks for the time, call "
                "get_device_time, then say the answer in one short sentence. The clock moves: never reuse an earlier "
                "result and never guess the time.")
# Each needs a fresh reading of the clock (asked for the date after the time, a model rightly reuses the first result).
QUESTIONS = ["What time is it on my device right now?", "Check my device's clock again: what time is it now?",
             "One more time, please read my device's clock: what time is it?"]


def get_device_time(_args):
    now = datetime.datetime.now().astimezone()
    return {"time": now.strftime("%H:%M"), "date": now.strftime("%A %d %B %Y"), "zone": now.tzname()}


class Ws:
    """A minimal RFC 6455 client: masked text frames out, text frames in."""

    def __init__(self, url, token):
        m = re.match(r"(?:http|ws)://([^:/]+)(?::(\d+))?", url)
        if not m:
            raise ValueError(f"not an http:// or ws:// URL: {url}")
        host, port = m.group(1), int(m.group(2) or 80)
        self.sock = socket.create_connection((host, port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        auth = f"Authorization: Bearer {token}\r\n" if token else ""
        self.sock.sendall((f"GET /v1/realtime HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                           f"Sec-WebSocket-Protocol: realtime\r\n{auth}\r\n").encode())
        self.buf = b""
        while b"\r\n\r\n" not in self.buf:
            more = self.sock.recv(4096)
            if not more:
                raise ConnectionError("closed during the handshake")
            self.buf += more
        head, self.buf = self.buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"no upgrade: {head[:200]!r}")

    def send(self, obj):
        data = json.dumps(obj).encode()
        n, mask = len(data), os.urandom(4)
        size = bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536 else \
            bytes([0x80 | 127]) + struct.pack(">Q", n)
        self.sock.sendall(bytes([0x81]) + size + mask + bytes(b ^ mask[i & 3] for i, b in enumerate(data)))

    def recv(self, timeout):
        """The next server event (a dict), or None after `timeout` seconds."""
        deadline = time.monotonic() + timeout
        while True:
            msg = self._frame()
            if msg is not None:
                return json.loads(msg)
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([self.sock], [], [], left)[0]:
                return None
            more = self.sock.recv(1 << 20)
            if not more:
                raise ConnectionError("the server closed the connection")
            self.buf += more

    def _frame(self):
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
                raise ConnectionError("the server closed the session")
            if op == 1:
                return payload.decode()


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def until(ws, want, timeout=120, seen=None):
    """Events until one whose type is in `want`; every event passes through `seen` (a list) on the way."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        e = ws.recv(deadline - time.monotonic())
        if e is None:
            break
        e["_t"] = time.monotonic()
        if seen is not None:
            seen.append(e)
        if e.get("type") == "error":
            log("error event:", e["error"].get("code"), e["error"].get("message"))
        if e.get("type") in want:
            return e
    raise TimeoutError(f"no {want} within {timeout} s")


def turn(ws, question, n):
    """One question: the call, run here, then the spoken answer. Returns the turn's record."""
    log(f"turn {n}: {question!r}")
    ws.send({"type": "conversation.item.create",
             "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": question}]}})
    ws.send({"type": "response.create"})
    events = []
    done = until(ws, {"response.done"}, seen=events)
    calls = [e for e in events if e["type"] == "response.function_call_arguments.done"]
    rec = {"question": question, "calls": [{"name": c["name"], "call_id": c["call_id"], "arguments": c["arguments"]} for c in calls],
           "first_status": done["response"]["status"]}
    if len(calls) != 1 or calls[0]["name"] != TOOL["name"]:
        log(f"  expected one {TOOL['name']} call, got {rec['calls']}")
        rec["ok"] = False
        return rec, b""
    call = calls[0]
    t_call = call["_t"]
    output = json.dumps(get_device_time(json.loads(call["arguments"] or "{}")))
    t_ran = time.monotonic()
    ws.send({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": call["call_id"], "output": output}})
    ws.send({"type": "response.create"})
    t_sent = time.monotonic()
    log(f"  call {call['call_id']} {call['name']}({call['arguments']}) -> {output}")
    events = []
    done = until(ws, {"response.done"}, seen=events)
    audio = b"".join(base64.b64decode(e["delta"]) for e in events if e["type"] == "response.audio.delta")
    first_audio = next((e["_t"] for e in events if e["type"] == "response.audio.delta"), None)
    first_text = next((e["_t"] for e in events if e["type"] in ("response.audio_transcript.delta", "response.text.delta")), None)
    said = next((e.get("transcript") or e.get("text") for e in events if e["type"] in ("response.audio_transcript.done", "response.text.done")), "")
    timing = next((e for e in events if e["type"] == "localrouter.response.timing"), {})
    ms = lambda a, b: round((b - a) * 1000) if a is not None and b is not None else None  # noqa: E731
    rec.update({
        "output": output, "answer": said, "status": done["response"]["status"], "answer_audio_s": round(len(audio) / 2 / 24000, 2),
        "run_locally_ms": ms(t_call, t_ran),
        "output_sent_to_first_text_ms": ms(t_sent, first_text),
        "output_sent_to_first_audio_ms": ms(t_sent, first_audio),
        "call_to_first_audio_ms": ms(t_call, first_audio),
        "server_timing": {k: v for k, v in timing.items() if k not in ("type", "event_id", "_t")},
    })
    rec["ok"] = rec["status"] == "completed" and len(audio) > 0 and bool(said.strip())
    log(f"  answer ({rec['answer_audio_s']} s of audio): {said!r}")
    log(f"  output sent -> first audio {rec['output_sent_to_first_audio_ms']} ms (first text {rec['output_sent_to_first_text_ms']} ms)")
    return rec, audio


def refused(ws, call_id, code):
    """Sends an output for `call_id` and expects an error event with `code`."""
    ws.send({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": call_id, "output": "{}"}})
    e = until(ws, {"error", "conversation.item.created"}, timeout=10)
    got = e.get("error", {}).get("code") if e["type"] == "error" else "accepted"
    log(f"  output for {call_id}: {got} (want {code})")
    return got == code


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--url", default=os.environ.get("LOCALROUTER_URL", "http://127.0.0.1:8190"))
    p.add_argument("--turns", type=int, default=1, help="questions in the session, each answered through the tool")
    p.add_argument("--wav", help="write the last spoken answer here")
    p.add_argument("--json", help="write the results here")
    args = p.parse_args()
    ws = Ws(args.url, os.environ.get("LOCALROUTER_TOKEN", ""))
    until(ws, {"session.created"}, timeout=30)
    ws.send({"type": "session.update", "session": {"instructions": INSTRUCTIONS, "tools": [TOOL], "turn_detection": None,
                                                   "localrouter": {"tool_execution": "client"}}})
    s = until(ws, {"session.updated", "error"}, timeout=30)
    mode = s.get("session", {}).get("localrouter", {}).get("tool_execution")
    if mode != "client":
        log(f"the server did not take client tool execution: {s}")
        return 1
    log(f"session {s['session']['id']}: tool_execution {mode}, tools {[t['name'] for t in s['session']['tools']]}")
    turns, audio = [], b""
    for n in range(args.turns):
        rec, audio = turn(ws, QUESTIONS[n % len(QUESTIONS)], n + 1)
        turns.append(rec)
    last = next((t for t in reversed(turns) if t["calls"]), None)
    checks = {
        "duplicate_output_refused": bool(last) and refused(ws, last["calls"][0]["call_id"], "duplicate_call_output"),
        "unknown_call_refused": refused(ws, "call_nobody_made", "unknown_call_id"),
    }
    if args.wav and audio:
        with wave.open(args.wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(audio)
    ok = all(t["ok"] for t in turns) and all(checks.values())
    result = {"url": args.url, "turns": turns, "checks": checks, "passed": ok}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    log("PASSED" if ok else "FAILED", json.dumps(checks))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
