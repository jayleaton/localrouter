"""A fake OpenAI-compatible chat server for the voice integration tests, in plain Python with no dependencies. It
shows what the voice session needs from `voice.chat_url`: POST /v1/chat/completions with `stream: true` answered by
server-sent events (`data: {chunk}` ... `data: [DONE]`, one HTTP chunk each), content deltas, a streamed tool call,
the usage chunk on `stream_options.include_usage`, OpenAI's error envelope, and a client disconnect that stops the
generation.

Run as `python3 voice_chat_server.py [--delay-ms N]`: it prints `listening <port>` once bound to 127.0.0.1 and exits when
its stdin closes. Test hooks, chosen by the last message: a `tool` message gets "Done. Your picture is ready."; user
text containing "fail" gets a 500, "picture" a `generate_image` tool call, "slow" twenty sentences at five times the
delay; anything else is echoed back a word at a time. GET /stats returns every request seen with its model, messages,
tool names and whether its stream completed or was cancelled by the client.
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as _ThreadingHTTPServer
import socketserver


class ThreadingHTTPServer(_ThreadingHTTPServer):
    """Without HTTPServer's reverse lookup of its own address (socket.getfqdn), which can take 30 s on a Mac."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

DELAY_MS = 10
LOCK = threading.Lock()
RECORDS = []


def chunk(delta, finish=None):
    return {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0, "model": "fake-chat",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def user_text(messages):
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content") or ""
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        return content
    return ""


def words(text):
    parts = text.split(" ")
    return [w + " " for w in parts[:-1]] + parts[-1:]


def tool_call_deltas():
    return [
        {"tool_calls": [{"index": 0, "id": "call_fake1", "type": "function",
                         "function": {"name": "generate_image", "arguments": ""}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": "{\"prompt\": "}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": "\"a red fox\"}"}}]},
    ]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def fail(self, status, kind, message):
        self.send_json(status, {"error": {"message": message, "type": kind}})

    def read_body(self):
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            data = b""
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip(), 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return data
                data += self.rfile.read(size)
                self.rfile.readline()
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_GET(self):
        if self.path.split("?")[0] == "/stats":
            with LOCK:
                self.send_json(200, {"requests": RECORDS})
        else:
            self.fail(404, "invalid_request_error", "no such route")

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/chat/completions":
            self.fail(404, "invalid_request_error", "no such route")
            return
        try:
            body = json.loads(self.read_body() or b"{}")
        except ValueError:
            self.fail(400, "invalid_request_error", "the body is not JSON")
            return
        if not isinstance(body, dict) or body.get("stream") is not True:
            self.fail(400, "invalid_request_error", "this fake only streams: set stream to true")
            return
        messages = body.get("messages") or []
        record = {"model": body.get("model"), "messages": messages,
                  "tools": [t["function"]["name"] for t in body.get("tools") or []],
                  "completed": False, "cancelled": False}
        with LOCK:
            RECORDS.append(record)
        delay, finish = DELAY_MS, "stop"
        if messages and messages[-1].get("role") == "tool":
            deltas = [{"content": w} for w in words("Done. Your picture is ready.")]
        else:
            text = user_text(messages)
            lower = text.lower()
            if "fail" in lower:
                self.fail(500, "server_error", "the fake model failed")
                return
            if "picture" in lower:
                deltas, finish = tool_call_deltas(), "tool_calls"
            elif "slow" in lower:
                answer = " ".join(f"This is sentence {n} of a long answer." for n in range(1, 21))
                deltas, delay = [{"content": w} for w in words(answer)], DELAY_MS * 5
            else:
                deltas = [{"content": w} for w in words(f"Hello! I heard you say: {text}. Anything else?")]
        usage = (body.get("stream_options") or {}).get("include_usage") is True
        self.stream(record, deltas, finish, delay, usage)

    def event(self, obj):
        data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def stream(self, record, deltas, finish, delay, usage):
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.event(chunk({"role": "assistant", "content": ""}))
            for d in deltas:
                time.sleep(delay / 1000)
                self.event(chunk(d))
            self.event(chunk({}, finish))
            if usage:
                n = len(deltas)
                self.event({"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0,
                            "model": "fake-chat", "choices": [],
                            "usage": {"prompt_tokens": 10, "completion_tokens": n, "total_tokens": 10 + n}})
            with LOCK:
                record["completed"] = True  # before [DONE], so a client that saw [DONE] sees it in /stats
            self.event(b"[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            with LOCK:
                record["completed"], record["cancelled"] = False, True
            self.close_connection = True


def watch_stdin():
    sys.stdin.buffer.read()
    os._exit(0)


def main():
    global DELAY_MS
    parser = argparse.ArgumentParser(description="A fake streaming OpenAI chat server for the integration tests.")
    parser.add_argument("--delay-ms", type=int, default=10, help="pause between streamed deltas")
    DELAY_MS = parser.parse_args().delay_ms
    threading.Thread(target=watch_stdin, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    sys.stdout.write(f"listening {server.server_address[1]}\n")
    sys.stdout.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()
