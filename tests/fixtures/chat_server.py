"""A fake TensorFold `serve` for the integration tests: the OpenAI chat completions server the `openai_server` engine
starts, in plain Python with no dependencies. It answers deterministically from the last user (or tool) message, so a
test can check what LocalRouter forwarded (`received` echoes the parsed request body) and how it relays replies.

CLI: --port N --name ID [--hold-mib 8] [--fail-start]. Binds 127.0.0.1, logs to stderr, exits 0 on SIGTERM.
Refusals: n != 1 and logprobs (400), a model other than --name (404). Test hooks by message text: "overload" (503),
"explode" (500), "die" (exits without replying), "slow" (40 words streamed 0.25 s apart), "midfail" (two content
chunks streamed, then the process exits mid-stream), "parallel" (with tools: two calls of the first tool). With tools,
the reply is a call of the first tool unless tool_choice is "none" or the last message is a tool's result, which is
answered as text (so a tool loop ends).
As a rank of a tensor-parallel group (FAKE_RANK, FAKE_WORLD and FAKE_MASTER=host:port in its environment, as a
tool's options.rank_env sets them), it keeps TensorFold's contract: a rank past 0 connects to rank 0 and serves no HTTP;
rank 0 waits for every other rank before it serves; a rank whose peer goes exits, so the group stops together. A
reply then says "(rank 0 of N)".
"""

import json
import os
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as _ThreadingHTTPServer
import socketserver


class ThreadingHTTPServer(_ThreadingHTTPServer):
    """Without HTTPServer's reverse lookup of its own address (socket.getfqdn), which can take 30 s on a Mac."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

NAME = ""
WORLD = 1


def log(msg):
    print(f"chat_server: {msg}", file=sys.stderr, flush=True)


def dumps(obj):
    return json.dumps(obj, separators=(",", ":"))


def error(message, kind):
    return {"error": {"message": message, "type": kind}}


def parse_args(argv):
    out = {"port": None, "name": None, "hold_mib": 8, "fail_start": False}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--port", "--name", "--hold-mib") and i + 1 < len(argv):
            key = arg[2:].replace("-", "_")
            out[key] = argv[i + 1] if key == "name" else int(argv[i + 1])
            i += 2
        elif arg == "--fail-start":
            out["fail_start"] = True
            i += 1
        else:
            log(f"unknown argument {arg}")
            sys.exit(2)
    if out["port"] is None or out["name"] is None:
        log("--port and --name are required")
        sys.exit(2)
    return out


def last_role(messages):
    last = messages[-1] if isinstance(messages, list) and messages else None
    return last.get("role") if isinstance(last, dict) else None


def last_text(messages):
    for m in reversed(messages if isinstance(messages, list) else []):
        if isinstance(m, dict) and m.get("role") in ("user", "tool"):
            content = m.get("content")
            if isinstance(content, list):  # text parts
                return "".join(p.get("text", "") for p in content if isinstance(p, dict))
            return content if isinstance(content, str) else ""
    return ""


def refusal(body, text):
    """(status, error body) when the request is refused, else None."""
    if "n" in body and body["n"] != 1:
        return 400, error("n must be 1; multiple choices are not supported", "invalid_request_error")
    if body.get("logprobs"):
        return 400, error("logprobs are not supported by this model", "invalid_request_error")
    if body.get("model") != NAME:
        return 404, error("unknown model", "invalid_request_error")
    if text == "overload":
        return 503, error("server busy", "server_error")
    if text == "explode":
        return 500, error("generation failed", "server_error")
    return None


def generate(body, text):
    """The deterministic reply: reasoning, content, tool call, finish reason and completion tokens."""
    kwargs = body.get("chat_template_kwargs")
    thinking = body.get("reasoning_effort") != "none" and not (isinstance(kwargs, dict) and kwargs.get("enable_thinking") is False)
    out = {"reasoning": "thinking about " + text if thinking else None, "content": None, "calls": [], "finish": "stop", "tokens": 0}
    limit = body.get("max_completion_tokens", body.get("max_tokens"))
    if not isinstance(limit, int) or isinstance(limit, bool):
        limit = None
    if text == "ponder" and thinking:  # a long think that counts against the limit, as a real reasoning model's does
        thought = ["hmm"] * 40
        if limit is not None and len(thought) >= limit:
            out.update(reasoning=" ".join(thought[:limit]), content="", finish="length", tokens=limit, reasoning_tokens=limit)
            return out
        out.update(reasoning=" ".join(thought), reasoning_tokens=len(thought))
        limit = None if limit is None else limit - len(thought)
    tools = body.get("tools")
    if isinstance(tools, list) and tools and body.get("tool_choice") != "none" and last_role(body.get("messages")) != "tool":
        name = tools[0]["function"]["name"]
        n = 2 if text == "parallel" else 1
        out["calls"] = [{"id": f"call_{i + 1}", "type": "function", "function": {"name": name, "arguments": json.dumps({"echo": text, "n": i + 1} if n > 1 else {"echo": text})}} for i in range(n)]
        out["finish"] = "tool_calls"
        return out
    fmt = body.get("response_format")
    if text in ("slow", "midfail"):
        content = " ".join(f"w{i}" for i in range(40))
    elif isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema"):
        content = json.dumps({"echo": text})
    else:
        content = "echo: " + text + (f" (rank 0 of {WORLD})" if WORLD > 1 else "")
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)] if isinstance(stop, list) else []
    cuts = [content.find(s) for s in stops if s and s in content]
    if cuts:
        content = content[: min(cuts)]
    tokens = content.split(" ")
    if limit is not None and len(tokens) > limit:
        tokens = tokens[:limit]
        content = " ".join(tokens)
        out["finish"] = "length"
    out["content"] = content
    out["tokens"] = len(tokens)
    return out


def usage(body, out):
    prompt = len(body.get("messages") or [])
    u = {"prompt_tokens": prompt, "completion_tokens": out["tokens"], "total_tokens": prompt + out["tokens"]}
    if "reasoning_tokens" in out:
        u["completion_tokens_details"] = {"reasoning_tokens": out["reasoning_tokens"]}
    return u


def whole(body, out):
    message = {"role": "assistant", "content": out["content"]}
    if out["reasoning"] is not None:
        message["reasoning_content"] = out["reasoning"]
    if out["calls"]:
        message["tool_calls"] = out["calls"]
    return {"id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": NAME,
            "choices": [{"index": 0, "message": message, "finish_reason": out["finish"]}],
            "usage": usage(body, out), "received": body}


def words(s):
    """`s` cut at its spaces, each piece after the first keeping its space: the pieces join back to `s`."""
    if not s:
        return []
    parts = s.split(" ")
    return [parts[0]] + [" " + p for p in parts[1:]]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "fake-tensorfold"

    def log_message(self, fmt, *args):
        log(fmt % args)

    def send_json(self, status, obj):
        data = dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def read_body(self):
        if "chunked" not in (self.headers.get("Transfer-Encoding") or "").lower():
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))
        data = b""
        while True:
            size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
            if size == 0:
                self.rfile.readline()
                return data
            data += self.rfile.read(size)
            self.rfile.readline()

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
        else:
            self.send_json(404, error("no such route", "invalid_request_error"))

    def do_POST(self):
        try:
            raw = self.read_body()
            if self.path != "/v1/chat/completions":
                return self.send_json(404, error("no such route", "invalid_request_error"))
            try:
                body = json.loads(raw)
            except ValueError:
                return self.send_json(400, error("the body is not JSON", "invalid_request_error"))
            if not isinstance(body, dict):
                return self.send_json(400, error("the body is not an object", "invalid_request_error"))
            text = last_text(body.get("messages"))
            refused = refusal(body, text)
            if refused:
                return self.send_json(*refused)
            if text == "die":
                log("die: exiting without a reply")
                os._exit(1)
            out = generate(body, text)
            if body.get("stream") is True:
                self.stream(body, out, text)
            else:
                self.send_json(200, whole(body, out))
        except (BrokenPipeError, ConnectionResetError):
            log("the client went away")

    def chunk(self, data):
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def event(self, obj, pause=0.0):
        self.chunk(b"data: " + dumps(obj).encode() + b"\n\n")
        if pause:
            time.sleep(pause)

    def delta(self, delta, finish=None, pause=0.0):
        self.event({"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0, "model": NAME,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}, pause)

    def stream(self, body, out, text):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        pause = 0.25 if text == "slow" else 0.0
        self.delta({"role": "assistant"}, pause=pause)
        for piece in words(out["reasoning"]):
            self.delta({"reasoning_content": piece}, pause=pause)
        for i, piece in enumerate(words(out["content"])):
            if text == "midfail" and i == 2:
                log("midfail: dropping the connection mid-stream")
                self.wfile.flush()
                os._exit(1)
            self.delta({"content": piece}, pause=pause)
        for i, call in enumerate(out["calls"]):
            fn = call["function"]
            self.delta({"tool_calls": [{"index": i, "id": call["id"], "type": "function", "function": {"name": fn["name"], "arguments": ""}}]})
            half = len(fn["arguments"]) // 2
            for part in (fn["arguments"][:half], fn["arguments"][half:]):
                self.delta({"tool_calls": [{"index": i, "function": {"arguments": part}}]})
        self.delta({}, finish=out["finish"])
        options = body.get("stream_options")
        if isinstance(options, dict) and options.get("include_usage"):
            self.event({"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0, "model": NAME,
                        "choices": [], "usage": usage(body, out)})
        self.chunk(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def group():
    """A tensor-parallel rank (see the module doc): returns once rank 0 may serve; never returns on another rank."""
    global WORLD
    import socket, threading
    if "FAKE_RANK" not in os.environ:
        return
    rank, WORLD = int(os.environ["FAKE_RANK"]), int(os.environ["FAKE_WORLD"])
    host, port = os.environ["FAKE_MASTER"].rsplit(":", 1)
    def watch(s, who):  # the peer's line closing stops this rank
        while s.recv(1):
            pass
        log(f"rank {rank}: {who} is gone, stopping")
        os._exit(1)
    if rank > 0:
        for _ in range(200):
            try:
                s = socket.create_connection((host, int(port)), timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        else:
            log(f"rank {rank}: rank 0 never answered on {host}:{port}")
            sys.exit(4)
        s.settimeout(None)
        s.sendall(f"{rank}\n".encode())
        log(f"rank {rank} of {WORLD}: joined rank 0 at {host}:{port}")
        watch(s, "rank 0")
    lst = socket.socket()
    lst.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lst.bind((host, int(port)))
    lst.listen(16)
    lst.settimeout(30)
    for _ in range(WORLD - 1):
        c, _addr = lst.accept()
        c.settimeout(None)
        threading.Thread(target=watch, args=(c, "a rank"), daemon=True).start()
    log(f"rank 0 of {WORLD}: every rank joined")


def main():
    global NAME
    args = parse_args(sys.argv[1:])
    NAME = args["name"]
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    if args["fail_start"]:
        log("--fail-start: refusing to start")
        sys.exit(3)
    held = bytearray(args["hold_mib"] << 20)
    group()
    server = ThreadingHTTPServer(("127.0.0.1", args["port"]), Handler)
    server.daemon_threads = True
    log(f"serving {NAME} on 127.0.0.1:{args['port']} holding {len(held) >> 20} MiB")
    server.serve_forever()


if __name__ == "__main__":
    main()
