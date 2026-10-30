"""A fake typed-decision server for the integration tests: `POST /v1/systemone` in the System One shape, the server the
`openai_server` engine starts for a decision tool, in plain Python with no dependencies. Deterministic: a choice picks
the first option whose id appears in the state's text (else the first option) at 0.9, the rest sharing 0.1; a noul is
0.8 when the state's text contains "yes", else 0.2; a score is level 1 of its levels at 0.7.

CLI: --port N --name ID. Binds 127.0.0.1, logs to stderr, exits 0 on SIGTERM. A model other than --name is a 404.
Test hooks by the state's text: "offbook" (a choice that was not offered), "badsum" (probabilities summing to 1.5),
"skip" (a question left unanswered), "overload" (503).
"""

import json
import signal
import sys
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as _ThreadingHTTPServer
import socketserver


class ThreadingHTTPServer(_ThreadingHTTPServer):
    """Without HTTPServer's reverse lookup of its own address (socket.getfqdn), which can take 30 s on a Mac."""

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

NAME = ""


def answer(q, text):
    kind = q.get("type")
    if kind == "noul":
        return {"type": "noul", "noul": 0.8 if "yes" in text else 0.2}
    if kind == "score":
        levels = q["criteria"]
        rest = 0.3 / (len(levels) - 1)
        probs = {str(i): (0.7 if i == 1 else rest) for i in range(len(levels))}
        return {"type": "score", "score": sum(i * p for i, p in enumerate(probs.values())), "confidence": 0.7,
                "probabilities": probs, "legend": {str(i): v for i, v in enumerate(levels)}}
    options = list(q["criteria"])
    chosen = next((o for o in options if o in text), options[0])
    rest = 0.1 / (len(options) - 1)
    probs = {o: (0.9 if o == chosen else rest) for o in options}
    if "badsum" in text:
        probs = {o: (0.9 if o == chosen else 0.6 / (len(options) - 1)) for o in options}
    if "offbook" in text:
        chosen = "not-offered"
    return {"type": "choice", "choice": chosen, "confidence": 0.9, "probabilities": probs}


class Handler(BaseHTTPRequestHandler):
    def send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self.send(200, {"ok": True})
        self.send(404, {"error": {"message": "no such route"}})

    def do_POST(self):
        if self.path != "/v1/systemone":
            return self.send(404, {"error": {"message": "no such route"}})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if req.get("model") != NAME:
            return self.send(404, {"error": {"message": f"unknown model {req.get('model')!r}"}})
        text = json.dumps(req["state"])
        if "overload" in text:
            return self.send(503, {"error": {"message": "busy"}})
        answers = {qid: answer(q, text) for qid, q in req["questions"].items()}
        if "skip" in text:
            answers.pop(next(iter(answers)))
        self.send(200, {"model": NAME, "answers": answers, "usage": {"input_tokens": len(text) // 4, "output_tokens": 0}})

    def log_message(self, fmt, *args):
        print("decide_server: " + fmt % args, file=sys.stderr, flush=True)


def main():
    global NAME
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    NAME = args["--name"]
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    ThreadingHTTPServer(("127.0.0.1", int(args["--port"])), Handler).serve_forever()


if __name__ == "__main__":
    main()
