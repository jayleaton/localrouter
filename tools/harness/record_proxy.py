"""Records what a real client sends to LocalRouter, for the harness contract's fixtures (docs/HARNESS.md).

A transparent proxy: LISTEN -> UPSTREAM. Every HTTP request (method, path, a few headers, the JSON body) and every
WebSocket text frame in both directions is appended to OUT as one JSON line. Secrets are redacted before anything is
written: Authorization, `?api_key=`, `openai-insecure-api-key.*` subprotocols, pairing codes and tokens. Audio and
image payloads are replaced by their sizes. Responses are piped through untouched (only their status is recorded), so
streaming (SSE, chunked speech) and the WebSocket upgrade behave as they do without the proxy.

Python 3 standard library only; run it on the box next to the daemon (`python3 -I`):

    python3 -I record_proxy.py --listen 127.0.0.1:18190 --upstream 127.0.0.1:8190 --label web --out web.jsonl
"""
import argparse
import json
import re
import socket
import struct
import sys
import threading
import time
import urllib.parse

LOCK = threading.Lock()
SECRET_KEYS = {"token", "code", "api_key", "authorization", "chat_api_key"}
BLOB_KEYS = {"audio", "delta", "b64_json", "image", "images", "input_reference", "file", "data", "screenshot"}
AUDIO_EVENTS = {"input_audio_buffer.append", "response.audio.delta"}  # audio of any length is a size
KEEP_HEADERS ={"content-type", "accept", "upgrade", "sec-websocket-protocol", "authorization", "user-agent"}


def redact_protocols(value):
    return ", ".join("openai-insecure-api-key.<redacted>" if p.strip().startswith("openai-insecure-api-key.") else p.strip()
                     for p in value.split(","))


def redact_target(target):
    u = urllib.parse.urlsplit(target)
    q = [(k, "<redacted>" if k == "api_key" else v) for k, v in urllib.parse.parse_qsl(u.query, keep_blank_values=True)]
    return urllib.parse.urlunsplit(("", "", u.path, urllib.parse.urlencode(q, safe="<>"), ""))


def scrub(v, key=""):
    """The value with secrets redacted and large payloads replaced by a size note."""
    if isinstance(v, dict):
        return {k: scrub(x, k.lower()) for k, x in v.items()}
    if isinstance(v, list):
        if key in BLOB_KEYS:
            return [scrub(x, key) for x in v[:8]]
        return [scrub(x, key) for x in v]
    if isinstance(v, str):
        if key in SECRET_KEYS:
            return "<redacted>"
        if key in BLOB_KEYS and len(v) > 96:
            return f"<{len(v)} chars>"
        if len(v) > 4000:
            return v[:4000] + f"...<{len(v)} chars>"
    return v


class Log:
    def __init__(self, path, label):
        self.f = open(path, "a", buffering=1)
        self.label = label
        self.t0 = time.time()

    def write(self, rec, conn=None):
        rec = {"client": self.label, "conn": conn if conn is not None else threading.current_thread().name,
               "ms": int((time.time() - self.t0) * 1000), **rec}
        with LOCK:
            self.f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def read_head(sock, buf):
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            return None, buf
        buf += chunk
    head, rest = buf.split(b"\r\n\r\n", 1)
    return head, rest


def parse_head(head):
    lines = head.decode("latin-1").split("\r\n")
    headers = []
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers.append((k.strip(), v.strip()))
    return lines[0], headers


def read_body(sock, rest, headers):
    h = {k.lower(): v for k, v in headers}
    if "content-length" in h:
        n = int(h["content-length"])
        while len(rest) < n:
            chunk = sock.recv(65536)
            if not chunk:
                break
            rest += chunk
        return rest[:n], rest[n:]
    if h.get("transfer-encoding", "").lower() == "chunked":
        body = b""
        while True:
            while b"\r\n" not in rest:
                rest += sock.recv(65536)
            size_line, rest = rest.split(b"\r\n", 1)
            n = int(size_line.split(b";")[0], 16)
            while len(rest) < n + 2:
                rest += sock.recv(65536)
            body, rest = body + rest[:n], rest[n + 2:]
            if n == 0:
                return body, rest
    return b"", rest


def body_record(body, ctype):
    if not body:
        return None
    if "json" in ctype:
        try:
            return scrub(json.loads(body))
        except ValueError:
            pass
    if "multipart/form-data" in ctype:
        fields = {}
        for name, filename, value in re.findall(rb'name="([^"]+)"(?:; filename="([^"]*)")?[^\r]*\r\n(?:[^\r]+\r\n)*\r\n(.*?)\r\n--', body, re.S):
            fields[name.decode()] = f"<file {len(value)} bytes>" if filename else scrub(value.decode("utf-8", "replace"), name.decode())
        return {"multipart": fields}
    return f"<{len(body)} bytes {ctype}>"


def ws_frames(data, masked):
    """Complete frames at the start of `data`: [(opcode, fin, payload)], the bytes left over."""
    out = []
    while len(data) >= 2:
        b0, b1 = data[0], data[1]
        n, i = b1 & 0x7F, 2
        if n == 126:
            if len(data) < 4:
                break
            n, i = struct.unpack(">H", data[2:4])[0], 4
        elif n == 127:
            if len(data) < 10:
                break
            n, i = struct.unpack(">Q", data[2:10])[0], 10
        mask = b""
        if b1 & 0x80:
            mask, i = data[i:i + 4], i + 4
        if len(data) < i + n:
            break
        payload = data[i:i + n]
        if mask:
            payload = bytes(c ^ mask[k % 4] for k, c in enumerate(payload))
        out.append((b0 & 0x0F, bool(b0 & 0x80), payload))
        data = data[i + n:]
    return out, data


def tap(src, dst, log, direction, masked, conn):
    """Pipes src to dst unchanged, recording each WebSocket text message under connection `conn`."""
    pending, partial = b"", b""
    threading.current_thread().name = conn
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
            frames, pending = ws_frames(pending + chunk, masked)
            for op, fin, payload in frames:
                if op in (1, 0):
                    partial += payload
                    if fin:
                        try:
                            ev = json.loads(partial.decode("utf-8"))
                            if isinstance(ev, dict) and ev.get("type") in AUDIO_EVENTS:
                                for k in ("audio", "delta"):
                                    if isinstance(ev.get(k), str):
                                        ev[k] = f"<{len(ev[k])} chars>"
                            log.write({"dir": direction, "ws": scrub(ev)})
                        except ValueError:
                            log.write({"dir": direction, "ws_text": f"<{len(partial)} bytes, not JSON>"})
                        partial = b""
                elif op == 2:
                    log.write({"dir": direction, "ws_binary": len(payload)})
                elif op == 8:
                    log.write({"dir": direction, "ws_close": struct.unpack(">H", payload[:2])[0] if len(payload) >= 2 else None})
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def pipe(src, dst):
    try:
        while True:
            chunk = src.recv(65536)
            if not chunk:
                break
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def serve(client, upstream, log):
    up = None
    try:
        head, rest = read_head(client, b"")
        if head is None:
            return
        start, headers = parse_head(head)
        method, target, version = start.split(" ", 2)
        body, rest = read_body(client, rest, headers)
        h = {k.lower(): v for k, v in headers}
        rec = {"dir": "c2s", "method": method, "path": redact_target(target)}
        kept = {}
        for k, v in headers:
            kl = k.lower()
            if kl not in KEEP_HEADERS:
                continue
            if kl == "authorization":
                v = v.split(" ", 1)[0] + " <redacted>"
            elif kl == "sec-websocket-protocol":
                v = redact_protocols(v)
            kept[kl] = v
        if kept:
            rec["headers"] = kept
        b = body_record(body, h.get("content-type", ""))
        if b is not None:
            rec["body"] = b
        log.write(rec)

        up = socket.create_connection(upstream)
        upgrade = h.get("upgrade", "").lower() == "websocket"
        out = [f"{method} {target} {version}"]
        for k, v in headers:
            kl = k.lower()
            if kl in ("transfer-encoding", "content-length", "keep-alive", "proxy-connection"):
                continue
            if kl == "connection" and not upgrade:
                continue
            out.append(f"{k}: {v}")
        if body:
            out.append(f"Content-Length: {len(body)}")
        if not upgrade:
            out.append("Connection: close")
        up.sendall(("\r\n".join(out) + "\r\n\r\n").encode("latin-1") + body)

        rhead, rrest = read_head(up, b"")
        if rhead is None:
            return
        status = parse_head(rhead)[0].split(" ", 2)[1]
        log.write({"dir": "s2c", "path": redact_target(target), "status": int(status)})
        if not upgrade:
            # One request a connection: the client sees the upstream's answer, then the close.
            rlines = rhead.decode("latin-1").split("\r\n")
            rlines = [ln for ln in rlines if not ln.lower().startswith("connection:")] + ["Connection: close"]
            client.sendall("\r\n".join(rlines).encode("latin-1") + b"\r\n\r\n" + rrest)
            pipe(up, client)
            return
        client.sendall(rhead + b"\r\n\r\n")
        if status != "101":
            client.sendall(rrest)
            pipe(up, client)
            return
        if rest:
            up.sendall(rest)
        conn = threading.current_thread().name
        log_frames_from = threading.Thread(target=tap, args=(up, client, log, "s2c", False, conn), daemon=True)
        log_frames_from.start()
        if rrest:
            client.sendall(rrest)
        tap(client, up, log, "c2s", True, conn)
        log_frames_from.join()
    except Exception as e:  # one broken connection never stops the proxy
        log.write({"proxy_error": repr(e)})
    finally:
        for s in (client, up):
            if s:
                try:
                    s.close()
                except OSError:
                    pass


def hostport(text):
    host, port = text.rsplit(":", 1)
    return host, int(port)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listen", default="127.0.0.1:18190")
    p.add_argument("--upstream", default="127.0.0.1:8190")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    o = p.parse_args()
    log = Log(o.out, o.label)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(hostport(o.listen))
    srv.listen(64)
    print(f"recording {o.listen} -> {o.upstream} as {o.label} into {o.out}", flush=True)
    upstream = hostport(o.upstream)
    n = 0
    while True:
        c, _ = srv.accept()
        n += 1
        threading.Thread(target=serve, args=(c, upstream, log), name=str(n), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
