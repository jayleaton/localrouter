"""ComfyUI's /ws progress events for tools/comfyui/worker.py: a minimal RFC 6455 client (text frames in, nothing out but
the close). Standard library only."""
import base64
import json
import os
import select
import socket
import struct
import urllib.parse


class Events:
    """ComfyUI's /ws progress events (a minimal RFC 6455 client: text frames in, nothing out but the close)."""

    def __init__(self, base, client_id):
        u = urllib.parse.urlparse(base)
        self.sock = socket.create_connection((u.hostname, u.port or 80), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET /ws?clientId={client_id} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\n"
                           f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                           "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("websocket closed during the handshake")
            head += chunk
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError("websocket upgrade refused")
        self.buf = bytearray(head.split(b"\r\n\r\n", 1)[1])
        self.sock.setblocking(False)

    def poll(self, timeout):
        """Text messages that arrived within `timeout` seconds, parsed."""
        out = []
        try:
            ready, _, _ = select.select([self.sock], [], [], timeout)
        except InterruptedError:
            ready = []
        if ready:
            try:
                chunk = self.sock.recv(1 << 20)
            except BlockingIOError:
                chunk = None
            if chunk == b"":
                raise ConnectionError("websocket closed")
            if chunk:
                self.buf.extend(chunk)
        while len(self.buf) >= 2:
            op, n, i = self.buf[0] & 0x0F, self.buf[1] & 0x7F, 2
            if n == 126:
                if len(self.buf) < 4:
                    break
                n, i = struct.unpack(">H", self.buf[2:4])[0], 4
            elif n == 127:
                if len(self.buf) < 10:
                    break
                n, i = struct.unpack(">Q", self.buf[2:10])[0], 10
            if len(self.buf) < i + n:
                break
            payload = bytes(self.buf[i:i + n])
            del self.buf[:i + n]
            if op == 1:
                try:
                    out.append(json.loads(payload))
                except json.JSONDecodeError:
                    pass
            elif op == 8:
                raise ConnectionError("websocket closed")
        return out

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
