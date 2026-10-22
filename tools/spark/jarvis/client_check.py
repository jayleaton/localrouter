"""The /voice web client in a real browser against a running daemon: headless Chrome driven over the DevTools protocol.

Python 3 standard library only (it runs on the GPU host with `python3 -I`, next to google-chrome). Steps:
  1. Load BASE/voice (optionally with PAGE written over it before it parses, so a branch's client is checked against the deployed
     daemon before it is deployed) and wait for the model list and the status strip.
  2. Chat: send a message, see the reply stream (more than one piece) into an assistant message.
  3. Image: ask for a picture, see the model call generate_image and the PNG render inline.
  4. Voice (with --wav): Chrome's fake microphone plays WAV once (padded with silence); start voice hands-free and
     see the transcript, a spoken reply and its measured "speech end to first audio".
Screenshots (desktop and phone width) and results.json go to OUT. Exit 0 only when every check passed.

    python3 -I client_check.py --base https://host:10443 --out ~/jarvis/client-shots/RUN [--page voice.html] [--wav p.wav]
"""
import argparse
import base64
import json
import os
import select
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import wave


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Cdp:
    """One DevTools WebSocket (to a page target): commands with ids, events kept in order."""

    def __init__(self, ws_url):
        u = urllib.parse.urlparse(ws_url)
        self.sock = socket.create_connection((u.hostname, u.port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nUpgrade: websocket\r\n"
                           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        self.buf = b""
        while b"\r\n\r\n" not in self.buf:
            self.buf += self.sock.recv(4096)
        head, self.buf = self.buf.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError(f"no upgrade: {head[:200]!r}")
        self.next_id = 0
        self.events = []
        self.handlers = {}

    def _send(self, obj):
        data = json.dumps(obj).encode()
        mask = os.urandom(4)
        n = len(data)
        head = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536
                                else bytes([0x80 | 127]) + struct.pack(">Q", n))
        self.sock.sendall(head + mask + bytes(b ^ mask[i & 3] for i, b in enumerate(data)))

    def _frame(self, timeout):
        """The next whole text message, or None after `timeout` s."""
        deadline = time.monotonic() + timeout
        parts = b""
        while True:
            b = self.buf
            if len(b) >= 2:
                op, n, at = b[0] & 0x0F, b[1] & 0x7F, 2
                if n == 126 and len(b) >= 4:
                    n, at = struct.unpack(">H", b[2:4])[0], 4
                elif n == 127 and len(b) >= 10:
                    n, at = struct.unpack(">Q", b[2:10])[0], 10
                if (b[1] & 0x7F) < 126 or at > 2:
                    if len(b) >= at + n:
                        payload, self.buf = b[at:at + n], b[at + n:]
                        if op in (1, 0):
                            parts += payload
                            if b[0] & 0x80:
                                return parts.decode()
                        elif op == 8:
                            raise ConnectionError("DevTools closed the connection")
                        continue
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            r, _, _ = select.select([self.sock], [], [], left)
            if not r:
                return None
            more = self.sock.recv(1 << 20)
            if not more:
                raise ConnectionError("DevTools closed the connection")
            self.buf += more

    def pump(self, timeout=0.05):
        msg = self._frame(timeout)
        while msg is not None:
            m = json.loads(msg)
            if "method" in m:
                self.events.append(m)
                h = self.handlers.get(m["method"])
                if h:
                    h(m.get("params", {}))
            else:
                self.events.append(m)
            msg = self._frame(0)

    def call(self, method, timeout=60, **params):
        self.next_id += 1
        my = self.next_id
        self._send({"id": my, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for i, e in enumerate(self.events):
                if e.get("id") == my:
                    del self.events[i]
                    if "error" in e:
                        raise RuntimeError(f"{method}: {e['error']}")
                    return e.get("result", {})
            self.pump(0.1)
        raise TimeoutError(f"{method}: no answer within {timeout} s")

    def js(self, expr, timeout=60):
        r = self.call("Runtime.evaluate", timeout=timeout, expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"page script failed: {r['exceptionDetails'].get('text')} {r['exceptionDetails'].get('exception', {}).get('description', '')}")
        return r.get("result", {}).get("value")

    def wait(self, expr, timeout, what):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            v = self.js(expr)
            if v:
                return v
            self.pump(0.5)
        raise TimeoutError(f"{what}: not within {timeout} s")


def padded_wav(src, dst, lead_s=1.5, tail_s=40):
    """src (16-bit mono WAV) with silence before it and a long tail, so Chrome's looping fake microphone says it once."""
    with wave.open(src, "rb") as w:
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("the WAV must be 16-bit")
    with wave.open(dst, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0" * int(rate * lead_s) * 2 * ch + frames + b"\0" * int(rate * tail_s) * 2 * ch)


def shot(cdp, out, name, width, height, mobile):
    cdp.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=2 if mobile else 1, mobile=mobile)
    time.sleep(0.6)
    cdp.js("document.getElementById('log').scrollTop = 1e9")
    time.sleep(0.3)
    data = cdp.call("Page.captureScreenshot", format="png")["data"]
    path = os.path.join(out, name + ".png")
    with open(path, "wb") as f:
        f.write(base64.b64decode(data))
    log("screenshot", path)
    return path


LAST_ASSISTANT = """(() => { const m = window.jarvis.conversation.msgs.filter(x => x.role === 'assistant').pop();
  return m ? { text: m.text, streaming: !!m.streaming, error: m.error || '', lat: m.lat || {}, via: m.via } : null; })()"""


def send(cdp, text):
    cdp.js(f"(() => {{ const t = document.getElementById('text'); t.value = {json.dumps(text)}; document.getElementById('send').click(); }})()")


def click(cdp, element_id):
    """A real mouse click (user activation, pointer events) in the middle of the element."""
    box = cdp.js(f"(() => {{ const r = document.getElementById('{element_id}').getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()")
    for t in ("mousePressed", "mouseReleased"):
        cdp.call("Input.dispatchMouseEvent", type=t, x=box[0], y=box[1], button="left", clickCount=1)
        time.sleep(0.08)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the daemon's origin, e.g. https://dgx-01.example.ts.net:10443")
    ap.add_argument("--out", required=True)
    ap.add_argument("--page", help="serve this file as /voice instead of the daemon's own (a branch's client)")
    ap.add_argument("--wav", help="a spoken prompt for the fake microphone (16-bit WAV); without it the voice step is skipped")
    ap.add_argument("--chrome", default=shutil.which("google-chrome") or shutil.which("chromium") or "google-chrome")
    ap.add_argument("--chat", default="In two short sentences: what is a GPU, and why does it suit neural networks?")
    ap.add_argument("--image", default="Please make me a picture of a red fox sitting in fresh snow.")
    ap.add_argument("--timeout", type=int, default=600, help="for the image (a cold image model loads first)")
    ap.add_argument("--steps", default="chat,image,voice", help="which of chat, image and voice to run, in that order")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results, checks = {"base": a.base, "page": a.page or "(the daemon's)"}, []

    def check(name, ok, detail=None):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        log(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))

    profile = tempfile.mkdtemp(prefix="jarvis-client-")
    flags = ["--headless=new", "--remote-debugging-port=0", f"--user-data-dir={profile}", "--no-first-run",
             "--no-default-browser-check", "--disable-gpu", "--window-size=1280,900", "--autoplay-policy=no-user-gesture-required",
             "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", "--mute-audio"]
    if a.wav:
        padded = os.path.join(profile, "mic.wav")
        padded_wav(a.wav, padded)
        flags.append(f"--use-file-for-fake-audio-capture={padded}%noloop")
    chrome = subprocess.Popen([a.chrome, *flags, "about:blank"], stdout=subprocess.DEVNULL, stderr=open(os.path.join(a.out, "chrome.log"), "w"))
    try:
        port_file = os.path.join(profile, "DevToolsActivePort")
        for _ in range(200):
            if os.path.exists(port_file) and open(port_file).read().strip():
                break
            time.sleep(0.1)
        port = int(open(port_file).read().split()[0])
        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
        page = next(t for t in targets if t["type"] == "page")
        cdp = Cdp(page["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("Network.enable")
        console = []
        cdp.handlers["Runtime.exceptionThrown"] = lambda p: console.append(p.get("exceptionDetails", {}).get("text", ""))
        cdp.handlers["Runtime.consoleAPICalled"] = lambda p: p.get("type") == "error" and console.append(
            " ".join(str(x.get("value", x.get("description", ""))) for x in p.get("args", [])))
        if a.page:  # the branch's client in place of the daemon's, on the real /voice document (its URL, its CSP): the
            # old page is discarded before it parses. (A document made by Fetch.fulfillRequest cannot open WebSockets.)
            cdp.call("Page.addScriptToEvaluateOnNewDocument", source="if (location.pathname.replace(/\\/$/, '') === '/voice' && !window.__swapped) {"
                     " window.__swapped = 1; document.open(); document.write(%s); document.close(); }" % json.dumps(open(a.page).read()))
        cdp.call("Page.navigate", url=a.base.rstrip("/") + "/voice")
        cdp.wait("window.jarvis && document.querySelectorAll('#strip .chip').length > 0", 60, "the page and the status strip")
        results["strip"] = cdp.js("[...document.querySelectorAll('#strip .chip')].map(c => c.textContent + ':' + (c.dataset.s || ''))")
        check("page: loads, lists the models and shows their state", len(results["strip"]) >= 2, results["strip"])
        cdp.js("localStorage.clear(); document.getElementById('new').click(); true")
        shot(cdp, a.out, "01-loaded-desktop", 1280, 900, False)

        steps = set(a.steps.split(","))
        # Chat: a streamed reply.
        if "chat" in steps:
            t0 = time.monotonic()
            send(cdp, a.chat)
            cdp.wait(LAST_ASSISTANT.replace("return m ?", "return m && !m.streaming && (m.text || m.error) ?"), 180, "the chat reply")
            r = cdp.js(LAST_ASSISTANT)
            done = cdp.js("window.jarvis.debug.filter(e => e.kind === 'chat.done').pop()")
            results["chat"] = {"reply": r["text"], "error": r["error"], "lat": r["lat"], "done": done, "s": round(time.monotonic() - t0, 2)}
            check("chat: a streamed reply in the conversation", r["text"] and not r["error"] and (done or {}).get("tokens", 0) > 1,
                  results["chat"])
            check("chat: first token time measured", (r["lat"] or {}).get("ttft") is not None, r["lat"])
            check("chat: the reply is rendered (markdown body)", cdp.js("!!document.querySelector('.msg.assistant .body p, .msg.assistant .body li')"))
            shot(cdp, a.out, "02-chat-desktop", 1280, 900, False)

        # Image: the model calls generate_image; the PNG renders inline.
        if "image" in steps:
            send(cdp, a.image)
            loaded = cdp.wait("(() => { const i = [...document.querySelectorAll('.media img')].pop(); return i && i.complete && i.naturalWidth > 0 ? [i.naturalWidth, i.naturalHeight, i.getAttribute('src')] : null; })()",
                              a.timeout, "the generated image")
            cdp.wait(LAST_ASSISTANT.replace("return m ?", "return m && !m.streaming ?"), 180, "the reply after the image")
            tool = cdp.js("window.jarvis.debug.filter(e => e.kind === 'tool.done').pop()")
            results["image"] = {"size": loaded[:2], "src": loaded[2], "tool": tool, "after": cdp.js(LAST_ASSISTANT)}
            check("image: generate_image ran and the PNG renders inline", loaded[0] > 0 and (tool or {}).get("name") == "generate_image", results["image"])
            shot(cdp, a.out, "03-image-desktop", 1280, 900, False)
            shot(cdp, a.out, "04-image-phone", 390, 844, True)

        # Voice: hands-free, the fake microphone says the prompt once.
        if a.wav and "voice" in steps:
            cdp.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=2, mobile=True)
            time.sleep(0.8)
            results["layout"] = cdp.js("[document.documentElement.scrollWidth, innerWidth]")
            check("phone: nothing wider than the screen", results["layout"][0] <= results["layout"][1], results["layout"])
            click(cdp, "mic")
            cdp.wait("window.jarvis.voice.connected", 20, "the voice session")
            m = cdp.wait("""(() => { const c = window.jarvis.conversation.msgs; const u = c.filter(x => x.role === 'user' && x.via === 'voice' && !x.partial && x.text).pop();
                const b = c.filter(x => x.role === 'assistant' && x.via === 'voice' && !x.streaming && x.text).pop();
                return u && b ? { heard: u.text, reply: b.text, lat: b.lat || {} } : null; })()""", 120, "a spoken turn and its reply")
            cdp.wait("window.jarvis.voice.live === 0", 60, "the reply's audio to finish")
            partials = cdp.js("window.jarvis.debug.filter(e => e.kind === 'voice.in' && e.type === 'localrouter.transcription.partial').length")
            results["voice"] = {**m, "partials": partials, "lat": {k: v for k, v in m["lat"].items() if k != "server"}}
            check("voice: the spoken prompt is transcribed into the conversation", len(m["heard"]) > 3, m["heard"])
            check("voice: a spoken reply, with speech end to first audio measured", m["lat"].get("firstAudio") is not None, results["voice"]["lat"])
            shot(cdp, a.out, "05-voice-phone", 390, 844, True)
            click(cdp, "hangup")
        results["console_errors"] = console[:20]
        check("page: no script errors", not [c for c in console if "Uncaught" in c or "Error" in c], console[:5])
        results["debug_tail"] = cdp.js("window.jarvis.debug.slice(-60)")
    except Exception as e:  # a failed step is a failed check, with its reason
        check(f"no step failed: {type(e).__name__}", False, str(e))
        try:
            results["console_errors"] = console[:20]
            results["debug_tail"] = cdp.js("window.jarvis.debug.slice(-80)")
            results["websocket"] = [{"method": e["method"], **{k: v for k, v in e["params"].items() if k in ("errorMessage", "response", "url")}}
                                    for e in cdp.events if e.get("method", "").startswith("Network.webSocket") and "Frame" not in e["method"]][:20]
            shot(cdp, a.out, "99-failure", 1280, 900, False)
        except Exception:
            pass
    finally:
        chrome.terminate()
        try:
            chrome.wait(10)
        except subprocess.TimeoutExpired:
            chrome.kill()
        shutil.rmtree(profile, ignore_errors=True)
    results["checks"] = checks
    results["passed"] = bool(checks) and all(c["ok"] for c in checks)
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    log("client check:", "PASSED" if results["passed"] else "FAILED")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
