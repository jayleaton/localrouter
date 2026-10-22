"""The /voice web client in a real browser against a running daemon: headless Chrome driven over the DevTools protocol.

Python 3 standard library only (it runs on the GPU host with `python3 -I`, next to google-chrome). Steps:
  1. Load BASE/voice (optionally with PAGE written over it before it parses, so a branch's client is checked against the deployed
     daemon before it is deployed) and wait for the model list and the status strip.
  2. Chat: send a message, see the reply stream into an assistant message with Markdown and a highlighted code block;
     light and dark themes; accessible names on every control.
  3. Image: ask for a picture, see the model call generate_image and the PNG render as a card that enlarges and downloads.
     Phone layout: nothing wider than the screen, touch targets large enough. Voice picker: engines, voices, a preview.
  4. Voice (with --wav): Chrome's fake microphone plays WAV once (padded with silence); start voice hands-free and
     see the transcript, a spoken reply and its measured "speech end to first audio"; each voice-view state photographed.
     Then typed prompts in voice mode (three one-sentence replies and one longer, contended one); the page's own output is recorded after its mixer, and no
     reply may have a playback gap over 30 ms or a click in its first 2 s. --start-wait-ms 0 measures the old start.
Screenshots (desktop and phone width) and results.json go to OUT. Exit 0 only when every check passed.

    python3 -I client_check.py --base https://host:10443 --out ~/jarvis/client-shots/RUN [--page voice.html] [--wav p.wav]
"""
import argparse
import base64
import json
import os
import re
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


def snap(cdp, out, name):
    """A screenshot of the page as it is now (no resize, so a moment in an animation is not lost)."""
    data = cdp.call("Page.captureScreenshot", format="png")["data"]
    path = os.path.join(out, name + ".png")
    with open(path, "wb") as f:
        f.write(base64.b64decode(data))
    log("screenshot", path)
    return path


def size(cdp, width, height, mobile):
    cdp.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=2 if mobile else 1, mobile=mobile)
    time.sleep(0.6)


def theme(cdp, dark):
    cdp.call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "dark" if dark else "light"}])
    time.sleep(0.3)


def shot(cdp, out, name, width, height, mobile):
    size(cdp, width, height, mobile)
    cdp.js("document.getElementById('scroll').scrollTop = 1e9")
    time.sleep(0.3)
    return snap(cdp, out, name)


LAST_ASSISTANT = """(() => { const m = window.jarvis.conversation.msgs.filter(x => x.role === 'assistant').pop();
  return m ? { text: m.text, streaming: !!m.streaming, error: m.error || '', lat: m.lat || {}, via: m.via } : null; })()"""


def send(cdp, text):
    cdp.js(f"(() => {{ const t = document.getElementById('text'); t.value = {json.dumps(text)}; t.dispatchEvent(new Event('input')); document.getElementById('send').click(); }})()")


def click(cdp, element_id=None, selector=None):
    """A real mouse click (user activation, pointer events) in the middle of the element."""
    q = f"document.getElementById('{element_id}')" if element_id else f"document.querySelector({json.dumps(selector)})"
    box = cdp.js(f"(() => {{ const r = {q}.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()")
    for t in ("mousePressed", "mouseReleased"):
        cdp.call("Input.dispatchMouseEvent", type=t, x=box[0], y=box[1], button="left", clickCount=1)
        time.sleep(0.08)
    time.sleep(0.2)


UNNAMED = """[...document.querySelectorAll('button, a[href], input, select, textarea')].filter(e => e.offsetParent !== null
  && !(e.getAttribute('aria-label') || e.textContent.trim() || e.labels && e.labels.length || e.getAttribute('title') || e.placeholder))
  .map(e => e.outerHTML.slice(0, 80))"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="the daemon's origin, e.g. https://dgx-01.example.ts.net:10443")
    ap.add_argument("--out", required=True)
    ap.add_argument("--page", help="write this file over the daemon's /voice before it parses (a branch's client)")
    ap.add_argument("--wav", help="a spoken prompt for the fake microphone (16-bit WAV); without it the voice step is skipped")
    ap.add_argument("--chrome", default=shutil.which("google-chrome") or shutil.which("chromium") or "google-chrome")
    ap.add_argument("--chat", default="In two short sentences: what is a GPU, and why does it suit neural networks? "
                                      "Then show a three-line Python example in a fenced code block.")
    ap.add_argument("--image", default="Please make me a picture of a red fox sitting in fresh snow.")
    ap.add_argument("--timeout", type=int, default=600, help="for the image (a cold image model loads first)")
    ap.add_argument("--steps", default="chat,image,picker,voice", help="which of chat, image, picker, voice and reel (its own --wav: a spoken picture request) to run, in that order")
    ap.add_argument("--reel-video", action="store_true", help="in the reel step, also generate and play a video (minutes)")
    ap.add_argument("--start-wait-ms", type=int, help="the client's hold on a reply's start (0: start on the first piece, as before)")
    ap.add_argument("--slow-cap-ms", type=int, help="the client's hold on a slow stream's start (0: none, as in #28)")
    ap.add_argument("--preview-max", type=int, default=8, help="voices to preview in the picker step")
    ap.add_argument("--stale-settings", action="store_true", help="start from settings an older client kept (a voice and an engine the daemon no longer lists): they must be cleaned")
    ap.add_argument("--replies", type=int, default=4, help="typed prompts spoken in voice mode after the spoken turn, for playback stats")
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
        padded_wav(a.wav, padded, lead_s=3)
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
        theme(cdp, False)
        size(cdp, 1280, 860, False)
        cdp.call("Page.navigate", url=a.base.rstrip("/") + "/voice")
        cdp.wait("window.jarvis && window.jarvis.ready", 60, "the page, the models and their state")
        results["models"] = cdp.js("[...document.querySelectorAll('#statusPop .model')].map(m => m.textContent)")
        check("page: loads, lists the models and shows their state", len(results["models"]) >= 2, results["models"])
        results["pill"] = cdp.js("[document.getElementById('statusPill').dataset.s, document.getElementById('statusPill').textContent]")
        check("page: the status pill shows the stack's state", results["pill"][0] in ("ready", "busy", "loading"), results["pill"])
        if a.stale_settings:  # what an older client kept: Drama 3's pre-#25 voice and an engine that is gone
            cdp.js("""localStorage.setItem('jarvis.settings', JSON.stringify(Object.assign(JSON.parse(localStorage.getItem('jarvis.settings') || '{}'),
                { speechModel: 'retired-tts-model', voices: { 'fishaudio/drama-3': 'default', 'retired-tts-model': 'Bruno' } }))); location.reload(); true""")
            time.sleep(1.5)
            cdp.wait("window.jarvis && window.jarvis.ready", 60, "the page after a reload")
            kept = cdp.js("({ speechModel: window.jarvis.settings.speechModel, voices: window.jarvis.settings.voices })")
            check("settings: a voice or engine the daemon no longer lists is dropped", kept["speechModel"] == "" and "default" not in (kept["voices"] or {}).values() and "retired-tts-model" not in (kept["voices"] or {}), kept)
        cdp.js("localStorage.clear(); document.getElementById('newChat').click(); true")
        snap(cdp, a.out, "01-empty-desktop-light")
        unnamed = cdp.js(UNNAMED)
        check("a11y: every visible control has an accessible name", not unnamed, unnamed)

        steps = set(a.steps.split(","))
        # Chat: a streamed reply, Markdown, a highlighted code block with its copy button.
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
            check("chat: the reply is rendered (markdown body)", cdp.js("!!document.querySelector('.row.assistant .md p, .row.assistant .md li')"))
            code = cdp.js("(() => { const c = document.querySelector('.row.assistant .code'); return c ? { spans: c.querySelectorAll('.tk, .tf, .ts, .tn, .tc').length, copy: !!c.querySelector('[data-act=copy-code]'), lang: c.querySelector('.code-h span').textContent } : null; })()")
            results["chat"]["code"] = code
            check("chat: a code block, highlighted, with a copy button", code and code["spans"] > 0 and code["copy"], code)
            check("chat: the reply has a timestamp and copy and regenerate actions",
                  cdp.js("(() => { const r = [...document.querySelectorAll('.row.assistant')].pop(); return !!(r && r.querySelector('time') && r.querySelector('[data-act=copy]') && !r.querySelector('[data-act=regen]').hidden); })()"))
            cdp.js("document.getElementById('scroll').scrollTop = 1e9")
            time.sleep(0.3)
            snap(cdp, a.out, "02-chat-desktop-light")
            theme(cdp, True)
            bg = cdp.js("getComputedStyle(document.body).backgroundColor")
            theme(cdp, False)
            bg_light = cdp.js("getComputedStyle(document.body).backgroundColor")
            results["themes"] = {"light": bg_light, "dark": bg}
            check("themes: light and dark follow the system setting", bg != bg_light, results["themes"])
            theme(cdp, True)
            snap(cdp, a.out, "03-chat-desktop-dark")
            shot(cdp, a.out, "04-chat-phone-dark", 390, 844, True)
            theme(cdp, False)
            size(cdp, 1280, 860, False)

        # Image: the model calls generate_image; the PNG renders as a card that enlarges and downloads.
        if "image" in steps:
            send(cdp, a.image)
            loaded = cdp.wait("(() => { const i = [...document.querySelectorAll('.card img')].pop(); return i && i.complete && i.naturalWidth > 0 ? [i.naturalWidth, i.naturalHeight, i.getAttribute('src')] : null; })()",
                              a.timeout, "the generated image")
            cdp.wait(LAST_ASSISTANT.replace("return m ?", "return m && !m.streaming ?"), 180, "the reply after the image")
            tool = cdp.js("window.jarvis.debug.filter(e => e.kind === 'tool.done').pop()")
            results["image"] = {"size": loaded[:2], "src": loaded[2], "tool": tool, "after": cdp.js(LAST_ASSISTANT),
                                "download": cdp.js("(() => { const a = [...document.querySelectorAll('.card a[download]')].pop(); return a ? a.getAttribute('href') : null; })()")}
            check("image: generate_image ran and the PNG renders inline", loaded[0] > 0 and (tool or {}).get("name") == "generate_image", results["image"])
            check("image: the card has a download link", bool(results["image"]["download"]), results["image"]["download"])
            shot(cdp, a.out, "05-image-card-desktop", 1280, 860, False)
            cdp.js("[...document.querySelectorAll('.card .thumb')].pop().click(); true")
            time.sleep(0.5)
            check("image: a click enlarges it", cdp.js("document.getElementById('lightbox').open && document.getElementById('lbImg').naturalWidth > 0"))
            snap(cdp, a.out, "06-image-enlarged")
            cdp.js("document.getElementById('lightbox').close(); true")
            shot(cdp, a.out, "07-image-card-phone", 390, 844, True)

        size(cdp, 390, 844, True)
        results["layout"] = cdp.js("[document.documentElement.scrollWidth, innerWidth]")
        check("phone: nothing wider than the screen", results["layout"][0] <= results["layout"][1], results["layout"])
        small = cdp.js("""[...document.querySelectorAll('.composer button, .top button')].filter(e => e.offsetParent !== null)
            .map(e => e.getBoundingClientRect()).filter(r => r.width < 36 || r.height < 36).length""")
        check("phone: the composer's and top bar's touch targets are large enough", small == 0, small)

        # The voice picker: the engines, their voices (from a voice-list API when there is one), a preview.
        if "picker" in steps:
            # The picker lists exactly what /v1/models lists; one engine: no engine choice; every voice previews.
            speech = cdp.js("fetch('/v1/models').then(r => r.json()).then(j => j.data.filter(m => m.kind === 'speech').map(m => ({ id: m.id, voices: m.voices || [], def: m.default_voice })))")
            cdp.js("document.getElementById('gear').click(); true")
            cdp.wait("document.querySelectorAll('#voices .vrow').length > 0", 20, "the voice picker")
            engine_hidden = cdp.js("document.getElementById('engines').parentElement.hidden")
            check("picker: the engine choice shows only when there is more than one engine", engine_hidden == (len(speech) < 2), {"engines": [s["id"] for s in speech], "hidden": engine_hidden})
            shown = cdp.js("[...document.querySelectorAll('#voices .vrow')].map(b => b.dataset.voice)")
            want = next((s["voices"] for s in speech if s["id"] == cdp.js("window.jarvis.engine")), [])
            check("picker: the voices are exactly what /v1/models lists", shown == want, {"shown": shown, "listed": want})
            cdp.js("document.getElementById('voiceSect').scrollIntoView(); true")
            time.sleep(0.3)
            snap(cdp, a.out, "08-voice-picker-phone")
            previews = []
            for v in shown[:a.preview_max]:
                mark = cdp.js("window.jarvis.debug.length")
                cdp.js(f"document.querySelector('#voices .play[data-voice=\"{v}\"]').click(); true")
                prev = cdp.wait(f"window.jarvis.debug.slice({mark}).find(e => e.kind === 'voice.preview' && e.voice === {json.dumps(v)})", 150, f"the preview of {v}")
                previews.append(prev)
                cdp.js("(() => { const p = document.querySelector('.play[data-playing=\"1\"]'); if (p) p.click(); return true; })()")
            results["picker"] = {"speech_models": speech, "shown": shown, "previews": previews}
            bad = [p for p in previews if not p or p.get("error") or p.get("bytes", 0) < 1000]
            check(f"picker: a preview of each voice plays ({len(previews)})", previews and not bad, bad or [(p["voice"], p["bytes"], p["ms"]) for p in previews])
            cdp.js("document.getElementById('drawer').close(); true")
            time.sleep(0.4)

        # Voice: hands-free, the fake microphone says the prompt once; each state of the voice view is photographed.
        if a.wav and "voice" in steps:
            theme(cdp, True)
            size(cdp, 390, 844, True)
            cdp.js("window.jarvis.recordOutput = true" + (f"; window.jarvis.settings.startWaitMs = {a.start_wait_ms}" if a.start_wait_ms is not None else "")
                   + (f"; window.jarvis.settings.slowCapMs = {a.slow_cap_ms}" if a.slow_cap_ms is not None else ""))
            click(cdp, "mic")
            cdp.wait("window.jarvis.voice.connected", 20, "the voice session")
            seen, order = {}, ["connecting", "listening", "hearing", "thinking", "speaking"]
            want = """(() => { const c = window.jarvis.conversation.msgs; const u = c.filter(x => x.role === 'user' && x.via === 'voice' && !x.partial && x.text).pop();
                const b = c.filter(x => x.role === 'assistant' && x.via === 'voice' && !x.streaming && x.text).pop();
                return u && b && window.jarvis.voice.live === 0 ? { heard: u.text, reply: b.text, lat: b.lat || {} } : null; })()"""
            deadline, m = time.monotonic() + 150, None
            while time.monotonic() < deadline:
                st = cdp.js("window.jarvis.ui.voiceState")
                if st in order and st not in seen:
                    seen[st] = snap(cdp, a.out, f"1{order.index(st)}-voice-{st}-phone")
                m = cdp.js(want)
                if m and len(seen) >= 4:
                    break
                cdp.pump(0.08)
            if not m:
                m = cdp.wait(want, 30, "a spoken turn and its reply")
            partials = cdp.js("window.jarvis.debug.filter(e => e.kind === 'voice.in' && e.type === 'localrouter.transcription.partial').length")
            results["voice"] = {**m, "partials": partials, "states": sorted(seen), "lat": {k: v for k, v in m["lat"].items() if k != "server"}}
            check("voice: the spoken prompt is transcribed into the conversation", len(m["heard"]) > 3, m["heard"])
            check("voice: a spoken reply, with speech end to first audio measured", m["lat"].get("firstAudio") is not None, results["voice"]["lat"])
            check("voice: the view showed listening, hearing, thinking and speaking", {"listening", "hearing", "thinking", "speaking"} <= set(seen), sorted(seen))
            # The connection drops mid-call (a service restart, a proxy): the page reconnects and the call goes on.
            mark = cdp.js("window.jarvis.debug.length")
            cdp.js("window.jarvis.dropVoiceSocket(); true")
            back = cdp.wait(f"window.jarvis.debug.slice({mark}).find(e => e.kind === 'voice.reconnected')", 40, "the voice session to reconnect")
            check("voice: a dropped connection reconnects and the call goes on", back and cdp.js("window.jarvis.voice.connected"), back)
            cdp.wait("window.jarvis.ui.voiceState === 'listening'", 30, "listening again")
            click(cdp, "talk")
            check("voice: the big button mutes", cdp.js("window.jarvis.ui.voiceState") == "muted")
            snap(cdp, a.out, "15-voice-muted-phone")
            click(cdp, "talk")
            click(cdp, selector=".seg [data-mode=ptt]")
            check("voice: push to talk is offered", cdp.js("window.jarvis.ui.voiceState") == "ptt")
            snap(cdp, a.out, "16-voice-push-to-talk-phone")
            click(cdp, selector=".seg [data-mode=vad]")
            theme(cdp, False)
            snap(cdp, a.out, "17-voice-listening-phone-light")
            click(cdp, "vmMin")
            snap(cdp, a.out, "18-chat-with-voice-phone-light")
            # Playback: typed prompts in voice mode get spoken replies too; the page's own output is analysed per reply.
            # Three one-sentence replies (defaults), then a three-sentence one with "smooth start" on: the chat model still
            # decodes it while its first sentence is spoken, and speech then shares the GPU (slower than playback).
            for n, text in enumerate(["Say good morning in one sentence.", "Name two colours of the rainbow, in a few words.",
                                      "What is two plus two? Answer in one short sentence.",
                                      "Tell me a little about lighthouses, in three sentences."][:a.replies]):
                if n == 3 and a.slow_cap_ms is None:  # the long reply: the "smooth start" setting is what keeps it gapless
                    cdp.js("window.jarvis.settings.smoothStart = true")
                before = cdp.js("window.jarvis.conversation.msgs.filter(x => x.role === 'assistant' && x.via === 'voice' && !x.streaming && x.text).length")
                send(cdp, text)
                cdp.wait(f"window.jarvis.conversation.msgs.filter(x => x.role === 'assistant' && x.via === 'voice' && !x.streaming && x.text).length > {before} && window.jarvis.voice.live === 0", 120, f"spoken reply {n + 1}")
                time.sleep(0.6)
            onsets = cdp.js("window.jarvis.output(2000, 30)") or []
            gaps = cdp.js("window.jarvis.voice.gaps")
            lat = cdp.js("window.jarvis.conversation.msgs.filter(x => x.role === 'assistant' && x.via === 'voice' && x.lat && x.lat.firstAudio != null).map(x => x.lat.firstAudio)")
            bad = [o for o in onsets if o["gaps_over"] or o["longest_gap_ms"] > 30 or o["clicks"]]
            results["playback"] = {"onsets": onsets, "schedule_gaps": gaps, "first_audio_ms": lat, "start_wait_ms": a.start_wait_ms, "slow_cap_ms": a.slow_cap_ms, "slow_starts": cdp.js("window.jarvis.debug.filter(e => e.kind === 'voice.slow_start').length"), "arrivals": cdp.js("window.jarvis.arrivals")}
            check("voice: replies were recorded as played", len(onsets) >= 1 + min(a.replies, 1), len(onsets))
            check("voice: no playback gap over 30 ms or click in the first 2 s of any reply", not bad, bad or onsets)
            click(cdp, "vbarEnd")
        # The reel, in a call: the fake microphone asks for a picture. The reel (and the chat) show a placeholder, then the
        # picture; nothing before it says it is ready, and a confirmation is spoken after it. A request the image model
        # refuses shows a failed card and is admitted aloud. The picture opens in the lightbox while a turn is answered;
        # then a video plays. The realtime socket stays open throughout.
        if a.wav and "reel" in steps:
            theme(cdp, True)
            size(cdp, 1440, 900, False)
            VOICE_REPLIES = "window.jarvis.conversation.msgs.filter(x => x.role === 'assistant' && x.via === 'voice' && !x.streaming && x.text)"
            FINISHED = "window.jarvis.debug.filter(e => e.kind === 'tool.finished')"
            click(cdp, "mic")
            cdp.wait("window.jarvis.voice.connected && window.jarvis.ui.voiceState === 'listening'", 30, "the voice session")
            cdp.wait("!!(document.querySelector('#reelList .rc.pending') && document.querySelector('#log .card.pending'))", 90, "a placeholder in the reel and the chat")
            check("reel: the spoken request shows a placeholder in the reel and in the chat", True)
            snap(cdp, a.out, "20-reel-placeholder-desktop")
            got = cdp.wait("(() => { const i = document.querySelector('#reelList .rc:first-child img'), c = [...document.querySelectorAll('#log .card img')].pop(); return i && i.complete && i.naturalWidth > 0 && c && c.naturalWidth > 0 ? [i.naturalWidth, c.naturalWidth] : null; })()", a.timeout, "the picture in the reel and the chat")
            check("reel: the picture replaces the placeholder, in the reel and the chat", bool(got), got)
            done_t = cdp.js(f"{FINISHED}.filter(e => e.status === 'completed').map(e => e.t)[0]")
            cdp.wait(f"{VOICE_REPLIES}.some(x => x.at > performance.timeOrigin + {done_t}) && window.jarvis.voice.live === 0", 120, "the confirmation after the picture")
            replies = cdp.js(f"{VOICE_REPLIES}.map(x => ({{ at: x.at - performance.timeOrigin, text: x.text }}))")
            before = [r["text"] for r in replies if r["at"] < done_t]
            after = [r["text"] for r in replies if r["at"] >= done_t]
            claims = [t for t in before if re.search(r"\b(here('s| is| it is| you go)|it'?s ready|is ready|all done|done!)", t, re.I)]
            results["reel_speech"] = {"before_picture": before, "after_picture": after}
            check("reel: nothing said before the picture claims it is ready", not claims, claims or before)
            check("reel: a confirmation is spoken after the picture is ready", bool(after), after)
            snap(cdp, a.out, "21-voice-reel-desktop")
            click(cdp, selector="#reelList .rc:first-child .rc-thumb")
            cdp.wait("document.getElementById('lightbox').open && document.getElementById('lbImg').naturalWidth > 0", 20, "the lightbox")
            state = cdp.js("({ socket: window.jarvis.voice.socket, vm: !document.getElementById('vm').hidden, pill: !document.getElementById('lbVoice').hidden })")
            check("reel: a click opens the lightbox with the call still connected", state["socket"] == 1 and state["vm"] and state["pill"], state)
            snap(cdp, a.out, "22-lightbox-during-call-desktop")
            n = cdp.js(f"{VOICE_REPLIES}.length")
            cdp.js("window.jarvis.say('Say the word hello, nothing else.'); true")
            cdp.wait(f"{VOICE_REPLIES}.length > {n} && window.jarvis.voice.live === 0", 90, "a turn with the lightbox open")
            check("reel: a turn during the lightbox is answered aloud", cdp.js("document.getElementById('lightbox').open && window.jarvis.voice.socket === 1"))
            cdp.js("document.getElementById('lbClose').click(); true")
            time.sleep(0.3)
            check("reel: closing the lightbox returns to the orb", cdp.js("!document.getElementById('lightbox').open && !document.getElementById('vm').hidden && document.activeElement === document.getElementById('orb')"))
            # A request the image model refuses (over its size limit): the failed card, and the reply admits it.
            nf = cdp.js(f"{FINISHED}.length")
            n = cdp.js(f"{VOICE_REPLIES}.length")
            cdp.js("window.jarvis.say('Make me a big poster of a penguin, size 2048x2048.'); true")
            fail = cdp.wait(f"{FINISHED}.slice({nf}).find(e => e.status !== 'completed')", 120, "the refused picture")
            fail_at = cdp.js(f"{FINISHED}.slice({nf}).find(e => e.status !== 'completed').t")
            cdp.wait("!!document.querySelector('#reelList .rc.failed .btn')", 10, "the failed card")
            cdp.wait(f"{VOICE_REPLIES}.length >= {n} + 2 && window.jarvis.voice.live === 0", 120, "the reply after the failure")
            said = cdp.js(f"{VOICE_REPLIES}.filter(x => x.at > performance.timeOrigin + {fail_at}).map(x => x.text)")
            check("reel: a refused picture shows a failed card with Retry", True, fail)
            check("reel: the reply after it admits the failure", any(re.search(r"\b(only|can'?t|cannot|couldn'?t|fail|sorry|unable|limit|up to|instead|too large)\b", t, re.I) for t in said), said)
            snap(cdp, a.out, "23-reel-failed-desktop")
            if a.reel_video:
                k = cdp.js("document.querySelectorAll('#reelList .rc').length")
                cdp.js("window.jarvis.say('Please make a two second video of ocean waves at sunset.'); true")
                cdp.wait(f"document.querySelectorAll('#reelList .rc').length > {k} && !!document.querySelector('#reelList .rc:first-child.pending')", 120, "a video placeholder")
                got = cdp.wait("(() => { const r = document.querySelector('#reelList .rc:first-child'), v = r && r.querySelector('video'); return r && r.classList.contains('failed') ? 'failed' : v && v.readyState >= 1 ? 'video' : null; })()", 1500, "the video in the reel")
                check("reel: the video arrives in the reel", got == "video", got if got == "video" else cdp.js(f"{FINISHED}.slice(-1)[0]"))
                if got != "video":
                    raise RuntimeError("the video failed")
                click(cdp, selector="#reelList .rc:first-child .rc-thumb")
                cdp.wait("(() => { const v = document.getElementById('lbVid'); return document.getElementById('lightbox').open && !v.hidden && v.currentTime > 0.2; })()", 30, "the video playing")
                check("reel: the video plays in the lightbox with the call connected", cdp.js("window.jarvis.voice.socket === 1"))
                snap(cdp, a.out, "24-lightbox-video-during-call-desktop")
                cdp.js("document.getElementById('lbClose').click(); true")
            size(cdp, 390, 844, True)
            time.sleep(0.6)
            orb = cdp.js("(() => { const o = document.getElementById('orb').getBoundingClientRect(), r = document.getElementById('reel').getBoundingClientRect(); return { orb_bottom: Math.round(o.bottom), reel_top: Math.round(r.top), visible: r.height > 0, reel: [r.x, r.y, r.width, r.height].map(Math.round), inner: [innerWidth, innerHeight], phone_css: matchMedia('(max-width: 860px)').matches, vm: getComputedStyle(document.getElementById('vm')).flexDirection, main: (b => [b.y, b.height].map(Math.round))(document.querySelector('.vm-main').getBoundingClientRect()), top_bottom: Math.round(document.querySelector('.vm-top').getBoundingClientRect().bottom), orb_top: Math.round(o.top), reply_bottom: Math.round(document.querySelector('#vm .vt').getBoundingClientRect().bottom), controls_top: Math.round(document.querySelector('#vm .controls').getBoundingClientRect().top) }; })()")
            check("reel: on a phone the reel sits below the orb, not over it", orb["visible"] and orb["orb_bottom"] <= orb["reel_top"], orb)
            check("reel: on a phone the stage fits between the top bar and the controls", orb["orb_top"] >= orb["top_bottom"] and orb["reply_bottom"] <= orb["controls_top"], orb)
            snap(cdp, a.out, "25-voice-reel-phone")
            closes = cdp.js("window.jarvis.debug.filter(e => e.kind === 'voice.closed').length")
            back = cdp.js("window.jarvis.debug.filter(e => e.kind === 'voice.reconnected').length")
            voices = cdp.js("window.jarvis.debug.filter(e => e.kind === 'voice.in' && e.type === 'session.updated').map(e => e.session && e.session.voice)")
            check("reel: the call stayed on throughout (any dropped connection reconnected)", closes == back and cdp.js("window.jarvis.voice.socket") == 1, {"drops": closes, "reconnects": back})
            check("voice: the session is never given a voice the model lacks (no 'default')", "default" not in voices, voices)
            results["reel"] = {"cards": cdp.js("[...document.querySelectorAll('#reelList .rc')].map(r => r.dataset.key + ' ' + r.className)"), "phone": orb}
            click(cdp, "vmEnd")
        results["console_errors"] = console[:20]
        check("page: no script errors", not [c for c in console if "Uncaught" in c or "Error" in c], console[:5])
        results["debug_tail"] = cdp.js("window.jarvis.debug.slice(-60)")
    except Exception as e:  # a failed step is a failed check, with its reason
        check(f"no step failed: {type(e).__name__}", False, str(e))
        try:
            shot(cdp, a.out, "99-failure", 1280, 900, False)
            results["console_errors"] = console[:20]
            results["websocket"] = [{"method": e["method"], **{k: v for k, v in e["params"].items() if k in ("errorMessage", "response", "url")}}
                                    for e in cdp.events if e.get("method", "").startswith("Network.webSocket") and "Frame" not in e["method"]][:20]
            results["debug_tail"] = cdp.js("window.jarvis && window.jarvis.debug.slice(-250)")
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
