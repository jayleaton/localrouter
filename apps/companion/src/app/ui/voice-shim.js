// LocalRouter app: runs before the bundled /voice page (src/voice/voice.html). Its realtime socket goes through the
// app's harness (client tool mode: the tools run on this computer), never straight to the box, and the device token
// stays in the app. Everything else the page does is unchanged: its fetches to /v1/... are answered by the app.
'use strict';
(() => {
  const bridge = window.companion && window.companion.realtime;
  const Native = window.WebSocket;
  class CompanionSocket {
    constructor(url, protocols) {
      this.url = String(url);
      this.protocol = 'realtime';
      this.readyState = CompanionSocket.CONNECTING;
      this.onopen = this.onmessage = this.onclose = this.onerror = null;
      this.id = null;
      this.queue = [];
      void protocols;
      bridge.open({
        onOpen: () => {
          if (this.readyState !== CompanionSocket.CONNECTING) return;
          this.readyState = CompanionSocket.OPEN;
          if (this.onopen) this.onopen({ type: 'open' });
        },
        onMessage: (data) => {
          hop(data);
          if (this.readyState === CompanionSocket.OPEN && this.onmessage) this.onmessage({ type: 'message', data });
        },
        onClose: (code, reason) => this.closed(code, reason),
      }).then((id) => {
        this.id = id;
        if (this.readyState === CompanionSocket.CLOSING) bridge.close(id);
      }, (error) => this.closed(1000, (error && error.message ? error.message : String(error)).replace(/^Error invoking remote method 'rt:open': (Error: )?/, '')));
    }
    send(data) {
      if (this.readyState !== CompanionSocket.OPEN || !this.id) throw new DOMException('not open', 'InvalidStateError');
      bridge.send(this.id, String(data));
    }
    close() {
      if (this.readyState >= CompanionSocket.CLOSING) return;
      this.readyState = CompanionSocket.CLOSING;
      if (this.id) bridge.close(this.id);
      this.closed(1000, '');
    }
    closed(code, reason) {
      if (this.readyState === CompanionSocket.CLOSED) return;
      this.readyState = CompanionSocket.CLOSED;
      if (code !== 1000 && this.onerror) this.onerror({ type: 'error' });
      if (this.onclose) this.onclose({ type: 'close', code, reason: reason || '', wasClean: code === 1000 });
    }
  }
  // The app's own hop for each piece of speech: from the app's process having it to this page having it, and the gaps
  // between pieces as they reach the page. Logged per reply into the page's debug log (Settings › Copy debug log), so a
  // slow box (gaps already there when the app got them) and a slow hop (lag here) can be told apart on the user's Mac.
  const H = { lags: [], gaps: [], last: 0, worst: null };
  window.__companionAudio = H;
  const now = () => performance.timeOrigin + performance.now();
  function hop(data) {
    if (typeof data !== 'string') return;
    if (data.indexOf('"lr_main_ms"') >= 0) {
      try {
        const ev = JSON.parse(data);
        const t = now();
        H.lags.push(t - ev.lr_main_ms);
        if (H.last) H.gaps.push(t - H.last);
        H.last = t;
      } catch { /* not ours */ }
    } else if (data.indexOf('"response.done"') >= 0 && H.lags.length) {
      const q = (a, f) => { const b = [...a].sort((x, y) => x - y); return b.length ? Math.round(b[Math.min(b.length - 1, Math.floor(f * b.length))]) : null; };
      const stats = { pieces: H.lags.length, hop_p50_ms: q(H.lags, 0.5), hop_p95_ms: q(H.lags, 0.95), hop_max_ms: q(H.lags, 1), gap_p95_ms: q(H.gaps, 0.95), gap_max_ms: q(H.gaps, 1) };
      H.lags = []; H.gaps = []; H.last = 0;
      if (!H.worst || stats.hop_max_ms > H.worst.hop_max_ms) H.worst = stats;
      if (typeof dbg === 'function') dbg('app.audio_hop', stats); // eslint-disable-line no-undef
    }
  }
  CompanionSocket.CONNECTING = 0;
  CompanionSocket.OPEN = 1;
  CompanionSocket.CLOSING = 2;
  CompanionSocket.CLOSED = 3;
  if (bridge) {
    window.WebSocket = function WebSocket(url, protocols) {
      const u = new URL(String(url), location.href);
      if (u.pathname === '/v1/realtime') return new CompanionSocket(u.href, protocols);
      return new Native(url, protocols);
    };
    Object.assign(window.WebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
    window.__companionShim = true;
  }
  // The device token is the app's: the page's own API key field does not apply here.
  document.addEventListener('DOMContentLoaded', () => {
    const key = document.getElementById('key');
    const field = key && key.closest('label');
    if (field) field.hidden = true;
    const style = document.createElement('style');
    style.textContent = 'label.field:has(#key){display:none}';
    document.head.appendChild(style);
  });
})();
