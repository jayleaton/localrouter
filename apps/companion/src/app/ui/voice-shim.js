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
        onMessage: (data) => { if (this.readyState === CompanionSocket.OPEN && this.onmessage) this.onmessage({ type: 'message', data }); },
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
