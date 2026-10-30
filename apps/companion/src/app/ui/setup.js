'use strict';
// The settings window. The Connect steps come computed from the main process (src/app/connect.ts): this page draws
// them, sends what is typed and shows what went wrong. It never sees the device token.
const $ = (id) => document.getElementById(id);
const api = window.companion.setup;
let last = null;
let confirming = false;
let discovered = null;
// The Settings name field while the person edits it: a refresh does not overwrite what they typed.
let assistantEdited = false;

const PERM_WORD = { granted: 'granted', denied: 'denied', restricted: 'restricted', 'not-determined': 'not asked yet', 'not-granted': 'not granted', stale: 'kept from an older build', 'n/a': 'not needed here' };
const PERM_DOT = { granted: 'ready', stale: 'failed', denied: 'failed' };

const ipcMessage = (error) => String((error && error.message) || error).replace(/^Error invoking remote method '[^']+': (Error: )?/, '');

/** Text with `commands` in backticks, as code. */
function withCode(el, text) {
  el.replaceChildren(...String(text).split('`').map((part, i) => {
    if (i % 2 === 0) return part;
    const c = document.createElement('code');
    c.textContent = part;
    return c;
  }));
  return el;
}

/** An error card: what went wrong, what to do, and the technical detail under it. */
function alertCard(el, error) {
  el.hidden = !error;
  if (!error) return el.replaceChildren();
  const title = document.createElement('b');
  title.textContent = error.title;
  const parts = [title];
  if (error.hint) {
    const hint = document.createElement('p');
    hint.className = 'hint';
    parts.push(withCode(hint, error.hint));
  }
  if (error.detail && error.detail !== error.hint) {
    const detail = document.createElement('p');
    detail.className = 'detail';
    detail.textContent = error.detail;
    parts.push(detail);
  }
  el.replaceChildren(...parts);
}

function chip(text, small) {
  const c = document.createElement('span');
  c.className = 'chip';
  c.textContent = text;
  if (small) {
    const s = document.createElement('small');
    s.textContent = ` ${small}`;
    c.append(s);
  }
  return c;
}

function renderSteps(s) {
  const c = s.connect;
  const byId = Object.fromEntries(c.steps.map((st) => [st.id, st]));
  for (const st of c.steps) {
    $(`step-${st.id}`).dataset.s = st.status;
    $(`sum-${st.id}`).textContent = st.summary;
  }
  // The name step is there only after a first pairing; the badges count the steps shown.
  $('step-assistant').hidden = !byId.assistant;
  let n = 0;
  for (const li of document.querySelectorAll('.steps > .step')) if (!li.hidden) li.querySelector('.badge span').textContent = String(++n);
  const repairing = Boolean(c.draft && c.draft.repair);

  // 1. Address: open until it is checked; Change goes back to it before pairing.
  $('open-address').hidden = byId.address.status === 'done';
  $('changeAddress').hidden = !(byId.address.status === 'done' && !s.paired);
  alertCard($('err-address'), c.errors.address);
  const typed = $('base').value.trim();
  if (!typed) $('base').value = (c.draft && c.draft.base) || s.base || '';

  // 2. Code: open once the address answers, while not paired or re-pairing.
  $('open-code').hidden = byId.address.status !== 'done' || (s.paired && !repairing);
  $('pairCmd').textContent = `deploy/jarvis/pair.sh add "${$('name').value.trim() || s.name}"`;
  $('cancelRepair').hidden = !repairing;
  $('pair').textContent = repairing ? 'Pair again' : 'Pair';
  $('repairNote').hidden = !(repairing && s.device);
  if (repairing && s.device) $('repairNote').textContent = `The Spark keeps the current entry (“${s.device.name}”) until you revoke it there: deploy/jarvis/pair.sh revoke ${s.device.id}`;
  alertCard($('err-code'), c.errors.code);

  // 3. Connection: the box, this token's scopes and what it can use; re-pair and disconnect.
  const conn = c.connection;
  $('open-connection').hidden = !s.paired || repairing;
  alertCard($('err-connection'), conn && !conn.ok ? conn.error : null);
  $('warn-connection').hidden = !(conn && conn.ok && conn.warning);
  $('warn-connection').textContent = conn && conn.ok && conn.warning ? conn.warning : '';
  $('linked').hidden = !(conn && conn.ok);
  $('pairedBase').textContent = s.base;
  $('pairedName').textContent = s.device ? s.device.name : s.name;
  if (conn && conn.ok) {
    $('serverText').textContent = [`LocalRouter ${conn.server.version}`, conn.server.machine && `on ${conn.server.machine}`, conn.server.contract && `· contract ${conn.server.contract}`].filter(Boolean).join(' ');
    const scopes = conn.scopes.length ? conn.scopes.map((sc) => chip(sc.name, sc.label)) : [chip('every service', '(paired before scopes)')];
    if (!conn.pairing_required) scopes.push(chip('pairing is off on the Spark', ''));
    $('scopes').replaceChildren(...scopes);
    $('caps').replaceChildren(...conn.capabilities.map((cap) => {
      const li = document.createElement('li');
      li.className = cap.ok ? '' : 'no';
      li.textContent = cap.label;
      const d = document.createElement('span');
      d.textContent = cap.detail;
      d.title = cap.detail;
      li.append(d);
      return li;
    }));
  }
  // A token the Spark refuses: re-pairing is the way on.
  $('open').hidden = Boolean(conn && !conn.ok);
  $('repair').classList.toggle('primary', Boolean(conn && !conn.ok && conn.error.kind === 'revoked'));
  $('checkedAt').textContent = conn ? `Checked ${new Date(conn.at).toLocaleTimeString()}` : 'Checking…';
  $('confirmDisconnect').hidden = !confirming || !s.paired;
  $('revokeCmd').textContent = `deploy/jarvis/pair.sh revoke ${s.device ? s.device.id : `"${s.name}"`}`;

  // The assistant's name: asked once after the first pairing, prefilled with the Spark's (Jarvis on a fresh box).
  $('open-assistant').hidden = !(byId.assistant && byId.assistant.status === 'active');
  if (!$('assistantName').value) $('assistantName').value = s.assistant.name;
  $('assistantStepNote').hidden = !s.assistant.unavailable;
  $('assistantStepNote').textContent = s.assistant.unavailable;

  // The header pill.
  const pill = $('linkPill');
  const host = (s.base || '').replace(/^https?:\/\//, '').split(/[.:]/)[0];
  if (!s.paired) [pill.dataset.s, $('linkText').textContent] = ['offline', 'Not connected'];
  else if (!conn) [pill.dataset.s, $('linkText').textContent] = ['loading', `Checking ${host}…`];
  else if (!conn.ok) [pill.dataset.s, $('linkText').textContent] = ['failed', `${host}: ${conn.error.kind === 'revoked' ? 'needs re-pairing' : 'not reachable'}`];
  else [pill.dataset.s, $('linkText').textContent] = ['ready', `Connected to ${host}`];
}

function renderPermissions(s) {
  const p = s.permissions;
  // Linux and Windows: the platform's own list of what it needs, each with how to get it.
  const reqs = p.requirements || null;
  $('macPerms').hidden = Boolean(reqs);
  $('macRestart').hidden = Boolean(reqs);
  $('osPerms').hidden = !reqs;
  if (reqs) {
    $('osBackend').textContent = p.backend || 'this computer';
    $('osReqs').replaceChildren(...reqs.map((r) => {
      const row = document.createElement('div');
      row.className = 'perm';
      const name = document.createElement('span');
      name.className = 'pname';
      name.textContent = r.label;
      const why = document.createElement('small');
      why.textContent = `for ${r.for}`;
      name.append(why);
      const pill = document.createElement('span');
      pill.className = 'pill';
      pill.dataset.s = r.granted === true ? 'ready' : r.granted === false ? 'failed' : 'loading';
      const dot = document.createElement('span');
      dot.className = 'dot';
      pill.append(dot, r.granted === true ? 'ready' : r.granted === false ? 'missing' : 'asked on first use');
      row.append(name, pill);
      if (r.granted === false && r.fix) {
        const fix = document.createElement('small');
        fix.className = 'note';
        fix.textContent = r.fix;
        row.append(fix);
      }
      return row;
    }));
    return;
  }
  for (const k of ['microphone', 'accessibility', 'screen']) {
    const row = $(`perm-${k}`);
    const v = p[k];
    row.querySelector('.pill').dataset.s = PERM_DOT[v] || (v === 'n/a' ? 'offline' : 'loading');
    row.querySelector('.pstate').textContent = PERM_WORD[v] || v.replace('-', ' ');
    row.querySelector('button').hidden = v === 'granted' || v === 'n/a';
  }
  const stale = p.stale || {};
  $('stale').hidden = !(stale.accessibility || stale.screen || stale.microphone);
  $('staleText').textContent = s.stale_message;
  $('staleAx').hidden = $('resetAx').hidden = !stale.accessibility;
  $('staleScreen').hidden = $('resetScreen').hidden = !stale.screen;
  $('resetMic').hidden = !stale.microphone;
}

function renderFound() {
  const box = $('found');
  const note = $('tailnetNote');
  if (!discovered) {
    box.hidden = true;
    note.textContent = 'Looking for LocalRouter on your tailnet…';
    return;
  }
  note.textContent = discovered.tailnet === 'off' ? 'Tailscale is not running on this computer (or not signed in): start it to find your Spark here.' : discovered.found.length ? '' : 'No LocalRouter found on your tailnet: enter its address.';
  box.hidden = discovered.found.length === 0;
  box.replaceChildren(...discovered.found.map((f) => {
    const b = document.createElement('button');
    b.className = 'pill';
    b.dataset.s = 'ready';
    const dot = document.createElement('span');
    dot.className = 'dot';
    const small = document.createElement('small');
    small.textContent = [f.version && `LocalRouter ${f.version}`, f.machine].filter(Boolean).join(' · ');
    b.append(dot, `${f.base.replace(/^https:\/\//, '')} `, small);
    b.onclick = () => {
      $('base').value = f.base;
      checkAddress();
    };
    return b;
  }));
}

function render(s) {
  last = s;
  $('version').textContent = s.version;
  if (!$('name').value) $('name').value = s.name;
  for (const n of document.querySelectorAll('.nm')) n.textContent = s.name;
  $('secrets').textContent = s.secrets === 'keychain' ? 'encrypted by the operating system (the Keychain, Windows DPAPI or the Secret Service)' : 'for this run only (no keychain here)';
  if (s.stop_words) $('stopWords').textContent = s.stop_words;
  for (const n of document.querySelectorAll('.an')) n.textContent = s.assistant.name;
  // A busy button's label is plain text for a moment (guarded), so this one is written whole.
  if (!$('grantControl').classList.contains('busy')) $('grantControl').textContent = `Let ${s.assistant.name} use this computer`;
  renderAssistant(s);
  renderSteps(s);
  renderPermissions(s);
  renderFound();
  $('control').checked = s.control;
  $('control').disabled = !s.paired;
  $('presence').textContent = s.presence_why ? `${s.presence} (${s.presence_why})` : s.presence;
  $('askEach').checked = s.ask_each;
  if ($('language').options.length !== s.languages.length) {
    $('language').replaceChildren(...s.languages.map((l) => Object.assign(document.createElement('option'), { value: l.code, textContent: l.name })));
  }
  $('language').value = s.language;
  $('grantState').textContent = s.granted_since ? `you allowed it on ${new Date(s.granted_since).toLocaleString()}` : 'not allowed yet: a task is refused until you allow it';
  $('revokeControl').hidden = !s.control_granted;
  $('allowControl').hidden = s.control_granted;
  $('open-control').hidden = s.control_granted;
  $('grantControl').disabled = !s.paired;
  $('folders').replaceChildren(...s.shared.map((f) => {
    const li = document.createElement('li');
    const t = document.createElement('span');
    t.textContent = f;
    const b = document.createElement('button');
    b.className = 'btn';
    b.textContent = 'Remove';
    b.onclick = () => guarded(() => api.removeFolder(f));
    li.append(t, b);
    return li;
  }));
  $('callers').replaceChildren(...s.callers.map((c) => {
    const li = document.createElement('li');
    const t = document.createElement('span');
    t.textContent = `${c.name} (since ${new Date(c.created).toLocaleDateString()})`;
    const b = document.createElement('button');
    b.className = 'btn';
    b.textContent = 'Revoke';
    b.onclick = () => guarded(() => api.revoke(c.id));
    li.append(t, b);
    return li;
  }));
  // The executor runs before pairing: harnesses can be added any time (computer use itself waits for the Spark).
  $('harnessNote').hidden = s.paired;
  $('legacy').checked = s.legacy.enabled;
  $('legacy').disabled = !s.paired;
  $('legacyStatus').textContent = s.legacy.enabled ? (s.legacy.connected ? 'connected to the Spark' : 'connecting…') : '';
}

/** Settings › Assistant: the name on the Spark, and its default voice as a hint (no voice is chosen here). */
function renderAssistant(s) {
  const a = s.assistant;
  if (!assistantEdited) $('assistantEdit').value = a.name;
  $('assistantEdit').placeholder = a.default_name;
  $('assistantEdit').disabled = $('renameAssistant').disabled = !s.paired || Boolean(a.unavailable);
  $('assistantVoice').hidden = !a.voice;
  $('assistantVoice').textContent = a.voice ? `${a.voice}: the Spark’s default voice for calls.` : '';
  $('assistantNote').textContent = a.unavailable || (s.paired ? `Kept on the Spark: every device paired with it shows and speaks this name. Leave it empty for ${a.default_name}.` : 'Kept on the Spark: pair with it to change the name.');
}

/** Runs `work` with `button` showing it is busy; an error the main process threw goes into `errorEl`. */
async function guarded(work, button, busyText, errorEl) {
  const label = button && button.textContent;
  if (button) {
    button.disabled = true;
    button.classList.add('busy');
    if (busyText) button.textContent = busyText;
  }
  if (errorEl) alertCard(errorEl, null);
  try {
    const s = await work();
    if (s && s.connect) render(s);
    return s;
  } catch (error) {
    const el = errorEl || $('err-general');
    alertCard(el, { title: ipcMessage(error) });
  } finally {
    if (button) {
      button.disabled = false;
      button.classList.remove('busy');
      if (busyText) button.textContent = label;
    }
  }
}

async function checkAddress() {
  const s = await guarded(() => api.checkAddress($('base').value), $('checkAddress'), 'Checking…');
  if (s && s.link_code) $('code').value = s.link_code;
  if (s && !s.connect.errors.address) $(s.link_code ? 'pair' : 'code').focus();
}

async function discover() {
  discovered = null;
  renderFound();
  try {
    discovered = await api.discover();
  } catch {
    discovered = { tailnet: 'off', found: [] };
  }
  // One box on the tailnet and nothing typed: that is the Spark.
  if (discovered.found.length === 1 && !$('base').value.trim()) $('base').value = discovered.found[0].base;
  renderFound();
}

$('checkAddress').onclick = checkAddress;
$('base').addEventListener('keydown', (e) => e.key === 'Enter' && checkAddress());
// A pasted pairing link carries the address and the code: check it straight away.
$('base').addEventListener('paste', () => setTimeout(() => /^localrouter:\/\//i.test($('base').value.trim()) && checkAddress()));
$('changeAddress').onclick = () => guarded(() => api.cancel()).then(() => $('base').focus());
$('name').addEventListener('input', () => last && renderSteps(last));
$('code').addEventListener('keydown', (e) => e.key === 'Enter' && $('pair').click());
$('copyCmd').onclick = async () => {
  await navigator.clipboard.writeText($('pairCmd').textContent).catch(() => undefined);
  $('copyCmd').textContent = 'Copied';
  setTimeout(() => ($('copyCmd').textContent = 'Copy'), 1500);
};
$('pair').onclick = async () => {
  const s = await guarded(() => api.pair($('code').value, $('name').value), $('pair'), 'Pairing…');
  if (s && !s.connect.errors.code) $('code').value = '';
};
$('cancelRepair').onclick = () => guarded(() => api.cancel());
$('check').onclick = () => guarded(() => api.check(), $('check'), 'Checking…');
$('repair').onclick = () => guarded(() => api.repair()).then(() => $('code').focus());
$('disconnect').onclick = () => {
  confirming = true;
  if (last) renderSteps(last);
};
$('disconnectNo').onclick = () => {
  confirming = false;
  if (last) renderSteps(last);
};
$('disconnectYes').onclick = async () => {
  confirming = false;
  await guarded(() => api.disconnect(), $('disconnectYes'), 'Disconnecting…');
  discover();
};
$('open').onclick = () => api.openVoice();
$('saveAssistant').onclick = () => guarded(() => api.assistant($('assistantName').value, true), $('saveAssistant'), 'Saving…', $('err-assistant'));
$('assistantName').addEventListener('keydown', (e) => e.key === 'Enter' && $('saveAssistant').click());
$('assistantEdit').addEventListener('input', () => (assistantEdited = true));
$('renameAssistant').onclick = async () => {
  const s = await guarded(() => api.assistant($('assistantEdit').value, false), $('renameAssistant'), 'Saving…', $('err-assistantEdit'));
  if (s) {
    assistantEdited = false;
    renderAssistant(s);
  }
};
$('assistantEdit').addEventListener('keydown', (e) => e.key === 'Enter' && $('renameAssistant').click());
$('recheck').onclick = () => guarded(() => api.state(), $('recheck'), 'Checking…');
for (const b of document.querySelectorAll('[data-perm]')) b.onclick = () => guarded(() => api.permission(b.dataset.perm), b);
for (const b of document.querySelectorAll('[data-pane]')) b.onclick = () => guarded(() => api.permission(b.dataset.pane), b);
for (const b of document.querySelectorAll('[data-reset]')) b.onclick = () => guarded(async () => {
  const r = await api.resetGrant(b.dataset.reset);
  const out = $('resetResult');
  out.hidden = false;
  out.textContent = r.ok ? 'Reset. Switch LocalRouter on in the list that opened, then Quit and reopen.' : `macOS did not reset it (${r.text}): remove LocalRouter in the list by hand, add it again, then Quit and reopen.`;
  return r.state;
}, b);
$('relaunch').onclick = () => api.relaunch();
$('control').onchange = () => guarded(() => api.control($('control').checked));
$('askEach').onchange = () => guarded(() => api.askEach($('askEach').checked));
$('language').onchange = () => guarded(() => api.language($('language').value));
$('revokeControl').onclick = () => guarded(() => api.revokeControl());
$('grantControl').onclick = () => guarded(() => api.grantControl('onboarding'), $('grantControl'), 'Asking macOS…');
$('allowControl').onclick = () => guarded(() => api.grantControl('settings'), $('allowControl'));
$('addFolder').onclick = () => guarded(() => api.addFolder());
$('enroll').onclick = () => guarded(async () => {
  const name = $('callerName').value.trim() || 'harness';
  const r = await api.enroll(name);
  const cfg = { mcpServers: { computer: { command: r.command, env: { LOCALROUTER_EXECUTOR_TOKEN: r.token } } } };
  $('enrolled').hidden = false;
  $('enrolled').textContent = `Token for ${r.caller.name} (shown once):\n${r.token}\n\nMCP config:\n${JSON.stringify(cfg, null, 2)}`;
  $('callerName').value = '';
  return r.state;
}, $('enroll'));
$('legacy').onchange = () => guarded(() => api.legacy($('legacy').checked, $('legacyCode').value), null, null, $('err-legacy'));
$('export').onclick = async () => {
  const path = await api.exportConversations();
  $('exported').textContent = path ? `saved to ${path}` : '';
};

// The connection is checked when the window opens and again on focus once it is a minute old.
async function refresh() {
  const s = await guarded(() => api.state());
  if (s && s.paired && (!s.connect.connection || Date.now() - s.connect.connection.at > 60_000)) await guarded(() => api.check());
}
window.addEventListener('focus', refresh);
refresh().then(() => {
  if (last && !last.paired) discover();
});
