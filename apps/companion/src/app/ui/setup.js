'use strict';
const $ = (id) => document.getElementById(id);
const api = window.companion.setup;

function show(error) {
  const e = $('pairError');
  e.hidden = !error;
  e.textContent = error ? String(error.message || error).replace(/^Error invoking remote method '[^']+': (Error: )?/, '') : '';
}

function render(s) {

  $('version').textContent = s.version;
  $('unpaired').hidden = s.paired;
  $('paired').hidden = !s.paired;
  $('pairedBase').textContent = s.base;
  $('pairedName').textContent = s.device ? s.device.name : s.name;
  $('secrets').textContent = s.secrets === 'keychain' ? 'in the macOS keychain' : 'for this run only (no keychain)';
  if (!$('name').value) $('name').value = s.name;
  $('suggestName').textContent = s.name;
  for (const n of document.querySelectorAll('.nm')) n.textContent = s.name;
  if (s.base && !$('base').value) $('base').value = s.base;
  const p = s.permissions;
  const word = (v) => (v === 'granted' ? 'granted' : v === 'n/a' ? 'not a Mac' : v === 'stale' ? 'kept from an older build: grant it again' : v.replace('-', ' '));
  const stale = p.stale || {};
  $('stale').hidden = !(stale.accessibility || stale.screen);
  $('staleText').textContent = s.stale_message;
  $('staleAx').hidden = $('resetAx').hidden = !stale.accessibility;
  $('staleScreen').hidden = $('resetScreen').hidden = !stale.screen;
  $('pMic').textContent = word(p.microphone);
  $('pAx').textContent = word(p.accessibility);
  $('pScreen').textContent = word(p.screen);
  $('control').checked = s.control;
  $('control').disabled = !s.paired;
  $('presence').textContent = s.presence;
  $('folders').replaceChildren(...s.shared.map((f) => {
    const li = document.createElement('li');
    li.textContent = f + ' ';
    const b = document.createElement('button');
    b.textContent = 'Remove';
    b.onclick = async () => render(await api.removeFolder(f));
    li.append(b);
    return li;
  }));
  $('callers').replaceChildren(...s.callers.map((c) => {
    const li = document.createElement('li');
    li.textContent = `${c.name} (since ${new Date(c.created).toLocaleDateString()}) `;
    const b = document.createElement('button');
    b.textContent = 'Revoke';
    b.onclick = async () => render(await api.revoke(c.id));
    li.append(b);
    return li;
  }));
  $('enroll').disabled = !s.paired;
  $('legacy').checked = s.legacy.enabled;
  $('legacy').disabled = !s.paired;
  $('legacyStatus').textContent = s.legacy.enabled ? (s.legacy.connected ? 'connected to the Spark' : 'connecting…') : '';
}

async function guarded(work) {
  show(null);
  try {
    render(await work());
  } catch (error) {
    show(error);
  }
}

$('pair').onclick = () => guarded(async () => {
  const s = await api.pair($('base').value, $('code').value, $('name').value);
  $('code').value = '';
  return s;
});
$('unpair').onclick = () => guarded(() => api.unpair());
$('open').onclick = () => api.openVoice();
for (const b of document.querySelectorAll('[data-perm]')) b.onclick = () => guarded(() => api.permission(b.dataset.perm));
for (const b of document.querySelectorAll('[data-pane]')) b.onclick = () => guarded(() => api.permission(b.dataset.pane));
for (const b of document.querySelectorAll('[data-reset]')) b.onclick = () => guarded(async () => {
  const r = await api.resetGrant(b.dataset.reset);
  const out = $('resetResult');
  out.hidden = false;
  out.textContent = r.ok ? 'Reset. Switch LocalRouter on in the list that opened, then Quit and reopen.' : `macOS did not reset it (${r.text}): remove LocalRouter in the list by hand, add it again, then Quit and reopen.`;
  return r.state;
});
$('relaunch').onclick = () => api.relaunch();
$('control').onchange = () => guarded(() => api.control($('control').checked));
$('addFolder').onclick = () => guarded(() => api.addFolder());
$('enroll').onclick = () => guarded(async () => {
  const name = $('callerName').value.trim() || 'harness';
  const r = await api.enroll(name);
  const cfg = { mcpServers: { computer: { command: r.command, env: { LOCALROUTER_EXECUTOR_TOKEN: r.token } } } };
  $('enrolled').hidden = false;
  $('enrolled').textContent = `Token for ${r.caller.name} (shown once):\n${r.token}\n\nMCP config:\n${JSON.stringify(cfg, null, 2)}`;
  $('callerName').value = '';
  return r.state;
});
$('legacy').onchange = () => guarded(() => api.legacy($('legacy').checked, $('legacyCode').value));
$('export').onclick = async () => {
  const path = await api.exportConversations();
  $('exported').textContent = path ? `saved to ${path}` : '';
};
window.addEventListener('focus', () => guarded(() => api.state()));
guarded(() => api.state());
