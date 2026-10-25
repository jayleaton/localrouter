# Computer control: Jarvis uses the user's Mac

The user asks Jarvis (by voice, or an agent asks over MCP) to do something on their computer: open an app, turn the
volume down, play the next track, find a page in a browser tab, type a search. The box stays the inference host; a small
helper on the Mac (`apps/desktop-helper`) owns everything that touches the Mac. The box owns decisions and authority:
every action the helper takes is granted by the box first, and consequential ones wait for the user's tap in the
LocalRouter web UI.

```
 voice session (Qwen tool call) ─┐                       web UI  /voice
 MCP computer_control / _use ────┤                         │  approvals: tap Approve / Deny, Stop, audit
                                 ▼                         ▼
             ┌──────────────── localrouter serve (the box) ───────────────────┐
             │ computer.Hub: paired helpers, tasks (one a helper), budgets     │
             │ computer.policy: auto / tap / refused, per action               │
             │ computer.approvals: one tap, one exact action, 60 s, single use │
             │ computer.decide.Provider: Qwen today, Jev later (same shape)    │
             │ computer audit log: <data>/computer/audit.jsonl                 │
             └──────────────▲─────────────────────────────────────────────────┘
                            │ WebSocket /v1/desktop (helper dials out; device token, scope desktop_helper)
             ┌──────────────┴──────────────── Mac: localrouter-desktop ────────┐
             │ fixed controls (open/quit app, volume, media, dark mode, lock)   │
             │ computer use loop: observe (AX, OCR fallback) → decide (box) →   │
             │   authorize (box) → act (CGEvent / keys) → settle; 15 steps, 90 s │
             │ local stop: pointer to the top-left corner, moving the pointer,  │
             │   `localrouter-desktop stop`, `localrouter-desktop disable`       │
             └──────────────────────────────────────────────────────────────────┘
```

Mac first. The helper's backend is the macOS one ported from SideBud's companion (JXA through `osascript`:
AXUIElement reads, CGEvent pointer and wheel events, System Events keys, Vision OCR when an app hides its window from
accessibility). Windows and Linux backends are separate work: the protocol below is not Mac specific, the backend is.

## Decisions

| Topic | Decision | Why |
| --- | --- | --- |
| Where the OS work runs | A helper on the controlled Mac. The box never runs AX, input or screenshots | macOS accessibility, input and TCC grants exist only on the Mac; the GB10 is headless Linux |
| Direction | The helper dials out to the box (`wss://<box>:10443/v1/desktop` on the Jarvis deploy) and keeps one socket open, reconnecting with backoff | No listening port on the Mac; works when the Mac's address changes; one place to authenticate |
| Language | TypeScript on Node 22.18+ (type stripping, so `src/*.ts` runs without a build), no runtime dependencies | The SideBud backend is TypeScript and JXA; a native rewrite buys nothing before the behaviour is qualified |
| Authority | The box. The helper asks `authorize` before every action (clicks, typing, keys, scrolls, bringing an app or tab forward) and acts only on a grant | The policy, approvals and audit live in one place, tested in Zig; the model, an MCP agent and screen text cannot grant anything |
| Pairing | The device tokens of LocalRouter's pairing (`src/pair.zig`), with per-device scopes: `desktop_helper` for the helper (its socket and nothing else), `computer` to call the tools, `approve` to answer approvals and stop tasks in the web UI. Computer control always needs paired devices, whether or not `pairing.required` gates the rest of the daemon | One credential system for phone, web, MCP and helper. A tailnet address alone must never reach someone's desktop, and an agent's MCP token can never approve its own step |
| Decisions | `computer.decide.Provider`, System One shaped: `ask(state, questions) -> answers` with Choice and Noul questions over closed candidate sets. Today: Qwen (the chat model) with a JSON schema whose enums are the candidates; later the open Jev decision service | The SideBud controller was written against System One; keeping its shape makes Jev a drop-in provider. Answers outside the candidates are refused |
| Probabilities | Qwen's answer carries `confidence` (`high`, `medium`, `low` = 0.9, 0.6, 0.3); the controller acts at ≥ 0.55 | Qwen has no calibrated scores; the threshold keeps SideBud's. Jev returns real probabilities through the same field |
| Bounds | One task a helper at a time; 15 steps; 90 s of working time (time waiting for a tap does not count); a step limit and deadline enforced on both sides; repeating the same action on an unchanged screen three times stops | SideBud's controller limits. The box refuses `decide` and `authorize` past them even if a helper ignores its own limits |
| What runs without a tap | Fixed controls `open_app`, `quit_app`, `volume`, `media`, `dark_mode`; on screen: scrolling, Space, Escape, Tab, bringing a named app or tab forward, clicks the policy finds benign, typing into a search or address field, Enter after typing into one | JJ's list: simple controls, no tap |
| What needs a tap, each time | `lock_screen`; typing into any other field; Enter after typing into one; any click whose label reads as sending, replying, posting, sharing, buying, paying, ordering, subscribing, deleting, removing, submitting, confirming, publishing (several languages), or that the decision provider judges could do one of those, or when that judgment fails | Sending, purchases, deletes, posting and typing into unknown fields need an explicit tap. The lexicon only adds taps; a miss falls through to the model judgment, and a failed judgment is a tap |
| An approval | Bound to the helper, task, step, observation (app, title, URL) and the exact action (kind, ref, role, label, text). Shown with all of it; expires after 60 s; used once; Deny ends the task | A tap approves one effect, not a goal. A spoken "yes" is not an approval |
| Voice | The voice session's chat model gets `computer_control` and `computer_use` when a helper is connected; progress lines are spoken ("Opening Safari", "I need your OK on screen to send that"), the outcome is spoken when the task ends | The user hears what Jarvis is doing on their computer |
| Audit | Every task, decision summary, grant, approval request and answer, action, refusal and outcome: one JSON line in `<data>/computer/audit.jsonl`, readable at `GET /v1/computer/audit`. Typed text is logged as its length and a SHA-256 prefix; labels are clipped to 80 characters | What happened on the user's computer is answerable later without keeping their screen text |
| Screen text | Untrusted data: sent to the decision provider as candidates and context, never as instructions; never leaves the box (no hosted provider) | Local only |

## Protocol (`/v1/desktop`, version 1)

A WebSocket, JSON text frames, one object a frame. The helper sends `Authorization: Bearer <token>` on the upgrade (any
way pairing reads a token works). A token without the `desktop_helper` scope is refused with a 401 before the upgrade,
whatever `pairing.required` says. The box checks the token again at least every 15 s while frames
arrive (the helper pings every 20 s), so revoking the device closes the socket.

### Scopes and tokens

```sh
localrouter pair token "JJ's MacBook" --scopes desktop_helper --config ~/jarvis/deploy/config.json   # the helper's token, printed once
localrouter pair scopes "JJ's phone" voice,mcp,computer,approve --config ~/jarvis/deploy/config.json  # a paired phone may ask and approve
localrouter pair list --config ~/jarvis/deploy/config.json                                           # every device and its scopes
```

A device with no scopes listed (every device paired before scopes existed) has `voice` and `mcp`. A list names each
scope it grants, so a helper's token reaches neither `/mcp` nor `/v1/realtime`. The voice session offers the computer
tools only to a device with `computer`. The web UI and the approval routes need `approve`: give it only to devices a
person holds, never to an agent's token.

### Handshake

```json
helper → box  {"type":"hello","protocol":1,"helper":{"name":"JJ's MacBook Pro","platform":"darwin","version":"0.1.0"},
               "capabilities":{"controls":["open_app","quit_app","volume","media","dark_mode","lock_screen"],"computer_use":true},
               "access":{"accessibility":true,"screen_recording":false}}
box → helper  {"type":"welcome","helper_id":"mac_3f2a","limits":{"max_steps":15,"timeout_ms":90000,"approval_timeout_ms":60000}}
```

A helper that sends anything else first, or another `protocol`, is closed with an `error` frame naming what was wrong.
`ping` / `pong` (`{"type":"ping"}`) either way every 20 s; a peer silent for 60 s is gone.

### Requests and replies

Both sides send requests with an `id` the sender chooses (unique on that socket) and get exactly one
`{"type":"result","id":…,"ok":true,…}` or `{"type":"result","id":…,"ok":false,"error":"…"}` back.

Box → helper:

| `type` | Fields | Reply |
| --- | --- | --- |
| `control` | `task_id`, `control`: one of `{"kind":"open_app","app"}`, `{"kind":"quit_app","app"}`, `{"kind":"volume","action":"up"\|"down"\|"mute"\|"unmute"\|"set","level"?,"target":"system"\|"Spotify"\|"Music"}`, `{"kind":"media","action":"play"\|"pause"\|"next"\|"previous","player":"Spotify"\|"Music"}`, `{"kind":"dark_mode","action":"on"\|"off"\|"toggle"}`, `{"kind":"lock_screen"}` | `done`: a sentence for the user ("Opened Safari") |
| `use.start` | `task_id`, `goal`, `text` (exact text to type, or null), `limits` | `{"ok":true}` at once; the task then runs and ends with `finished` |
| `stop` | `task_id` | `{"ok":true}`; the task ends with `finished` `stopped` |

Helper → box, during a `use` task:

| `type` | Fields | Reply |
| --- | --- | --- |
| `decide` | `task_id`, `state` (goal, the screen's app/title/url/visible text, steps so far), `questions`: `{id: {"type":"choice","instructions","criteria":{key: description}}}` or `{"type":"noul","instructions"}` | `answers`: `{id: {"type":"choice","choice":key,"probability":p}}` / `{"type":"noul","probability":p}`; a question the provider could not answer is absent |
| `authorize` | `task_id`, `step`, `observation` (`app`, `title`, `url`?), `action`: `{"kind":"click","ref","role","label"}`, `{"kind":"type","ref","role","label","text"}`, `{"kind":"press","key","field"?:{role,label}}` (`field`: the box text was typed into before Enter), `{"kind":"scroll","direction"}`, `{"kind":"surface","what":"tab"\|"app","name"}` | `grant`: `"allow"` or `"approved"`; or `ok:false` with `denied` (a reason for the user). May take until the approval times out |

Helper → box, no reply:

| `type` | Fields |
| --- | --- |
| `progress` | `task_id`, `step` (what it just did, in words: "clicked the button “Search”") |
| `finished` | `task_id`, `outcome`: `{"status":"done"\|"stuck"\|"blocked"\|"unavailable"\|"stopped"\|"failed","reason","steps":[…],"surface"}` |

The box refuses `decide` and `authorize` for a task that is not this helper's running task, past `max_steps`
authorizations, or past its working-time deadline; the helper then ends the task `stuck`.

## HTTP for the web UI (scope `approve`)

| Route | Does |
| --- | --- |
| `GET /v1/computer/status` | `computers` (connected helpers: name, controls, access, busy task), `tasks` (recent, newest first: status, steps, `waiting_for_approval`), `pending_approvals` |
| `GET /v1/computer/approvals` | approval cards, pending first: `action` (the sentence), `kind`, `text` (the exact text to type), `app`, `title`, `url`, `why`, `goal`, `expires_in_s`, `state` |
| `POST /v1/computer/approvals/{id}` | `{"decision": "approve"}` or `{"decision": "deny"}`; 409 when it expired or was already answered |
| `POST /v1/computer/stop` | `{"task_id": …}`, or `{}` for every running task |
| `GET /v1/computer/audit?limit=100` | the audit log's newest lines (at most 1000) |

The `/voice` page polls the status and shows each pending approval as a card (Approve, Deny, its countdown, the typed
text in full) and each running task with a Stop button; Settings → Computer lists the computers and recent activity.
It sends its API key field as the bearer token, so a device token allowed to approve goes there.

## Config (`computer`)

| Field | Default | |
| --- | --- | --- |
| `enabled` | true | the routes and tools exist (they always need paired devices) |
| `voice` | true | voice sessions whose device has the `computer` scope get the tools while a helper is connected |
| `decision_model` | "" | the chat model that decides steps; "": the `chat_completion` default |
| `max_steps` | 15 | at most 15 |
| `timeout_s` | 90 | working time, at most 90; taps not counted |
| `approval_timeout_s` | 60 | how long a tap may take |

## Replacing the decision provider

`src/computer/decide.zig` defines `Provider.ask(state, questions) -> answers`, the System One shape: Choice questions
over closed candidate sets and Noul (yes/no) questions, each answer with a probability. `decide.Qwen` implements it with
the chat model. A Jev provider implements the same function against the open decision service (its `/v1/systemone`)
and is set as `hub.provider` in `api/desktop.zig`; `Provider.ask` re-validates every answer against the offered keys
whichever provider answers, and the policy, approvals and budgets do not change.

## The local stop on the Mac

While a task runs, the helper checks the pointer every 250 ms and before each action:

- the pointer in a top-left corner (within 4 points): the main display's, or with displays arranged around it, the
  top-left corner of each display with no display to its left, above it or diagonally (a corner the pointer is held
  in, not an edge it passes through): the task stops ("You stopped it on the Mac");
- a deliberate move: more than 24 points from where the helper last left it on two readings in a row (one stray
  reading is not a move), counting only steps of more than 6 points between readings, so a hand resting on a
  trackpad or a drifting mouse never adds up to one: the person took over, the task stops;
- a click after the helper last acted (the probe reads how long ago the left button went down; the helper's own clicks
  come before it reports where it left the pointer): the person took over, the task stops;
- `localrouter-desktop stop` stops the task now; `localrouter-desktop disable` refuses every request until `enable`.

A task start shows a macOS notification saying what Jarvis is doing and how to stop it. A stop also ends the action
under way: the running script is killed, and long typing stops between chunks.

## Acting on the Mac: typing, scrolling, reading

| Action | How | Why |
| --- | --- | --- |
| Type | Click the box; check the app's focused element is that box (else nothing is typed); select its text through the app's Edit › Select All item by its ⌘ character; printable one-line ASCII is typed in chunks of 50 (a stop comes between chunks), anything else (other scripts, emoji, line breaks) is pasted through a pasteboard item marked transient and concealed, and the pasteboard is put back; then the box is read back, and text that did not land fails the step | Key code 0 with ⌘ is ⌘A on a US layout and ⌘Q on AZERTY; System Events keystrokes mangle characters the layout lacks; a typed line break is a Return, which sends in a chat. The typed text is never in an error or the audit |
| Scroll | The pointer goes over the window's largest scroll or web area (or the control `ref` names), then wheel events | The view under the window's middle is often a sidebar |
| Click | Refused when the control's middle is off every display ("scroll it into view first") | CGEvent would clamp it to the nearest edge and click something else |
| Unchanged screen | Three times the same action on a screen whose app, title, page, controls and text are all the same stops the task | A scrolled list keeps its count of rows, so counting controls called progress "stuck" |
| Reading | One AX answer at most 1.5 s (`AXUIElementSetMessagingTimeout`), the walk 4 s, 2,500 elements, 150 controls; past any of them the observation says `truncated`. The page's URL comes from the web area's `AXURL`, so approvals in browsers bind to it; password fields are listed without their value | A hung app answered each call after 6 s |

The executor also offers two reads to the harnesses the person enrolled (MCP `computer_observe` and
`computer_screenshot`, socket ops `observe` and `screenshot`): the front window's controls with their bounds, and a PNG
of it scaled so its longer edge is at most 1568 pixels (`max_edge`, 200 to 3136), with `region` and `scale` (screen
point = region origin + pixel / scale). They need no approval and change nothing; they are refused while control is
switched off, for the box's bridge and for a voice turn, and when LocalRouter itself is in front (its approval cards).
Each read is audited (app, title, size; never the content). A screenshot needs Screen Recording and is refused, not
blank, without it.

## On the device: the companion's executor (the new placement)

JJ's decision moves authority to the controlled computer. `apps/companion` carries an **executor** that is the final
authority on its own computer, whoever asks: our harness, a voice session in client tool mode, a third-party harness
over MCP, or this box through `/v1/desktop`. It reuses the helper's controller and Mac backend above, and enforces
beneath every API:

| Rule | Where |
| --- | --- |
| The policy (auto, approval, judged, refused) | `src/executor/policy.ts`, a port of `src/computer/policy.zig`; both read `src/computer/policy_golden.json` in their tests |
| Approvals: bound to the executor installation, task, step, screen and exact action by SHA-256; 60 s; single use | `src/executor/approvals.ts` (a port of `approvals.zig`) |
| One task at a time (the lease), 15 steps, 90 s working time, the decision budget | `src/executor/broker.ts` |
| Approval only on this computer, through the app's card, confirmed by the presence gate (Touch ID where the Mac has it); never while the executor itself is injecting input; never in the first moment a card shows | `Broker.answer`; no MCP tool, socket operation or box frame can answer |
| After an approval, the screen must still be the approved one (app, window title, page, the exact control; for Enter the field) | `Broker.revalidate` |
| LocalRouter's own windows are never a target | `ownApps` |
| Stop and the local switch work with no network and no model | `Broker.stop`, `setDisabled` |
| Audit on the device, typed text as length and digest | `src/executor/audit.ts` |

### One executor, a platform per OS

What is the same on every computer lives in the executor: the policy, approvals, budgets, audit, revalidation, the
lease and the local stop's rules (`src/executor/`, `apps/desktop-helper/src/stop.ts`, the controller). What differs is a
`Platform` (`apps/companion/src/platform/types.ts`), chosen by `createPlatform` at start:

| Part | Does | macOS | Linux | Windows |
| --- | --- | --- | --- | --- |
| `screen` | `observe` (tree), `act` (click, type, press, scroll), `pointer`, `displays`, `screenshot` | AX + CGEvent (`platform/mac.ts`) | AT-SPI + XTest / portal / ydotool | UI Automation + SendInput |
| `tabs`, `launcher`, `apps` | a browser's tabs; the person's own app list; running apps and bringing one forward | Chrome scripting; the Dock | AT-SPI tab strip; dash favourites | UIA tab strip; pinned taskbar apps |
| `controls` | the fixed controls; `unsupported` lists the ones the OS cannot run, refused before the policy with the reason | `MacControls` | | |
| `pointer`, `displays` | the local stop's readings; null where the OS hides the pointer (the stop is then the shortcut and the tray) | | | |
| `presence` | an OS check for approvals; null: the app's Touch ID or its dialog | (Touch ID in the app) | | Windows Hello |
| `access()` | what it needs, granted or not, with the fix, for Settings | | | |

Every coordinate a platform takes or reports is in one global space (points on macOS, physical pixels with per-monitor
DPI awareness on Windows, compositor pixels on Linux), so the stop's corners, screenshots' `region` and controls'
`bounds` agree. The keyboard stop is ⌘⇧⎋ on macOS and Ctrl+Shift+Alt+Esc elsewhere (Ctrl+Shift+Esc is the Task
Manager on Windows, Ctrl+Alt+Esc xkill on KDE). The executor's socket is a Unix socket in the data folder, or a per-user
named pipe on Windows. An OS with no platform yet reads nothing and refuses every control, saying so.

The box's decisions (the chat model asked as `decide.Qwen` asks it) are advice: they choose steps and judge clicks, and
a failed judgment asks. With the legacy bridge on, the box's `authorize` grant is one more input: a refusal there holds,
an allow does not replace the local rules or approval. New harnesses need none of `desktop_helper`, `computer` or
`approve`. Other harnesses on the computer use the executor through `localrouter-executor-mcp` (MCP stdio), after the
person enrolls them in the app.

## Status

Not yet run on a real Mac: AX observation, OCR fallback, CGEvent clicks and keys, the corner stop and the TCC grants
need JJ's MacBook. The helper's tests use a fake backend; the box's policy and approvals have Zig unit tests.

## Linux

`apps/companion/src/platform/linux/` (`createLinuxPlatform`). The tree is AT-SPI on every session, read by one python3
script (python3-gi, Atspi 2.0) run per call, its payload on stdin so typed text never shows in a process list. Refs are
child-index paths from the active window, re-resolved at act time; the walk is breadth first and bounded (2,500 nodes,
150 controls, 3 s: `truncated`). Actions are AT-SPI first and need no pointer (`background: true`): click = the
element's Action (`click`, `press`, `activate`, `jump`…) or `Selection` on its parent for tabs and rows; type =
`EditableText.set_text_contents`, read back, else focus + delete + insert, read back; scroll = the nearest vertical
scroll bar's `Value`. Only a control with none of these falls back to the injector at its centre; typing that way is
Ctrl+A then keys, read back, and refuses text with line breaks (they would be Enter). Password fields are listed
without a value and never read back. The URL is the browser address bar's text as shown (Chromium hides the scheme).

| | X11 | GNOME Wayland | KDE Wayland | Hyprland | sway / wlroots |
| --- | --- | --- | --- | --- | --- |
| Tree | AT-SPI; extents are global | AT-SPI; extents are window-relative and the origin is unknown: no `bounds` | as GNOME | AT-SPI + `hyprctl activewindow -j` (`at`, matched by pid): global bounds | AT-SPI + `swaymsg -t get_tree` (`rect` + `window_rect`, pid): global bounds |
| Keys, text | xdotool (XTest); without it AT-SPI key events | ydotool (needs ydotoold + uinput), else the RemoteDesktop portal (asks once; restore token in the data dir) | as GNOME | wtype, else ydotool | wtype, else ydotool |
| Pointer fallback | xdotool | none: refused with the reason | none | `hyprctl dispatch movecursor` + `ydotool click` (needs ydotoold) | `swaymsg seat - cursor set/press/release` (no setup) |
| Screenshot | ImageMagick `import -window` (else `xwd` + `convert`), the window | Screenshot portal (may ask once), the whole desktop: no window position to crop to | as GNOME | `grim -g` of the window | `grim -g` of the window |
| Pointer for the local stop | `xdotool getmouselocation` | null: the stop is the keyboard shortcut and the tray | null | `hyprctl cursorpos` | null (sway's IPC has no cursor) |
| App focus | `xdotool search --pid` + `windowactivate` | AT-SPI `grab_focus`, else `gtk-launch` again (single-instance apps raise) | as GNOME | `hyprctl dispatch focuswindow pid:` | `swaymsg [pid=…] focus` |
| Quit an app | `wmctrl -i -c` (a close request) | unsupported | unsupported | `hyprctl dispatch closewindow` | `swaymsg [con_id=…] kill` (a close request) |
| Impossible | | reading the pointer, raising another app's window, placing the pointer, window positions | as GNOME | | reading the pointer |

| Fixed control | How |
| --- | --- |
| `open_app` | `.desktop` entries in `$XDG_DATA_HOME/applications`, `$XDG_DATA_DIRS/*/applications`, flatpak exports; Name, id, aliases (Chrome, VS Code, Settings, Files…), GenericName/Keywords, word prefix; equal matches are a question. `gtk-launch <id>`, else `gio launch <path>`; never a shell |
| `volume` | `wpctl` (PipeWire), else `pactl`, else `amixer`; Spotify `playerctl -p spotify volume`, Music the default MPRIS player |
| `media` | `playerctl [-p spotify] play/pause/next/previous` |
| `dark_mode` | KDE `plasma-apply-colorscheme BreezeDark/BreezeLight`; elsewhere `gsettings … color-scheme prefer-dark/default` (+ Adwaita gtk-theme), which GTK/libadwaita apps follow on Hyprland and sway too |
| `lock_screen` | `loginctl lock-session [$XDG_SESSION_ID]`, else `xdg-screensaver lock` (the desktop's locker must answer logind's Lock) |
| notice | `notify-send` |

Setup the access report names: python3-gi + Atspi typelib + at-spi2-core (Debian/Ubuntu `python3-gi gir1.2-atspi-2.0
at-spi2-core xdotool`, Fedora `python3-gobject at-spi2-core xdotool`, Arch `python-gobject at-spi2-core xdotool`);
accessibility on (`gsettings set org.gnome.desktop.interface toolkit-accessibility true`; Qt apps
`QT_LINUX_ACCESSIBILITY_ALWAYS_ON=1`/`ACCESSIBILITY_ENABLED=1`; Electron apps `--force-renderer-accessibility`; Chrome and
Chromium build their web tree only with accessibility on, by the setting above or the same flag). AT-SPI actions work on every compositor for GTK, Qt and Chromium
with accessibility on. ydotool needs ydotoold with `/dev/uinput` (the user in `input` or a udev rule); its "absolute"
move is relative from a corner and bent by acceleration, so it is used for clicks only where the compositor places the
pointer (Hyprland). The RemoteDesktop portal is used for keys only: a button would land wherever the person's pointer
is, and absolute motion needs a ScreenCast stream.

Not yet run on a real Linux desktop: everything above is unit tested with fake runners, and the AT-SPI script against
a fake `gi.repository.Atspi`.

## Windows

The companion's Windows platform (`apps/companion/src/platform/windows/`, `createWindowsPlatform()`): UI Automation and
SendInput through persistent Windows PowerShell 5.1 workers. No pwsh, no .NET SDK, no runtime dependency: every
Windows 10 and 11 has `powershell.exe` and the .NET Framework compiler its `Add-Type` uses.

| Topic | How | Why |
| --- | --- | --- |
| Worker | `%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command -`, by full path. One bootstrap line on stdin compiles a C# helper (`helper.cs.ts`, C# 5) with `Add-Type`, then JSON lines: `{id, op, args}` in, `{id, ok, result\|error}` out, ASCII only (`\uXXXX` both ways). Requests run one at a time; a timeout or a stop kills the process, a crash too, and the next request starts a fresh one | Request data is parsed as JSON by the helper, never spliced into script text; the console code page cannot change typed text; a hung UIA call or a stop never leaves the worker wedged |
| Two workers | `screen` (observe, act, screenshots, pointer, displays) and `system` (Start menu, windows, fixed controls, tabs, Windows Hello, access) | The local stop's pointer reads never wait behind a Hello prompt or Get-StartApps |
| Reading the window | The foreground window (`GetForegroundWindow`, a Store app's real process found inside its ApplicationFrameWindow), its app (the executable's file description: "Google Chrome") and title; a breadth-first walk of UIA's ControlView with a `CacheRequest`, 2,500 nodes, 150 controls, 3 s (`truncated` past any); off-screen, zero-size, disabled and out-of-window elements skipped | One cross-process call per container instead of one per property |
| Roles | Button, SplitButton → button; Hyperlink → link; Edit, editable Document or ComboBox → textbox (a read-only Edit is text); TabItem → tab; ListItem, DataItem, TreeItem → row; MenuItem → menuitem; CheckBox, RadioButton → checkbox; other ComboBox → button; Text → the text excerpt (1,500 characters) | The same closed set as the Mac |
| Passwords | An `IsPassword` field is listed with its label only: its value is never read, and the TS side drops one if it came | Screen text leaves the helper; a password must not |
| Refs | `<hwnd>:<child-index path>#<RuntimeId>`; act walks the path again from that window and checks the RuntimeId (when the provider gives one): a mismatch is "That control is gone; look again." | A list that changed under the path never lets a click land on the neighbour |
| Browser URL | Chrome, Edge, Firefox, Brave, Opera, Vivaldi: the address bar's Value ("Address and search bar", "Search or enter address", …) is `url`, unless it has focus (then it holds what is being typed). It is the address as the bar shows it, often without `https://` | Approvals bind to it |
| Background actions | Click: Invoke, else SelectionItem.Select (tabs, list items), Toggle, Expand/Collapse, or focus for a field; the call runs on its own thread and one still running after 2.5 s (a modal dialog it opened holds it) counts as done (`pending`), the worker is not killed. Type: ValuePattern.SetValue, read back; a value that did not land fails. Scroll: ScrollPattern on the ref or its nearest scrollable ancestor (the window's first scrollable part without a ref), a page or to the top/bottom. All `background: true`: the person's pointer and keyboard are untouched | Most native and browser controls have a pattern; acting through it needs no foreground and cannot be misdirected by a moving window |
| Input fallback | `SendInput`: the window is brought forward first (restore, `SetForegroundWindow`; refused under the foreground lock rules → a sentence, no AttachThreadInput or key tricks) and checked again before each event; pointer moves are absolute over the virtual desktop (`MOUSEEVENTF_VIRTUALDESK`); typing is Ctrl+A then `KEYEVENTF_UNICODE` per UTF-16 unit, 200 a chunk, a stop checked between chunks; Enter, Escape, Tab, Space as virtual keys; the wheel over the ref or the window. `pointer` reports where it was left | Only for controls without a pattern; a line break is never typed as keys (it would press Enter): such text fails unless SetValue took it |
| DPI and coordinates | The worker makes itself per-monitor DPI aware (v2; fallbacks per-monitor, system) before any UIA or GDI call, so BoundingRectangle, GetCursorPos, SendInput, monitor rectangles and screen capture all use physical pixels of the virtual desktop: the global space of `Point`. The primary display's top-left is (0, 0); others can be negative | One space, no per-monitor scaling, on mixed-DPI setups too. `access()` reports the awareness it got |
| Screenshots | The foreground window's DWM extended frame bounds (no shadow), clipped to the virtual desktop, `Graphics.CopyFromScreen`, scaled with `fitScale` (HighQualityBicubic), PNG. `region` and `scale` per the contract | What is on screen at that rectangle, including anything on top of the window |
| Displays | `EnumDisplayMonitors` rectangles, primary first | The local stop's corners |
| Local stop | The pointer every 250 ms through `GetCursorPos` (the corner of the primary display is (0, 0)); no `sinceClickS` (Windows has no reliable time since the last click), so takeover is by movement; plus the app's own shortcut and Stop | Same rules as the Mac |
| Elevation (UIPI) | Input and most UIA to a process of higher integrity are dropped silently. Before acting (and observing) the helper compares integrity levels (`OpenProcessToken`, `TokenIntegrityLevel`; an unreadable token counts as elevated) and fails: "<App> runs as administrator, so LocalRouter cannot see, click or type in it. Run <App> normally (not as administrator), or do this step yourself." | A silent no-op would look like success |
| Secure desktop | A UAC prompt, the lock or sign-in screen (`OpenInputDesktop` refused or not "Default"): every screen operation fails with a sentence asking the person to answer it | Nothing can read or drive it, by design |
| Windows Hello | `UserConsentVerifier` (WinRT, from the system worker): `available()` only when Windows says `Available` (a PIN, face or fingerprint is set up); an approval is confirmed only on `Verified` | The OS confirms the person, as Touch ID does |
| Fixed controls | `open_app`: Get-StartApps (Win32 and Store apps, `explorer.exe shell:AppsFolder\<AppUserModelID>`) plus Start menu shortcuts, Windows aliases, equal matches asked as a question; `quit_app`: WM_CLOSE to the app's windows, never a kill; `volume`: Core Audio endpoint volume, Spotify's own audio session, Music refused (Windows has no Music app); `media`: the system media session (Spotify's when asked), else the media keys; `dark_mode`: `AppsUseLightTheme`/`SystemUsesLightTheme` and the `ImmersiveColorSet` broadcast; `lock_screen`: `LockWorkStation` | No user text on a command line except one argv entry naming something from those lists |
| Launcher, apps, tabs | The taskbar's pinned shortcuts; running apps are those with a visible, uncloaked top-level window; Chrome's and Edge's tab strip read through UIA (titles only, no URLs), switched with SelectionItem.Select, then the browser brought forward | |

Impossible on Windows, by design: driving elevated apps from a normal LocalRouter (run both at the same level), the
secure desktop (UAC prompts, the lock screen, Ctrl+Alt+Del), apps that draw without UI Automation (some games and
remote-desktop windows: only the pointer fallback at their window's middle, no controls), and reading a password
field.

Status: written and unit-tested on Linux with a fake transport; the PowerShell scripts are parse-checked and the C#
compiled as C# 5 against the .NET Framework 4.8 reference assemblies. `windows.win32.test.ts` runs on a Windows runner:
it parses every script with Windows PowerShell's own parser, compiles the helper with `Add-Type`, and runs the
read-only operations against the desktop.
