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

- the pointer in the top-left corner of the main display (within 4 points): the task stops ("You stopped it on the Mac");
- the pointer more than 24 points from where the helper last left it: the person took over, the task stops;
- `localrouter-desktop stop` stops the task now; `localrouter-desktop disable` refuses every request until `enable`.

A task start shows a macOS notification saying what Jarvis is doing and how to stop it.

## Status

Not yet run on a real Mac: AX observation, OCR fallback, CGEvent clicks and keys, the corner stop and the TCC grants
need JJ's MacBook. The helper's tests use a fake backend; the box's policy and approvals have Zig unit tests.
