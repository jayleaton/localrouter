# The harness contract

**Contract `2026-10-23`.** What a LocalRouter box offers a harness running on someone else's device: the routes, their
shapes, how they are authorized, and what they do not do. A harness is the software that holds the conversation,
offers tools, decides what runs and asks the user; LocalRouter is the model and media service it calls. Our own clients
(the `/voice` web page, the mobile app, the desktop helper) are harnesses like any other, and nothing here needs them.

This is a baseline: it documents what the daemon does today, not a plan. Where something is planned it is marked
**coming**, with the change that brings it. Anything not listed here is not part of the contract, even if it happens to
work.

| Surface | Route | Standard | Section |
| --- | --- | --- | --- |
| Discovery | `GET /v1/capabilities` | LocalRouter | [Discovery](#discovery) |
| Models | `GET /v1/models` | OpenAI, with extensions | [Models](#models-and-model-management) |
| Chat | `POST /v1/chat/completions` | OpenAI Chat Completions | [Chat](#chat-completions) |
| Voice | `GET /v1/realtime` (WebSocket) | OpenAI Realtime, beta events | [Realtime](#realtime-profile-openai-realtime-beta) |
| Tools for agents | `POST /mcp`, `localrouter mcp-stdio` | MCP, Streamable HTTP (stateless) | [MCP](#mcp) |
| Decisions | `POST /v1/systemone` | TypeSafe System One | [Decisions](#decisions) |
| Media | `/v1/images/*`, `/v1/videos`, `/v1/audio/*`, `/v1/jobs`, `/v1/files` | OpenAI Images, Videos, Audio | [Media](#media-and-jobs) |
| Pairing | `/v1/pair`, bearer tokens | LocalRouter | [Authorization](#authorization) |
| Computer control | `/v1/desktop`, `/v1/computer/*` | LocalRouter | [Computer control](#computer-control-legacy-placement) |

The field-level reference for each route stays where it was: [AGENTS.md](AGENTS.md) (HTTP), [MCP.md](MCP.md) (MCP
tools), [dev/VOICE.md](dev/VOICE.md) (the voice loop) and [dev/COMPUTER.md](dev/COMPUTER.md) (computer control). This
document says which of it a harness may rely on.

## Principles

- **Standard shapes first.** Where OpenAI or MCP has a shape for the job, the route takes it. A LocalRouter addition
  is named as one (`localrouter.*` events, extension fields); a client that ignores every extension still works.
- **Refuse, never approximate.** A field or value the daemon cannot honour is a 400 (an `error` event on the
  WebSocket, a tool error over MCP) naming it, before any model loads. Nothing is silently dropped or rounded. The one
  exception is Chat Completions, whose body goes to the model's own server: a field that server does not know may be
  ignored there rather than refused (see [Chat](#chat-completions)).
- **The service never executes the harness's tools.** Function calls a model makes for the caller's tools come back to
  the caller. The one exception is a realtime session left in its default `server` tool mode, where the box's
  built-ins (generation and computer tools) run on the box; a harness sets `client` mode and owns every tool.
- **Results are facts, not permission.** A job's output, a decision's probabilities and a model's tool call never
  authorize anything on the user's computer. The device that acts decides.

## Discovery

`GET /v1/capabilities` describes this daemon: the contract version, which surfaces exist with which profile, how each
is authorized and what is not offered. It needs no token, even with pairing required, and holds no secrets. Read it once at
startup; a harness that needs something this box lacks should say so and stop, rather than find out mid-task.

```bash
curl -s http://<host>:8190/v1/capabilities | jq '{contract, chat: .chat.available, realtime: .realtime.profile, mcp: .mcp.tools}'
```

| Field | Meaning |
| --- | --- |
| `contract` | this document's version (a date). A change to a documented shape changes the date; an addition a client may ignore does not |
| `server` | `name`, `version`, `machine` |
| `auth` | `scheme` (`bearer`), `pairing_required`, `scopes` (every scope a token can have), `service_scopes`, `legacy_scopes` (either grants every service scope), `harness_scopes` (what a harness is usually issued), `guarded` (route, scope, and `when`: `always` or `pairing_required`), `open` (routes that need no token even with pairing), `jobs_owned` |
| `models` | `list` (`/v1/models`), the `kinds` and `capabilities` this box's models have |
| `chat` | `available`, `api` (`chat_completions`), `path`, `models`, `stream`, `stream_usage`, `tools`, `tool_choice`, `response_format`, `reasoning` (the field reasoning arrives in, and how to switch it off), `input` (`text`), `n` (1), `cancel` |
| `realtime` | `path`, `transport`, `profile` (`openai-realtime-beta`), `remote` (whether non-loopback peers are served), `audio`, `turn_detection`, `modalities`, `tool_execution` (the modes: `server`, `client`), `tool_execution_default`, `tool_execution_set`, `server_tools` (which built-ins a session gets), `client_events`, `server_events`, `extension_events`, `token_transports` |
| `mcp` | `path`, `transport` (`streamable_http`), `stateless`, `protocol_versions`, `stdio_bridge`, `tools` (names, in `tools/list` order) |
| `decisions` | `available`, `path`, `shape` (`system_one`), `mcp_tool` |
| `media` | the media and job routes |
| `model_management` | the operator routes and the MCP tools that change what is loaded |
| `computer` | `enabled`, the helper and approval routes, and `placement` (`box_policy` today) |
| `unsupported` | `[{feature, status, note}]`: `status` is `unsupported` or `coming`; `note` says what to use instead |

A model's own limits are in `/v1/models` (its voices, `max_input_chars`, `max_audio_s`, `qualified`...), not here.

## Authorization

Tokens come from pairing, done by the operator on the box:

```sh
localrouter pair token NAME --scopes inference,media,read --config FILE   # a harness's token, shown once
localrouter pair add NAME --config FILE           # a one-time code (10 min) a client trades at POST /v1/pair
localrouter pair revoke ID|NAME --config FILE     # refused from the next request
```

`POST /v1/pair {"code", "name"}` returns `{token, device: {id, name}, required}`; `GET /v1/pair` returns
`{required, paired, device}` (with the token's scopes) for whatever token is presented. A token is sent as
`Authorization: Bearer <token>`. On `/v1/realtime`, where browsers cannot set headers, it may instead be a
`Sec-WebSocket-Protocol` entry `openai-insecure-api-key.<token>` beside `realtime`, or `?api_key=<token>` (which can
end up in logs: prefer the others).

With `pairing.required` (off by default), every route but `/health`, `/v1/pair`, `/v1/capabilities` and `/voice` needs
a paired token with the scope it serves, over HTTP, the realtime socket and MCP alike. `auth.guarded` in discovery
lists each route's scope, computed from the same table the daemon enforces.

| Scope | Grants |
| --- | --- |
| `inference` | `/v1/chat/completions`, `/v1/realtime`, MCP `chat` |
| `media` | images, videos, speech and transcription over HTTP and MCP, and a voice session's picture and video tools |
| `decisions` | `/v1/systemone`, MCP `decide` |
| `read` | `/v1/models`, `/v1/tools`, `list_models`, `list_voices` |
| `manage` | loading, unloading, reserving and releasing models, however asked |
| `voice`, `mcp` (legacy) | `/v1/realtime`, `/mcp`; either, or none listed (a device paired before scopes), grants every service scope |
| `desktop_helper` | `/v1/desktop`: connecting as a desktop helper, nothing else (needed always) |
| `computer` | `computer_control`, `computer_use`, `computer_status`, `computer_stop` (needed always) |
| `approve` | `/v1/computer/*`: answering approvals, stopping tasks, the audit (needed always) |

Issue a harness least privilege: `inference,media,read` covers chat, voice and media; add `decisions` or `manage` only
when it needs them. `/mcp` itself takes any service scope, and its `tools/list` shows only the tools the token may
call. A job and its output files (`/v1/jobs`, `/v1/videos/{id}`, `/v1/files`, `get_job`, `cancel_job`) answer only the
device that asked for them; others get 404. Wrong or missing token: 401 `invalid_api_key` with
`WWW-Authenticate: Bearer`, before the body or the upgrade. A valid token without the scope: 403 naming it.

With pairing off every route answers whoever reaches the daemon. Either way keep it on loopback or the tailnet
(`--host tailscale`, or `tailscale serve`), never on a LAN or public address.

## Models and model management

`GET /v1/models` is OpenAI's list with LocalRouter's fields: `kind` (`chat`, `image`, `video`, `speech`,
`transcription`, `decision`), `capabilities`, `default_for`, `running`, `resident_bytes`, `qualified` (false: refused
with a 503 until the operator qualifies it on this GPU), and, for speech models, `voices`, `default_voice`, `presets`,
`speed`, `max_input_chars`, `stream`, `references`; for transcription, `max_audio_s`. Model ids are stable for a
deployment; a request without `model` goes to the model whose `default_for` lists the capability it needs.

The scheduler decides what is loaded. A harness may ask, never insist:

| Ask | HTTP (operators) | MCP |
| --- | --- | --- |
| what is loaded, memory | `GET /v1/tools` | `list_models` |
| make room for a heavy job | | `reserve_for` (`dry_run` first) |
| load / unload one | `POST /v1/tools/{id}/load`, `/unload` | `load_model`, `unload_model` |
| unload everything | `POST /v1/tools/release` | `release_models` |

The chat model is never unloaded on request. Models that make room for a job come back by themselves after it. Voice
sessions hear `localrouter.voice.paused` before a voice model goes and `localrouter.voice.resumed` when it is back.

## Chat Completions

`POST /v1/chat/completions` is OpenAI's Chat Completions, proxied to the chat model's own server: the daemon checks
that the body is an object with a non-empty `messages`, sets `model` to the model it routes to and passes every other
field through. What the model's server refuses comes back as a 400 with its message. For the deployed Qwen3.8 27B
(TensorFold):

| Feature | Contract |
| --- | --- |
| Messages | `system`, `user`, `assistant` (with `tool_calls`), `tool` (`tool_call_id`). Text only: image or audio parts are a 400 |
| Tools | OpenAI function tools; `tool_choice` `none`, `auto`, `required` or `{"type":"function","function":{"name"}}`; `parallel_tool_calls`. Calls come back as `message.tool_calls` (`delta.tool_calls` when streaming) with `finish_reason: "tool_calls"`; the caller runs them and sends `tool` messages |
| Reasoning | arrives in `reasoning_content` (`delta.reasoning_content`), never in `content`. Off: `"reasoning_effort": "none"` or `"chat_template_kwargs": {"enable_thinking": false}` |
| Structured output | `response_format` `json_object` or `json_schema` |
| Sampling | `temperature` (0: greedy, the same request gives the same tokens), `top_p`, `top_k`, `min_p`, `seed`, `stop`, `max_tokens` / `max_completion_tokens` (default 4096) |
| Streaming | `stream: true`: server-sent events `data: {chunk}` then `data: [DONE]`; `stream_options.include_usage` adds a last chunk with `usage` and `choices: []` |
| `n` | 1 only |
| Errors | a refused streamed request gets an ordinary JSON error, not a stream; a stream that breaks after it began ends with `data: {"error": {...}}` and `data: [DONE]` |
| Cancel | close the connection; the model stays loaded |
| Unknown fields | passed to the model's server, which may ignore them: DeepSeek's `thinking: {"type": "disabled"}` is accepted and ignored (measured: the reply still reasons); use `reasoning_effort` |
| Model names | must be one of `/v1/models` (or omitted for the default); another name, such as `deepseek-chat`, is a 404 listing the chat models |

There is no `/v1/responses`, `/v1/completions` or `/v1/embeddings` (**coming**: Responses, H5 and H6). Harnesses that
speak only the Responses API (Codex) cannot use the box until then.

## Realtime profile `openai-realtime-beta`

`ws://<host>:8190/v1/realtime` (`wss://` behind `tailscale serve`) speaks the OpenAI Realtime API's **beta** event
vocabulary over a WebSocket, JSON text frames only (a binary frame is an `error`). Audio is pcm16, 24 kHz, mono, base64,
both ways. Offer the subprotocol `realtime`. `?model=` sets the chat model. By default the route answers loopback peers
only (`realtime.remote` in discovery; the operator's `voice.remote`); behind `tailscale serve` every peer arrives as
loopback.

**Client events** (anything else is an `error` with code `unknown_event_type`; the session stays open):

| Event | Contract |
| --- | --- |
| `session.update` | `{session: {...}}`, the fields below; answered by `session.updated` |
| `input_audio_buffer.append` | `{audio}` base64 pcm16 at 24 kHz |
| `input_audio_buffer.commit` | ends a push-to-talk turn (`turn_detection: null`) |
| `input_audio_buffer.clear` | drops buffered audio |
| `conversation.item.create` | `message` (`user` with `input_text`, `assistant` with `text`), `function_call` `{call_id, name, arguments}` and `function_call_output` `{call_id, output}`: history a client seeds after reconnecting; a seeded call is not run again |
| `conversation.item.truncate` | `{item_id, content_index: 0, audio_end_ms}`: what the user actually heard of an interrupted reply |
| `conversation.item.delete` | `{item_id}` |
| `response.create` | answer now (push to talk, text turns, after a function's output) |
| `response.cancel` | stop the current response |

**`session.update` fields:** `modalities` (`["audio","text"]` or `["text"]`), `instructions`, `voice`, `model`,
`input_audio_format` and `output_audio_format` (`pcm16` only), `input_audio_transcription` (`{model, language,
prompt}`), `turn_detection` (`{type: "server_vad", threshold, prefix_padding_ms, silence_duration_ms, create_response,
interrupt_response}` or `null`), `tools` (function tools only), `tool_choice`, `temperature`,
`max_response_output_tokens`. Extensions: `speech_model` (which speech model speaks), `partial_interval_ms` and
`localrouter` (`{tool_execution}`, below).

**Server events:** `session.created`, `session.updated`, `input_audio_buffer.speech_started`, `.speech_stopped`,
`.committed`, `.cleared`, `conversation.item.created`, `.deleted`, `.truncated`,
`conversation.item.input_audio_transcription.completed`, `.failed`, `response.created`, `response.output_item.added`,
`.done`, `response.content_part.added`, `.done`, `response.text.delta`, `.done`, `response.audio_transcript.delta`,
`.done`, `response.audio.delta`, `.done`, `response.function_call_arguments.delta`, `.done`, `response.done`, and
`error` (`{type: "error", event_id, error: {type, code, message, param, event_id}}`).

**Extension events** (a basic client may ignore every one):

| Event | Fields | For |
| --- | --- | --- |
| `localrouter.transcription.partial` | `item_id`, `transcript` | live captions, with `partial_interval_ms` |
| `localrouter.response.text_done` | `response_id`, `item_id`, `tool_calls`, `chars`, `speech_rate` | the reply's text is complete; how fast its audio will arrive (a playback hint) |
| `localrouter.response.speech_done` | `response_id`, `item_id` | every sentence is synthesized |
| `localrouter.response.timing` | `response_id`, `from`, `ms` (per stage), `stt_worker_ms`, `voiced_audio_ms` | latency measurement |
| `localrouter.job.created` / `.job.done` | `call_id`, `job_id`, `kind`; then `status`, `urls`, `error` | a built-in generation tool's job (below) |
| `localrouter.voice.paused` / `.voice.resumed` | `model`, `for_model`, `message` | a voice model made room for a job; the session stays open |
| `localrouter.function_call.cancelled` | `item_id`, `call_id`, `name`, `response_id`, `reason` (`turn_detected`, `response_started`) | client tool mode: stop that call's work, its output will be refused |

**Tool execution** (`realtime.tool_execution`), chosen per session with the extension
`session.update {session: {localrouter: {tool_execution: "server" | "client"}}}` before its first response, and echoed
in `session.created` / `session.updated`. Any other `localrouter` key or value is `invalid_session`, so a client asking
for a mode the box lacks finds out at once.

- **`client`: the harness owns its tools (use this for a new harness).** Only the client's `tools` exist (any names,
  built-in ones included) and none runs on the box. Each call arrives once, complete, as
  `response.function_call_arguments.done {call_id, name, arguments}` before `response.done`; the harness runs it,
  sends `conversation.item.create {type: "function_call_output", call_id, output}` for each, then `response.create`.
  An output for an unknown call, a second output, or one after the call expired is refused (`unknown_call_id`,
  `duplicate_call_output`, `call_expired`). A call expires when the user speaks again or another response begins, and
  `localrouter.function_call.cancelled` says so, so a late result is never spoken in a later turn.
- **`server` (default, what our web page and app use).** The client's `tools` plus, unless the operator turned them
  off, the box's built-ins (`realtime.server_tools`): `generate_image`, `generate_video`, the model management tools
  with `voice.manage_models`, and `computer_control` / `computer_use` while a desktop helper is paired. Client calls
  return as above; a built-in runs on the box, its job's events (`localrouter.job.*`) arrive and the assistant follows
  up by itself. A client tool may not reuse a built-in's name.

Details: [dev/VOICE.md](dev/VOICE.md#client-tool-execution); a one-file harness: `examples/realtime-client-tools/`.

**Not in this profile:** the GA event names and nested `session.audio` settings (`response.output_audio.delta`),
WebRTC, ephemeral keys (`/v1/realtime/sessions`), `semantic_vad`, g711 or other sample rates, and a transcription-only
session. A client written for the GA API needs an adapter.

## MCP

`POST /mcp` is MCP over Streamable HTTP, stateless: one JSON-RPC message a request (no batches), the reply in the HTTP
body, no `Mcp-Session-Id`, no server-sent notifications, no resources or prompts. `initialize` accepts protocol
versions `2025-11-25`, `2025-06-18`, `2025-03-26` and `2024-11-05` (`mcp.protocol_versions`). Clients that only start
a command use `localrouter mcp-stdio --url URL [--token TOKEN]`, a stateless bridge (`$LOCALROUTER_URL`,
`$LOCALROUTER_TOKEN`).

Tools (`mcp.tools`; schemas strict, `additionalProperties: false`): `list_models`, `list_voices`, `reserve_for`,
`unload_model`, `load_model`, `generate_image`, `generate_video`, `transcribe_audio`, `synthesize_speech`, `chat`,
`decide`, `get_job`, `cancel_job`, `release_models`, `computer_control`, `computer_use`, `computer_status`,
`computer_stop`. Arguments and results: [MCP.md](MCP.md#tools).

- **Failures** are tool results with `isError: true` whose text says what to change; a JSON-RPC error means the call
  itself was malformed (unknown tool, bad params).
- **Long jobs never need a long call:** `generate_image`, `generate_video` and `transcribe_audio` take `wait_s`
  (default 600, at most 3000) and return the job id when it passes; `get_job {id, wait_s}` picks it up and
  `cancel_job` ends it. A client with a short tool timeout passes a small `wait_s` and polls.
- **Outputs** come back as `resource_link` URLs on the box (`/v1/files/...`, kept 24 h) and, for images and speech,
  inline as well (`inline_images`, `inline_audio` switch that off).

## Decisions

`POST /v1/systemone` takes TypeSafe's System One request `{model, state, questions}` (`choice`, `noul` and `score`
questions) and returns typed answers with probabilities, checked against the questions before they are returned
([AGENTS.md](AGENTS.md#decisions-the-system-one-shape)). The MCP `decide` tool is the same. A decision acts on nothing.

## Media and jobs

OpenAI's Images (`/v1/images/generations`, `/v1/images/edits`), Videos (`/v1/videos`, `/v1/videos/{id}`,
`/content`, `DELETE`) and Audio (`/v1/audio/speech`, `/v1/audio/transcriptions`) APIs with documented extensions
(`seed`, `steps`, `stream` for PCM speech...): [AGENTS.md](AGENTS.md). Any job's state: `GET /v1/jobs/{id}` `{id, kind,
status, progress, phase, urls, error}`. Outputs: `GET /v1/files/{job}/{name}`, kept `keep_outputs_s` (24 h by
default). Absolute output URLs (MCP results, `response_format: url`, `localrouter.job.done`) are built from the
request's `Host` and the scheme the client used: `https` when a TLS proxy in front (`tailscale serve`) sends
`X-Forwarded-Proto: https`. Input URLs are fetched by the daemon and may not name private addresses
([AGENTS.md](AGENTS.md#input-urls)); behind `tailscale serve` that includes the daemon's own https output URLs, so send
an output back as an input in base64.

## Computer control (legacy placement)

Today the box decides each step of a computer task, grants it by its policy and waits for the user's tap on
consequential ones; the desktop helper on the Mac connects out to `/v1/desktop` and does what it is granted
([dev/COMPUTER.md](dev/COMPUTER.md)). `computer.placement` is `box_policy`. The direction (JJ's decision) is that
permissions live on the controlled computer: the executor there becomes the final authority and the box's approvals
become, at most, one more input to it. Harnesses should not build on `/v1/computer/*`.

## Not supported

`/v1/capabilities` lists each with what to use instead (`unsupported`):

| Feature | Status |
| --- | --- |
| Responses API (`/v1/responses`) | coming (H5, H6) |
| Realtime GA schema, WebRTC, ephemeral keys, `semantic_vad`, audio formats other than pcm16 24 kHz | unsupported |
| Images or audio in chat messages; `n` > 1; DeepSeek's `thinking` field (ignored) | unsupported |
| The daemon's own https output URLs as inputs, behind a TLS proxy | unsupported |
| `/v1/audio/translations`; streamed transcription; speech `instructions` and SSE | unsupported |
| `/v1/embeddings`, `/v1/completions`, file uploads to `/v1/files` | unsupported |
| MCP OAuth, MCP sessions, resources and prompts | unsupported |

## Fixtures

`tests/fixtures/harness/{web,mobile,helper}.jsonl` are the requests our own three clients actually sent to the Jarvis
box, recorded by `tools/harness/record_proxy.py` (a transparent proxy that redacts tokens, codes and media payloads)
while each ran its own end-to-end check: the web page under `tools/spark/jarvis/client_check.py`, the mobile app's
protocol layer under `apps/mobile` `pnpm e2e`, and the desktop helper (`LOCALROUTER_DESKTOP_FAKE=1`) driven by the MCP
computer tools. `tests/contract.zig` replays their client side against a test daemon: every event type and HTTP route
they use must still be accepted. A change that breaks one breaks a deployed client; change the contract date and the
client together, never the fixture alone. The recorded runs did not happen to send `input_audio_buffer.commit`,
`conversation.item.truncate`, `conversation.item.delete` or `response.cancel` (no push to talk or barge-in), nor the
helper's `authorize`: the contract test sends those four events in their documented shapes, and the hub's unit tests
cover `authorize`. To re-record, run the proxy in front of the daemon and point the client at
it:

```sh
python3 -I tools/harness/record_proxy.py --listen 127.0.0.1:18191 --upstream 127.0.0.1:8190 --label web --out web.jsonl
```

## Harnesses

Qualified recipes for outside harnesses, each with its pinned version and what was run:
[harnesses/OPENCODE.md](harnesses/OPENCODE.md) (OpenCode 1.19.4) and [harnesses/DEEPSEEK.md](harnesses/DEEPSEEK.md)
(DeepSeek Harness 0.2.1-rc.2, and what DeepSeek-API clients need to change). `tools/harness/qualify.sh` repeats the
qualification from the machine a harness runs on. Moving a user between harnesses is export and cancel: export the
conversation from the old one, cancel its running work, then start the new one; no live session or approval moves.
