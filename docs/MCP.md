# LocalRouter over MCP

Every capability of a LocalRouter box is an MCP tool, so any agent or harness can use it: chat, images, video,
speech, transcription, the model and voice lists, and job control. Two transports reach the same tools:

| Transport | Address | For |
| --- | --- | --- |
| Streamable HTTP (stateless) | `http://<host>:8190/mcp`; on the Jarvis box `https://dgx-01.<tailnet>.ts.net:10443/mcp` | clients that speak MCP over HTTP |
| stdio | `localrouter mcp-stdio --url http://<host>:8190` | clients that only start a command; a stateless bridge to the HTTP endpoint |

## Security: pairing, and who can reach it

By default LocalRouter has no API keys: anyone who can reach `/mcp` can run every tool, use the GPU, read the outputs,
and read audio files under the directories listed in `local_files`. Keep it on loopback (the default, `127.0.0.1`) or
your tailnet (`--host tailscale`, or `tailscale serve` as `deploy/jarvis/install.sh` sets up), where the tailnet's ACLs
decide who connects. Never bind it to `0.0.0.0` or a LAN address, and never put it behind a public reverse proxy.

To narrow that to your own devices and harnesses, set `"pairing": {"required": true}` in the config. Every route but
`/health`, pairing itself (`/v1/pair`) and the `/voice` page then answers only a paired device's token, and only for
what its scopes allow: no token (or a revoked one) is HTTP 401 (`invalid_api_key`, `WWW-Authenticate: Bearer`), a
token without the scope 403, both before the request is read. That holds for HTTP, the `/v1/realtime` socket and `/mcp`
alike. There is no loopback exemption: behind `tailscale serve` every peer arrives as loopback. Tokens are made on the
box:

```sh
localrouter pair token opencode --scopes inference,media,read --config FILE   # a harness: prints a token once
localrouter pair add "JJ's phone" --config FILE    # a code (XXXX-XXXX, 10 minutes, once) for the app: POST /v1/pair
localrouter pair list --config FILE                # id, name, created, scopes (never tokens)
localrouter pair scopes ID|NAME inference,read --config FILE   # replaces what a device may do
localrouter pair revoke ID|NAME --config FILE      # refused from the next request, no restart
```

### Scopes

Issue a harness the least it needs (`--scopes LIST` on `token` or `add`; `GET /v1/pair` shows a token's own):

| Scope | HTTP | MCP tools | Voice session (`/v1/realtime`) |
| --- | --- | --- | --- |
| `inference` | `POST /v1/chat/completions` | `chat` | opens the socket (its speech in and out is part of it) |
| `media` | `/v1/images/*`, `POST /v1/videos`, `/v1/audio/*` | `generate_image`, `edit_image`, `generate_video`, `transcribe_audio`, `synthesize_speech` | the picture, edit and video tools |
| `decisions` | `POST /v1/systemone` | `decide` | |
| `read` | `GET /v1/models`, `/v1/tools`, `/v1/skills/model-manager`, `/v1/voices` | `list_models`, `list_voices` | `list_models` |
| `manage` | `POST /v1/tools/{id}/load`, `/unload`, `/v1/tools/release`; `POST /v1/voices`, `PATCH` and `DELETE /v1/voices/{id}` (cloning a voice) | `load_model`, `unload_model`, `reserve_for`, `release_models` | the model tools |

Any of these opens `/mcp`, whose `tools/list` then shows only the tools the token may call; a tool outside its scopes
is a tool error naming the scope. Changing which models are loaded always needs `manage`: no other scope implies it.

**Jobs and files are their device's own.** A job (an image, a video, speech, a transcript, a completion, a decision)
belongs to the device whose token asked for it, over HTTP, MCP or a voice session. `GET /v1/jobs/{id}`,
`/v1/videos/{id}` (and its `DELETE`), `/v1/files/{id}/...`, `get_job` and `cancel_job` answer only that device (with
`read` or the job's own scope); any other token gets 404, as for a job that does not exist. Fetch an output URL with
the same `Authorization` header. Jobs the daemon makes for itself are nobody's.

**Computer control** keeps its own scopes, checked whether or not pairing is required: `desktop_helper`, `computer` and
`approve` ([dev/COMPUTER.md](dev/COMPUTER.md)). They grant no service scope.

**Legacy tokens keep working unchanged.** A device paired before service scopes holds `voice` and `mcp` (or one of
them, or either beside computer control's). `voice` opens `/v1/realtime`, `mcp` opens `/mcp`, and either grants every
service scope above, because before service scopes the other routes answered anyone and `/mcp` offered every tool.
Such a device is not narrowed silently: to give it less, re-issue its scopes (`localrouter pair scopes NAME LIST`) or
pair a new token. With pairing off (the default) scopes are not checked at all, except computer control's.

On the Jarvis box use `deploy/jarvis/pair.sh token claude-code --scopes inference,media,read` (the deployed config and
data). Only hashes of the tokens are kept, in `<data_dir>/devices.json` (0600). A token goes in
`Authorization: Bearer <token>`; for `/v1/realtime` it may also be a `Sec-WebSocket-Protocol` entry
`openai-insecure-api-key.<token>` (the OpenAI browser SDK's) or `?api_key=<token>`. Keep it in the harness's secret
store, never in a URL you share: an MCP client that cannot send headers uses the stdio bridge (`--token`).

## Connect

**Claude Code**

```sh
claude mcp add --transport http localrouter http://127.0.0.1:8190/mcp            # on the box itself
claude mcp add --transport http localrouter https://dgx-01.<tailnet>.ts.net:10443/mcp   # from another tailnet machine
claude mcp add localrouter -- localrouter mcp-stdio --url http://127.0.0.1:8190   # stdio, through the bridge
```

With `pairing.required`, pass the token:

```sh
claude mcp add --transport http localrouter https://<host>/mcp --header "Authorization: Bearer <token>"
claude mcp add localrouter -- localrouter mcp-stdio --url https://<host> --token <token>
```

**Codex** (`~/.codex/config.toml`)

```toml
[mcp_servers.localrouter]
command = "localrouter"
args = ["mcp-stdio", "--url", "http://127.0.0.1:8190"]
tool_timeout_sec = 900   # image and video loads; or pass a small wait_s and poll get_job
```

**Cursor** (`~/.cursor/mcp.json`, or `.cursor/mcp.json` in a project)

```json
{"mcpServers": {"localrouter": {"url": "http://127.0.0.1:8190/mcp"}}}
```

**Any MCP client** (the common `mcpServers` shape)

```json
{"mcpServers": {
  "localrouter": {"type": "http", "url": "http://127.0.0.1:8190/mcp"},
  "localrouter-stdio": {"command": "localrouter", "args": ["mcp-stdio", "--url", "http://127.0.0.1:8190"]}
}}
```

The bridge reads `$LOCALROUTER_URL` when `--url` is not given, and `$LOCALROUTER_TOKEN` when `--token` is not (sent as
`Authorization: Bearer`; an unpaired bridge's requests come back as JSON-RPC errors that say so). It keeps no state;
a daemon it cannot reach answers each request with a JSON-RPC error rather than hanging.

## Tools

Call `list_models` first. A model that is not loaded loads on its first call; the chat model and the voice models are
kept warm. Long jobs never need a long tool call: `generate_image`, `edit_image`, `generate_video` and `transcribe_audio` take
`wait_s` (default 600, at most 3000) and return the job id when it passes; `get_job` with its own `wait_s` picks the
job up. Output files come back as `resource_link` URLs on the server (kept 24 h), images and speech also inline.

Failures come back as tool errors (`isError: true`) whose text says what to change: a missing or mistyped argument
names the tool and points at its schema, an unknown voice lists the model's voices, a size the video model does not
take gives its limit and says to omit size for its default, an unqualified model says to run its qualify script, and a
full machine says `insufficient_memory`.

| Tool | Arguments (required in bold) | Returns |
| --- | --- | --- |
| `list_models` | | every model: id, kind, capabilities, `default_for`, `state` (unloaded, loading, ready, busy), `resident_bytes`, `lane`, priority, `keep_loaded`, `qualified`; speech models' voices; the machine's `memory` |
| `list_voices` | | each speech model's `voices` (voices cloned for it on this machine included), `default_voice`, `state` and whether it is the default |
| `reserve_for` | **`model`**, `seconds`, `size`, `dry_run`, `hold_s` (180), `wait_s` | what it unloads (or would) to fit the model's job, the bytes needed and free; the model loaded and held for its job. What it unloads comes back by itself after the job. Never a chat model |
| `unload_model` | **`model`**, `hold_s` (120) | one model unloaded now, back by itself after `hold_s`. A chat model is refused |
| `load_model` | **`model`**, `wait_s` | loads it (refused if it would need the chat model's memory) |
| `chat` | `prompt` and/or `messages`, `system`, `model`, `reasoning`, `max_tokens`, `temperature`, `top_p`, `stop`, `seed`, `tools`, `tool_choice`, `response_format` | the reply's text (or its tool calls as JSON), then the whole completion |
| `decide` | **`state`**, **`questions`** (choice, noul, score; [AGENTS.md](AGENTS.md#decisions-the-system-one-shape)), `model` | one line a question (`route: generate_image (0.912)`), then the model's reply, checked against the questions. Acts on nothing |
| `generate_image` | **`prompt`**, `size` (1024x1024; for an edit, the input's shape), `n`, `seed`, `steps`, `negative_prompt`, `images` (edit inputs), `inline_images`, `wait_s` | PNGs inline and as URLs, the seed; or the job id |
| `edit_image` | **`prompt`**, `image` or `images` (1 to 5), `mask`, `size` (omit: the input's shape), `model`, `n`, `seed`, `steps`, `negative_prompt`, `inline_images`, `wait_s` | as `generate_image`; refused when no model lists `image_edit` |
| `generate_video` | **`prompt`**, `size` (omit: the model's own), `seconds`, `fps`, `seed`, `steps`, `audio`, `image`, `wait_s` | the MP4's URL; or the job id |
| `synthesize_speech` | **`input`**, `voice`, `model`, `speed`, `preset`, `instructions` (for a model whose `instructions` is true; Drama 3 speaks them as `[...]` before the text), `format` (wav), `seed`, `references`, `inline_audio` | the audio inline and as a URL |
| `transcribe_audio` | `audio` (base64, data URL or URL) or `path` (a file on the box inside `local_files`), `language`, `prompt`, `timestamps`, `temperature`, `wait_s` | the text, then JSON (language, duration, segments, words) |
| `get_job` | **`id`**, `wait_s` | status, progress, output URLs; a transcript's text |
| `cancel_job` | **`id`** | the job, cancelled |
| `release_models` | | unloads every model now |
| `computer_control` | **`control`** (`{"kind":"open_app","app":"Safari"}`, `quit_app`, `volume`, `media`, `dark_mode`, `lock_screen`), `computer` | the paired computer's sentence ("Opened Safari"); locking waits for a tap in the web UI |
| `computer_use` | **`goal`**, `text` (exact text to type), `computer`, `wait_s` (120, at most 600) | the outcome (`done`, `stuck`, `blocked`, `unavailable`, `stopped`, `failed`) with its steps, or the task id while it runs |
| `computer_status` | `task_id`, `wait_s` | a task's state (and the approval it waits for), or the paired computers and recent tasks |
| `computer_stop` | `task_id` (none: every running task) | how many it stopped |

The computer tools act on the user's own computer through the LocalRouter desktop helper, and need a paired device token
with the `computer` scope (`Authorization: Bearer`), even when `pairing.required` is off; without one they refuse. Sending, buying, deleting, posting and
typing into a field that is not a search box each wait for the user's tap in the web UI: an agent's token cannot approve
(that needs the `approve` scope). Details: [dev/COMPUTER.md](dev/COMPUTER.md).

Every schema is strict (`additionalProperties: false`), so a misspelled argument shows up in the client rather than
being dropped.

### Editing a picture

`edit_image` changes a picture as its `prompt` says. Send the picture as `image`: a base64 string, a data URL or an
http(s) URL (this server's own `/v1/files/...` output URLs too, so the URL `generate_image` returned works as is; with
pairing on, or behind `tailscale serve`, send base64 instead: the daemon fetches a URL without your token). Several
inputs go in `images` (1 to 5; the model's own limit may be lower): the first is the one edited, the others guide it.
PNG, JPEG or WebP, up to 32 MB each.

Without `size` the result keeps the input's shape (the last input's), at a size the model picks; with `size` it is
that size. `mask` (the same forms) is a PNG of the first image's size whose alpha is 0 where the edit goes; wherever
it is opaque the first image's pixels are kept (exactly, when `size` is the input's own). It needs a model whose
`capabilities` in `list_models` include `image_edit`; an image model lists it only once its edit pack is installed
([dev/IMAGE-EDIT.md](dev/IMAGE-EDIT.md)). Without one the call is a tool error naming the models and what each can do.
On a GB10 (Qwen-Image 2.1 Turbo, FP8) a warm 1024x1024 edit of one input takes 11 to 12.5 s, of two inputs about 16 s.

### Making room for a heavy job

The procedure an agent follows to make room for a video and restore the models after (check with `reserve_for`
`dry_run`, tell the user, reserve, run the job, confirm the models came back) is a skill:
[skills/jarvis-model-manager/SKILL.md](../skills/jarvis-model-manager/SKILL.md). The Jarvis chat model does not
(`voice.manage_models` offers it the tools for when the user asks): its session checks memory itself. Before a video
it runs the `reserve_for` dry run and, when a voice model would go, says that the voice pauses; after a job it lists
any model still coming back in the call's result. Voice sessions on the box are told before a voice model goes
(`localrouter.voice.paused`) and stay open; the models come back by themselves when the job is done, even if no one
calls `load_model`.

## Examples

Each line is a `tools/call` request body for `POST /mcp` (or a line on the bridge's stdin):

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_models","arguments":{}}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_voices","arguments":{}}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"chat","arguments":{"prompt":"Three facts about Canberra.","reasoning":false,"max_tokens":200}}}
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"generate_image","arguments":{"prompt":"a red fox in fresh snow","size":"1024x1024","seed":7,"wait_s":120}}}
{"jsonrpc":"2.0","id":15,"method":"tools/call","params":{"name":"edit_image","arguments":{"prompt":"make it night","image":"http://127.0.0.1:8190/v1/files/img_…/0.png","wait_s":300}}}
{"jsonrpc":"2.0","id":16,"method":"tools/call","params":{"name":"edit_image","arguments":{"prompt":"a lake at night","image":"<base64 PNG>","mask":"<base64 PNG, left half transparent>","size":"1024x1024"}}}
{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"generate_video","arguments":{"prompt":"waves at dusk, gulls calling","seconds":2,"wait_s":30}}}
{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"get_job","arguments":{"id":"video_…","wait_s":300}}}
{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"synthesize_speech","arguments":{"input":"Hello from the GPU box.","voice":"Mae"}}}
{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"transcribe_audio","arguments":{"audio":"http://127.0.0.1:8190/v1/files/speech_…/speech.wav"}}}
{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"transcribe_audio","arguments":{"path":"/home/dgx01/jarvis/deploy/inbox/meeting.m4a","timestamps":"segment"}}}
{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"cancel_job","arguments":{"id":"video_…"}}}
{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"reserve_for","arguments":{"model":"minimax-h3","seconds":3,"dry_run":true}}}
{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"unload_model","arguments":{"model":"qwen-image-2.1-turbo","hold_s":300}}}
{"jsonrpc":"2.0","id":13,"method":"tools/call","params":{"name":"load_model","arguments":{"model":"qwen-image-2.1-turbo"}}}
{"jsonrpc":"2.0","id":14,"method":"tools/call","params":{"name":"release_models","arguments":{}}}
```

With curl: `curl -s http://127.0.0.1:8190/mcp -H 'content-type: application/json' -d '<one of the lines>'` (with
pairing: add `-H "authorization: Bearer $TOKEN"`).

## Files on the box

`transcribe_audio` reads a local file by `path` only inside a directory the config's `local_files` lists (none by
default; the Jarvis deploy lists `~/jarvis/deploy/inbox`). The path is resolved first, so `..` and symlinks cannot
leave the directory. Anything else is refused with a message naming the setting.
