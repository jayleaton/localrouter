# LocalRouter over MCP

Every capability of a LocalRouter box is an MCP tool, so any agent or harness can use it: chat, images, video,
speech, transcription, the model and voice lists, and job control. Two transports reach the same tools:

| Transport | Address | For |
| --- | --- | --- |
| Streamable HTTP (stateless) | `http://<host>:8190/mcp`; on the Jarvis box `https://dgx-01.<tailnet>.ts.net:10443/mcp` | clients that speak MCP over HTTP |
| stdio | `localrouter mcp-stdio --url http://<host>:8190` | clients that only start a command; a stateless bridge to the HTTP endpoint |

## Security: there is no authentication

LocalRouter has no API keys. Anyone who can reach `/mcp` can run every tool: use the GPU, read the outputs, and read
audio files under the directories listed in `local_files`. Keep it on loopback (the default, `127.0.0.1`) or your
tailnet (`--host tailscale`, or `tailscale serve` as `deploy/jarvis/install.sh` sets up), where the tailnet's ACLs
decide who connects. Never bind it to `0.0.0.0` or a LAN address, and never put it behind a public reverse proxy.

## Connect

**Claude Code**

```sh
claude mcp add --transport http localrouter http://127.0.0.1:8190/mcp            # on the box itself
claude mcp add --transport http localrouter https://dgx-01.<tailnet>.ts.net:10443/mcp   # from another tailnet machine
claude mcp add localrouter -- localrouter mcp-stdio --url http://127.0.0.1:8190   # stdio, through the bridge
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

The bridge reads `$LOCALROUTER_URL` when `--url` is not given. It keeps no state; a daemon it cannot reach answers
each request with a JSON-RPC error rather than hanging.

## Tools

Call `list_models` first. A model that is not loaded loads on its first call; the chat model and the voice models are
kept warm. Long jobs never need a long tool call: `generate_image`, `generate_video` and `transcribe_audio` take
`wait_s` (default 600, at most 3000) and return the job id when it passes; `get_job` with its own `wait_s` picks the
job up. Output files come back as `resource_link` URLs on the server (kept 24 h), images and speech also inline.

Failures come back as tool errors (`isError: true`) whose text says what to change: a missing or mistyped argument
names the tool and points at its schema, an unknown voice lists the model's voices, a size the video model does not
take gives its limit and says to omit size for its default, an unqualified model says to run its qualify script, and a
full machine says `insufficient_memory`.

| Tool | Arguments (required in bold) | Returns |
| --- | --- | --- |
| `list_models` | | every model: id, kind, capabilities, `default_for`, `state` (unloaded, loading, ready, busy), `resident_bytes`, `lane`, priority, `keep_loaded`, `qualified`; speech models' voices; the machine's `memory` |
| `list_voices` | | each speech model's `voices`, `default_voice`, `state` and whether it is the default |
| `reserve_for` | **`model`**, `seconds`, `size`, `dry_run`, `hold_s` (180), `wait_s` | what it unloads (or would) to fit the model's job, the bytes needed and free; the model loaded and held for its job. What it unloads comes back by itself after the job. Never a chat model |
| `unload_model` | **`model`**, `hold_s` (120) | one model unloaded now, back by itself after `hold_s`. A chat model is refused |
| `load_model` | **`model`**, `wait_s` | loads it (refused if it would need the chat model's memory) |
| `chat` | `prompt` and/or `messages`, `system`, `model`, `reasoning`, `max_tokens`, `temperature`, `top_p`, `stop`, `seed`, `tools`, `tool_choice`, `response_format` | the reply's text (or its tool calls as JSON), then the whole completion |
| `generate_image` | **`prompt`**, `size` (1024x1024), `n`, `seed`, `steps`, `negative_prompt`, `images` (edit inputs), `inline_images`, `wait_s` | PNGs inline and as URLs, the seed; or the job id |
| `generate_video` | **`prompt`**, `size` (omit: the model's own), `seconds`, `fps`, `seed`, `steps`, `audio`, `image`, `wait_s` | the MP4's URL; or the job id |
| `synthesize_speech` | **`input`**, `voice`, `model`, `speed`, `preset`, `format` (wav), `seed`, `references`, `inline_audio` | the audio inline and as a URL |
| `transcribe_audio` | `audio` (base64, data URL or URL) or `path` (a file on the box inside `local_files`), `language`, `prompt`, `timestamps`, `temperature`, `wait_s` | the text, then JSON (language, duration, segments, words) |
| `get_job` | **`id`**, `wait_s` | status, progress, output URLs; a transcript's text |
| `cancel_job` | **`id`** | the job, cancelled |
| `release_models` | | unloads every model now |

Every schema is strict (`additionalProperties: false`), so a misspelled argument shows up in the client rather than
being dropped.

### Making room for a heavy job

The procedure an agent follows to make room for a video and restore the models after (check with `reserve_for`
`dry_run`, tell the user, reserve, run the job, confirm the models came back) is a skill:
[skills/jarvis-model-manager/SKILL.md](../skills/jarvis-model-manager/SKILL.md). The Jarvis chat model follows the same
steps (`voice.manage_models`). Voice sessions on the box are told before a voice model goes (`localrouter.voice.paused`)
and stay open; the models come back by themselves when the job is done, even if no one calls `load_model`.

## Examples

Each line is a `tools/call` request body for `POST /mcp` (or a line on the bridge's stdin):

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"list_models","arguments":{}}}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"list_voices","arguments":{}}}
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"chat","arguments":{"prompt":"Three facts about Canberra.","reasoning":false,"max_tokens":200}}}
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"generate_image","arguments":{"prompt":"a red fox in fresh snow","size":"1024x1024","seed":7,"wait_s":120}}}
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

With curl: `curl -s http://127.0.0.1:8190/mcp -H 'content-type: application/json' -d '<one of the lines>'`.

## Files on the box

`transcribe_audio` reads a local file by `path` only inside a directory the config's `local_files` lists (none by
default; the Jarvis deploy lists `~/jarvis/deploy/inbox`). The path is resolved first, so `..` and symlinks cannot
leave the directory. Anything else is refused with a message naming the setting.
