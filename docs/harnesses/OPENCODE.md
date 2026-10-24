# OpenCode on a LocalRouter box

OpenCode runs on your machine, with your files, shell and permissions; the box serves the model (Chat Completions)
and its media tools (MCP). Nothing of OpenCode runs on the box.

| | Qualified |
| --- | --- |
| OpenCode | **1.19.4** (`anomalyco/opencode`, release `v1.19.4`, `opencode-linux-arm64.tar.gz` sha256 `8568230ccd847dc93e95b9f6894379077d3e014a16b48040743573e77c23e85f`) |
| Box | LocalRouter contract `2026-10-23` ([HARNESS.md](../HARNESS.md)), chat model `qwen3.8-27b` (32k context, one request at a time) |
| Client | DGX-01 as the client (Ubuntu, aarch64), reaching the box over the tailnet through `tailscale serve` (https) |
| Date | 2026-10-23, `tools/harness/qualify.sh opencode` |

## Install

```sh
curl -fsSLo opencode.tar.gz https://github.com/anomalyco/opencode/releases/download/v1.19.4/opencode-linux-arm64.tar.gz
echo "8568230ccd847dc93e95b9f6894379077d3e014a16b48040743573e77c23e85f  opencode.tar.gz" | sha256sum -c
tar -xzf opencode.tar.gz && ./opencode --version   # 1.19.4
```

Or `npm install -g opencode-ai@1.19.4`, or `curl -fsSL https://opencode.ai/install | VERSION=1.19.4 bash`. Other
platforms take the same version's asset for their OS and architecture. Keep auto-update off (`"autoupdate": false`
below) so the version stays the one qualified.

## Connect

1. A token, made on the box (needed on every route once the operator turns on `pairing.required`; harmless
   otherwise). Least privilege: `inference` (chat), `media` (the media tools), `read` (`list_models`); add
   `decisions` for `spark_decide`. Never `computer`, `approve` or the legacy `mcp` (which grants every service scope).

   ```sh
   deploy/jarvis/pair.sh token "opencode on my laptop" --scopes inference,media,read   # prints the token once
   ```

   The box's MCP `tools/list` then shows only the tools the token may call.

2. In your shell (or a secret manager OpenCode reads `{env:...}` from):

   ```sh
   export LOCALROUTER_URL=https://<box>.<tailnet>.ts.net:10443   # the box's tailscale serve address, no /v1
   export LOCALROUTER_TOKEN=<the token>
   ```

3. [`opencode.json`](opencode.json) in your project (or `~/.config/opencode/opencode.json`, or `OPENCODE_CONFIG`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "spark/qwen3.8-27b",
  "small_model": "spark/qwen3.8-27b",
  "autoupdate": false,
  "share": "disabled",
  "provider": {
    "spark": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LocalRouter (Spark)",
      "options": { "baseURL": "{env:LOCALROUTER_URL}/v1", "apiKey": "{env:LOCALROUTER_TOKEN}", "timeout": 600000 },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen3.8 27B on the Spark",
          "tool_call": true,
          "reasoning": true,
          "interleaved": { "field": "reasoning_content" },
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  },
  "mcp": {
    "spark": {
      "type": "remote",
      "url": "{env:LOCALROUTER_URL}/mcp",
      "headers": { "Authorization": "Bearer {env:LOCALROUTER_TOKEN}" },
      "oauth": false,
      "timeout": 30000,
      "enabled": true
    }
  },
  "permission": { "edit": "allow", "bash": "deny", "webfetch": "deny", "external_directory": "deny" }
}
```

| Setting | Why |
| --- | --- |
| `@ai-sdk/openai-compatible` | Chat Completions, what was qualified; `@ai-sdk/openai` would use the Responses API, which the box serves too but this recipe has not qualified |
| `reasoning`, `interleaved.field` | the model's thinking arrives in `reasoning_content`; this keeps it out of the answer and sends it back with tool calls |
| `limit` | the deployed model's context is 32768 tokens; OpenCode's own prompt with every Spark tool is about 11k of it (measured) |
| `mcp.spark` | the box's tools appear as `spark_generate_image`, `spark_synthesize_speech`, `spark_transcribe_audio`, `spark_generate_video`, `spark_decide`, `spark_get_job`... `oauth: false`: the box takes a bearer token, not OAuth |
| `permission` | what the qualification ran with: file edits allowed, shell and web denied. Choose your own; OpenCode enforces it on your machine, the box never sees it |

Run `opencode` in your project, or one task: `opencode run "..."`. For media, ask for it ("make a 1024x1024 picture
of ..."): the model calls `spark_generate_image` and gets the PNG's URL on the box (and the image inline unless
`inline_images` is false).

## What was qualified

`tools/harness/qualify.sh opencode --bin ./opencode --config docs/harnesses/opencode.json --url $LOCALROUTER_URL`,
from DGX-01:

| Run | Result |
| --- | --- |
| Text with a tool call: "create answer.txt with the sum of 1..100, read it back" | OpenCode's `write` then `read` ran on the client; `answer.txt` is `5050`; 47 s |
| Image through MCP: "use the spark generate_image tool, 512x512, a red fox in snow" | one `spark_generate_image` call; a 512x512 PNG from `qwen-image-2.1-turbo`; 39 s; the returned URL opens as a PNG on the client |

## Limits

- **One request at a time.** The box's chat server runs one request at a time: OpenCode's parallel subagents and a
  voice session on the same box queue behind each other.
- **Context.** 32k tokens. Long sessions compact sooner than with hosted models. Every Spark tool's schema is in the
  prompt (about 11k tokens with OpenCode's own); if space runs short, connect fewer MCP servers.
- **Chat Completions, no images in chat.** Text in, text and tool calls out ([HARNESS.md](../HARNESS.md#chat-completions)). The box also serves the Responses API now; this recipe stays on Chat Completions, which is what was qualified.
- **Media URLs** are https behind `tailscale serve`. Passing one back as an input (an image to edit or animate) is
  refused; the model should send the file as base64 or start from the prompt.
- **No voice.** OpenCode has no client for the box's realtime voice; it is a text and tools harness here.
- **Computer tools** need the `computer` scope; this recipe does not grant it. Computer permissions belong to the
  computer being controlled.

## Switching harnesses

Export what you want to keep (`opencode export [sessionID]` writes a session as JSON), stop or cancel OpenCode's running work,
then start the other harness. Its own conversation does not move, and nothing on the box needs undoing; revoke the
token (`pair.sh revoke NAME`) if this machine should stop using the box.
