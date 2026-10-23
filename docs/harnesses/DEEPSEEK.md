# DeepSeek on a LocalRouter box

**What "DeepSeek" means here.** DeepSeek ships its own agent harness, **DeepSeek Harness** (`dsh`,
[deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness), npm `@deepseek-ai/dsh`, MIT, a
developer preview). That is the recipe below: `dsh` on your machine, with the box as its model and media tools. For
other harnesses already set up for DeepSeek's API, [the second part](#harnesses-configured-for-deepseeks-api) says
what carries over to the box unchanged and the two settings that must change.

| | Qualified |
| --- | --- |
| DeepSeek Harness | **0.2.1-rc.2** (npm `latest` on 2026-10-23; git tag `dsh-v0.2.1-rc.2`, commit `687f282e`). A developer preview: its config format may break between versions, so pin it |
| Node | 22.23.4 (dsh wants `^22.19.0` or `>=24`) |
| Box | LocalRouter contract `2026-10-23` ([HARNESS.md](../HARNESS.md)), chat model `qwen3.8-27b` (32k context, one request at a time) |
| Client | DGX-01 as the client (Ubuntu, aarch64), reaching the box over the tailnet through `tailscale serve` (https) |
| Date | 2026-10-23, `tools/harness/qualify.sh dsh` |

## Install

```sh
mkdir -p ~/opt/dsh && npm install --prefix ~/opt/dsh @deepseek-ai/dsh@0.2.1-rc.2
~/opt/dsh/node_modules/.bin/dsh --version   # 0.2.1-rc.2
```

## Connect

dsh is configured with YAML patches over its profiles. Its built-in `deepseek` provider speaks Anthropic Messages to
`api.deepseek.com/anthropic`, so it cannot point at the box; declare the box as an OpenAI-compatible provider of its
`llm-pi-ai` plugin instead, make it the default model, and add the box's MCP server.

1. A token from the box with least privilege (needed on every route with `pairing.required`; harmless otherwise):
   `deploy/jarvis/pair.sh token "dsh on my laptop" --scopes inference,media,read` (as in the
   [OpenCode recipe](OPENCODE.md#connect)).
2. `export LOCALROUTER_URL=https://<box>.<tailnet>.ts.net:10443 LOCALROUTER_TOKEN=<the token>` (dsh also reads a
   `.env` in the working directory or in `$DSH_HOME`).
3. [`dsh.cordis.patch.yml`](dsh.cordis.patch.yml) as `~/.dsh/profiles/headless/cordis.patch.yml` (and the same file
   under `profiles/web/` for the web UI):

```yaml
- id: llm-pi-ai
  config:
    providers:
      spark:
        api: openai-completions
        baseURL: !!js process.env.LOCALROUTER_URL + '/v1'
        apiKeyEnv: LOCALROUTER_TOKEN
        timeoutMs: 600000
        streamIdleTimeoutMs: 300000
        compat:
          supportsDeveloperRole: false
          maxTokensField: max_tokens
        models:
          - id: qwen3.8-27b
            contextWindow: 32768
            maxTokens: 8192
            input: [text]
- id: agent-default-model
  config:
    provider: spark
    model: qwen3.8-27b
- insert:
    - id: mcp-spark
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        transport: streamable-http
        serverName: spark
        url: !!js process.env.LOCALROUTER_URL + '/mcp'
        headers:
          Authorization: !!js "'Bearer ' + process.env.LOCALROUTER_TOKEN"
        toolCallTimeoutMs: 600000
        failOnStartupError: true
```

`dsh --profile headless --dump-config` shows the composed result without starting anything. A patch replaces a row's
whole `config`, so this `llm-pi-ai` row holds every provider you want (add DeepSeek's own beside `spark` to keep it).
`supportsDeveloperRole: false` keeps the system prompt a `system` message.

Run one task with `dsh --profile headless "..."` (`--json` for its events), or the web UI with `dsh web --no-open`
(`127.0.0.1:3080`). The box's tools appear as `mcp__spark__generate_image`, `mcp__spark__synthesize_speech`...

## What was qualified

`tools/harness/qualify.sh dsh --bin ~/opt/dsh/node_modules/.bin/dsh --config docs/harnesses/dsh.cordis.patch.yml --url $LOCALROUTER_URL`,
from DGX-01, with `DSH_TELEMETRY_DISABLED=1` and dsh's default permission mode (`workspace-write`):

| Run | Result |
| --- | --- |
| Text with a tool call: "create answer.txt with the sum of 1..100, read it back" | dsh's `write` and `read` ran on the client, and `bash` (`ls`, then `od` to check the bytes) in its workspace sandbox without asking; `answer.txt` is `5050`; 67 s |
| Image through MCP: "use the spark generate_image tool, 512x512, a red fox in snow" | one `mcp__spark__generate_image` call; a 512x512 PNG from `qwen-image-2.1-turbo`; 46 s; the returned URL opens as a PNG on the client |

## Notes

- **Permissions are dsh's own, on your machine.** `workspace-write` (the default) lets it run shell commands in the
  working directory without asking (measured); headless, anything that needs an approval is refused (dsh's
`user-approval` docs). `DSH_PERMISSION_MODE=read-only`
  stops writes; `danger-full-access` removes the sandbox. The box never sees or relaxes these.
- **Telemetry.** dsh uploads session logs only for requests to DeepSeek's own API; `DSH_TELEMETRY_DISABLED=1` turns
  its telemetry off entirely.
- The [OpenCode recipe's limits](OPENCODE.md#limits) apply as they are: one request at a time, 32k context, no
  Responses API, https media URLs not accepted back as inputs, no voice, no computer scope.

## Harnesses configured for DeepSeek's API

A harness with a DeepSeek preset (an OpenAI-compatible client with DeepSeek's base URL and model names) talks to the
box with **two changes**: the base URL (`https://<box>...:10443/v1`) and the model id (`qwen3.8-27b`: `deepseek-chat`,
like any name the box does not serve, is a 404 listing the box's chat models). Measured on the box with DeepSeek-shaped requests:

| DeepSeek API behaviour a client relies on | On the box |
| --- | --- |
| Chat Completions with OpenAI function tools, streamed `delta.tool_calls` | the same |
| `reasoning_content` beside `content` (and `delta.reasoning_content`) | the same |
| Sending the assistant message back as it came (with `reasoning_content` and `tool_calls`), then the `tool` result | accepted; the model continues from the tool result |
| `response_format: {"type": "json_object"}` | the same |
| `thinking: {"type": "enabled" or "disabled"}` | accepted and **ignored**: the model thinks unless `reasoning_effort: "none"` (or `chat_template_kwargs.enable_thinking: false`) |
| `usage.prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` | absent: the box reports `usage.prompt_tokens_details.cached_tokens` and `completion_tokens_details.reasoning_tokens` |
| `/beta` strict function calling, FIM (`/completions` with `suffix`), chat prefix completion | not offered |
| The Anthropic-format endpoint (`/anthropic`) | not offered: use the OpenAI-compatible one |

## Switching harnesses

dsh keeps its sessions under `~/.dsh/sessions/`; keep what you need, stop its running task, then start the other
harness. Nothing on the box needs undoing; revoke the token if this machine should stop using the box.
