# Codex on a LocalRouter box

Codex CLI runs on your machine, with your files, shell, sandbox and approvals; the box serves the model through its
Responses API (`/v1/responses`) and its media tools over MCP. Nothing of Codex runs on the box.

| | Qualified |
| --- | --- |
| Codex CLI | **0.164.0** (npm `@openai/codex@0.164.0`; `codex --version` prints `codex-cli 0.164.0`) |
| Box | LocalRouter contract `2026-10-23` ([HARNESS.md](../HARNESS.md#responses)), chat model `qwen3.8-27b` (32k context, one request at a time) |
| Client | DGX-01 as the client (Ubuntu, aarch64) |
| Date | 2026-10-24, `tools/harness/qualify.sh codex` |

## Install

```sh
npm install -g @openai/codex@0.164.0
codex --version   # codex-cli 0.164.0
```

Keep the version pinned (`check_for_update_on_startup = false` below). A newer Codex may send fields this box refuses
by name (a 400 that says which); qualify it before you move.

## Connect

1. A token, made on the box (needed on every route once the operator turns on `pairing.required`; harmless
   otherwise). Least privilege: `inference` (Responses), `media` (the media tools), `read` (`list_models`). Never
   `computer`, `approve` or the legacy `mcp`.

   ```sh
   deploy/jarvis/pair.sh token "codex on my laptop" --scopes inference,media,read   # prints the token once
   ```

2. In your shell: `export LOCALROUTER_TOKEN=<the token>`.

3. [`codex.config.toml`](codex.config.toml) as `~/.codex/config.toml` (or `$CODEX_HOME/config.toml`), with your box's
   `tailscale serve` address in `base_url` and the MCP `url`:

```toml
model = "qwen3.8-27b"
model_provider = "spark"
model_context_window = 32768
web_search = "disabled"
check_for_update_on_startup = false

[features]
multi_agent = false

[model_providers.spark]
name = "LocalRouter (Spark)"
base_url = "https://BOX.TAILNET.ts.net:10443/v1"
env_key = "LOCALROUTER_TOKEN"
wire_api = "responses"
stream_idle_timeout_ms = 600000
request_max_retries = 1
stream_max_retries = 1

[mcp_servers.spark]
url = "https://BOX.TAILNET.ts.net:10443/mcp"
bearer_token_env_var = "LOCALROUTER_TOKEN"
tool_timeout_sec = 600
```

| Setting | Why |
| --- | --- |
| `wire_api = "responses"` | Codex's custom providers speak the Responses API; the box serves it statelessly, and Codex replays the whole conversation each turn (`store: false`), which is what the box needs |
| `web_search = "disabled"` | hosted web search is an OpenAI service; the box refuses hosted tools by name (400 `unsupported_tool`) rather than pretend |
| `multi_agent = false` | sub-agents send parallel requests, and the box runs one at a time; they would queue behind each other in a 32k context. Not required: their tools are plain functions the box takes |
| `model_context_window` | the deployed model's context; Codex's own prompt and tools take about 10k of it |
| `stream_idle_timeout_ms` | a turn may wait behind a voice reply or another harness's request before its first event |
| `mcp_servers.spark` | the box's tools reach the model as `mcp__spark__generate_image`, `...synthesize_speech`, `...transcribe_audio`... Codex sends an MCP server's tools as a `namespace`; the box names each function `<namespace>__<name>` for the model and returns calls with `namespace` and `name`, so Codex runs them as it does against OpenAI |

Codex warns `Model metadata for qwen3.8-27b not found. Defaulting to fallback metadata`: it knows nothing of this
model's family, and the fallback works (that is what was qualified). Its sandbox is its own: on Linux it needs user
namespaces for bubblewrap (DGX-01 refuses them, so the qualification ran with `sandbox_mode = "danger-full-access"`
in fresh scratch directories). Keep your own sandbox setting; the box never sees it.

Run `codex` in your project, or one task: `codex exec "..."`, and continue it with `codex exec resume --last "..."`.

## What was qualified

`tools/harness/qualify.sh codex --bin $(which codex) --config docs/harnesses/codex.config.toml --url $URL --sandbox
danger-full-access`, from DGX-01, against the merged and deployed box:

| Run | Result |
| --- | --- |
| Text with a tool call: "create answer.txt with the sum of 1..100, read it back" | Codex's own `exec_command` wrote and read the file on the client; `answer.txt` is `5050` |
| Continuation: the same session resumed, "write next.txt: the number you wrote plus one" | `next.txt` is `5051`, written from the conversation so far; the box got the whole first turn back (its reasoning, calls and outputs) |
| Image through MCP: "use the spark generate_image tool, 512x512, a red fox in snow" | one `spark.generate_image` call; the returned URL opens as a PNG on the client |

What Codex sent during the first two runs (before the deploy, through `tools/harness/record_proxy.py` to the PR's
daemon on the same chat server) is the contract fixture `tests/fixtures/harness/codex.jsonl`: five streamed
`/v1/responses` requests and its MCP handshake, replayed by `tests/contract.zig` on every build.

## Limits

- **One request at a time.** The box's chat server runs one request at a time: Codex's turns, other harnesses and a
  voice session queue behind each other.
- **Context.** 32k tokens. Long sessions reach Codex's compaction sooner than with hosted models. Codex's remote
  compaction is OpenAI's service and is not offered here.
- **No images in prompts, no hosted tools.** Text in, text, reasoning and function calls out; `view_image` results
  and `-i` image attachments are refused (the chat model takes text only), as are web search and other hosted tools.
- **Reasoning** comes back as raw `reasoning_text`, never summarized or encrypted; `model_reasoning_summary`
  `concise` or `detailed` is refused. `model_reasoning_effort` maps to the model's own effort (`none` switches
  thinking off).
- **No voice.** Codex's voice features do not target this box's realtime endpoint.
- **Computer tools** need the `computer` scope; this recipe does not grant it.

## Switching harnesses

Codex keeps its sessions on your machine (`~/.codex/sessions`); nothing about them is on the box. Stop or interrupt
Codex's running turn (that closes its stream, which cancels the request on the box), then start the other harness.
Revoke the token (`pair.sh revoke NAME`) if this machine should stop using the box.
