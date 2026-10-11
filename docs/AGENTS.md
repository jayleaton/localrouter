# Using LocalRouter from an agent

LocalRouter is an HTTP server (default port 8190), OpenAI-compatible where OpenAI has an API for the job. Tools load
on demand, so the first request to a tool waits for it to load. Later requests are fast until the tool has been idle
for its TTL.

Base URL: `http://<host>:8190/v1` (any API key, or none).

By default the daemon listens on `127.0.0.1` only, so `<host>` is `127.0.0.1` on the machine itself. To reach it from
other machines the operator starts it with `--host tailscale` (the tailnet address; `<host>` is then the machine's
tailnet name or IP), or `--host all` / an IP (`host` in the config takes the same values).

## MCP (the easy way)

The same server speaks the Model Context Protocol at `http://<host>:8190/mcp` (Streamable HTTP, stateless):

```bash
claude mcp add --transport http localrouter http://<host>:8190/mcp
```

| Tool | Does |
| --- | --- |
| `list_models` | the models: kind, capabilities, `default_for`, priority, `keep_loaded`, and whether each is loaded |
| `generate_image` | `prompt`, `size`, `n`, `seed`, `steps`, `model`: waits, returns the PNG(s) inline and as URLs, and the seed. `images` (up to 5 base64 strings or http(s) URLs) makes it an edit of those images |
| `generate_video` | `prompt`, `size`, `seconds`, `seed`, `audio`, `wait_s`: the MP4's URL when done within `wait_s`, else a job id. `image` (one base64 string or URL) makes it image to video |
| `get_job` | `id`, `wait_s`: status, progress, output URLs |
| `cancel_job` | `id` |
| `release_models` | unloads every model now |

Clients that only speak MCP over stdio use the bridge: `localrouter mcp-stdio --url http://<host>:8190` (below).

Refusals (bad size, no memory free right now, a failed load) come back as tool errors with the reason. The skill for
agents is `skill/localrouter/SKILL.md`. The HTTP API below is the same jobs without MCP.

## Find the tools

```bash
curl -s $BASE/models | jq '.data[] | {id, kind, running}'
```

`kind` is `image` or `video`. `running: true` means it is loaded now. `capabilities` lists what the model can do:
`text_to_image`, `image_edit`, `text_to_video`, `image_to_video`. `default_for` lists the capabilities the model serves
when a request names no model (the operator's `defaults` in the config, else the first model that has the capability).
Name a model with `model` to choose one yourself, per request. A request that needs a capability the model lacks is
refused with a 400 naming the model and the capability; a model of the other kind is a 400 and an unknown one a 404,
both listing the models that can do it. A model's own limits (a larger size than it was set up for, a step count its
schedule lacks) are refused before it loads. `localrouter models` prints the same table on the command line.

| Image model | Steps | Notes |
| --- | --- | --- |
| `qwen-image-2.1` | 25 (any 1 to 200) | the base model; FP8 (`-nvfp4`: the faster NVFP4) |
| `qwen-image-2.1-turbo` | 8 (only 8) | the Turbo checkpoint and its own schedule: about 3x fewer steps. Omit `steps` |

Both run without classifier-free guidance (CFG 1, the models' default): `guidance` and `negative_prompt` are not used.

## Images (OpenAI Images API)

```bash
curl -s -m 900 $BASE/images/generations -H 'content-type: application/json' -d '{
  "model": "qwen-image-2.1", "prompt": "a red fox in fresh snow", "size": "1024x1024", "n": 1, "seed": 42
}' | jq -r '.data[0].b64_json' | base64 -d > fox.png
```

| Field | Notes |
| --- | --- |
| `size` | `WIDTHxHEIGHT`, 256 to 2048 a side, multiples of 16 |
| `n` | 1 to 4 |
| `seed` | Extension. Omit it for a random one; the response's `seed` reproduces the image |
| `steps`, `guidance`, `negative_prompt` | Extensions; 0 or absent means the tool's default |
| `response_format` | `b64_json` (default) or `url` (served from `/v1/files/...`, kept 24 h) |

The request is synchronous: it returns when the image is done. Use a client timeout of at least 15 minutes, so a
cold load fits.

## Image edits (OpenAI Images API)

```bash
curl -s -m 900 $BASE/images/edits -F model=<edit-model> -F prompt='make it night' -F 'image[]=@fox.png' \
  | jq -r '.data[0].b64_json' | base64 -d > night.png
```

`POST /v1/images/edits` takes `image` or `image[]` (1 to 5 files) and the fields of a generation (`prompt`, `size`,
`n`, `seed`, `response_format`...). A JSON body works too: `"image": "<base64>"` or `"images": ["<base64>", "data:image/png;base64,...", "https://..."]`.
Inputs must be PNG, JPEG or WebP, at most 32 MB each. URLs are fetched by the server (http and https only, 32 MB,
30 s; see [Input URLs](#input-urls)). The model must list `image_edit` in its capabilities.

## Videos (OpenAI Videos API)

```bash
id=$(curl -s $BASE/videos -H 'content-type: application/json' \
  -d '{"model": "testpattern-video", "prompt": "waves at dusk", "size": "1344x768", "seconds": "5"}' | jq -r .id)
curl -s $BASE/videos/$id            # status: queued, in_progress, completed, failed or cancelled; progress 0-100
curl -s -o out.mp4 $BASE/videos/$id/content   # once completed
curl -s -X DELETE $BASE/videos/$id  # cancel
```

Extensions: `seed`, `steps`, `guidance`, `fps`, `audio` (true by default). Poll every few seconds.

Image to video: add `input_reference`, a file in a multipart body (`-F input_reference=@frame.png`, with the other
fields as form fields) or a base64 string, data URL or URL in a JSON body. The model must list `image_to_video`.

## Input URLs

A URL in `image`, `images` or `input_reference` (HTTP) or `images` / `image` (MCP) is fetched by the daemon, so by default
it may only name public hosts. The host is resolved by the daemon, and the request is refused with a 400 (a tool error
over MCP) if any address it resolves to is loopback, in RFC 1918 space (10/8, 172.16/12, 192.168/16), link-local
(169.254/16, which includes cloud metadata at 169.254.169.254; fe80::/10), carrier-grade NAT (100.64/10, where tailnet
addresses live), unique-local IPv6 (fc00::/7), multicast, reserved, or unspecified (`0.0.0.0`, `::`). IPv4 carried inside
IPv6 (`::ffff:a.b.c.d`, NAT64, 6to4) is judged as the IPv4 address. The connection goes to the address that was checked,
not to a second lookup, so DNS cannot swap it afterwards.

Redirects are not followed: a 3xx answer is an error, so a public URL cannot bounce the daemon to a private one.

Two exceptions. The daemon's own output URLs (`/v1/files/...`, as returned by `response_format: url` and the MCP tools)
work, because the URL's host and port are the address the request itself came to; the daemon fetches them from itself.
And the operator can set `"allow_private_urls": true` in the config to lift the check (a lab where inputs live on a
private image server); redirects stay off.

## Command line

```bash
localrouter gen image "a red fox in fresh snow" -o fox.png --size 1360x768 --seed 42
localrouter gen video "waves at dusk" -o waves.mp4 --seconds 5
localrouter gen image "a red fox in fresh snow" -o fox.png --model qwen-image-2.1   # a model of your choice
localrouter models                                                                  # ids, capabilities, defaults
```

The server address comes from `$LOCALROUTER_URL` (default `http://127.0.0.1:8190`) or `--url`.

## MCP over stdio

`localrouter mcp-stdio [--url http://host:8190]` reads JSON-RPC lines on stdin, POSTs each to the daemon's `/mcp`, and writes
the response as one line on stdout (notifications produce none; if the daemon cannot be reached, the request gets a
JSON-RPC error). It keeps no state. In a client's config: `{"command": "localrouter", "args": ["mcp-stdio", "--url", "http://<host>:8190"]}`.
The URL defaults to `$LOCALROUTER_URL`, else `http://127.0.0.1:8190`.

## Priority and keep_loaded (operators)

Per tool in the config: `priority` (integer, default 0; higher is kept longer) and `keep_loaded` (default false).
When a request needs memory, idle tools are unloaded lowest priority first, then least recently used, until it fits;
a tool that is loading or running is never unloaded. Priority only orders the victims: a request is never refused while
unloading idle tools, of any priority, would make room. A `keep_loaded` tool is loaded when the daemon starts if it
fits (best effort, logged), is not unloaded by its idle TTL, and is loaded again once memory frees up (the scheduler
idle for a couple of seconds and the tool fits without unloading anything). It can still be evicted by a request that
needs the room, and `POST /v1/tools/{id}/unload` (or `release`) keeps it down until it is next used or loaded.
`GET /v1/tools` shows `priority`, `keep_loaded` and `capabilities`.

## Errors

Errors use OpenAI's shape: `{"error": {"message", "type"}}`.

| Status | Meaning | Do |
| --- | --- | --- |
| 400 | Invalid request (size, n, prompt, input images, a capability the model lacks, a model of the other kind, a size or step count the model does not take) | Fix the request |
| 404 | Unknown model (the message lists the models that can do it) or route | Re-check `GET /models` |
| 503 `insufficient_memory` | The machine has no room for this tool right now (another service may have priority) | Retry later; don't loop |
| 503 `server_error` | The tool failed to load or its worker died; the message has the log tail | Report it |
| 504 | A deadline passed | Retry once, then report |

## Being considerate

- One job runs at a time. Send requests one after another, not in parallel bursts.
- Batch work for one tool together. Alternating tools can force reloads when memory is tight.
- `POST /v1/tools/release` unloads everything, and `GET /v1/tools` shows what is loaded. These are for operators;
  agents should not need them.
