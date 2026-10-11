---
name: localrouter
description: Generate images (and video with audio) with local models on your own GPU box through LocalRouter's MCP server, instead of a paid image or video API. Use when a task needs a new image, illustration, thumbnail, concept art, or a short video clip.
---

# LocalRouter: images and video from local models on your own GPU box

The GPU box (a DGX Spark, first of all) runs `localrouter` (installed with `tools/spark/install.sh`). It is an MCP
server at `http://<host>:8190/mcp` and an OpenAI-compatible API at `http://<host>:8190/v1`, reachable from the
owner's network. No API key. Models load on their first call and unload by themselves when idle.

## Connect (once per machine)

```bash
localrouter mcp-stdio --url http://<host>:8190
```

Other clients: `{"mcpServers": {"localrouter": {"type": "http", "url": "http://<host>:8190/mcp"}}}`, or for
stdio-only clients `npx -y mcp-remote http://<host>:8190/mcp --allow-http`.

## Use

1. `list_models`: the ids, their kind (`image` / `video`), what each can do (`capabilities`), which one a request
   without `model` gets (`default_for`) and whether each is loaded (`running`). Name `model` to pick one: where it is
   installed, `qwen-image-2.1-turbo` (8 steps, the faster) is usually the default and `qwen-image-2.1` (25 steps) the
   base model. Leave `steps` out: each model uses its own (Turbo takes only 8).
2. `generate_image` with `prompt` (and `size`, `seed`, `n`, `model`): returns the PNG inline plus its URL, and the
   seed. About 1 megapixel is best: `1024x1024`, `1360x768`, `768x1360`. A warm 1024x1024 image takes about 17 s
   (FP8, the default; the NVFP4 model, `qwen-image-2.1-nvfp4` where installed, takes about 13 s); the first call also
   loads the model (about 5 s).
3. If `list_models` shows a video model: `generate_video` with `prompt` (and `size`, `seconds`, `seed`): waits up to `wait_s` (600) and returns the MP4's
   URL, or a job id; then `get_job` with `wait_s` until `status` is `completed`.
4. `release_models` when you are done with a long batch and the machine is shared; otherwise let them idle out.

Without MCP, the same over HTTP: `POST /v1/images/generations` (OpenAI Images API, extensions `seed`, `steps`) and
`POST /v1/videos` (OpenAI Videos API); see the repo's `docs/AGENTS.md`.

## Be considerate

- One request at a time; the machine runs one job at once and queues the rest.
- Batch work for one model; switching models can force a reload.
- A tool error says why (bad size, no memory free right now, model failed to load): fix the request or report it;
  don't loop on it.
- Report the model, size, seed and prompt you used, so the result can be reproduced. Save files under the project
  (e.g. `assets/`) named after prompt and seed.
