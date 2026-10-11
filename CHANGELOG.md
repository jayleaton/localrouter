# Changelog

## Unreleased

- Qwen-Image 2.1 Turbo (`qwen_image_turbo`, installed as `qwen-image-2.1-turbo`): the official Turbo checkpoint on
  the same native Zig engine, with its own 8-step schedule (the checkpoint's `sample_sigmas`, shift 1, CFG 1). On GB10
  its sigmas, context, latents and pixels equal the twin's; the bf16 twin's cosine gate against diffusers fails at
  sigma 0.2 for Turbo and for Qwen-Image 2.1 alike (`docs/dev/RESULTS.md`). Its FP8 pack is calibrated and digest-checked like the base model's; the text encoder and
  VAE packs are shared (the same tensors). The installer adds it beside Qwen-Image 2.1 and makes it the default image
  model (`IMAGE_MODELS`, `IMAGE_DEFAULT`).
- Model choice per capability: the config's `defaults` names the model a request without `model` gets (checked at
  start); `/v1/models` and `list_models` show each model's `default_for`; `localrouter models` prints them. Unknown and
  wrong-kind models are refused with the models that can serve the request; engines can refuse requests they cannot
  serve (sizes, step counts) before loading (`Entry.check`).

## v0.1.0

First public release.

- One daemon with per-model worker processes, memory budgeting, priority eviction, idle unloading and warm models.
- MCP over HTTP and a stdio bridge; image and video HTTP endpoints, job polling and cancellation.
- Qwen-Image 2.1 text-to-image and MiniMax H3 text-to-video with audio engines on NVIDIA DGX Spark.
- Local conversion of publisher checkpoints, container service installation and a GPU-free test engine.
- Input URL validation, public-address checks, pinned connections and no redirects.
- Listens on `127.0.0.1` by default. `host` and `--host` take an IP, `localhost`, `all` or `tailscale` (the machine's
  tailnet IPv4, found from the network interfaces). The container runs `--host all` behind a published port that
  defaults to `127.0.0.1`; the installer's `LOCALROUTER_BIND` takes `localhost`, `all`, `tailscale` or an IP, asks once on
  a terminal, keeps the choice in `.env`, and serves a tailnet-only install from the host network so it starts at boot
  whichever of Docker and Tailscale comes up first.
- Qwen-Image 2.1 defaults to FP8 (closer to the original model): 1024x1024 in 16.7 s warm, about 22 s from cold on a DGX
  Spark. NVFP4 is the faster option (12.7 s warm, 16.1 s cold): `PRECISIONS="fp8s nvfp4"` installs it as
  `qwen-image-2.1-nvfp4`.

Image edit and image-to-video inputs are accepted by the API but refused by both model engines until their ports
land. The full two-model installer still needs an end-to-end hardware check. There is no authentication.
