# Changelog

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
