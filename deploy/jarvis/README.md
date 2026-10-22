# The Jarvis box on DGX-01

The whole of LocalRouter on one GB10 as a systemd user service, from a checkout:

| | |
| --- | --- |
| Chat | `qwen3.8-27b`: TensorFold 1.1.1, Qwen3.8-27B MLX 4-bit + DFlash2 (qualified), priority 100, kept warm |
| Speech to text | `whisper-large-v3-turbo`, priority 90, kept warm, 4 CPU threads (PR #19) |
| Text to speech | `fishaudio/drama-3`: 42 GiB (qualified), priority 95 (the last voice model to go), kept warm, `realtime` lane: the only speech model on the box (Kitten TTS stays in the repo but is not deployed: nothing lists or loads it) |
| Image | `qwen-image-2.1-turbo`, **kept warm** (loaded at start, no idle unload), priority 10: a picture skips the cold load |
| Video | `minimax-h3`, on demand (idle 30 s), priority 0: it evicts the image model to run, and the image model comes back after it |
| Voice | `/voice` and `/v1/realtime` with PR #19's settings: draft after 100 ms, sentence end after 300 ms, greedy transcripts, primed chat, thinking off |

Memory, measured on DGX-01 (2026-10-22):
- **With the warm set at full size** (chat 24 GiB with its 4 GiB prompt cache, Drama 3 43, Whisper 3.6, image 35.2):
  about 9 GiB `MemAvailable` of about 115, and 13.7 GiB just after a start.
- **The reserve is 6 GiB.** That leaves room for Whisper's 2 GiB working memory beside the full warm set; with 8 GiB,
  each voice turn evicted the image model.
- **A 1024x1024 image** on the warm model evicts nothing; the lowest `MemAvailable` was 11.8 GiB.
- **A 1 s video** (40 GiB) evicts only the image model; the lowest was 11.7 GiB. The image model is back about 30 s
  later.
- **Chat, Drama 3 and Whisper** are never unloaded for an image or a video.

## Install, start, stop

```sh
deploy/jarvis/install.sh   # preflight, build into ~/jarvis/deploy/src/<commit>, config, unit, tailscale serve
deploy/jarvis/start.sh     # start and wait until chat, Drama 3, Whisper and the image model are warm
deploy/jarvis/status.sh    # unit, build, each model's state and memory, URLs
deploy/jarvis/stop.sh      # stop every model (and a Drama 3 server a killed worker left); --disable: off at boot too
journalctl --user -u localrouter-jarvis -f          # the daemon; models log to ~/jarvis/deploy/data/logs/
tools/jarvis/smoke.sh [--quick]                       # every capability against the live service
```

`install.sh` refuses unless each qualified model's marker says `"qualified": true` and was made from the pins in
this tree. When a model's `pins.json` changes, re-run that model's `qualify.sh` first. Re-running `install.sh` on a
new commit builds it and restarts the running service onto it, once no voice call is open (`voice_sessions` in
`/v1/tools`; it waits up to `IDLE_WAIT_S`, 30 min). The unit restarts on failure, and lingering keeps it
running without a login.

Settings live in `env.sh`: the paths of the qualified installs, `PORT` (8190) and `HTTPS_PORT` (10443).

## Reaching it

The daemon listens on `127.0.0.1:8190` only. Browsers allow the microphone only in a secure context, so remote use
goes through HTTPS:

- **On the tailnet:** `https://dgx-01.prawn-penny.ts.net:10443`. The voice page is at `/voice`, the API at `/v1`
  and MCP at `/mcp`. `install.sh` sets this up with `tailscale serve` (tailnet only; DGX-01's other serve entries
  are untouched).
- **Without tailscale:** `ssh -N -L 8190:127.0.0.1:8190 dgx01@192.168.1.116`, then `http://127.0.0.1:8190/voice` on
  your machine. Loopback counts as a secure context.

LocalRouter has no API keys. Who may connect is decided by the tailnet (its ACLs) or by who can ssh in. Do not bind
the daemon to `0.0.0.0` or a LAN address: nothing would authenticate the requests.
