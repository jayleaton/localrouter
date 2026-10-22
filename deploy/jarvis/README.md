# The Jarvis box on DGX-01

The whole of LocalRouter on one GB10 as a systemd user service, from a checkout:

| | |
| --- | --- |
| Chat | `qwen3.8-27b`: TensorFold 1.1.1, Qwen3.8-27B MLX 4-bit + DFlash2 (qualified), priority 100, kept warm |
| Speech to text | `whisper-large-v3-turbo`, priority 90, kept warm, 4 CPU threads (PR #19) |
| Text to speech | `fishaudio/drama-3`: 42 GiB (qualified), priority 95 (the last voice model to go), kept warm, `realtime` lane; `kitten-tts-2-graphs`: the fallback (qualified, CUDA-graph decode), priority 50, kept warm, on the `main` lane: its reload (about 30 s with the graph capture) would otherwise hold the realtime lane and Drama 3's speech behind it |
| Image, video | `qwen-image-2.1-turbo` (idle 60 s), `minimax-h3` (idle 30 s), on demand |
| Voice | `/voice` and `/v1/realtime` with PR #19's settings: draft after 100 ms, sentence end after 300 ms, greedy transcripts, primed chat, thinking off |

Measured with everything warm: 42.4 GiB `MemAvailable` of about 115. A 1024x1024 image evicts only Kitten (the image needs 36 GiB plus the 8 GiB reserve; the lowest `MemAvailable` during it was 14 GiB); Drama 3 and Whisper stay, and the first voice turn after it reached first audio in 951 ms. The image job takes the fallback voice's memory
(Kitten), and a video job takes part of the voice set. The scheduler brings the kept models back as soon as it is
idle, within about 30 s of a clip.

## Install, start, stop

```sh
deploy/jarvis/install.sh   # preflight, build into ~/jarvis/deploy/src/<commit>, config, unit, tailscale serve
deploy/jarvis/start.sh     # start and wait until chat, Drama 3, Whisper and Kitten are warm (about 40 s)
deploy/jarvis/status.sh    # unit, build, each model's state and memory, URLs
deploy/jarvis/stop.sh      # stop every model (and a Drama 3 server a killed worker left); --disable: off at boot too
journalctl --user -u localrouter-jarvis -f          # the daemon; models log to ~/jarvis/deploy/data/logs/
tools/jarvis/smoke.sh [--quick]                       # every capability against the live service
```

`install.sh` refuses unless each qualified model's marker says `"qualified": true` and was made from the pins in
this tree. When a model's `pins.json` changes, re-run that model's `qualify.sh` first. Re-running `install.sh` on a
new commit builds it and restarts the running service onto it. The unit restarts on failure, and lingering keeps it
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
