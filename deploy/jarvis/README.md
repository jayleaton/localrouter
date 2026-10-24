# The Jarvis box on DGX-01

The whole of LocalRouter on one GB10 as a systemd user service, from a checkout:

| | |
| --- | --- |
| Chat | `qwen3.8-27b`: TensorFold 1.1.1, Qwen3.8-27B MLX 4-bit + DFlash2 (qualified), priority 100, kept warm |
| Speech to text | `whisper-large-v3-turbo`, priority 90, kept warm, 4 CPU threads (PR #19) |
| Text to speech | `fishaudio/drama-3` on the NVFP4 pruned weights (`DRAMA_VARIANT=nvfp4`, lossy, approved by JJ 2026-10-24; [measured against the full model](../../docs/dev/DRAMA3.md#the-nvfp4-pruned-weights)): 20.56 GiB (qualified), priority 95 (the last voice model to go), kept warm, `realtime` lane: the only speech model on the box (Kitten TTS stays in the repo but is not deployed: nothing lists or loads it) |
| Image | `qwen-image-2.1-turbo`, **kept warm** (loaded at start, no idle unload), priority 10: a picture skips the cold load. Text to image and edits (`edit_refs` 2: one or two input images, optionally a mask) once the edit pack is installed (`tools/spark/prepare.sh` adds it beside the packs) |
| Video | `minimax-h3`, on demand (idle 30 s), priority 0: text and image to video with audio, up to 1280x704 and 8 s (768x448 without a size). It evicts the image model to run, and the image model comes back after it |
| Decisions | `knowline-4b` (KnowLine-4B Gen5.5, bf16, [tools/decide/knowline](../../tools/decide/knowline/README.md)), **kept warm** since Drama 3's NVFP4 weights freed 18.7 GiB, priority 15 (above the image model's 10, so a video job evicts only the image model, as before), `spare_room_only` (coming back, it never unloads a kept-warm model: without spare room a decision is refused with a 503, and shadow decisions skip), `decide` lane. Shadow only: while it is loaded, each voice reply logs how it would have routed the turn (`voice shadow decision`); nothing acts on a decision |
| Voice | `/voice` and `/v1/realtime` with PR #19's settings: draft after 100 ms, sentence end after 300 ms, greedy transcripts, primed chat, thinking off |

Memory, measured on DGX-01 (2026-10-23), per process: GPU `used_memory` (nvidia-smi) plus host anonymous and shared
memory, sampled every 0.5 s at idle, through a long voice reply (chat and Drama 3 at once), transcription, 1024x1024
and 1664x1664 images and a 25k-token chat. Each server allocates its GPU memory at load, so the peaks are flat. The
bookings in `config.json.in` are each peak plus 256 MiB:

| Model | Measured peak | Booked |
| --- | --- | --- |
| Chat | 18.25 GiB GPU + 0.6 host + its prompt cache in host shared memory (76 KiB a token; capped at 3 GiB, one full 32k context) | 22.1 GiB |
| Drama 3, NVFP4 (2026-10-24, qualify `nvfp4/run1`: upstream's bench, 144 replies, beside chat, Whisper and the image model) | 19.69 GiB GPU (19.05 GiB weights and codec, the guidance sequence, two prefix-cache entries) + 0.62 host | 20.56 GiB |
| Drama 3, full (`DRAMA_VARIANT=full`) | 38.3 GiB GPU (the 36.9 GiB weights and the 1.7 GiB codec, nothing else) + 0.6 host | 39.25 GiB |
| Whisper | 2.0 GiB GPU + 1.5 host | 3.75 GiB, + 0.5 working |
| Image | 41.3 GiB with the edit pack at `edit_refs` 2 (2026-10-24: edits of one and two 1024x1024 inputs, 1344x768 and 1024x576; 38.2 GiB at `edit_refs` 1, 44.1 at 3); 31.2 GiB without it | 41.55 GiB |
| Video (on demand) | 52.06 GiB GPU + host at the caps 1280x704 and 8 s (2026-10-24: image to video at 1280x704, 8 s, the largest job); the buffers are sized at load for the caps, so a smaller clip takes as much | 52.3 GiB |
| Decisions | 10.6 GiB GPU + 1.5 host on the largest prompt (a 14 KB state read twice); 11.9 GiB while it loads | 12.4 GiB |

- **The reserve is 3 GiB.** earlyoom sends SIGTERM below 2% `MemAvailable` (2.4 GiB on this box); the rest covers the
  measured drift of what is not booked (under 0.6 GiB).
- **The image model's scratch** is sized by `max_pixels`, about 4.6 KiB a pixel: 1024x1024, 1360x768 and 768x1360 are
  served, a larger area is refused before it loads, and the PNG bytes are the same as with the larger scratch.
- **A 1024x1024 image** on the warm model evicts nothing; the lowest `MemAvailable` was 11.8 GiB.
- **The largest video** (image to video, 1280x704, 8 s: 445 s through the API) evicts only the image model; the lowest
  `MemAvailable` was 8.5 GiB, with chat, Drama 3, Whisper and KnowLine warm. The image model is back about 30 s later.
- **Chat, Drama 3 and Whisper** are never unloaded for an image or a video.

## Install, start, stop

From nothing, every model and then these scripts: `tools/install/bootstrap.sh --profile jarvis-gb10` ([INSTALL.md](../../INSTALL.md)).
Once the models are installed and qualified:

```sh
deploy/jarvis/install.sh   # preflight, build into ~/jarvis/deploy/src/<commit>, config, unit, tailscale serve
deploy/jarvis/start.sh     # start and wait until chat, Drama 3, Whisper and the image model are warm and answer at warm speed
deploy/jarvis/status.sh    # unit, build, each model's state and memory, URLs
deploy/jarvis/pair.sh      # paired devices: add NAME (a code for the app), token NAME, list, revoke ID|NAME
deploy/jarvis/stop.sh      # stop every model (and a Drama 3 server a killed worker left); --disable: off at boot too
journalctl --user -u localrouter-jarvis -f          # the daemon; models log to ~/jarvis/deploy/data/logs/
tools/jarvis/smoke.sh [--quick]                       # every capability against the live service
```

`install.sh` refuses unless each qualified model's marker says `"qualified": true` and was made from the pins in
this tree. When a model's `pins.json` changes, re-run that model's `qualify.sh` first. Re-running `install.sh` on a
new commit builds it and restarts the running service onto it, once no voice call is open (`voice_sessions` in
`/v1/tools`; it waits up to `IDLE_WAIT_S`, 30 min); on the same commit with the same config it changes nothing. The
unit restarts on failure, and lingering keeps it running without a login. `TOOLS` (tool ids) serves a subset;
`BUILD_ONLY=1` only builds.

Settings live in `env.sh`: the paths of the qualified installs, `PORT` (8190) and `HTTPS_PORT` (10443), and
`DRAMA_VARIANT`, Drama 3's weights: `nvfp4` (the default) or `full`. Each has its own install under
`qualify-drama3/root` (`drama-3-NVFP4-Pruned`, `build/tensorfold-nvfp4`, and `drama-3`, `build/tensorfold`), its own
marker and its own booking, so going back to the full model is `DRAMA_VARIANT=full deploy/jarvis/install.sh` (its
marker must still match `pins.json`, which the NVFP4 pins leave alone).

## Reaching it

The daemon listens on `127.0.0.1:8190` only. Browsers allow the microphone only in a secure context, so remote use
goes through HTTPS:

- **On the tailnet:** `https://dgx-01.prawn-penny.ts.net:10443`. The voice page is at `/voice`, the API at `/v1`
  and MCP at `/mcp`. `install.sh` sets this up with `tailscale serve` (tailnet only; DGX-01's other serve entries
  are untouched).
- **Without tailscale:** `ssh -N -L 8190:127.0.0.1:8190 dgx01@192.168.1.116`, then `http://127.0.0.1:8190/voice` on
  your machine. Loopback counts as a secure context.

Who may connect is decided by the tailnet (its ACLs) or by who can ssh in. Do not bind the daemon to `0.0.0.0` or a
LAN address: nothing would authenticate the requests.

## Pairing

`tailscale serve` hands every tailnet peer to the daemon as loopback, so by default any tailnet device can use every
route: voice (`/v1/realtime`), the MCP tools (`/mcp`), the `/v1` API and its output files. With device pairing on, every
route but `/health`, pairing and the `/voice` page answers only paired devices, each within its scopes (`inference`,
`media`, `decisions`, `read`, `manage`; [docs/MCP.md](../../docs/MCP.md#scopes)), and each device sees only its own
jobs and files. It is off by default.

```sh
deploy/jarvis/pair.sh add "JJ's phone"     # a code for the app (XXXX-XXXX, 10 minutes, once) and its localrouter:// link
deploy/jarvis/pair.sh token opencode --scopes inference,media,read   # a harness's token (printed once)
deploy/jarvis/pair.sh token smoke          # no --scopes: every service scope (scripts that check everything)
deploy/jarvis/pair.sh list                 # id, name, created, scopes
deploy/jarvis/pair.sh revoke ID|NAME       # refused from the next request
```

`pair.sh` runs the deployed binary on `$DEPLOY/config.json` and `$DEPLOY/data` (devices in `data/devices.json`, hashes
only); none of it restarts the service. Five wrong codes within 10 minutes discard the pending codes (logged).

To turn it on: pair what you use first, then `PAIRING_REQUIRED=1 deploy/jarvis/install.sh` (it restarts onto
`"pairing": {"required": true}` once no voice call is open). Later installs keep the deployed setting;
`PAIRING_REQUIRED=0` turns it off. Once it is on:

- **The web page** (`/voice`): paste a token (`pair.sh token browser`) into Settings → API key; the page sends it on
  the socket (`?api_key=`) and its API calls.
- **MCP clients**: `claude mcp add --transport http localrouter https://dgx-01.prawn-penny.ts.net:10443/mcp --header
  "Authorization: Bearer <token>"`, or `localrouter mcp-stdio --token <token>` (or `LOCALROUTER_TOKEN`).
- **Smoke and check scripts**: `tools/jarvis/smoke.sh`, `tools/jarvis/mcp_check.py`, `tools/spark/jarvis/voice_e2e.py`
  and `tools/spark/jarvis/model_manager_check.py` send `$LOCALROUTER_TOKEN` when it is set:
  `LOCALROUTER_TOKEN=$(deploy/jarvis/pair.sh token smoke) tools/jarvis/smoke.sh`. `tools/spark/jarvis/client_check.py`
  drives the web page in Chrome and does not set the page's API key: run it with pairing off.
- **These scripts** (`status.sh`, `start.sh`'s warm wait, `install.sh`'s idle check) read `/v1/tools` with
  `$DEPLOY/operator.token`, a `read`-only token install.sh makes when pairing is on (delete it to make a new one), or
  `$LOCALROUTER_TOKEN` when set.
