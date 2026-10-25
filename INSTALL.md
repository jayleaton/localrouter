# Installing LocalRouter Jarvis on a box (for coding agents)

You are a coding agent asked to "set up LocalRouter Jarvis on this box". Follow these steps in order. Each step says
what to run, what success looks like and what to do otherwise. Do not add steps of your own, do not skip a failed
checksum, and stop and ask the user wherever a step says so.

The end state is one systemd user service, `localrouter-jarvis`, on `http://127.0.0.1:8190`. It serves:

- chat: Qwen3.8 27B;
- speech: Fish Audio Drama 3;
- transcription: Whisper large-v3-turbo;
- images: Qwen-Image 2.1 Turbo;
- video: MiniMax H3, with a voice page at `/voice`.

The service starts at boot.

Everything goes under one prefix, `~/jarvis` by default. One script does the install:
[`tools/install/bootstrap.sh`](tools/install/bootstrap.sh). It reads the profile's manifest,
[`deploy/profiles/jarvis-gb10.json`](deploy/profiles/jarvis-gb10.json), which lists every step, size, pin and command.
A second script, [`tools/install/doctor.sh`](tools/install/doctor.sh), says what is missing and how to fix it.

## Rules

- Run commands as the user who will own the service, never as root. Run `sudo` only where a fix below names it, and
  only after the user agrees.
- Never bind the service to `0.0.0.0` or a LAN address. LocalRouter has no API keys. It listens on loopback only,
  and `tailscale serve` (HTTPS on the tailnet) or an ssh tunnel reaches it.
- The models need the whole GPU while they are qualified. If something else uses the GPU (another LocalRouter, a
  training job), do not stop it yourself: ask the user (step 6).

## 1. Get the code

```sh
git clone https://github.com/jayleaton/localrouter ~/localrouter && cd ~/localrouter
```

Use `main`, or the tag the user names. If `INSTALL.md` and `tools/install/` are not in your checkout, it is not a
version this guide is for. Keep the checkout clean: the build refuses uncommitted changes. Run every command below from
this directory.

The commands below use the defaults: prefix `~/jarvis`, port 8190, tailnet HTTPS port 10443 and unit
`localrouter-jarvis`. If the user wants others, add the same `--prefix DIR --port N --https-port N --unit NAME` to
**every** `bootstrap.sh` and `doctor.sh` command ([Options](#options)).

## 2. Look at the box

```sh
tools/install/doctor.sh --profile jarvis-gb10
```

The first line is the box: arch, OS, GPU, driver, CUDA, memory (and "unified" when the GPU shares the system's memory,
as on GB10), and free memory. The `MISSING` lines come with a `fix:` line. Before anything is installed, every
install step is `MISSING`: that is expected. Read the `== box` and `== prerequisites` sections only.

The same facts by hand, if the doctor cannot run:

| Fact | Command | jarvis-gb10 needs |
| --- | --- | --- |
| arch | `uname -m` | `aarch64` |
| GPU | `nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv` | `NVIDIA GB10`, driver 580 or newer (memory `[N/A]`: unified) |
| memory | `grep MemTotal /proc/meminfo` | at least 110 GiB |
| CUDA | `/usr/local/cuda/bin/nvcc --version` | release 13.x |
| OS | `. /etc/os-release; echo $ID $VERSION_ID` | tested on `ubuntu 24.04` (DGX OS) |
| disk | `df -h ~` | about 290 GiB free (step 4 gives the exact figure) |

## 3. Pick the profile

| The box | Profile | Status |
| --- | --- | --- |
| `aarch64`, NVIDIA GB10 (DGX Spark and its OEM versions), at least 110 GiB | `jarvis-gb10` | tested: the full stack, end to end |
| anything else | none | untested; no profile yet |

On any other box, stop. Tell the user what step 2 found, and that no profile has been tested for it. The chat and
Drama 3 engines here are aarch64 GB10 builds, and their qualification refuses other GPUs. Do not `--force` unless
the user asks for an experiment.

## 4. Read the plan

```sh
tools/install/bootstrap.sh --profile jarvis-gb10 --dry-run
```

It changes nothing. It prints:

- every step, `done` or `todo`, with `[GPU, N GiB free]` on the steps that need the GPU;
- what the remaining steps need: download, disk (with a 30 GiB margin) and the memory of the warm models;
- the prerequisites, `ok` or `MISSING` with a `fix:`;
- **needs acceptance**: the model licences, and Drama 3's numerics, that the user must accept.

If the disk line says there is not enough, stop and tell the user how much is missing. To install fewer models, see
[Choosing models](#choosing-models).

## 5. Ask the user, then fix the prerequisites

1. Show the user the **needs acceptance** list exactly as printed, with its links, and ask whether they accept it.
   Yes: add `--accept` to the commands below. No: leave those models out (`--models`, below).
2. For each `MISSING` prerequisite, show the user its `fix:` line. A fix with `sudo` needs their agreement. Run
   it, then run step 4 again, until no prerequisite is `MISSING`.

## 6. Make sure the GPU is free

```sh
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
grep MemAvailable /proc/meminfo
```

The qualification steps need 64 GiB free (88 GiB for Drama 3), and the warm models take about 107 GiB. If the GPU is
busy, ask the user. Either:

- they free it, and you go on to step 7; or
- you run step 7 with `--no-gpu` now: everything that needs no GPU (downloads and builds, nearly all of the time).
  Then run step 7 again without `--no-gpu` once the GPU is free.

## 7. Install

```sh
tools/install/bootstrap.sh --profile jarvis-gb10 --accept
```

It takes about 1.5 to 3 hours, mostly downloads (about 200 GiB) and the GPU steps. Run it so that it outlives your
session, and read its log as it goes. Over ssh, `systemd-run --user` keeps it going after the connection drops:

```sh
systemd-run --user --unit jarvis-bootstrap bash -lc "cd $PWD && tools/install/bootstrap.sh --profile jarvis-gb10 --accept > ~/jarvis-bootstrap.log 2>&1; echo EXIT=\$? >> ~/jarvis-bootstrap.log"
tail -f ~/jarvis-bootstrap.log    # until a line EXIT=0 (done) or EXIT=1 (see the FAILED lines above it)
```

In a terminal of your own, `tmux` works as well. Each step prints `run`, then `ok` or `FAILED`. A
finished step prints `ok ... (done)` and is skipped.

On `FAILED`, the script prints:

- the end of that step's log (`~/jarvis/install/logs/<model>-<step>.log`);
- a `next:` line saying what to do;
- the exact command to run again. It resumes at the failed step.

Do what `next:` says, then run the same command again. If the same step fails twice in the same way, stop and give
the user the log.

`STOP ... needs the user's acceptance` means step 5 was skipped. Go back to it.

If it exits with an error but no `FAILED` line (for example `bootstrap.sh: the profile manifest ... is not valid`),
the checkout itself is broken: stop and give the user the last lines it printed.

## 8. Check it

```sh
tools/install/doctor.sh --profile jarvis-gb10
LOCK=/dev/null tools/jarvis/smoke.sh --quick
```

The doctor must end with `== nothing missing`. The smoke must end with `== quick: N passed, 0 failed`. It takes
about a minute and exercises chat, transcription, speech (every voice), the warm set and three voice turns.

## 9. Tell the user

Tell the user:

- **The URL.** It is `http://127.0.0.1:8190`, with the voice page at `/voice`, the API at `/v1` and MCP at `/mcp`. On
  a tailnet it is `https://<this machine>.<tailnet>.ts.net:10443`, if `tailscale serve` worked. `deploy/jarvis/status.sh`
  prints both.
- **How to reach it from another machine.** Use the tailnet URL, or `ssh -N -L 8190:127.0.0.1:8190 <user>@<this box>`.
- **How to stop and start it.** `deploy/jarvis/stop.sh` and `deploy/jarvis/start.sh`; `deploy/jarvis/status.sh`
  shows it at a glance.
- **For their agent.** `claude mcp add --transport http localrouter http://127.0.0.1:8190/mcp`, as in
  [docs/AGENTS.md](docs/AGENTS.md).
- **For their phone.** The phone app ([apps/mobile](apps/mobile/README.md)) reaches the box over the tailnet URL; the
  phone needs Tailscale on the same tailnet. Pair it from the box with
  `deploy/jarvis/pair.sh add "<their phone>" --url https://<this machine>.<tailnet>.ts.net:10443` and give them the
  code or the `localrouter://pair?…` link it prints (valid 10 minutes). The README says how to install the app.

## Pairing devices

Pairing is off by default: anyone who can reach the box can use its API, voice and MCP, as before. With
`"pairing": {"required": true}` in the config (on a Jarvis box: `PAIRING_REQUIRED=1 deploy/jarvis/install.sh`, which
restarts onto it once no voice session is open; later installs keep it, `PAIRING_REQUIRED=0` turns it off), every
route but `/health` and pairing answers only paired devices, each within its scopes ([docs/MCP.md](docs/MCP.md#scopes)),
and each device sees only its own jobs and files. Devices paired before scopes keep everything they had. The deploy
scripts use their own read-only token (`$DEPLOY/operator.token`, made by install.sh). Before turning it on:

- pair every phone (`pair.sh add NAME --url URL`, the app enters the code);
- mint a token for each harness, MCP client and script with what it needs (`pair.sh token NAME --scopes
  inference,media,read`, printed once; `manage` only for what loads and unloads models) and give it to them:
  `claude mcp add --transport http localrouter <url>/mcp --header "Authorization: Bearer <token>"`,
  `localrouter mcp-stdio --token <token>`, and `LOCALROUTER_TOKEN=<token>` for the smoke and check scripts (a token
  with no `--scopes`, which can do everything they check);
- in the web page (`/voice`), paste a token into **Settings → API key**.

`pair.sh list` shows the devices and `pair.sh revoke <id or name>` removes one (its next connection is refused).
[docs/MCP.md](docs/MCP.md) has the details.

## The models

Each model is one key for `--models`. Its manifest entry has the same facts, machine-readable: `download_bytes`,
`disk_bytes`, `memory_bytes` and its `steps`, each with `run`, `done` and `fix`. bootstrap.sh runs each step's `run`
from the checkout with `J` set to the prefix. Running one by hand works the same way. The `verify` line is the
step's `done` test.

### chat: `qwen3.8-27b`

- **What it is.** Qwen3.8 27B, served by TensorFold 1.1.4's native server: MLX 4-bit weights with the DFlash2
  drafter. Licence Apache-2.0. Kept warm.
- **Size.** About 20 GB to download and 21 GB on disk. 24 GiB in memory.
- **Steps.** `tools/chat/qwen38-27b/fetch.sh $J/qualify-chat/root`, then `setup.sh`, then `qualify.sh`. Every file
  is checked against `tools/chat/qwen38-27b/pins.json`. qualify runs the model on the GPU for about 5 minutes and
  writes `$J/qualify-chat/qwen38-27b-qualification.json`. Without that marker the service refuses the model.
- **Verify.** `jq .qualified $J/qualify-chat/qwen38-27b-qualification.json` is `true`.
- **Skip it.** Leave `chat` out. The voice loop then has nothing to answer with.

### tts: `fishaudio/drama-3`

- **What it is.** Fish Audio Drama 3 on its TensorFold native engine, with the NVFP4 pruned weights
  ([plotarmordev/drama-3-NVFP4-Pruned](https://huggingface.co/plotarmordev/drama-3-NVFP4-Pruned)): six fixed voices
  (Mae, Arlo, Ezra, Nora, Theo, Wren), streamed 44.1 kHz audio. Kept warm.
- **Licence.** FACL v1.0, which grants no licence for a hosted service, so keep the service on loopback or your
  tailnet. The quant is an unofficial community quant under the same licence. Its acceptance covers the licence, the
  engine's numerics (not bit-exact with the SGLang-Omni reference) and lossy weights: routed experts in NVFP4, 7,369 of
  10,240 kept ([measured against the full model](docs/dev/DRAMA3.md#the-nvfp4-pruned-weights)).
- **Size.** About 22.6 GB to download and 25 GB on disk. 20.6 GiB in memory.
- **Steps.**
  - `tools/speech/drama3/fetch.sh --variant nvfp4`: every file against `pins.json` and `pins-nvfp4.json`.
  - `build.sh --variant nvfp4`: the engine for these weights, about 15 minutes.
  - A qualify config made from the deployed tool.
  - `qualify.sh --accept-native-numerics --accept-lossy`: about 20 minutes on the GPU. It needs 32 GiB free and writes
    `$J/qualify-drama3/drama3-nvfp4-qualification.json`.
- **Verify.** `jq .qualified $J/qualify-drama3/drama3-nvfp4-qualification.json` is `true`.
- **The full model instead.** `DRAMA_VARIANT=full` (`deploy/jarvis/env.sh`) serves fishaudio/drama-3 (41.5 GB, 39.25
  GiB in memory), fetched, built and qualified without `--variant` (`reference.sh` and 88 GiB free for its accuracy
  checks); both installs live side by side under `$J/qualify-drama3/root`.
- **Skip it.** Leave `tts` out. The box then has no voice.

### stt: `whisper-large-v3-turbo`

- **What it is.** Whisper large-v3-turbo in PyTorch (fp16 on the GPU) with the Silero VAD. Licence MIT. Kept warm.
- **Size.** About 7 GB to download (1.6 GB of weights, the rest the torch wheels) and 15 GB on disk. 4 GiB in
  memory, plus 2 GiB while transcribing.
- **Steps.**
  - `uv sync --locked` builds `$J/venvs/whisper` from `tools/speech/whisper/uv.lock`, with every wheel checked by its
    hash.
  - `tools/speech/whisper/fetch_model.py` fetches the checkpoint, each file by sha256.
  - The VAD file is fetched at a pinned commit and checked by sha256.
- **Verify.** `ls $J/models/whisper/whisper-large-v3-turbo/` shows one revision directory.
- **Skip it.** Leave `stt` out. The voice loop then cannot hear.

### image: `qwen-image-2.1-turbo`

- **What it is.** Qwen-Image 2.1 Turbo, FP8 with 8 steps, on LocalRouter's own engine. Its licence is the Qwen Research License: non-commercial research or
  evaluation only. Kept warm.
- **Size.** About 67 GB to download: 47 GB of checkpoints and the 20 GB NVIDIA PyTorch container, pulled once. About
  95 GB on disk. 35 GiB in memory.
- **Steps.** Both run `tools/spark/install.sh` with `WEIGHTS_ONLY=1` inside the container.
  - `download`: `DOWNLOAD_ONLY=1`. Every checkpoint file is checked against `tools/twin/packs/sources/` before
    anything is built from it.
  - `packs`: on the GPU, about 30 minutes. It makes the packs, checks them against `tools/twin/packs/built/`, and
    compiles the Triton kernels for this GPU.
- **Verify.** `$J/lr/weights/qwen-image-2.1/turbo-fp8s/triton/triton.json` exists.
- **Skip it.** Leave `image` out. Picture requests then fail with "no image model".

### video: `minimax-h3`

- **What it is.** MiniMax H3 text to video with audio: the NVFP4 Turbo pack, 8 steps. Its licence is the MiniMax H3 Community License, which excludes the
  US, EU, UK and South Korea and sets downstream use conditions. It loads on demand and unloads the image model while it runs.
- **Size.** About 66 GB to download and 90 GB on disk. 39 GiB in memory while it runs.
- **Steps.**
  - `download`: the checkpoints, each checked against `tools/twin/packs/sources/minimax-h3.json`, and the video venv.
  - `packs`: the Turbo-LoRA pack, calibrated on the GPU in about 15 minutes and checked against
    `tools/twin/packs/built/minimax-h3.json`.
- **Verify.** `$J/lr/weights/minimax-h3/h3-turbo-nvfp4/manifest.json` exists.
- **Skip it.** Leave `video` out. Video requests then fail with "no video model".

## Choosing models

`--models chat,tts,stt,image,video` is the default. A shorter list installs and serves only those models, and the
service's config lists only them.

- The voice loop needs `chat,tts,stt`.
- `image` and `video` are independent of it and of each other.

## Options

| Option | Default | What it does |
| --- | --- | --- |
| `--prefix DIR` | `~/jarvis` | where everything goes: tools, weights, venvs, the deploy, state and logs |
| `--port N` | `8190` | the service's loopback port. The Drama 3 server uses N + 10603 |
| `--https-port N` | `10443` | `tailscale serve` on the tailnet. `0` sets up none |
| `--unit NAME` | `localrouter-jarvis` | the systemd user unit |
| `--models-dir DIR` | `~/jarvis` when `--prefix` is elsewhere | an existing install whose model files are reused instead of downloaded again. `none`: no reuse |
| `--no-gpu` | | everything that needs no GPU, then stop |
| `--dry-run` | | the plan only |

A second install on the same box needs its own `--prefix`, `--port`, `--https-port 0` (or another port) and `--unit`.
It also needs the memory for a second stack, which one GB10 does not have.

## Models already on the box

Never download a model the box already has. When an install exists (`~/jarvis`, or the one `--models-dir` names),
bootstrap.sh reuses its model files in a new prefix:

- **How.** It hard-links the files, so they take no second copy on disk. The image and video packs were written by
  root in their container, so root links them in that container. Each model's own step then checks every reused file
  against its pin, and fetches only a file that is missing or differs. The uv and zig caches are shared too.
- **In the plan.** `--dry-run` prints `reuse ... GiB already present, reusing` for each reused directory, and the
  `needs:` line gives the bytes reused instead of downloaded.
- **When it does not link.** Hard links need both installs on one filesystem. On another filesystem, nothing is linked
  and the step downloads as usual.
- **Deleting one install.** Removing one install never affects the other: each keeps its links.

What it reuses, per model: the chat checkpoint and archive, the Drama 3 checkpoint, the Whisper checkpoint and VAD,
and the finished image and video packs (checked against `tools/twin/packs/built/`). The image and video download steps
then have nothing left to fetch.

To re-check the install logic on a box that already runs the stack, point a new prefix at the existing install. It
downloads nothing big:

```sh
tools/install/bootstrap.sh --profile jarvis-gb10 --prefix ~/jarvis-check --port 8291 --https-port 0 \
  --unit localrouter-jarvis-check --models-dir ~/jarvis --accept --no-gpu
```

## Remove it

```sh
JARVIS=~/jarvis UNIT=localrouter-jarvis deploy/jarvis/stop.sh --disable
rm ~/.config/systemd/user/localrouter-jarvis.service && systemctl --user daemon-reload
docker run --rm -v ~/jarvis/lr:/x nvcr.io/nvidia/pytorch:26.07-py3 rm -rf /x/models /x/weights /x/venv /x/venv-video /x/src /x/hf-cache /x/captures /x/torch-ext
rm -rf ~/jarvis
```

The `docker run` line removes the image and video files, which the container wrote as root. Run it only if the user
asks to remove everything. `sudo tailscale serve --https=10443 off` removes the tailnet entry.

## Where things are

| Path under the prefix | What |
| --- | --- |
| `bin/`, `opt/`, `dl/` | zig, uv and ffmpeg (pinned by sha256) and their downloads |
| `deploy/` | the service: `src/<commit>` (the build), `current`, `config.json`, `data/` (outputs and model logs) |
| `qualify-chat/`, `qualify-drama3/` | the chat and Drama 3 installs, their qualification runs and markers |
| `venvs/whisper`, `models/whisper`, `models/vad` | Whisper |
| `lr/` | the image and video checkpoints (`models/`), packs (`weights/`) and their build environment |
| `install/logs/`, `install/state/` | bootstrap's logs, and what it has finished |

The deeper detail per model is in [deploy/jarvis/README.md](deploy/jarvis/README.md),
[tools/chat/qwen38-27b/README.md](tools/chat/qwen38-27b/README.md), [docs/dev/DRAMA3.md](docs/dev/DRAMA3.md) and
[tools/speech/whisper/README.md](tools/speech/whisper/README.md).
