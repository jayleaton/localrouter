# Packaging a model as an on-demand service

These scripts turn one LocalRouter model into an on-demand service on a GPU node: built once on a build machine,
installed from files already on the node, started and stopped by the operator, and checked by a lifecycle smoke test
that leaves a JSON receipt. The daemon holds no GPU memory while idle. The model loads on a request and unloads after it
(`idle_ttl_s: 0`), so the device stays available to other tools between jobs.

| File | Does |
| --- | --- |
| `build-image.sh` | Builds the arm64 image from a clean commit (`docker/Dockerfile`, every engine compiled in; the model is chosen by config) and exports it as `localrouter-<commit>-arm64.tar.zst` with a receipt: commit, image id, SHA-256. It refuses to run on an arm64 host: GPU nodes build nothing |
| `release-batch.sh` | One exact commit, checked out alone: `zig fmt --check` compared with its base by identity (a new finding fails), `zig build`, `zig build test`, then `build-image.sh`; `batch.json` records every step. Run it in the machine's build slot when one is shared |
| `install.sh` | On the node: checks GPU, Docker and disk; checks the weights against the model's pinned manifest; checks the image file against its receipt and loads it; writes `tools.json` with the one model and `env` for `localrouterctl`. `--start` starts the daemon, `--smoke` runs the lifecycle smoke. `--uninstall` removes the image and the state, and keeps the weights |
| `localrouterctl` | `start`, `status`, `generate MODEL PROMPT -o out.mp4`, `unload MODEL`, `release`, `stop`, `pids`, `logs`, in docker or binary mode (`LR_MODE`). `stop` returns only once the daemon and every worker and ffmpeg it started have exited |
| `image_digests.py` | An image file's manifest and config digests, read and hashed from the archive itself. A Docker daemon names a loaded image by one or the other (containerd image store: the manifest; classic store: the config), so `install.sh` accepts either and the build receipt records both |
| `verify_assets.py` | A weight directory against a manifest: every listed file's size and SHA-256 (`lfs_sha256`), or git blob hash (`git_blob_sha1`) for small files kept in git |
| `lifecycle_smoke.py` | The lifecycle on the running package, written as a receipt (below) |
| `profiles/` | Per model: the `tools.json` entry, the smoke request, the output it must produce, and the requests it must refuse |

## Kandinsky-6 Pro distilled

`profiles/kandinsky6-pro-distill.json`: engine `kandinsky6`, model `kandinsky6-pro-distill`. It makes 864x480 5 s
video at 24 fps (121 frames) with jointly generated mono 44.1 kHz audio, from the unmodified Hugging Face snapshot of
`kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers` at revision `1018d5828716e8bea3de99609f45c924de9c5564`. Any other
size, length, frame rate, step count or guidance is refused, as are `audio: false`, a negative prompt and a first frame.

The image build runs in a builder of its own whose container is capped (`LR_BUILDER_MEMORY`, default 7g with no swap;
`LR_BUILDER_CPUS`, default 0-3). A build that needs more fails with an out-of-memory error instead of pushing the build
machine into swap. Its receipt records the cap, the builder's actual `memory.peak` and any OOM kills. That builder
cannot see the Docker daemon's images. Name already-pulled base images in `LR_BASE_IMAGES`: they are exported once as
OCI layouts (about their size on disk, under `~/.cache/localrouter-oci`) instead of pulled again. They are removed
after a successful build and kept after a failure for the retry. Before exporting or building, the script checks the
free disk against what the run can take (twice each base image, plus `LR_DISK_MARGIN_GB`, default 30) and stops if it
does not fit. The receipt records each layout's size and the largest drop in free space during the run.

```bash
# build machine (the NGC builder image already pulled)
docker pull --platform linux/arm64 nvidia/cuda:13.0.1-base-ubuntu24.04   # the runtime base, once
LR_BASE_IMAGES="nvcr.io/nvidia/pytorch:26.07-py3 nvidia/cuda:13.0.1-base-ubuntu24.04=linux/arm64" \
  bash tools/package/release-batch.sh <commit>
# copy localrouter-<commit>-arm64.tar.zst and .json to the node; the weights are already there

# the node
TOOL_OPTIONS='{"fp32_linear": "<mode>", "clip_linear": "<mode>", "qwen_rope_freq": "<mode>", "qwen_inv_freq": "<file>",
               "audio": {<every audio policy field>}}' \
  bash tools/package/install.sh --profile tools/package/profiles/kandinsky6-pro-distill.json \
    --weights <snapshot dir> --image localrouter-<commit>-arm64.tar.zst \
    [--manifest tools/kandinsky6/staging/assets.json] --smoke
LR_HOME=~/.localrouter tools/package/localrouterctl start
LR_HOME=~/.localrouter tools/package/localrouterctl generate kandinsky6-pro-distill "a prompt" -o out.mp4 --size 864x480 --seconds 5
LR_HOME=~/.localrouter tools/package/localrouterctl stop
```

`fp32_linear`, `clip_linear` and `qwen_rope_freq` (`tf32_rna`, `tf32_truncate` or `fp32`) and `qwen_inv_freq` (a
256-byte file: 64 little-endian f32, Qwen2.5-VL's rotary `inv_freq` as captured on the target) have no default: the
engine refuses to load without them, and their values come from the model's GPU qualification. So does `audio`: the
audio decoder's and vocoder's numeric choices as one object with every field named (its shape is in
`src/engines/kandinsky6_tool.zig`); the engine refuses a partial one, and `install.sh` refuses a `null` at any depth. `install.sh` copies
the `qwen_inv_freq` file (a profile's `file_options`) into `$LR_HOME/files`, checks its size, records its SHA-256 and
gives it to the container read-only. The profile marks these options, and the stage names the cancel checks target
(`load`, `text`, `denoise`, `decode`, `audio`, `encode`), as `provisional`.

## The lifecycle smoke

```bash
LR_HOME=~/.localrouter python3 -I tools/package/lifecycle_smoke.py --profile <profile> --out receipt.json \
  [--cycles N] [--after-model ID] [--explicit-unload] [--probe-bodiless-post] [--skip-cancel]
```

It starts the daemon through `localrouterctl`, and checks, in order:

1. The model is configured on demand and unloaded, and no worker runs.
2. Each unsupported request gets its 400 (or 404) and nothing loads.
3. A cold generation completes. Its MP4 has the expected video (size, frame count, fps) and an audio stream of the
   expected rate and channels, as long as the video.
4. After each job the model unloads. The worker and its children are reaped, none of their pids holds GPU memory, and
   MemAvailable is back within `--mem-tolerance-gib`.
5. `--cycles` more load, generate and unload cycles show no memory drift.
6. A job cancelled in each named phase ends `cancelled`, serves no output, and its worker tree is reaped.
7. A worker killed mid-job fails that job only, and the next job succeeds.
8. With `--explicit-unload`, the operator's `/v1/tools/{id}/unload` releases the model the same way.
9. With `--after-model`, another model generates on the freed device.
10. `stop` ends the daemon and every process it started, and closes the port.

Each check is `pass`, `fail` or `skipped`, with the reason. GPU checks are skipped, not passed, without `nvidia-smi`
(`gpu_verified: false`). A profile with `provisional` fields yields `provisional`, never `pass`. Exit status: 0 pass,
1 fail, 2 provisional. `profiles/testpattern.json` (with `profiles/testpattern.tools.json`) runs the whole sequence
on any machine without a GPU, in binary mode:

```bash
zig build
LR_HOME=$(mktemp -d) LR_CONFIG=tools/package/profiles/testpattern.tools.json \
  python3 -I tools/package/lifecycle_smoke.py --profile tools/package/profiles/testpattern.json --out receipt.json \
  --after-model testpattern-after --explicit-unload
```

That checks the lifecycle mechanics only. A model's release still needs its own receipt from the target GPU, plus
that model's correctness gates (see CONTRIBUTING.md).
