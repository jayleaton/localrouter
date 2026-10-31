# Idol worker (image to 3D Gaussian splat)

A `cmd` worker for [Idol](https://huggingface.co/speridlabs/idol): one image in, a standard 3D Gaussian Splatting PLY
out. It runs the unmodified [idol-sdk](https://github.com/speridlabs/idol-sdk/tree/4702a7f00850e34b6e83a2170cc123c3e9570744)
(PyTorch) stage by stage, in the same order as `IdolPipeline.__call__`. On macOS it uses MPS; on Linux it uses CUDA.
The kind is `asset3d` and the capability is `image_to_3d`.

On a Mac it has a second backend, `tensorfold` ([below](#native-backend-tensorfold-metal)): background removal stays
in PyTorch, and everything after it runs in TensorFold's native Metal `tf-idol-generate`. It takes about 30 s an asset
instead of about 100 s, and books 7.5 GiB instead of 12.5 GiB. It is opt-in until a TensorFold release ships
`tf-idol-generate`; PyTorch stays the default.

| File | Does |
| --- | --- |
| `worker.py` | the worker: `load`, `generate` (writes `asset.ply`, `input.png`, `asset.json`), `exit`, soft cancel on SIGUSR1 |
| `fetch.sh` | downloads the pinned weights (no token) and checks every sha256 (`pins.json`); `--venv` also builds the venv |
| `pins.json` | each file's repo, revision, size and digest (`tools/package/verify_assets.py` form) |
| `pyproject.toml`, `uv.lock` | the environment: macOS arm64 (PyPI torch, MPS) and Linux aarch64 (PyTorch cu130 wheels) |
| `test_worker.py` | protocol and unit tests against a fake engine (no weights, no GPU) |
| `config.example.json` | `idol` (PyTorch, the `image_to_3d` default) and `idol-native` (the native backend, disabled: opt-in), each with its measured bookings |

## Sources and licences

Every repo is ungated and pinned. Nobody has to apply for access to anything.

| Part | Source | Licence |
| --- | --- | --- |
| Denoiser, octree and Gaussian decoders, BiRefNet copy, config | [`speridlabs/idol@2731c22`](https://huggingface.co/speridlabs/idol/tree/2731c2291fc65a797c5cb627a5e5526fc1490902) | Apache-2.0 (model card) |
| Code | `speridlabs/idol-sdk@4702a7f` (a git dependency in the lock) | Apache-2.0 |
| DINOv3 ViT-H+/16 image encoder | [`timm/vit_huge_plus_patch16_dinov3.lvd1689m@dd0addc`](https://huggingface.co/timm/vit_huge_plus_patch16_dinov3.lvd1689m/tree/dd0addc09788111fa893d24799a44744ba022eee) `model.safetensors` | **DINOv3 License** (Meta's custom licence, Last Updated August 19, 2025). Its `LICENSE.md` is fetched beside the weights |
| FLUX.2 autoencoder image encoder | [`black-forest-labs/FLUX.2-klein-4B@e7b7dc2`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294), only `vae/config.json` and `vae/diffusion_pytorch_model.safetensors` | Apache-2.0 (the repo's `LICENSE.md`, fetched too) |

**The tool is not non-commercial.** Idol and the FLUX.2 autoencoder are Apache-2.0. Nothing from FLUX.2-dev is used.
The binding terms come from DINOv3. Its licence allows use, copying, redistribution and derivatives (§1a). You may
distribute the weights, or anything derived from them, only under that Agreement and with a copy of it (§1b.i).
Publications must acknowledge DINOv3 (§1b.ii). Law, privacy and Trade Controls apply (§1b.iii, §1b.v). Reverse
engineering is prohibited (§1b.iv). ITAR uses and the end uses Trade Controls prohibit are barred (§1b.v). Keep
`dinov3/LICENSE.md` with the weights. The tool's `license.notice` repeats this.

### Why these encoders are the same model

Idol's own config names the gated `facebook/dinov3-vith16plus-pretrain-lvd1689m` and `black-forest-labs/FLUX.2-dev`.
The worker uses ungated copies that compute the same values:

- **DINOv3.** timm's checkpoint stores the official weights in a different layout. The worker remaps them at load, so
  no converted file is written to disk (`dinov3_from_timm`). The official config is embedded as `DINOV3_CONFIG`.
  Transformers' DINOv3 graph is built in BF16, and the timm tensors are copied in:
  - `cls_token`, `reg_token`, `patch_embed.proj` and `norm` go to their `embeddings.*` / `norm.*` names.
  - `blocks.N.norm1/2` goes to `layer.N.norm1/2`.
  - `gamma_1/2` goes to `layer_scale1/2.lambda1`.
  - `attn.proj` goes to `attention.o_proj`.
  - `mlp.fc1_g/fc1_x/fc2` go to `gate_proj/up_proj/down_proj`.
  - Each fused `attn.qkv.weight` is split into contiguous Q, K and V thirds.
  - Q/V biases are zero, as they are in the official file. The mask token, which an unmasked forward never uses, is
    zero. Any other unmapped tensor fails the load.

  Research verified this: all 550 mapped segments are byte-identical to the official file, and the `[1,4101,1280]`
  features are bit-identical.
- **FLUX.2 autoencoder.** Klein-4B's `vae/` is the BF16 copy of FLUX.2-dev's `vae/`. Loaded in BF16, which is how Idol
  runs it, the two state dicts are bit-identical, and so are the conditioning tokens.

The whole pipeline confirms both. With these sources, on this worker, the gallery avocado (`food-avocado.webp`,
synthetic, seed 42, 50 steps) gives a PLY byte-identical to the research reference made with the official gated
encoders: SHA-256 `b6e814f5ab3ff31639c472b5e56bbcb30ea0972ae98f7be0030c70389f7b70cc`.

## Install

```sh
export UV_PROJECT_ENVIRONMENT=$HOME/.cache/localrouter-asset3d/venv   # outside the repository
uv sync --locked --python 3.12 --project tools/asset3d/idol
tools/asset3d/idol/fetch.sh ~/.cache/localrouter-asset3d/weights      # 6.2 GiB; or add --venv DIR to do both
```

`fetch.sh` downloads only the eleven pinned files from `https://huggingface.co/<repo>/resolve/<revision>/<path>`. It
keeps any file already there with the pinned size, resumes partial downloads, and then checks every digest with
`tools/package/verify_assets.py`. It works with bash 3.2. The worker sets `HF_HUB_OFFLINE=1`, so it never contacts the
network. If a pinned file is missing or has the wrong size, `load` fails with a message naming the file.

## Protocol

```
{"op":"load"}  -> {"ok":true,"resident":<phys footprint bytes>,"cancel":"signal"}
{"op":"generate","id":"a3d_…","dir":"<job>","request":{"asset3d":{"image":"image_0.png","input_mode":"synthetic",
  "format":"ply","seed":42,"steps":50,"guidance":3.0,"num_gaussians":262144}}}
  -> {"progress":{"phase":"preprocess|encode|denoise|decode|write","step":k,"of":n}} …
  -> {"ok":true,"files":["asset.ply","asset.json","input.png"],"seed":42,"ms":98673}
{"op":"exit"}  -> the process exits (os._exit: every weight is released at once)
```

- **Request fields.** `image` is a PNG, JPEG or WebP in the job directory. `input_mode` maps to Idol's `mode`:
  `synthetic` crops at 1.2, `real` at 1.7. A missing `steps`, `guidance` or `num_gaussians` takes the tool's defaults
  (50, 3.0, 262,144). `guidance` 0 turns off the negative pass, as in the SDK. `rescale_t` is always 3.0. A missing seed
  is drawn at random and reported in the reply.
- **Refusals** (`invalid_request_error`, never quietly changed):
  - an unreadable or unsupported image, or a path outside the job directory;
  - a fully transparent image, or one where background removal leaves an empty alpha;
  - `steps` outside 1..`max_steps` (100);
  - `guidance` outside 0..`max_guidance` (10);
  - `num_gaussians` outside `min_gaussians`..`max_gaussians`, or not a multiple of 32;
  - an unknown `input_mode` or `format`, or a non-integer count.
- **Errors.** An out-of-memory error replies `service_unavailable`. Any other failure replies `server_error`. In both
  cases the worker stays loaded.
- **Progress.** Each phase emits `step 0 of 1` when it starts. The denoise phase emits `step 0 of n`, then step k after
  each Euler step. The worker synchronises the device at each step boundary, so the line reports real progress, not
  queued work.
- **Cancel.** SIGUSR1 during a generate sets a flag. The worker checks it at the start of every Euler step (a wrapper
  around the denoiser counts its calls: two a step with CFG in eager mode) and between phases. It replies
  `{"ok":false,"type":"cancelled","error":"cancelled"}`, empties the device cache and stays loaded. When no generate is
  running, SIGUSR1 is ignored. Measured: about 2 s from the signal to the reply, which is at most one Euler step.
- **`{"op":"stats"}`** is used only by measurement scripts. It returns the footprint now and its lifetime peak. With the
  option `memory_stats`, it also returns the torch allocator peak of the last generate.

Options: `device` (`auto` picks cuda, else mps), `optimize` (default false; on CUDA only, the SDK's compiled denoiser
plus batched CFG), and the limits and defaults above. `working_bytes_by_gaussians` is read by the daemon.

### Outputs

- `asset.ply` is the SDK's own `save_ply` output, unchanged: binary little-endian, 17 float32 properties, 68 bytes
  per Gaussian.
- `input.png` is the prepared 1024² view the model was conditioned on.
- `asset.json` is the manifest, version 1, with model-neutral keys. It contains:
  - `gaussians`: the count and field meanings;
  - `frame`: up axis, bounds read back from the written file, and `from_model`;
  - `camera`;
  - `model`: the code commit and each component's repo, revision and licence;
  - `settings`, `runtime` and `timing_ms` per phase.

  `runtime` also records `optimize`.

### Frame and camera derivation

The SDK's Gaussians are z-up inside [-0.5, 0.5]³. `save_ply` calls its transform "z-up to y-up", but the code maps
`PLY (x, y, z) = model (x, -z, y)`, a +90° rotation about x. Each quaternion is left-multiplied by that rotation. So
the object's up, model +z, lands on **PLY −y** (the COLMAP/INRIA y-down convention), and `frame.up` is `[0,-1,0]`.

`camera.py` turns the camera latent into a normalised direction `d`, with `yaw = atan2(dx, dy)` and
`pitch = asin(dz)`. The radius is `1 / inverse_distance`. When the inverse distance is below 1e-3, the camera is
orthographic and the SDK uses radius 1000 with a narrow FOV.

Inverting that gives `d = (sin yaw · cos pitch, cos yaw · cos pitch, sin pitch)`. The camera is placed at
`radius · d`, looks at the origin, and takes the model's +z as its up. Position and up then go through the same
`(x, y, z) → (x, −z, y)` map. That gives `camera.position` and `camera.up = [0,-1,0]`.

A render of the PLY with Spark settles the direction: `d` points from the object to the camera, and with `up`
flipped, or the other sign, the render's mask matches `input.png` at an IoU of 0.53 or less, or comes out rolled 180°.
The decoded FOV is the wrong scale, though. It frames 1.4 to 1.6 times too much, by a factor that changes from sample
to sample. So `frame_camera()` frames the view the way preprocessing framed the input: it projects the splats that
are at least half opaque into the predicted camera (tangent space for a perspective camera), takes their box times
the input mode's crop margin (1.2 synthetic, 1.7 real) as the view, and moves the target to the box's centre. An
orthographic camera gets `half_height` and a matching `fov_y_deg`; a perspective one gets `fov_y_deg`. Against
`input.png`, mask IoU goes from about 0.38 to 0.92 for the avocado (orthographic: radius 1000, decoded FOV 0.11°,
fitted half-height 0.602) and 0.86 for the coffee photo (perspective: 80.2° decoded, 60.6° fitted). `camera.framing`
says which was used; the raw `yaw_rad`, `pitch_rad`, `radius` and `fov_deg` stay under `camera.model`.

## Measured on the Mac (PyTorch backend: M5 Ultra, 96 GiB, MPS, torch 2.13.0, eager)

Footprint is `proc_pid_rusage` `ri_phys_footprint`, sampled every 25 ms by a driver over the protocol. It includes
Metal buffers. The weights were in the page cache.

| Gaussians | Resident after load | Peak footprint | Working (peak − resident) | MPS driver peak | Time (pre / enc / denoise / dec / write) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 32,768 | 5.03 GB | 12.64 GB | 7.61 GB | 12.64 GB | 97.6 s (0.5 / 0.4 / 96.3 / 0.4 / 0.02) |
| 262,144 (avocado) | 5.03 GB | 12.26 GB | 7.24 GB | 12.63 GB | 98.7 s (0.5 / 0.4 / 96.6 / 1.2 / 0.04) |
| 262,144 (real photo) | 5.03 GB | 12.37 GB | 7.34 GB | 12.63 GB | 99.0 s (0.5 / 0.4 / 96.5 / 1.6 / 0.04) |
| 1,048,576 | 5.03 GB | 12.64 GB | 7.61 GB | 12.63 GB | 102.8 s (0.5 / 0.4 / 96.3 / 5.6 / 0.1) |

The worst 262,144 peak, seen in a cancel run, is 12.69 GB, or 7.61 GB working.

- **Load** takes 3.6 s. The footprint settles at 4.5–5.1 GB once the device cache is emptied.
- **The peak is the denoiser**, not the decoders. That is why it is flat across counts: the 24-block DiT over
  12,294 tokens runs twice per step. 1,048,576 Gaussians is qualified. Only the decode phase grows with the count.
- **Bookings in `config.example.json` (`idol`):** `resident_bytes` is 5 GiB (5.08 GB measured, plus 5%). `working_bytes` is
  7.5 GiB at every count (7.61 GB worst, plus 5.8%).
- **Unload:** `exit` ends the process in under 0.1 s, and its footprint goes to 0.

## Native backend (TensorFold Metal)

With `"backend": "tensorfold"`, the worker loads only BiRefNet (PyTorch MPS). Each generate then works like this:

1. The worker runs the SDK's own `preprocess` (background removal and the crop to 1,024 x 1,024) and empties the MPS
   cache, so BiRefNet's peak and the child's peak never add up.
2. It writes the picture as `<job>/.tf-idol/picture.safetensors` (`image`, [1024, 1024, 3] float32 in [0, 1]).
3. It runs `tf_idol_generate SNAPSHOT VAE DINO PICTURE OUT --seed --steps --cfg --gaussians` from the same weights
   directory. The child runs both encoders, the denoiser, the decoders and the PLY writer. It frees each model after
   its stage.
4. It moves `generate.ply` to `asset.ply`, reads `camera.json`, writes `input.png` and `asset.json` as the PyTorch path
   does, and removes the work directory.

The PLY uses the SDK's `save_ply` frame and properties, and the camera follows `decode_camera`'s convention, so the
manifest, the asset check and the viewer are unchanged. The manifest's `runtime` says `backend: tensorfold`, the
PyTorch version used for preprocessing, the child's stage times and its peak footprint.

- **Not bit-identical to the SDK.** The starting noise is torch's CPU-seeded randn, but the other draws are native. It
  is the same model, sampled differently. TensorFold's `docs/recipes/idol.md` compares the two as distributions. The
  predicted camera can differ too: for the avocado at seed 42 the SDK predicts a distant (orthographic) camera and the
  native run a near one, 130 units away at 0.8 degrees.
- **Opacity varies by seed, on both backends.** The avocado at seed 42 is 90% opaque splats (alpha > 0.5) on
  PyTorch and 42% native, with the skin thin from behind and below. That gap is the noise draw, not the port.
  TensorFold's recipe gives the SDK the same features and the CPU-seeded noise, and it also gets 42%. Over other seeds,
  through the daemon at 262,144, PyTorch gives 49%, 71% and 51% (seeds 1–3) and native gives 74%, 49%, 56%, 51% and
  49% (seeds 1–5).
- **Progress** comes by phase, not by step: `encode` when the child starts, then `denoise` after the encoders,
  `decode` after the denoiser and `write` after the decoders. The child prints its stages only as each one ends.
- **Cancel.** SIGUSR1 terminates the child within 50 ms. The worker removes the work directory and replies
  `cancelled`, and it stays loaded. Measured: 0.1 s from the signal to the reply.
- **Errors.** A child that exits non-zero gives `server_error` with its last lines. Zig's `OutOfMemory` gives
  `service_unavailable`.
- **Options.** `tf_idol_generate` (required) is the binary's path. `preprocess` is `pytorch`, the only value until
  TensorFold's BiRefNet lands. Then `Preprocess` in `worker.py` is the one class that changes, and the torch import
  and venv go with it. The child needs the timm `dinov3/config.json` (pinned in `pins.json`) beside
  `model.safetensors`.
- **Which binary.** `tf-idol-generate` is not in a TensorFold release yet. It is in TensorFold PR #1780, a draft stacked
  on #1766, #1773, #1774 and #1775. The numbers below come from a build of `192c4e902`
  (`zig build tf-idol-generate -Dcpu=apple_m1 -Doptimize=ReleaseFast`). The Mac install (`deploy/mac/config.json.in`)
  keeps the PyTorch `idol` until a release ships it.

**Switching.** `config.example.json` lists both. `idol` (PyTorch) is the `image_to_3d` default. `idol-native` has
`residency` `disabled`, because no TensorFold release ships `tf-idol-generate` yet; set its `tf_idol_generate` to a
built binary first. To opt in, turn the native one on and the PyTorch one off, either in the config or live:

```sh
curl -s localhost:8190/v1/tools/idol-native/settings -d '{"residency":"on_demand"}'
curl -s localhost:8190/v1/tools/idol/settings -d '{"residency":"disabled"}'
```

A request that names no model goes to the next enabled `image_to_3d` model. `{"residency":"on_demand"}` alone, without
disabling `idol`, lets a request pick `idol-native` by name. To go back, reverse the two (or send `{"reset":true}` to
each). Once a release ships the binary, a follow-up makes the native backend the default.

### Measured (native, M5 Ultra, 96 GiB)

The binary was built from TensorFold `192c4e902`; preprocessing used torch 2.13.0 MPS. Footprint is
`ri_phys_footprint` of the worker plus its child, sampled every 25 ms by a driver over the protocol. Inputs: the
avocado (`synthetic`) and a real photo (`real`), seed 42, 50 steps, guidance 3.

| Gaussians | Resident after load | Peak footprint (worker + child) | Working (peak − resident) | Child peak | Time (pre / child / write) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 32,768 (first job) | 2.77 GB | 7.12 GB | 4.35 GB | 2.71 GB | 29.4 s (0.47 / 28.9 / 0.02) |
| 262,144 (avocado) | 2.77 GB | 6.48 GB | 3.71 GB | 2.71 GB | 29.7 s (0.42 / 29.2 / 0.02) |
| 262,144 (real photo) | 2.77 GB | 6.65 GB | 3.88 GB | 2.71 GB | 29.8 s (0.44 / 29.3 / 0.03) |
| 1,048,576 | 2.77 GB | 6.50 GB | 3.73 GB | 2.71 GB | 31.4 s (0.41 / 30.9 / 0.05) |

- **Load** takes 1.4–1.9 s (BiRefNet only). The child loads its weights in each job, in about 0.7 s.
- **The peak is BiRefNet**, not the child. It peaks highest on the first job after load. The child peaks at 2.71 GB
  at every count. Before the worker emptied the MPS cache ahead of the child, the two peaks stacked to 9.41 GB.
- **Bookings in `config.example.json` (`idol-native`):** `resident_bytes` is 3 GiB (2.77 GB measured, plus 16%). `working_bytes` is
  4.5 GiB at every count (4.35 GB worst, plus 11%). The total is 7.5 GiB, against 12.5 GiB for the PyTorch `idol`.

## Spark (GB10, sm_121, aarch64)

**Untested until qualified.** The lock resolves torch 2.13.0+cu130 and torchvision 0.28.0+cu130 aarch64 wheels from
the PyTorch cu130 index, the same index the Whisper worker uses. The code path is the SDK's own CUDA path:

- BiRefNet runs in FP16 on CUDA.
- `optimize` stays off unless the option asks for it. When on, it uses `torch.compile` plus cuDNN attention plus TF32,
  none of which has been tried on sm_121.
- Resident memory is the resident set plus `torch.cuda.memory_reserved()`.

No Spark numbers exist. CUDA output is not bit-comparable with MPS. Qualify it on its own reference.

## Tests

```sh
$UV_PROJECT_ENVIRONMENT/bin/python -I tools/asset3d/idol/test_worker.py
```

The tests use a fake engine. They cover load and refused loads (missing weights, bad config), every refusal, defaults,
`guidance` 0, cancel mid-denoise followed by a working generate, an idle SIGUSR1 being ignored, server and OOM errors,
the manifest's shape, camera math, the DINOv3 key map, and stdout carrying protocol lines only. For the `tensorfold`
backend, a fake `tf-idol-generate` script covers: option checks, a non-executable binary, the child's arguments and
picture, the work directory's removal, the far-camera convention, a failing child (`OutOfMemory` gives
`service_unavailable`), and cancel ending the child mid-run followed by a working generate.
