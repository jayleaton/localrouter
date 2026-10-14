# Kandinsky-6 Pro-distill oracle capture and bit gate

Development and validation only. The production runtime is native Zig; Python here only records the reference and
compares captures. ComfyUI is not used.

The oracle is upstream kandinsky-6 at `eac06e39b904940baff2c408e925cdb0c4529269`. `capture` imports it at run time from
the pinned source tree passed as `--src`. The code is never copied into this repository. Before it imports anything,
`capture` checks the tree's commit (`.k6-source-commit` or the checkout's `.git`) and records a sha256 over every file
it imports.

## Commands

```sh
python -I -m stk_twin.kandinsky6.capture --model SNAPSHOT --src KANDINSKY_SRC \
  --stages noise,rope,piflow,clip,text,vae,dit0,e2e --seed 42 --prompt-file PROMPTS --out DIR [--oracle-prefix /opt/k6-oracle]
python -I -m stk_twin.kandinsky6.compare ORACLE_DIR NATIVE_DIR [--ops k6.vae.,k6.clip.] [--json report.json] [--allow-alone]
```

`-I` keeps `PYTHONPATH` and the user site out of `sys.path`, so `stk_twin` has to be importable some other way. The
node kit puts `tools/twin` on `sys.path` with a `-c` bootstrap; see `k6-job.sh`.

## Oracle interpreter

The oracle runs on one pinned Python closure: the shared, Captain-verified prefix `k6-oracle-prefix.tar.gz`
(3,344,315,223 bytes, sha256 `74118af7…0c474`), staged once per node by Cody. The node kit extracts it after T0 into
its own workspace, checks `MANIFEST.sha256` (sha256 `a1af1b3b…24b90`) and `sha256sum -c`s it, and mounts the tree
read-only at `/opt/k6-oracle` (python `/opt/k6-oracle/bin/python3.13`).

`capture` asserts the closure before anything runs: CPython 3.13.16 (python-build-standalone), torch 2.14.1+cu130,
CUDA 13, NumPy 2.5.4. With `--oracle-prefix /opt/k6-oracle`, `sys.prefix` must be that prefix and its
`MANIFEST.sha256` must have the pinned sha256. The `oracle` note records:

- the pins: archive size and sha256, manifest sha256, freeze sha256 `d793ad20…90936` (includes filelock 4.1.0 and
  safetensors 0.8.0), metadata-correction receipt sha256 `38cd6454…580a9`;
- the observed versions, and a freeze sha256 recomputed from the installed metadata (informational; the gate on the
  prefix is its manifest);
- `install`: a corrected install (one wheel's installed WHEEL tag and RECORD entry rewritten, library bytes
  unchanged). It is not byte-identical official metadata and is never called official;
- `qualification: unqualified`. Every capture, compare and job result carries this label until a T0 GPU result
  exists. A CPU import or relocation pass of the prefix does not qualify the GPU/NGC ABI or precision.

`SNAPSHOT` is the HF snapshot `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers@1018d582`. It is read only.

## How the pipeline is built

The pipeline is built the way `pipeline/factory.py` `_build_pipeline` builds it:

- The config is `configs/checkpoints/pro-distill.yaml`, loaded through `load_config`.
- Component paths are bound to `--model` as absolute paths. `bind_checkpoint_paths` therefore never downloads, and
  `HF_HUB_OFFLINE=1` is set as well.
- `set_float32_matmul_precision("high")` is applied, as in factory.py:159-162.

These choices are explicit and recorded in the `runtime` note:

- Attention engine `--attention`, default `sdpa`. `auto` is not offered.
- Torch SDPA backend priority `--sdpa-backends`. `default` reads torch's built-in order and pins it with
  `sdpa_kernel(..., set_priority=True)`.
- Compile `--compile`, default `none`, which is eager.
- Beautifier `none`.
- Offload `none`, cache `none`, NF4 off, SR off.

The upstream `pipeline/config.py` imports `kandinsky_sr.pipeline.config.SRConfig`. The oracle closure leaves SR out,
so a frozen stand-in with no fields fills that `sr` section. The `upstream` note records which one was used.

`capture` refuses to run in any of these cases:

- `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`, `NVIDIA_TF32_OVERRIDE` or `LD_PRELOAD` is set.
- The interpreter is not the pinned closure (CPython 3.13.16, torch `2.14.1+cu130`, CUDA 13, NumPy 2.5.4), or, with
  `--oracle-prefix`, it does not run from that prefix with the pinned manifest.
- `--out` already exists.

It sets these logs, written in `--out`:

- `CUBLASLT_LOG_LEVEL=5`, written to `cublaslt.log`.
- cuBLAS API logging, written to `cublas.log`.

## Capture format

The format is the twin's own, `stk_twin/rec.py`, and capture records through `REC` itself. Zig reads it with
`src/engines/qwen_image/capture.zig` (`Capture.open`, `note`, `Op.in`/`Op.out`).

```
DIR/ops.jsonl        one record per line, in run order:
                       op:   {"i", "name", "kind", "attrs": {...}, "ins": {role: ref}, "outs": {role: ref}, "launches": []}
                       note: {"i", "note": name, ...data}
DIR/blobs/<sha256>   raw tensor bytes: C-contiguous, little-endian; the file name is the sha256 of the bytes
ref                  {"sha": hex, "dtype": "bfloat16|float16|float32|uint8|bool|int16|int32|int64", "shape": [...]}
DIR/cublaslt.log, DIR/cublas.log, DIR/e2e/output.mp4 (+ output.txt)   when produced
```

The native side writes the same format (`localrouter check k6-components --stages ...` and `localrouter check k6-e2e`,
src/cli/k6.zig):

- The same op names and the same role names.
- Its own `stage` notes, if it has stages.
- Notes are free-form. `compare` only reads `stage` notes.
- Attrs whose name starts with `note_` are informational and never compared.
- Ops under `k6.aux.*` are diagnostics. They are compared only when both sides emit them.

### Ops

Shapes are for the preset call: 480x864, 31 latent frames, 218 audio latents and 1024 text tokens.

| op | ins | outs | attrs |
|---|---|---|---|
| `k6.noise.video` | | `x` bf16 [31,60,108,16] | seed |
| `k6.noise.audio` | | `x` bf16 [218,40] | seed (= seed+1) |
| `k6.freqs.time_v` / `time_a` | | `f` f32 [2048] / [1024] | buffer |
| `k6.freqs.rope_t` / `rope_h` / `rope_w` | | `f` f32 [128,16] / [128,24] / [128,24] | buffer |
| `k6.freqs.rope_text` / `rope_audio` | | `f` f32 [2048,64] | buffer |
| `k6.rope.visual` | | `table` f32 [50220,256] | source_shape |
| `k6.rope.audio` | | `table` f32 [218,256] | source_shape |
| `k6.rope.text_v` / `text_a` | | `table` f32 [1024,256] | source_shape |
| `k6.piflow.step{i}`, i=0..9 | `state_v`, `state_a`, `pred_v` bf16 [1,10,31,60,108,16], `pred_a` bf16 [1,10,218,40] | `state_v`, `state_a` | step, sigma, tau_src, tau_dst, segment, substeps, time |
| `k6.clip.pooled` (per prompt) | `ids` i64 [1,77] | `pooled` f32 [1,768] | prompt |
| `k6.qwen.embeds` (per prompt) | `ids`, `mask` i64 [1,1153] | `embeds` bf16 [1024,3584] | prompt, crop_start, max_length |
| `k6.vae.decode` | `latent` [1,31,60,108,16] | `frames` u8 [1,3,121,480,W] (W 864, or the plan's shorter extent: 832 at 2,301,403,136 free bytes) | the decode plan (tile sizes and strides, tiling flags, scaling_factor, dtype, note_free_bytes) |
| `k6.dit.step{i}` | `x_video` [1,31,60,108,33], `x_audio` [1,218,40], `time` f32 [2] | `video` [1,10,31,60,108,16], `audio` [1,10,218,40] | step |
| `k6.audio.latent_in` | | `latent` f32 [1,218,40] | chained |
| `k6.audio.mel` | `latent` | `mel` f32 [1,128,436] | chained |
| `k6.audio.wave` | `mel` | `wave` f32 [1,1,223232] | chained, sample_rate |
| `k6.audio.pcm16` | `wave` | `pcm` i16 [223232] | chained, sample_rate, normalization |
| `k6.e2e.frames` | | `frames` u8 [1,3,121,480,864] | fps, prompt |

The four `k6.audio.*` ops are the audio chain in pipeline order, shared with D3's audio exporter:

- `latent_in` is the final PiFlow audio state exactly as handed to `postprocess_audio`, before the transpose and the
  VAE de-normalization.
- `mel` is the audio VAE decoder output (`wrapped_decode`) as fed to the vocoder. Its `latent` input is
  `latent_in`'s output: the op covers the de-normalization, transpose and decode.
- `wave` is the vocoder output for that mel; `pcm16` is the int16 the pipeline returns.

`capture` writes `chained: true` on all four, and each op's input is blob-identical to the previous op's output, so
one run proves the chain. An in-run check confirms that the vocoder's input equals the VAE's mel.

Rope tables keep upstream's rotary layout, flattened: one row per position, and per frequency the four values
`(cos, -sin, sin, cos)`. Piflow and e2e states are bf16 at step 0 and float32 after that. That is the upstream
loop's own promotion, recorded as it happens.

The diagnostic ops are:

- `k6.aux.freqs.rope_text_a`
- `k6.aux.clip.mask`
- `k6.aux.qwen.extra`: cropped mask, cu_seqlens, pooled, CLIP ids.
- `k6.aux.vae.decoder_in`: the fp16 `vae.decode` input.
- `k6.aux.vae.gn.fixture` (diagnostic; `python -I -m stk_twin.kandinsky6.vae_gn_isolate` and `localrouter check
  k6-components --stages vae_gn --oracle ISOLATE`): one torch GroupNorm + SiLU of the video VAE on a trace input; outs
  `x`, `norm`, `silu` fp16 [P][C] channels last, `gamma`, `beta` fp16 [C]. Native records the first exact statistics
  candidate (note `gn_fixture` lists every candidate's unequal counts).
- `k6.aux.vae.conv.fixture.<target>` (diagnostic; `python -I -m stk_twin.kandinsky6.vae_conv_isolate` and `localrouter
  check k6-components --stages vae_conv --oracle ISOLATE`): one decoder convolution a shape group on its trace input;
  outs `x` [ti*hi*wi][ci], `w` [co][ci][kt][kh][kw], `bias`, `acc` (cudnn before the bias add) and `y` [to*ho*wo][co],
  fp16 channels last; attrs the geometry. Native notes `conv_fixture` per target (every K-schedule candidate's unequal
  count, the production kernel tap-major and under the exact candidate's rule).
- `k6.aux.vae.conv.fixture.attn_to_{q,k,v,out}` and `k6.aux.vae.attn.sdpa` (diagnostic; `python -I -m
  stk_twin.kandinsky6.vae_attn_isolate`): the mid-block attention on its trace input, which must reproduce the trace's
  group_norm, to_q, to_k, to_v and output. The projections in the conv fixture's format (a 1x1x1 conv over one row of N
  positions, attr `linear`); the native fixture also runs the production kernel on `y` with the bias separate and fused
  (`production_y_unequal`). The SDPA's `q`, `k`, `v`, `out` [N][C] and `mask` [N][N]; notes `attn_isolate` (layouts,
  SDPA backend flags), `attn_kernels` (each projection's, the SDPA's and the module's CUDA kernels) and
  `attn_sdpa_backend` (torch._fused_sdp_choice, and each backend forced on the call against its bits). Native
  `localrouter check k6-components --stages vae_sdpa --oracle ISOLATE` replays `k6.aux.vae.attn.sdpa` under every
  k6v_sdpa_memeff variant (note `sdpa_fixture`, with the closest variant's differences localised).
- `k6.aux.vae.attn.sdpa.<probe>` (diagnostic; `python -I -m stk_twin.kandinsky6.vae_sdpa_probe --isolate ATTN_ISOLATE`):
  crafted inputs through the same torch kernel (SDPBackend.EFFICIENT_ATTENTION forced, the kernel checked by name):
  `real`, `q0` (P = 1), `nomask`, `one_block`, `q0_one_block`; replayed by `--stages vae_sdpa` like the recorded call.
  Note `sdpa_pv_steps`: P V's rounding steps measured directly (classes of keys, agreement with the k8 / k16 models).
- `k6.aux.vae.conv.oracle.<target>` and `k6.aux.vae.conv.spec.<target>` (diagnostic; `python -I -m
  stk_twin.kandinsky6.vae_conv_oracle`): a convolution's summation order measured on the device with fp16 lattice
  queries (signatures int8 [pairs][elements]); when the steps hold, the spec: `elements` int32 [steps][16] (tap * ci +
  c, -1 padded), `chains` int32 [steps], attrs `atom`, `chains`, `join`. `localrouter check k6-components --stages
  vae_conv --oracle ISOLATE --conv-spec ORACLE` runs it as that target's candidates (`spec_*`).
- `k6.aux.vae.gn.fixture.<case>` (diagnostic; `python -I -m stk_twin.kandinsky6.vae_gn_probe`): torch's fp16
  GroupNorm(32, C) + SiLU on other data and shapes, in the fixture's format; `--stages vae_gn` replays every
  `k6.aux.vae.gn.fixture[.<case>]` op (one `gn_fixture` note each). A policy is a candidate exact on all of them: on one
  input several candidates can tie.
- `k6.aux.vae.blend.fixture.<case>` (diagnostic; `python -I -m stk_twin.kandinsky6.vae_blend_isolate` and `localrouter
  check k6-components --stages vae_blend --oracle DIR`): the pinned upstream blend_t / blend_v / blend_h on fp16 tiles;
  attrs axis, extent and both tiles' dims, outs a, b (before) and y (b after), [t*h*w][3]. Native runs the decode's
  k6v_blend launch on each (note `blend_fixture`).
- `k6.aux.vae.trace.<name>` (`--vae-trace first-tile` or `pass-N` on both sides; diagnostic; the decoder checkpoints are
  those of decoder pass N, first-tile = 0, passes in the reference's order): role `x`, fp16 [positions][channels],
  channels last. `<name>` is `decode.input`, then a decoder checkpoint of kernels/cuda/kandinsky6/video_vae/QUALIFICATION.md at its first call
  (`post_quant_conv`, `decoder.conv_in`, `decoder.mid_block.resnets.0.nonlinearity.0`, ..., `decoder.conv_out`: the first
  spatial tile of the first temporal tile), then `blend_v`, `blend_h`, `tiled_decode`, `blend_t` (each method's first
  return). Native adds `note_thw`, the positions' [t, h, w]. Ops run in decode order: the first unequal one in the
  native capture's `i` order is the first divergent operation.
- `k6.aux.dit.cond`: step-0 text embeds, pooled, mask, and all four rope tables as the DiT received them.
- `k6.aux.e2e.latents`: the final video and audio latents. `--vae-latent DIR` decodes this op's video.
- `k6.aux.audio.decoder.<name>` (stage `audio_decoder`): every native decoder hook, role `raw` (the tensor's
  contiguous bytes as uint8), in this order:
  - `latent_unscale`, `first_conv`;
  - per residual block, `<block>_norm`, `_silu`, `_res_conv1`, `_res_silu`, `_res_conv2`, `_shortcut` (blocks whose
    channels change), `_residual`, and `_clamp` where a clamp follows. The blocks are mid.block_1 `first`,
    mid.block_2 `mid2`, then up[L].block[i] `upL_first|second|third` for L = 2, 1, 0.
  - after mid.block_1, `attention_qkv`, `_norm`, `_out`, `_merged`, `_proj`, `_residual`;
  - after up[1]'s blocks, `up1_upsample` (nearest-exact x2) and `up1_upsample_conv`;
  - `final_silu`, `conv_out`, and `mel` (after unnormalize).

  `--audio-trace VALUE` stops after one op. The values are those of native `--audio-trace`; the table is
  `audio_trace.STOPS` here and `audio_stops` in `src/cli/k6.zig`. The names are launch.zig's `decoder_blocks`.
- `k6.aux.audio.vocoder.<name>` (stage `vocoder`): `conv_pre`, then per stage i (0-5) `upsample<i>`,
  `stage<i>_amp<j>` (j 0-2) and `stage<i>_avg`, then `post_act`, `conv_post`, `wave`. Native `check k6-components
  --stages vocoder` records the same names from the same mel.

### Notes

The notes are:

- `request`: argv, stages, seed, prompts and their sha256, geometry.
- `upstream`: commit, tree sha256, SR stand-in, the full resolved config, HF transformer config.
- `runtime`: attention engine, SDPA order, compile and the other policies.
- `device`: every scalar CUDA device property, including `multiProcessorCount`, `maxThreadsPerMultiProcessor`, name
  and uuid.
- `precision` and `precision_after`.
- `env`: the full environment, with secrets redacted.
- `pipeline`: DiT class, cache mode, beautifier, offload, the attention functions bound, compiled or not.
- `stage`: per stage, `ok`, `skipped: missing <file>` or `failed: ...`, with seconds and the traceback.
- `check`: in-run equalities, for example rope stage vs DiT step-0 ropes, or the piflow time vs the schedule.
- `vae_latent` and `e2e_mp4` (sha256).
- `versions`: python, torch, CUDA, cuDNN, transformers, diffusers and the rest, plus every loaded
  cuBLAS/cuDNN/driver library with its path, sha256 and version. cuBLAS and cuBLASLt versions come from the loaded
  library; the driver version comes from `cuDriverGetVersion` and `nvidia-smi`.

## Stages and what they need from the snapshot

| stage | records | snapshot files |
|---|---|---|
| noise | `k6.noise.*` | none |
| rope | `k6.freqs.*`, `k6.rope.*` | none (dims from upstream's `configs/modules/transformer_pro.json`) |
| piflow | `k6.piflow.step0..9` | none |
| clip | `k6.clip.pooled` | `text_encoder_2/{config.json,model.safetensors}`, `tokenizer_2/{tokenizer.json,tokenizer_config.json}` |
| text | `k6.qwen.embeds` | `text_encoder/*` (config, generation_config, index, 5 shards, tokenizer, vocab, merges, preprocessor_config, chat_template) + the clip files |
| vae | `k6.vae.decode` | `vae/{config.json,diffusion_pytorch_model.safetensors}` |
| dit0 | `k6.dit.step0` | everything |
| e2e | `k6.dit.step0..9`, `k6.vae.decode`, `k6.audio.*`, `k6.e2e.frames`, mp4 | everything |
| audio_decoder | `k6.aux.audio.decoder.*` from `--audio-input`'s `k6.audio.latent_in`, `placement` notes | `audio_vae/{config.json,diffusion_pytorch_model.safetensors}` |
| vocoder | `k6.aux.audio.vocoder.*` from `--audio-input`'s `k6.audio.mel`, `placement` notes | `vocoder/{config.json,diffusion_pytorch_model.safetensors}` |

How each stage is produced:

- **piflow** runs upstream `piflow_denoise_loop` on the stage's noise. A stand-in DiT returns synthetic bf16 grid
  predictions from a CUDA generator seeded `seed+1000+i`. The loop's schedule and `policy_rollout_fm` and
  `MultimodalDXPolicy` are therefore upstream's own, not a copy.
- **dit0** and **e2e** use forward hooks on the real pipeline's bare DiT and the vocoder. `postprocess_video` is
  wrapped in the pipeline module only to record the decode. When both stages are requested, one generate records both.
- **clip** follows the CLIP section of `Kandinsky6TextEmbedder.encode`.
- **text** calls `encode` itself and checks its pooled output against the clip stage.
- **audio_decoder** builds the audio VAE with upstream `build_audio_vae` on the CPU and then moves it to the device.
  Each parameter and buffer gets a `placement` note with its CPU and device-readback sha256, plus a check. The stage
  then unscales the latent as `postprocess_audio` does. `audio_trace.decode` runs upstream's `Decoder1D` / block
  forwards statement by statement (the same functions and modules in the same order) to record each intermediate.
  These two stages need no `--prompt-file` and need `--audio-input`.
- **vocoder** builds BigVGAN with upstream `build_vocoder`, runs it on the input mel with forward hooks only, and
  checks the wave against that capture's `k6.audio.wave` when the capture has one.

A stage whose files are missing is skipped as `skipped: missing <file>`. Capture exits as follows:

- 0 when all stages are ok.
- 2 when some stages were skipped.
- 1 when a stage failed or an in-run check failed.

## Gate

`compare` pairs ops by name and occurrence. It then checks each pair:

- Every role in `ins` and in `outs` must exist on both sides.
- Each role must have the same dtype and shape, and equal bits. A same sha256 counts as equal; otherwise it does an
  element-wise bitwise comparison and reports the unequal count, the first mismatch (index and both values) and the
  maximum absolute difference.
- Shared attrs must agree, with floats compared at float32. Attrs named `note_*` are skipped.
- Any `stage` note that is not ok, on either side, fails the stage's ops.

The gate passes only with zero unequal values and zero unmatched ops or roles. Otherwise it exits 1.

The audio chain has two more rules:

- **Full-chain qualification** needs all four chain ops on both sides with `chained: true`, each op's input equal by
  sha256 to the previous op's output on the same side (`latent_in.latent → mel.latent`, `mel.mel → wave.mel`,
  `wave.wave → pcm16.wave`), and every role of every chain op bit-equal. A chain op missing on either side, while
  any chain op is present, is a failure.
- **Alone checks.** An op with `chained: false` on either side (D3's per-job exporter seeds job B with its own mel)
  is an alone check of that operator. It is still compared role by role and the report labels it per op, but it
  never counts toward chain qualification. A chain that holds only alone checks fails the exit code unless
  `--allow-alone` is given, and the summary's `audio_chain` says why. `chained` itself is not compared as an attr.

Every compare summary carries `qualification: unqualified` and the oracle prefix pins from the oracle's `oracle` note.
