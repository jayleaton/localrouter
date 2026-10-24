# M6: image edits for Qwen-Image 2.1, findings and design

Text + 1 to 5 reference images -> image. Same method as M2 to M4: a Python twin on LocalRouter's own kernels, gated
against the published reference; the Zig engine bit-exact with the twin op by op (`localrouter check qwen-edit`); the
GPU runs happen later, in a granted window (`tools/spark/image-edit-window.sh`).

## 1. Does Qwen-Image 2.1 itself edit?

**Yes. Editing is the same checkpoint, not a separate model.** What was read, and where:

- The model card (`Qwen/Qwen-Image-2.1`, README): "a unified text-to-image generation and image editing model", up to
  10 reference images, transparent (RGBA) in and out. Its Quick Start edit call is
  `QwenImage21Pipeline(prompt=..., image=PIL)`: the same pipeline class and the same weights as text to image.
- diffusers 0.41 `pipelines/qwenimage21/pipeline_qwenimage21.py`: `image=` is a list of condition images, "encoded by
  the text encoder as vision context and by the VAE into latent tokens". The transformer
  (`transformer_qwenimage21.py`) takes `img_shapes` and `img_mask` and splices the reference latents into the text
  stream at the positions the vision-language encoder reserved for them.
- ComfyUI 0.37.0: `comfy_extras/nodes_qwen.py: TextEncodeQwenImage21` (`image_slots` and `reference_latents`),
  `comfy/ldm/qwen_image21/model.py: build_sequence`, `comfy/text_encoders/qwen_image21.py`.
- TensorFold's `tfimage/qwen_image21.py` already takes `ref_latents` and `image_slots` (its README: the edit forward is
  verified, not timed end to end). The repo's DiT twin keeps its `image_ids` parity rule for exactly this.

The older "Qwen-Image-Edit" line (`QwenImageEditPipeline`, `-2509`, `-2511`) is a different 60-layer, 24-head DiT with
a Qwen2.5-VL encoder and its own `pack`. It is **not** what the pack converts and it is not ported. If it is ever
wanted it is a separate engine entry (`qwen-image-edit`) with its own pack; nothing here blocks that.

**Weights.** Nothing new to download. The 31 GB `Qwen/Qwen-Image-2.1` snapshot the packs were built from already has
both missing pieces; the existing packs simply dropped them:

| Piece | Where in the snapshot | Size | Pack today | Pack after M6 |
| --- | --- | --- | --- | --- |
| Qwen3-VL vision tower (27 ViT blocks, hidden 1152, 16 heads x 72, 3 deepstack mergers, final merger to 4096) | `text_encoder/model-00001-of-00004.safetensors`, keys `model.visual.*` | about 0.58 B parameters, 1.15 GB bf16 | `te/` has the language model only | `te/` gains `vis.*` |
| VAE encoder (conv_in 4->96, 5 residual down blocks, mid block with attention, conv_out 768->128) and `quant_conv` | `vae/diffusion_pytorch_model.safetensors`, keys `encoder.*`, `quant_conv.*` | 78.7 M parameters, 0.16 GB bf16 | `vae/` has the decoder only | `vae/` gains `vae.encoder.*` |
| Tokenizer, `<|vision_start|>` etc. | `processor/` | | already in `te/` | unchanged |
| DiT | `transformer/` | | `nvfp4/`, `fp8s/` | **unchanged** (`img_in` is reused for reference latents) |

The extra pack bytes are about 1.3 GB, loaded only when the pack has them (`edit.json`, section 6). A text-only
install keeps working and does not declare `image_edit`.

## 2. What an edit does (reference behaviour, diffusers 0.41 unless noted)

Inputs: prompt, images `I_1 .. I_n` (n <= 5 in LocalRouter's API), optional size, seed, steps.

1. **Target size.** Given in the request, else from the last image: `calculate_dimensions(1024*1024, w/h)` =
   `round(sqrt(1048576 * r) / 32) * 32` wide, height `round(w / r / 32) * 32` (ComfyUI's node uses the first image;
   diffusers, the twin and the engine use the last). Sides are floored to multiples of 32 for the pipeline; LocalRouter
   accepts 16 and keeps the t2i behaviour for explicit sizes.
2. **Per reference** (`img.convert("RGBA")` first, so every image has an alpha channel, 255 when it had none):
   `(w_i, h_i) = calculate_dimensions(1024*1024, w/h)` of that image, then PIL `resize((w_i, h_i), LANCZOS)`. PIL
   premultiplies RGBA for every filter but NEAREST (`RGBA -> RGBa`, resample, `RGBa -> RGBA`) and returns a copy when
   the size is unchanged. One resized image feeds both paths below.
3. **Vision path.** RGBA composited over white (`Image.paste` with the alpha as mask), then Qwen3-VL's image
   processor: `smart_resize` to multiples of 32 (a no-op here: sides are multiples of 32 and 65536 <= pixels <=
   16777216 hold for ~1 MP; a bicubic resize to the same size is the identity), `x / 255`, `(x - 0.5) / 0.5` in fp32,
   patches of 16x16 with 2 identical temporal slots, merge-block order. Grid `(1, h_i/16, w_i/16)`.
4. **Vision tower** (`Qwen3VLVisionModel`, bf16): patch embed (a GEMM, K = 3*2*16*16 = 1536) + position embedding
   (the 48x48 learned table bilinearly resampled to the image grid, align_corners, merge-block order) -> 27 blocks of
   LayerNorm / attention (qkv with bias, 2D rotary on the 72-wide head, bidirectional inside one image) / LayerNorm /
   GELU(tanh) MLP; taps after blocks 8, 16, 24 go through the three deepstack mergers (LayerNorm over 4608, fc1, GELU,
   fc2); the last hidden state through the merger (LayerNorm over 1152 *then* reshape to 4608). Output: `n_i =
   h_i*w_i/1024` rows of 4096 for each of the 4 outputs (embeds + 3 deepstack features).
5. **Prompt** (note: not the same template as t2i). With `P_k = "<image{k}><|vision_start|><|image_pad|><|vision_end|>"`:
   `<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n` + `P_1 P_2 ...`
   (joined by a space) + prompt + `<|im_end|>\n<|im_start|>assistant\n`. Each `<|image_pad|>` is expanded to `n_i`
   copies. The system turn is the same as t2i, so `drop` is unchanged.
6. **Text encoder** (Qwen3-VL 8B language model, 36 layers): the embedding rows at pad positions are replaced by the
   vision embeds; positions are 3D mrope (below); after layers 0, 1, 2 the deepstack features are *added* to the
   hidden states at pad positions; the output is the last layer before the final norm (as t2i). Then: drop the first
   `drop` rows (system turn), **delete the pad rows** and record for each image its `slot` = number of kept rows before
   it. (diffusers keeps them and replaces them in the DiT; ComfyUI deletes them. Same math, and ours follows ComfyUI's
   `image_slots` because TensorFold's DiT and the existing context buffers want it.)
7. **VAE encode** (per reference, same pixels as the vision path before compositing, 4 channels): `x/255` fp32,
   `2x - 1`, bf16; encoder (single frame); `quant_conv`; the mean (first 64 of 128 channels, `latent_dist.mode()`);
   `(z - mean) / std` in bf16 per channel. Result `[1, 64, h_i/16, w_i/16]`, the reference latent.
8. **DiT** (`QwenImage21Transformer2DModel`, `causal_condition`): the sequence is
   `[text_0, ref_1, text_1, ..., ref_n, text_n, target]`, where `text_k` are the kept text rows between slots and
   `ref_k` the reference latents through `img_in` ((h_i/16)*(w_i/16) rows each for pixel dimensions h_i,w_i); target is the noisy latent. Text and
   reference rows modulate from t = 0 (they are step-independent), the target rows from the step's sigma. Attention is
   block-causal: text rows are causal and see everything before them, every image block sees everything before its
   end including itself, the target sees all. So the **prefix** (everything but the target) is computed once per
   (prompt, images, size) and its K/V kept, exactly as t2i keeps the text prefix, and each step runs the target rows
   only. RoPE ids: text rows `pos` on all three axes, advancing 1 a row; a reference block sits at frame `pos` with
   centred rows / columns (`image_ids`, half a token of shift where the parity differs from the target's) and then
   advances `pos` by `max(h_i, w_i)`; the target uses the position after the last text row.
9. **Sampling and decode** are unchanged: shift `mu` from the *target's* token count only, Euler on the portable noise,
   VAE decode (4-channel RGBA out). No CFG (the 2.1 pipeline's `true_cfg_scale` is off, as in t2i).

## 3. What is new for the engine

| Part | Twin (Python) | Zig / CUDA | Reuse |
| --- | --- | --- | --- |
| Image decode | PIL | Internal fallback `src/media/png_decode_edit.zig`: RGB8 plus alpha-preserving RGBA8, non-interlaced 8-bit RGB/RGBA PNG; replace with the shared `src/media/png_decode.zig` interface when it lands in main | `std.compress.flate` |
| Resize and composite | PIL | Lanczos-style host preparation in `src/engines/qwen_image/edit_prepare.zig`: 22-bit coefficients, premultiplied RGBA, white composite, twin patch order | no shared resize kernel |
| Vision tower | `vision.py` | `vision.zig`: GEMM + bias (`gemm.cu`), new `stk_vis_*` kernels in `vision.cu` (LayerNorm with bias, exact GELU, 2D rope, position gather, head padding 72 -> 128), attention `pattn2` with scale `1/sqrt(72)` on zero-padded heads | `stk_gemm_bf16`, `stk_gelu_tanh`, `stk_add`, `pattn2_kernel` |
| mrope text encoder | `te.py` `forward_mm` | `te.zig`: tables from 3D positions (host), row scatter, deepstack adds | every kernel of the t2i encoder, unchanged |
| VAE encoder | `vae_enc.py` | `vae_enc.zig`; new `stk_avg_down`, `stk_chan_normalize`, `stk_from_u8_rgba` in `vae_enc.cu` | im2col + GEMM convs, RMSNorm, silu, add, the mid-block attention (all `vae.zig`) |
| DiT with references | `dit.py` `_build_prefix(refs, slots)` | `dit.zig`: segmented prefix, `next_pos` | the t2i blocks and step; `img_in` |
| Orchestration | `qwen_image.py` `generate_edit` | `pipeline.zig` `edit`; tool `image_edit` capability | |

Everything that is not new arithmetic is the t2i code path with a longer sequence.

### Memory and cost (the tool's `needs`)

A 1 MP reference is 4096 latent tokens plus 1024 pad rows in the text encoder. The prefix grows to
`text + sum(h_i*w_i)` rows: with `r` references at 1 MP, about `1024 + 4096 r` rows, and the kept K/V is
`2 * 32 layers * rows * 4096 * 2 B` = 0.5 MiB per row (1 reference: 2.5 GiB; 5 references: 10.5 GiB, on top of the
DiT scratch, which scales with the same rows). The tool takes the option `edit_refs` (default 3) and sizes the DiT
scratch for it; a request with more references than `edit_refs` is refused with a message. Every step attends over the
whole prefix, so a step with `r` references costs about `(1 + r)` times a t2i step's attention. The vision tower runs
once per request (ViT over `4096 r` patches, about 1.4 TFLOP per 1 MP image).

## 4. How it is gated

| Level | What | Where |
| --- | --- | --- |
| L0 | twin logic against the published code: sizes, PIL pipeline, tokens and slots, mrope positions, patchify, position-embedding taps, vision rope, against local ComfyUI sequence/mrope methods and transformers vision plans; exact Zig/Pillow preprocessing parity remains unproven | Deferred: `python -m stk_twin.test_edit --comfy COMFY`, `python -m stk_twin.test_vision`; not run on the dev box |
| L1 | twin kernels against torch ops (`test_edit --gpu`, `test_vision --gpu`, `test_vae_enc`): each new `.cu` op against its formula; vision tower, multimodal encoder and VAE encoder against `transformers` / `diffusers` (cosine, relative L2, with the floor of two backends) | GPU window |
| L2 | `test_edit --gpu`: tiny reference DiT / multimodal TE; `python -m stk_twin.edit_gate`: full bf16 twin image against fresh, unpatched diffusers pipeline (PSNR). Full-checkpoint per-step DiT gate still pending | GPU window |
| L3 | `capture-edit` -> `localrouter check qwen-edit PACKS CAPTURE`: vision, text encoder, VAE encoder, prefix and a step, each op alone and chained, byte for byte | GPU window (sm_120 then sm_121) |
| L4 | end to end: the twin's final pixels against the engine's, then timing and the memory peak | GPU window |

Declared deviations from the diffusers pipeline: the new kernels replace cuDNN / SDPA / torch ops with fixed-order
fp32 kernels, so L1 gates are numeric (cosine, relative L2) and L3 is bitwise only against the twin. The current
fallback decoder accepts non-interlaced 8-bit RGB/RGBA PNG only. JPEG, WebP, interlaced PNG, and other PNG color/depth
variants are refused. Host preprocessing follows Pillow's Lanczos approach, but exact pixel agreement with Pillow has
not been established. L3 and L4 use PNG inputs.

## 5. Order of work and status

1. Twin reference and deferred GPU gates are in `tools/twin/stk_twin`.
2. Pack conversion and capture entry points are `pack_edit.py` and `capture_edit.py`.
3. Zig edit execution is in `src/engines/qwen_image` and `src/engines/qwen_image_tool.zig`; `src/cli/qwen_edit.zig`
   replays captures alone and chained and compares final pixels.
4. `tools/spark/image-edit-window.sh` runs the local GPU and reference gates. CUDA builds and GPU checks still need that window.

Status is at the end of this file.

## 6. Pack layout and capability

- `te/weights.safetensors` + `vis.*` (bf16; names in `pack_edit.py`); `te/manifest.json` gains
  `"vision": {...config, "template_edit": ...}`.
- `vae/weights.safetensors` + `vae.encoder.*`, `vae.quant_conv.*` (all 128 moment channels; the twin selects the first 64 outputs),
  `vae.enc_latents_*`; `vae/manifest.json` gains an `encoder` member with its own convs / norms / downs maps; top-level decoder maps stay separate.
- `<weights>/edit.json`: `{"format": "stk-edit/1", "refs_max": 5, "manifest_sha256": {"te": ..., "vae": ...}}`, written by the pack step only when both parts
  were written. The tool declares `image_edit` only when this marker matches both current manifests and the pack supports
  the requested reference count. Requests resolve image file names relative to the job directory.


## 7. Offline implementation checkpoint

The twin and Zig edit paths are implemented: vision, multimodal text, VAE encoding, reference-latent prefixes and RoPE,
sampling, decode, tool capability gating, and edit capture replay. Reference image names resolve in the job directory.
The internal PNG fallback returns RGB8 and RGBA8 so transparent inputs reach the VAE encoder unchanged when resize is
an identity. It handles non-interlaced 8-bit RGB/RGBA PNG only and is intended to be replaced by the shared i2v decoder
once that decoder lands in main. JPEG and WebP are refused. The host preparation matches the twin's dimensions,
compositing and patch layout; exact Pillow Lanczos pixel parity remains unproven.

Deferred tests: `test_edit` checks published ComfyUI sequence construction and mrope positions, then tiny random DiT
outputs against TensorFold and multimodal text outputs against transformers with `--gpu`; `test_vision` checks host
plans, kernels and a tiny tower against two reference attention backends; `test_vae_enc` checks encoder outputs against
the unpatched diffusers VAE. The new host preparation has unit tests for dimensions, identity transparency, and constant
image resizing. The prescribed CPU Zig build/test wrapper passed for the execution, tool/PNG, and replay checkpoints.
All GPU, CUDA-kernel, twin-reference, and full edit checks remain unrun. Neither model loading nor complete edit generation has run.

Vision learned-position interpolation deliberately follows the local ComfyUI bf16 coefficient/product/add boundaries.
The locally installed newer transformers implementation uses fp32 coefficients and a reduction before the final cast;
whole-tower comparison is numerical, while engine replay must match the twin bitwise. Merger GELU is exact erf GELU;
vision block MLP uses tanh GELU. New kernels preserve these distinct operations.

Corrections to the earlier design: reference latents include batch dimension; DiT row counts use latent spatial sizes,
not pixel sizes; published-reference CPU gates remain deferred; the local fallback decoder is named separately from the shared
i2v decoder; and the interpolation version difference is recorded.


Pack conversion: `python -m stk_twin pack --model MODEL --precision edit --out PACKS` augments existing `PACKS/te`
and `PACKS/vae` packs (create those with `--precision te` / `vae` first). It verifies all base tensor hashes before
rewriting, flattens the vision patch projection to [1152,1536], freezes vision rotary frequencies, records separate encoder
convolution/normalization/downsample metadata, and writes `edit.json` only after both pack writes succeed. Conversion
has not run. Full quant_conv preserves the twin's exact output shape/weight naming; posterior variance computation
is still skipped. `refs_max=5` matches the local request limit.

Capture entry: `python -m stk_twin.capture_edit --model MODEL --pack PACKS/nvfp4 --image REF.png --prompt '...'`
`--steps 25 --seed 7 --out CAPTURE` (repeat `--image` up to five times; optional `--size WIDTHxHEIGHT`). It requires
all own-kernel backends, records vision/text/encoder and the first DiT prefix/step, final latents and decoder, and
keeps canonical input PNGs plus final pixels. It refuses a nonempty output directory so op streams cannot append
across requests. Capture and replay have not run.


Run `tools/spark/image-edit-window.sh REF1.png [REF2.png ... REF5.png]` on the GPU window after local setup. It checks
the local model, pack manifests, inputs, ComfyUI reference tree, CUDA compiler, and Zig executable without downloading
dependencies; then it runs twin tests, pack conversion, capture, the full twin reference gate, the Zig build, and
`check qwen-edit`. Set `SIZE`, `STEPS`, `PROMPT`, `SEED`, `ONLY`, `SKIP_PACK`, or `SKIP_CAPTURE` to select request and
window behavior. The full check must pass alone and chained with final pixels matching. Exercise both sm_120 and sm_121,
1 to 5 references, mixed dimensions, transparent inputs, repeated insertion slots, and explicit target sizes. All these
GPU/CUDA/reference checks remain unrun. PIL-exact preprocessing remains unproven and needs an independent comparison.


Encoder metadata is separate from the existing decoder maps. The decoder loader iterates top-level convs/norms;
merging encoder entries into those maps would load unused edit weights on text-only requests and reject the encoder
conv_in's K=36 (the twin pads that to 40). `vae_enc.zig` reads the `encoder` member and reuses decoder operations with
the same weight padding as the twin.

## Status (2026-10-24, DGX-01, GB10 sm_121)

Qwen-Image 2.1 Turbo, FP8 DiT pack, one 1024x1024 reference, 8 steps, seed 7, the edit pack of section 6:

| Gate | Result |
| --- | --- |
| L1 vision tower and kernels (`test_vision --gpu`) | pass; the merger GELU against `F.gelu` max abs 2.4e-7 |
| L3 `localrouter check qwen-edit` | 2164 ops equal alone and chained, sigmas equal, final pixels and the saved PNG equal |
| L2/L4 the twin against diffusers' `QwenImage21Pipeline` (`edit_gate`) | 38.64 dB |
| Through the API (a daemon with the pack) | edits of one input 11 to 12.5 s, 1024x576 7.2 s, two inputs 15.9 s; a mask keeps every opaque pixel exactly |

The merger GELU uses `kernels/cuda/common/erf.cuh` (correctly rounded operations only) instead of libdevice's `erff`,
which differs between CUDA 13.0 (the box's nvcc, which builds the engine) and 13.3 (the twin's container): with `erff`
four GELU ops differed by a few bytes in 66,709 and the chained replay diverged from there. The same header serves the
H3 vision tower. References are resized to about one megapixel with sides up to 2048 (4:1), independent of the
output's `max_side`; the worker's memory is 38.2 / 41.3 / 44.1 GiB at `edit_refs` 1 / 2 / 3 (31.2 GiB without the pack).
