# H3-Image: what ComfyUI e9afbccc does, and what LocalRouter builds

MiniMax-H3-Image (`Comfy-Org/MiniMax-H3-Image`) is a second checkpoint of the H3 DiT with no audio stream, six PDD output heads and its own image
VAE decoder. All paths below are under ComfyUI at `e9afbccc2669677076feb08e52658f129cd8c12b`, the revision #157 pinned (`COMFYUI_REV` in
`deploy/strix-halo/comfyui/install.sh`). The workflows are #157's `deploy/strix-halo/comfyui/minimax-h3-image-{t2i,edit,edit-unmasked}.json`.
The weights' headers were read (`minimax_h3_image_pruned_bf16`, `minimax_h3_image_vae_fp16`), but no weights. The video model is in
docs/dev/M7.md and H3-I2V.md; this file covers only what the image model changes. The plan is docs/dev/ROCM.md (PRs 1.5 to 1.7).

## 1. Findings (read from ComfyUI's code; none of it run, there is no GPU here)

**1. Detection and configuration.** A checkpoint with `video_patch_proj` and `condition_proj` but no `audio_patch_proj` is `minimax_h3_image`
(`comfy/model_detection.py:401-404`). The config comes from the safetensors metadata: `pdd_steps` sizes the head bank
(`model_detection.py:409-411`), and the whole `transformer` dict is merged into the model's kwargs (`:433-434`), which is how `pdd_sigmas`
reaches `MiniMaxH3ImageModel(pdd_sigmas, ...)` (`comfy/ldm/minimax/model.py:833-835`). The bf16 file's `__metadata__.config` is
`{"transformer": {"pdd_steps": 6, "pdd_sigmas": [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]}}`.
(#157's GGUF lacks this metadata. Its loader node injects the same dict: `deploy/strix-halo/comfyui/nodes/nodes_minimax_h3_image_gguf.py:7-8`.)
- The header gives 50 blocks, and every shape equals the video DiT's: `qkv_proj` [21504, 5376], `out_proj` [5376, 7168] (56 x 128),
  `fc1` [28672, 5376], `fc2` [5376, 14336]. Also 2 refiner blocks, `condition_proj` [5376, 5120] + bias, and `adaln_t_table` F32 [1025, 8].
  The adaLN linears are stored F16: `blocks.*.adaln_proj.linear.weight` [96768 = 6 x 3 x 5376, 8], and the final layer's is [10752, 8].
  `rope.inv_freq` is F32 [16]. `video_patch_proj` is F32 [5376, 96] + bias, and `final_layer.video_out` is F32 [576, 5376] + bias. There
  are no `audio_*` tensors. Every block also has `attn.comfy_attention.config`, U8 [37], a JSON attention selector that `ComfyAttention`
  decodes (`comfy/ldm/modules/attention.py:89-106`: `comfy_kitchen_int8` or `comfy_kitchen_sol` when kitchen reports it available on the
  device, any other method ignored with a warning; otherwise the default attention). Its 37 bytes are not in the header, so **which
  method it names is unread** (section 3).
- `latents_dim` = 576 / (4 x 6) = 24 (`model_detection.py:412`). The latent format is `MiniMaxH3Video`: 24 channels, /16 spatially, scale 1.0
  (`comfy/latent_formats.py:622-627`), so `process_latent_in/out` are the identity. `MiniMaxH3Image` is a `ModelType.FLOW` model with its
  own sampling object and no `shift` (`comfy/supported_models.py:1015-1024`, `comfy/model_base.py:2430-2433`).

**2. The conditioning node, `MiniMaxH3ImageConditioning`** (`comfy_extras/nodes_minimax_h3.py:711-746`). Its inputs are `clip`, an
optional `vae`, `prompt`, `width` and `height` (0 = derive, step 32), and `images` (autogrow `image_1` .. `image_100`, where `image_1` is
the picture to edit). Its outputs are the positive conditioning and `{"samples": zeros[1, 24, 1, height/16, width/16]}` (`:745`): one
latent frame, no audio, no NestedTensor.
- Size: when `width` or `height` is 0 it comes from `image_1` after its VAE scaling (below), or 2048 x 2048 without images (`IMAGE_SIZE`,
  `:38, 742-744`; the tooltip says the model "is tuned for 2048x2048 pixels of area", `:726`). There is no pixel cap and no rounding beyond
  the widget's step of 32. #157's text-to-image passes 1024 x 1024.
- Text to image: `clip.tokenize(prompt, images=[], picture_label="Picture %d: ")` (`:739`), so the tokens are **the raw prompt**, with no
  chat template, BOS or system prompt (`comfy/text_encoders/minimax.py:187-241`). An empty prompt becomes the single pad token 151643 (`:237-238`).
- The text encoder is the H3 one: the CLIP file `qwen3vl_32b_minimax_h3_int8_convrot` is detected as `QWEN3VL_32B` (`comfy/sd.py:1851-1853`)
  and loads `MiniMaxH3TEModel` / `MiniMaxH3Tokenizer` (`sd.py:2140-2142`). Its output is the raw residual after LM layer 50 with no final
  norm (`minimax.py:16-18`), as in M7. (`PH3ImageQwen3VL` in `minimax.py:120-131`, a Qwen3-VL-4B with a five-layer text connector, belongs
  to a different model, "p-h3-image", with its own node `PH3ImageConditioning`, `nodes_minimax_h3.py:749-788`. It is not this checkpoint.)
- Negative: #157's workflows feed `ConditioningZeroOut` of the positive as the negative at `cfg 1.0`. With cfg 1 ComfyUI never evaluates the
  negative (`comfy/samplers.py:610`, the cfg1 optimization), so **one DiT forward a step**.

**3. The DiT's packed sequence: `[text | reference images | target image]`** (`model.py:826-893`).
- Layout (`PackedLayout(text_len, 1, h, w, 0, refs=...)`, `model_base.py:2435-2443`; `model.py:383-504`, with `audio_t` 0, so there is no
  audio segment and nothing replaces it). The text rows take positions `(i, 0, 0)`. Reference image j (0-based) takes time `text_len + j` and
  its own area-normalized grid. The target image takes time `text_len + n_refs` and the target grid (`:401-403, 428-443, 481-486`).
  `_frame_grid(h, w)` gives the 2x2-patch rows' `(h, w)` as `(k (r / n) + (1 - r) / 2) * 32`, with r = side / sqrt(h w) and n = side / 2,
  in float64 (`model.py:77-99`). The RoPE is the video model's: positions cast to fp32, times `inv_freq` [16] per axis, `[t | h | w]` (48),
  repeated to 96, rotating 96 of 128 dims by kitchen's split-half `rms_rope` (`model.py:558-565, 185-199`). Text-to-image has
  `text_len + (H/32)(W/32)` rows.
- Timesteps (`model.py:870-875`): `sigma = timestep / 1000` and `t = 1 - sigma`. The text and the target are at `t`; reference rows are at
  `max(t, 0.999)`. **No flow shift and no audio timestep.** `unique_t` has one value (text to image) or two (edits). The time embedding is
  the `adaln_t_table` lerp at `t` (`model.py:639-646`). Mod rows are `t_row * 3 + tag`, with tags text 1 and `video`/`ref_img` 0
  (`model.py:239, 877-884`). The text span is split into tag runs from the TE's tags (vision tokens and their two markers 0, the rest 1:
  `minimax.py:69-76`, `model.py:242-251`).
- Embedding (`_embed_and_pack`, `model.py:595-637`, with `audio_x` None): the target latent is patchified 1x2x2 to 96 values a row
  (`model.py:49-56`, order `c, pt, ph, pw`) in fp32. Reference rows go before it (`img_update` False). `video_patch_proj` runs in fp32 and
  rounds to the compute dtype. The text goes through `condition_proj` + the 2 refiner blocks + `final_norm`, once per sampling
  (`model_base.py:2257-2262`, `model.py:552-556, 273-299`). Blocks: the video model's `DiTBlock` (`model.py:302-321`).
- Output: the last segment (the target) through `FinalLayer` with the target's `t` row (`model.py:890-893`), unpatchified, **negated**
  (`-out`). The `CONST` model turns it into `denoised = x - out * sigma` (`comfy/model_sampling.py:90-92`).

**4. PDD heads and the schedule.**
- `FinalLayer` (`model.py:324-367`): RMSNorm, then adaLN (expand 2, 1 modality), then `_modulate` in the stored dtype, cast to fp32
  (`:344-347`). The six heads are stacked in `video_out`: rows `[96 k, 96 k + 96)` are head k. Head 0 is a full head; heads 1..5 are offsets
  from it. A step from grid index `start` to `stop` uses weight `rows[0] + sum_k w_k rows[k]` and the same for the bias, over k in
  `[max(start, 1), stop)`, with `w_k = dt_k / sum dt` (dt = grid[k] - grid[k+1], over `[start, stop)`), in fp32 (`model.py:370-380`).
  `start` and `stop` are the grid points nearest the step's sigma and the next sigma in the sampler's schedule (`:356-362`; the sampler
  passes `sample_sigmas` through transformer_options, `:891`). For one-interval steps the step's head is `rows[0]` (k = 0) or
  `rows[0] + rows[k]`.
- Sampling object (`comfy/ldm/minimax/sampling.py:6-32`): `sigma(timestep)` is linear in the grid index of the ascending grid (lerp between
  neighbours), so ComfyUI's 1000-entry `sigmas` table (`model_sampling.py:308-312`) follows `pdd_sigmas` piecewise linearly. The flow shift is
  unused (`sampling.py:9-10`).
- #157 runs `KSampler euler / simple` (`comfy/samplers.py:645-652`; `comfy/k_diffusion/sampling.py:190-212`). The schedule below is
  arithmetic, recomputed by hand from those lines in fp32, not run:
  - 6 steps: `[1.0, 0.967654, 0.887708, 0.75, 0.533643, 0.23261, 0]`. The table's entries are sampled at index `-(1 + int(x * 1000/6))`, so
    the step sigmas sit **near** the grid (`0.967654` vs `0.967524`), not on it. Heads (start, stop): (0,1), (1,2) .. (5,6), one each.
  - 30 steps (#157's whole-scene edit): each grid interval is split into about 5 sub-steps, and each sub-step uses the head of the grid
    interval it falls in, by the nearest-point rule: heads 0, 0, 0, 1 x 5, 2 x 5, 3 x 5, 4 x 5, 5 x 7.
- Initial noise: `torch.randn` of the latent shape, fp32 on the CPU, from `torch.manual_seed(seed)` (`comfy/sample.py:22-36, 9-11`), then
  `sigma * noise + (1 - sigma) * 0` (`model_sampling.py:94-97`). With cfg 1 and euler there is no other randomness in sampling.
- Versus the video model: no shift (12 / 3), no audio stream, no `res_multistep`, no Turbo LoRA. The heads are distilled on `pdd_sigmas`, not
  on the shifted uniform grid `pdd_grid` (`model.py:44-46`), and the steps follow the stock schedulers over the PDD table.

**5. The image VAE** (`minimax_h3_image_vae_fp16`, 415 tensors; `comfy/sd.py:1053-1087`, `comfy/ldm/minimax/vae.py:1016-1056`).
- Detection: `decoder.proj_out` [768, 1024] with `x_embedder` [1024, 24] gives `MiniMaxH3ImageVAE(num_layers=24)` with `latents_mean/std`
  set to the H3 tables (`sd.py:1071-1077`, `vae.py:25-41`). The VAE runs in fp16 (`working_dtypes` `[float16, float32]`, `sd.py:1093`), and
  `process_input` is `x * 2 - 1` (`sd.py:523`).
- Encoder: the video VAE's `EncoderFCN3D` + `quant_conv` [48, 48, 1, 1, 1], the header's `encoder.*` (`vae.py:614-626, 191`). For one frame
  it is the 2-D network H3-I2V.md describes, with `_normalize_pixels` (ImageNet mean / std after `(x + 1) / 2`, `vae.py:654-655`) and 256-px
  tiles with blended overlaps (`vae.py:735-765`). **Unlike I2V's mean-only latent, `H3ImageVAE.encode` samples the posterior:**
  `mean + exp(0.5 clamp(logvar, -30, 20)) * noise`, where noise is `torch.randn` on a CPU generator seeded **42** (`IMAGE_ENCODE_SEED`)
  whatever the request's seed. The result is then normalized by `(z - latents_mean) / latents_std`, in fp32 (`vae.py:22-23, 1022-1028`).
- Decoder (`vae.py:1045-1056, 378-428, 351-375, 284-349`): `post_quant_conv` is a 2-D 1x1 conv [24, 24] + bias. Then `x_embedder` 24 -> 1024
  per latent pixel. Then 4 `register_tokens` and one **zero** token are appended (`:414`; `mask_token` is loaded but unused, `:392-393`).
  Then 24 `TransformerBlock`s, each `x += to_out(attn(RMSNorm1(x))) * scale1` and `x += w2(silu(g) * u) * scale2`, where `[g | u] =
  w1(RMSNorm2(x))` (8192 = 2 x 4096, gate first, `comfy/ops.py:1046-1048`) and the residuals are `addcmul` (`ops.py:1085-1086`). RMSNorms are
  affine with eps 1e-5. Attention has 16 heads x 64 on `to_qkv` [3072, 1024] + bias. Per head, Q and K are RMS-normalized (unit weight,
  eps 1e-5) and rotated by kitchen's split-half RoPE over 32 of the 64 dims (`rope_dim_ratio` 0.5, 2 axes (h, w), 8 frequencies an axis,
  `inv_freq = 100^-(0, 0.125, .., 0.875)`, angle `2 pi * coord * inv_freq`, where coord is `2 (k + 0.5) / n - 1` over the latent grid,
  `vae.py:252-281`; the 5 suffix tokens sit at coordinate 0, i.e. no rotation). Then plain fp16 attention (`optimized_attention`) and
  `nan_to_num` (`vae.py:340-348`). Then `norm_out` (LayerNorm affine, eps 1e-5), `proj_out` 1024 -> 768 = 3 x 16 x 16
  (`c, py, px` order, `vae.py:246-249`), the suffix tokens dropped. Pixels are `clamp(out * imagenet_std + imagenet_mean, 0, 1)` in fp32
  (`vae.py:657-660`). The decoded latent is first un-normalized: `z * std + mean` (`vae.py:992-996`).
- Decode tiling: `tiled_decode` cuts 256-px tiles (16 x 16 latents, so 256 + 5 tokens an attention) with overlaps of at least 64 px
  (`vae.py:686-708, 776-815`). It blends each tile into the canvas with linear ramps in the tile's dtype (`vae.py:710-725`) and decodes up to 4
  tiles a call, depending on free memory (`vae.py:767-774`). For 1024 px a side the tiles start at 0, 192, 384, 576 and 768 (5 a side;
  arithmetic from `split_tiles`). **The decoder never sees the whole image**, so the result depends on the tiling.

**6. Edits** (`nodes_minimax_h3.py:692-746`, `minimax.py:187-241`, `model.py:567-580, 428-443`).
- Each input image `image_k` (sorted by k, first frame of its batch) is used twice:
  1. **Qwen** sees it resized so its short edge is exactly 1024 (`_size_short_edge(w, h, 1024, 1024)`, rounded to 32 a side, `:679-682`)
     through `_resize_to`, i.e. the node's 8-bit lanczos (PIL; the identity when the size already matches, `:685-689`, `comfy/utils.py:1130-1138`).
     Then Qwen3-VL's own preprocessing (`smart_resize` to multiples of 32, which is a no-op here; bilinear to the same size; `(x - 0.5) / 0.5`;
     the frame repeated for the temporal patch), with patch 16 and merge 2 (`comfy/text_encoders/qwen3vl.py:63-69`,
     `comfy/text_encoders/qwen_vl.py:9-21, 30-64`). The tokens are `"Picture k: "` (no angle brackets, unlike I2V's `<Picture 1>: `),
     `<|vision_start|>`, the vision embeddings ((H/32)(W/32) of them), `<|vision_end|>`, once per image in order, then the prompt
     (`minimax.py:8, 231-236`). MRoPE and deepstack are as in H3-I2V.md.
  2. **The VAE** sees it resized so its short edge lies in [1024, 2048] (`IMAGE_REF_VAE_MAX_SHORT_EDGE`, `:40-41, 737`), and encodes it as in
     5 (seed-42 posterior sample). Each encoded image becomes a **reference block** `{"kind": "image", "latent", "latent_h", "latent_w"}`
     (`:707`) in `minimax_refs`, not a `minimax_keyframes` cond row as in I2V.
- In the DiT, reference rows are the normalized latent patchified in fp32 with the condition noise augmentation
  `r = 0.999 r + 0.001 noise`, where noise is `torch.randn(r.shape)` from a CPU generator seeded with **the sampler's seed**, restarted for
  every reference (`model.py:567-580`, `model_base.py:2280, 2285`). They are constant over the steps, at `max(t, 0.999)` with tag 0, at the
  positions in 3, and take no part in the output.
- Canvas: with `width`/`height` 0 the target is `image_1`'s VAE size. #157 passes the size of `ImageScaleToTotalPixels(area, megapixels,
  resolution_steps 32)` (`comfy_extras/nodes_post_processing.py:249-259`: `round(side * sqrt(MP * 2^20 / area) / 32) * 32`, torch `area`
  interpolation): 1.0 MP for masked edits (1024 x 1024 for a square input) and 2.0 MP for whole-scene ones (1440 x 1440). So the target, the
  VAE reference (short edge already within [1024, 2048]) and the image are one size; Qwen's copy is 1024 short edge (resized for 1440).
- **The mask never reaches the model.** The masked edit is a whole-frame edit. #157's worker binds the request's mask to `ImageCompositeMasked.mask`
  (node 14) as a `LoadImage` MASK output, `1 - alpha / 255` (1 where transparent, `nodes.py:1795-1797`; `tools/comfyui/worker.py:30-33`). The
  composite (`comfy_extras/nodes_mask.py:12-46, 100-104`) is `out = m * decoded + (1 - m) * dest`, in fp32, where `m` is the mask
  bilinear-resized (`F.interpolate`, align_corners False) to the decoded size, and `dest` is `ImageScale(lanczos, crop center)` of the scaled
  input to its own size (node 15). A same-size lanczos still goes through `np.clip(255 x).astype(uint8)` (truncation) and back, so `dest` is the
  input quantized to 8 bits. The PNG write rounds the result.
- Several references are supported (`image_2` .. `image_100`; #157 caps at 5); `<Picture n>` in the prompt names them. Whole-scene edits run the
  same graph without the composite (`-edit-unmasked.json`), at #157's 30 steps on the PDD table (4).

**7. What matters for the bit-exact work.**
- Random draws: the initial noise (seed, `[1, 24, 1, h, w]`), one condition draw a reference (the same seed, the reference's row shape), and
  **one fixed-seed (42) posterior draw a reference in the VAE encoder**. All three are CPU `torch.randn`. As in M7, the twin and engine replace them with LocalRouter's portable generator (declared substitution), the
  encoder's with seed 42 kept.
- Dtypes: the patch projection and the PDD heads (weights, bias, the dt mix) are fp32. The adaLN linears are stored fp16 and run at the
  compute dtype: ComfyUI casts them, but **the cast dtype was not traced**. The blocks are bf16. The VAE is fp16 throughout, with fp32
  normalization, posterior and finalize.
- ComfyUI-only arithmetic the twin does not copy: kitchen's fused fp16 `group_norm_silu_pad3d` and fp16-accumulate convs in the encoder
  (as I2V), the attention method named by `comfy_attention.config`, and the memory-dependent tile batching of the decoder. These are gated
  by cosine against ComfyUI, not by bits (ROCM.md "Baseline").

## 2. What LocalRouter builds (ROCM.md PRs 1.5-1.7)

| PR | Twin (`tools/twin/stk_twin/h3i/`) | Engine (`src/engines/minimax_h3/`) |
| --- | --- | --- |
| 1.5 | `dit.py`: the H3 twin's DiT with `audio=False`, `[text | refs | target]`, the PDD bank. `vae_image.py`: the 24-block decoder and its 256-px tiling. `sampler.py`: the PDD table, `simple` schedule, euler, head pick. `build.py`: the bf16 pack (linears as `weights.zig` `.bf16`: `{key}.weight`). Captures and the CPU structure gate at 256², 2 blocks against ComfyUI's module code | none |
| 1.6 | `generate.py --image` (text to image) | `dit.zig` image variant (no audio segment, one or two timestep rows, PDD head mix in fp32), `vae_image.zig` (the tile ViT parameterized by depth 24, width 1024, 16 heads, 2-D RoPE over half the head, 768 out), `sampler.zig` PDD schedule; the `minimax_h3_image` entry, `text_to_image`, licence "MiniMax H3 Image Community License" (hosted services to third parties need MiniMax's authorization) |
| 1.7 | `--edit`: references through `te_vision` and the VAE encoder (posterior, seed 42), cond noise, the composite | `image_edit`: `te_vision.zig` and `vae_video_enc.zig` reused, reference blocks in `layout.zig`, the posterior draw, the masked composite and the 8-bit lanczos in `image.zig` |

Reused unchanged: the 32B text encoder (`te32.zig`, layer 50), the vision tower and MRoPE (`te_vision.zig`), the refiner, `DiTBlock`, the
split-half RoPE, the VAE encoder. New: the PDD sampler and head mix, the image decoder's parameterization (the video decoder is 36 blocks of
2048, 32 heads, 3-D RoPE over 0.75 of the head), the posterior sample, reference blocks (I2V has cond rows), and the composite.

Declared substitutions, as in M7 and H3-I2V: LocalRouter's portable generator for the three draws; no `ImageScaleToTotalPixels` (`area`)
pre-scale, the request's size instead; the decoder's tiles decoded one at a time (no batching by free memory); the bf16 attention of ROCM.md
in place of whatever `comfy_attention.config` selects.

Gates (GPU, later; ROCM.md section 5): twin vs ComfyUI bf16 on gfx1151, cosine a step (G-baseline), decoded image LPIPS / DINO against #157's
outputs at the same seeds. Engine vs twin bit for bit (check names proposed, as in ROCM.md section 5): G1 `h3i-replay` / `h3-te` /
`h3i-vae`, G2 one step, G3 `h3i-e2e`, G4 `h3i-edit-e2e` (masked and whole scene, after the composite). Host tests (no GPU): the PDD schedule and head picks at 6 and 30 steps against the
arithmetic above, `PackedLayout` positions and segments against ComfyUI's function on CPU (as `test_vision_position`), the composite and the
8-bit lanczos against the twin's numpy path.

## 3. Open questions

- **The attention selector.** `blocks.*.attn.comfy_attention.config` (37 bytes) is in the bf16 file, but its value was not read (no
  weights here). If it names `comfy_kitchen_int8`, ComfyUI on CUDA runs INT8 attention for this checkpoint. On gfx1151 kitchen has no ROCm
  backend, so its availability check fails, `self.function` stays unset and the default attention runs (`attention.py:99-106`). Either way the twin's bf16 attention is a declared
  substitution, but the baseline must say which one ComfyUI ran. Read the 37 bytes with a ranged read on the gfx1151 box.
- **Tiling as reference.** Both encoder and decoder tile at 256 px, so ComfyUI's image depends on its tile plan. The twin should copy
  `split_tiles` exactly (it is pure integer arithmetic); whether the engine also tiles (ViT attention per 261-token tile) or decodes whole is
  a choice. Tiling is cheaper and matches the reference, so the default is to tile.
- The adaLN linear's compute dtype (stored fp16, model bf16): read `comfy.ops` cast rules for `adaln_dtype` / `**curve` (`model.py:546-550`)
  before the twin fixes it.
- `ImageScale(lanczos)` to the same size: confirm on a real run that node 15's 8-bit round trip is applied (`common_upscale` with
  `crop="center"` at equal size reaches `lanczos`, `utils.py:1140-1164`). If so, LocalRouter's composite background is the input
  truncated to 8 bits, i.e. the PNG's own values.
- Size policy for LocalRouter: the node defaults to 2048 x 2048 ("tuned for 2048x2048 pixels of area"), while #157 ran 1024 x 1024 for
  speed. Pick the tool's default and `max_pixels` against the measured time and GTT (PR 1.8).
- Text length with references: about 1,024 vision tokens per 1024² reference, plus the prompt, so `max_l` and the TE's limit must cover
  5 references (#157's cap) or the engine must refuse past a stated count.
