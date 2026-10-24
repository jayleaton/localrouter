# H3 image to video: what ComfyUI 0.37.0 does, and what LocalRouter builds

The checkpoint `minimax_h3_fl2va_pruned_*` is the first/last-frame model ("fl2va"), so image to video is one of its native tasks, not a
different network. All paths are under `stk-ref/ComfyUI-0.37.0/` and `stk-ref/minimax-h3-tensorfold-rtx/` (the TensorFold workflows).
Text to video is in `tools/twin/VIDEO-PORT.md` and `docs/dev/M7.md`; this file is only what a first frame changes.

## 1. Findings (read from ComfyUI's code; none of it run, there is no GPU here)

**The first frame is not a concat channel and there is no denoise mask.** The sampler, the noise, the schedule (`simple`, shift 12 / 3,
`res_multistep`, 8 steps with the Turbo LoRA, 20 without) and the audio carry are exactly text to video. A first frame changes three things:

1. **The text encoder sees the image** (`comfy_extras/nodes_minimax_h3.py:155`, `clip.tokenize(prompt, images=images)`). The token list is
   `"<Picture 1>: "`, `<|vision_start|>` (151652), the image's Qwen3-VL vision embeddings, `<|vision_end|>` (151653), then the prompt
   (`comfy/text_encoders/minimax.py:148-202`, the branch at 192-195). No chat template, no BOS. The vision tower is Qwen3-VL's
   (27 layers, hidden 1152, 16 heads, patch 16, temporal patch 2, merge 2, deepstack taps after layers 8, 16, 24; the 32B and the 8B share it,
   out_hidden 5120 for the 32B; `comfy/text_encoders/qwen3vl.py:14-28`, `qwen35.py:460-694`). The merged vision embeddings are inserted into the
   token embedding sequence (`comfy/sd1_clip.py:236-243`, inserted at the dict's position, so between the two markers), the LM runs
   **causal** with 3-axis interleaved MRoPE for the image span (`qwen_vl.py:91-114`, `llama.py:514-560`, rope_dims 24/20/20, theta 5e6) and
   adds the deepstack features after LM layers 0, 1 and 2 at the image positions (`llama.py:1006-1008`). The conditioning is, as in text to
   video, the raw residual after layer 50 (`minimax.py:15-16`). The tokens of the vision block, **including the two markers**, are tagged
   video modality (0) for the DiT's modulation, the rest tag 1 (`minimax.py:75-82`, `model.py:667,676-687`).
2. **The first frame's VAE latent is appended as extra rows of the packed sequence** (`nodes_minimax_h3.py:158-161`: `vae.encode(image)` stored as
   the conditioning key `minimax_keyframes` with `resolved_frame_index` 0; `model_base.py:2188-2192` hands it to the DiT as `cond_video_latents`).
   The packed order becomes `[text | cond rows | audio | video]` (`model.py:4-7, 372-394, 440-452`). The cond rows are one latent frame
   (T = 1) patchified like the target video (`patchify_video`, 1x2x2, 96 values per row, `model.py:42-49`), `(H/32) * (W/32)` rows:
   - positions: time `cursor + FRAME_RESCALE * resolved_frame_index = L` (L = text length; `FRAME_RESCALE = 5/3`; the target audio and video
     also start at `L`), height/width the target frame's grid (`_frame_grid`, `model.py:88-92`);
   - values (`_cond_video_rows`, `model.py:532-545`): the **normalised** latent (`vae.encode` already subtracts the mean and divides by the std;
     no `process_latent_in`) as fp32, then the condition noise augmentation `r = 0.999 * r + (1 - 0.999) * noise` (`aug =
     VISUAL_COND_TIMESTEP = 0.999`, `model.py:32`; `aug < 1.0` so it is applied) with `noise = randn(r.shape)` from a CPU generator seeded
     with the sampler's seed (`model_base.py:2202`, `payload["seed"]`); the rows are constant over the steps (same seed, every call);
   - timestep and modulation: the cond segment is video modality (tag 0) at `t = max(t_v, 0.999)` (`model.py:632-637, 667`), so the table of
     distinct timesteps has up to three entries `{t_v, t_a, 0.999}`; the cond rows are never updated (`img_update` False) and take no part in the
     output (`model.py:766-782` reads only the last two segments).
   - the embedding is the same fp32 `video_patch_proj` as the target video's, rounded to the compute dtype (`model.py:714`).
   - audio plays no role: there is no `audio_latent` in a first-frame keyframe; the audio is generated from the text and noise alone.
3. **Nothing else.** `denoise_mask` is None (`model_base.py:2206`; `scale_latent_inpaint`, `:2253`, is only reached with a mask), there is no
   `concat_cond` for this model (`concat_keys` is empty, `model_base.py:202,276`), the latent given to the sampler is the same empty AV latent
   (`_empty_av_latent`, `nodes_minimax_h3.py:84-90`), and `process_latent_in/out` are the audio scale only (`model_base.py:2163-2167`).
   The workflow's sparse attention (`sol-attn` with `sink_conditioning`) is a speed lever ComfyUI applies to the cond rows; the twin and engine
   attend densely (docs/dev/M7.md).

**Preprocessing of the image.**
- The node stretches the first frame to the canvas, `_resize(first_frame, width, height, "disabled")` (`nodes_minimax_h3.py:67-71, 146`):
  `common_upscale(..., "lanczos", crop="disabled")` (`comfy/utils.py:1097-1135`). `lanczos()` is PIL: `np.clip(255. * image, 0, 255).astype(uint8)`
  (a float32 multiply, then **truncation**), `Image.resize((w, h), Image.Resampling.LANCZOS)`, then `/ 255.0` to float32. PIL returns a copy
  when the size is unchanged. Pixels are then RGB in [0, 1] (alpha dropped, `image[..., :3]`).
- The same canvas-sized image goes to the text encoder: `process_qwen2vl_images(patch 16, merge 2, mean = std = 0.5)` (`qwen3vl.py:66-72`,
  `qwen_vl.py:9-88`): `h_bar = round(H / 32) * 32` (unchanged for a canvas of multiples of 32), bilinear resize to the same size (identity),
  `(x - 0.5) / 0.5`, the frame repeated twice for the temporal patch, patches laid out as `[gh * gw, 3 * 2 * 16 * 16]` in merge-block order.
  The vision block has `(H/32) * (W/32)` tokens, the same count as the cond rows.
- In the TensorFold workflow (`workflows/Image to Video (MiniMax H3, TensorFold).json`: nodes 114, 119, 120, 1132) `LoadImage` ->
  `ImageScaleToTotalPixels` (`nearest-exact`, 0.9 MP, multiple 32) -> `GetImageSize` -> the node's width and height. So the canvas follows the
  image's aspect ratio and the node's lanczos is an identity. LocalRouter takes the request's width and height and stretches the image with the
  node's lanczos (declared substitution: no `nearest-exact` pre-scale; better quality, the same canvas rule). Everything below is defined on
  the canvas-sized RGB image.
- Official canvas: 768 short edge, capped at 768 x 1344 pixels, multiples of 32 (`adapt_canvas`, `nodes_minimax_h3.py:53-64`; the node does not call it).
- LocalRouter's vision encoder accepts canvases with both sides divisible by 32 and area from 3,136 through 12,845,056 pixels; it rejects canvases outside that range and does not apply a secondary resize.

**The video VAE encoder** (`comfy/ldm/minimax/vae.py`). It is **not** a ViT: the encoder is a 3-D causal CNN, `EncoderFCN3D` (`vae.py:176-225`),
`ch` 128, `ch_mult (1, 2, 2, 4, 4, 8)`, 2 ResNet blocks a level, `space_down (2, 2, 2, 2, 1, 1)`, `time_down (1, 2, 2, 1, 1, 1)`, then
`quant_conv` 48 -> 48 (1x1x1) and the mean half (24 channels) is the latent (`vae.py:407-469, 767-784`). Weights in the fp16 VAE file
(already read by the decoder's loader, which skips these rank-5 tensors: commit cb053da): `encoder.conv_in`, `encoder.down.{l}.block.{i}.{norm1,
conv1,norm2,conv2,nin_shortcut}`, `encoder.down.{l}.downsample.conv` (levels 0-3), `encoder.norm_out`, `encoder.conv_out`, `quant_conv`.
- One frame (`encode`, `vae.py:774-778`, `x.shape[2] == 1`): `moments = tiled_encode(normalize(x))`, no clip loop. The causal convolutions
  see a single frame: `CausalConv3d.forward` skips the front padding (`front = 0 if T == 1`, `vae.py:83`) and calls the conv with
  `autopad="causal_zero"`, which keeps only the **last temporal tap** (`weight[:, :, -T:]`, `comfy/ops.py:613-615`, `vae.py:101-103`); the
  1x1x1 shortcut is a plain conv. Time strides leave T = 1. So for one frame the whole encoder is a **2-D network** whose 3x3 kernels are the
  3-D kernels' last temporal slice `w[:, :, 2]`, and the downsampling convs are 3x3 stride 2 on a frame reflect-padded by 1 on the
  right and bottom only (`Downsample3D`, `vae.py:149-153`); every other 3x3 conv reflect-pads 1 on all sides (`vae.py:82-97`).
- GroupNorm: 32 groups, eps 1e-6, per frame (`TemporalIsolatedGroupNorm`, `vae.py:122-134`; with T = 1 it is the plain GroupNorm), then SiLU, as the
  block's pre-norm of the conv (`ResnetBlock3D`, `vae.py:155-173`); the second conv adds the (shortcut) input as its residual.
- Input: `image * 2 - 1` (`VAE.process_input`) cast to fp16, then `_normalize_pixels`: `(x + 1) * 0.5`, minus the ImageNet mean, over the std
  (`vae.py:17-18, 478-479`; fp16 ops). Output: `mean` of the moments in fp32, `(mean - latents_mean) / latents_std` (`vae.py:780-784`,
  tables at `vae.py:20-36`, the same as the decoder's). Latent T for one frame is **1**; for frames the formula is `(F - 5) // 17 * 5 + 2`
  (`sd.py:1034`), so a first frame alone is T = 1 and `[1, 24, 1, H/16, W/16]`.
- Spatial tiling (`tiled_encode`, `vae.py:556-586`): tile 256, overlap at least 64 (`split_tiles`, `vae.py:507-529`, the decoder's table), each tile
  encoded alone, the moments blended across the overlaps in latent units (overlap / 16) with fp16 weights `1 - p / ext`, `p / ext`
  (`blend`, `vae.py:531-554`), the trailing overlaps cut. A side shorter than 256 is one tile. ComfyUI runs the fp16 file with kitchen's
  fused `group_norm_silu_pad3d` and fp16-accumulate `conv3d` (`vae.py:41-70`) that the twin does not have; the twin's convs are fp16 tensor-core
  GEMMs (im2col) with fp32 accumulation. A declared substitution like the decoder's: gated against ComfyUI's encode by cosine / relative
  error of the latent, not by bits.

**Differences from text to video, in one list** (everything else is identical): the tokens (+ `<Picture 1>: ` and a vision block, ids and
count), the text encoder (+ vision tower, MRoPE, deepstack, `text_token_tags`), the DiT's layout (+ cond rows, positions, a third timestep row,
text rows split into tag runs), one VAE encode, one noise-augmentation draw, the image's preprocessing. Steps, shift, sampler, audio VAE,
video VAE decode, MP4 unchanged. `length` and the 17 k + 5 grid unchanged; the clip's frame 0 is the (re-decoded) first frame.

## 2. What LocalRouter builds

| Piece | Twin (`tools/twin/stk_twin/h3/`) | Engine (`src/engines/minimax_h3/`) |
| --- | --- | --- |
| Image to RGB, PNG, the node's lanczos | `image.py` (PIL's exact algorithm, no PIL) | `image.zig` (PNG via `src/media/png_decode.zig`; JPEG and WebP are not decoded) |
| Video VAE encoder, one frame | `vae_video_enc.py` (+ `kernels/cuda/minimax/vae_enc.cu`) | `vae_video_enc.zig` |
| Text encoder vision tower, MRoPE, deepstack | `te_vision.py`, `te32.py` | `te_vision.zig`, `te32.zig` |
| DiT cond rows, tag runs, third timestep | `dit.py`, `generate.py --image` | `layout.zig`, `dit.zig`, `pipeline.zig` |
| Gates | `test_vae_video_enc.py`, `test_te_vision.py`, `capture_i2v.py` | `localrouter check h3-i2v`, `h3-e2e` with an image |
| Spark window | `tools/spark/h3-i2v-window.sh` (sourced by `h3-inner.sh` with `H3_I2V=1`) | |

Declared substitutions (as in docs/dev/M7.md): the condition-noise draw uses LocalRouter's portable generator (`noise(seed, rows * 96)`, from the
start of the stream, as ComfyUI restarts its stream per condition); the encoder's fp16 arithmetic is the twin's; the vision tower runs fp32 on
the toolkit's fp32 GEMM and attention (ComfyUI casts the pixels to fp32 and the weights follow); the `nearest-exact` pre-scale is skipped.

Gates against ComfyUI (GPU, later): encoder latent vs `MiniMaxH3VideoVAE.encode` of the same canvas image (cosine and relative L2 per channel);
vision embeddings and deepstack features vs `Qwen3VL.visual`; the encoder's hidden states vs the LM with an image (cosine); the packed DiT inputs
(cond rows, positions, rows' timesteps) vs `PackedLayout` / `_forward`; per-step velocity cosine as in `h3.gate`. Gates of the engine against the
twin: the same capture replay, op by op, alone and chained.

## 3. Unknowns that need the GPU or the checkpoints

- The text encoder checkpoint's vision tensors: names, dtype, whether any are NVFP4. Read the safetensors header on the Spark; the twin
  reads `visual.*` as ComfyUI's module names give them (`patch_embed.proj`, `pos_embed`, `blocks.{i}.{norm1,norm2,attn.qkv,attn.proj,
  mlp.linear_fc1,mlp.linear_fc2}`, `merger.{norm,linear_fc1,linear_fc2}`, `deepstack_merger_list.{j}.{norm,linear_fc1,linear_fc2}`, with the
  LM's own prefix) and fails with the missing name.
- Whether ComfyUI's fused fp16 encoder differs from the twin's im2col GEMM beyond fp16 noise (the gate above says).
- The size of the text context with an image: 1,008 vision tokens at 1344 x 768, 336 at 768 x 448; the engine's token limit and the DiT's `max_l`
  must cover them.


## 4. Offline implementation status (2026-10-08)

The twin and Zig paths are wired through first-frame PNG decode, portable Lanczos canvas conversion, the one-frame VAE encoder,
vision/LM encoding and conditioning rows. The tool declares `image_to_video`; the existing AV sampler, decoders and MP4 writer consume
its output. PNG supports every non-interlaced color type and valid bit depth, discards alpha, validates chunk CRC and zlib checksum;
JPEG, WebP and Adam7 PNG remain unsupported. The decoder is general RGB infrastructure for other engines.

`capture_venc` isolates encoder recording; `capture_i2v` records a complete conditioned step. Both are GPU entry points.
`localrouter check h3-i2v PACK TE VAE CAPTURE` verifies the archived PNG hash, canvas, token ids/tags and all recorded encoder,
vision/LM, refiner and conditioned DiT operations, alone and chained, plus exact video/audio velocities. `h3-e2e` accepts an image
capture from `generate --image`, compares the sampler and final frame/waveform hashes and repeats warm for deterministic output.
The standalone conditioned-step capture uses portable initial noise at the selected sigma; it isolates a step rather than claiming
that this input came from an earlier sampler trajectory.

Offline validation uses the required heavy-wrapper build/test command and Python syntax compilation. No CUDA source was compiled,
no checkpoint was loaded and no GPU operation was run here. Consequently checkpoint vision names/storage, new CUDA launch ABI,
encoder substitution tolerances, vision/deepstack/LM agreement, conditioned replay and complete AV output remain unverified.
The encoder gate's per-channel cosine >= 0.999 and relative L2 <= 0.02, and the vision/LM gate's tolerances are provisional until
measured against the real checkpoints; a failing gate must be investigated, not silently relaxed.

Run the opt-in window with `H3_I2V=1` and a real PNG at `H3_I2V_IMAGE` (container path; default `$STK/input/h3-i2v.png`). It must prove:
encoder kernel/tiling correctness and ComfyUI latent agreement; vision preprocessing, positional tables, main/deepstack features and
image-conditioned layer-50 states; conditioned DiT structure/velocities against ComfyUI with shared conditioning noise; exact twin/Zig
replay in both modes; and full frame/waveform hash equality including a warm repeat. Inspect the produced clip for coherent first-frame
identity and an audible, valid stereo track, and measure peak memory and encode latency. Baseline text-only checks must still pass.
