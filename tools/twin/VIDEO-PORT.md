# MiniMax H3 text-to-video(+audio): what ComfyUI 0.37.0 does (bit-exact port notes)

Read-only research. Nothing was run on a GPU and no checkpoint was available on this machine, so everything that lives
only in a checkpoint header (PDD head count, adaln curve table, LoRA key names, which TE/VAE layers carry `comfy_quant`,
norm dtypes) is marked **UNDETERMINED FROM CODE**.

Path conventions (all read-only):

* `C/` = `ref/ComfyUI-0.37.0/` (written `comfy/...:line` for short).
* `TF/` = `ref/minimax-h3-tensorfold-rtx/` (`tfvideo/...`, `workflows/...`, `bench/...`).
* `CK/` = `ref/comfy-kitchen/comfy_kitchen/` (version 0.2.35, `pyproject.toml:19`, the version `TF/AGENTS.md` pins).
  ComfyUI calls compiled kitchen kernels in several places, so a few "exact arithmetic" statements below are taken from
  the kitchen CUDA source (`CK/backends/cuda/ops/*.cu`) rather than from ComfyUI.

## 0. Things that differ from the brief, or that you must not miss

1. **The sampler really is `res_multistep` with eta = 0.** `KSamplerSelect` -> `comfy.samplers.sampler_object("res_multistep")`
   (`comfy_extras/nodes_custom_sampler.py:391-393`, `comfy/samplers.py:1388-1397`) -> `ksampler(name)` ->
   `sample_res_multistep` (`comfy/k_diffusion/sampling.py:1527-1528`), which calls `res_multistep(..., eta=0., cfg_pp=False)`.
   (The inner function's own default is `eta=1.`; it is not used.) So `sigma_down = sigma_to`, `sigma_up = 0`, no noise is
   ever injected after the initial noise (`sampling.py:68-76`).
2. **The saved T2V workflow is 864x480, not 1344x768.** `ResolutionSelector` widgets are `['16:9 (Widescreen)', 0.4, 32]`
   (workflow node 115). The `MiniMaxH3ImageToVideo` widget values `1344, 768, 73` are overridden by the links (links 246,
   247, 1264). 1344x768 needs `megapixels = 0.98` (the workflow's own note table says so). 124 frames comes from
   `PrimitiveFloat 5` -> math expression (section 1).
3. **The workflow uses the int8_convrot video VAE**, not the fp16 one (`VAELoader` widget
   `minimax_h3_video_vae_int8_convrot.safetensors`, nodes 1140). `bench/comfy_h3.py:155` defaults to the fp16 VAE. The two
   take different kernels (section 6).
4. **The DiT loader is `TFMiniMaxH3Loader` (precision `nvfp4`, attention `auto`), not `UNETLoader`**, and the model
   passes through ComfyUI's `BlockSparseAttention` node (`sol-attn`, tau 1.3, start 0.2, min_tokens 12288, extra_tokens 256,
   `exact_kv_and_rows`) before the scheduler and the guider. Attention is therefore **not** plain SDPA: steps 0 and 1 are dense
   (engine int8 attention), steps 2..7 go through `comfy_kitchen.sol_attn_chunked` (section 5.9).
5. **The audio latent is carried at 4x scale on the video sigma grid** (`audio_scale = 12/3 = 4`), undone inside the DiT
   forward. If you port only `_forward` you will be wrong; the wrapper `forward()` is part of the numerics (section 5.1).
6. **Everything is fp32 only where it says so.** The sampler state `x` is fp32 (packed `[1,1,N]`), but every DiT call casts
   `x` to bf16 first, so the video/audio latents the network sees are bf16-rounded copies of the fp32 state; the network's
   velocity comes back bf16-rounded (section 4).

---

## 1. Node graph of "Text to Video (MiniMax H3, TensorFold).json"

Source: `TF/workflows/Text to Video (MiniMax H3, TensorFold).json` (link ids in brackets). No node is bypassed (all `mode 0`).

| Node | Type | Widgets (effective) | Inputs |
|---|---|---|---|
| 115 | `ResolutionSelector` | `16:9 (Widescreen)`, megapixels `0.4`, multiple `32` (**saved = 864x480; use 0.98 for 1344x768**) | out0 width [246] and out1 height [247] -> node 1152 |
| 1154 | `PrimitiveFloat` | `5` (seconds) | -> 1153.`values.a` [1267] |
| 1153 | `ComfyMathExpression` | `max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17` | out1 (INT) -> 1152.length [1264] |
| 1149 | `CLIPLoader` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, type `minimax`, device `default` | -> 1152.clip [1258] |
| 1140 | `VAELoader` (video) | `minimax_h3_video_vae_int8_convrot.safetensors` | -> 1152.vae [1259] (only used to encode keyframes: unused in T2V), -> 1143.vae [1249] |
| 1141 | `VAELoader` (audio) | `minimax_h3_audio_vae_fp32.safetensors` | -> 1142.vae [1248] |
| 1152 | `MiniMaxH3ImageToVideo` | prompt (1 string), width/height/length linked; `first_frame`/`last_frame` unconnected -> T2V | out0 `positive` -> 1147.conditioning [1255]; out1 latent -> 1146.latent_image [1254] |
| 1148 | `TFMiniMaxH3Loader` (custom, `TF/tfvideo/comfy_nodes.py:244-273`) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors`, precision `nvfp4`, attention `auto` | model -> 1155.model [1276] and 1156.on_false [1277] |
| 1155 | `LoraLoaderModelOnly` | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`, strength_model `1` | model -> 1156.on_true [1278] |
| 1160 | `PrimitiveBoolean` | `True` | -> 1156.switch [1283], 1157.switch [1284] |
| 1156 | `ComfySwitchNode` (model) | switch=True -> picks `on_true` = LoRA model (`comfy_extras/nodes_logic.py:106-116`) | -> 1161.model [1287] |
| 1158 / 1159 | `PrimitiveInt` | `20` / `8` | -> 1157.on_false [1281] / on_true [1282] |
| 1157 | `ComfySwitchNode` (steps) | switch=True -> **8** | -> 1145.steps [1280] |
| 1161 | `BlockSparseAttention` | `sol-attn`, tau `1.3`, start_percent `0.2`, end_percent `1`, dense_blocks `''`, min_tokens `12288`, extra_tokens `256`, sink_conditioning `exact_kv_and_rows`, verbose `False` | -> 1145.model [1279], 1147.model [1286] |
| 1145 | `BasicScheduler` | scheduler `simple`, steps (widget 4, **overridden by link: 8**), denoise `1` | sigmas -> 1146 [1253] |
| 1144 | `KSamplerSelect` | `res_multistep` | -> 1146.sampler [1252] |
| 1150 | `RandomNoise` | seed `757358688076805`, control `fixed` | -> 1146.noise [1250] |
| 1147 | `BasicGuider` | none (cfg stays 1.0, no negative) | -> 1146.guider [1251] |
| 1146 | `SamplerCustomAdvanced` | | out0 "output" -> 1143.samples [1273] and 1142.samples [1274] (out1 denoised unused) |
| 1143 | `VAEDecode` | | -> 1151.images [1256] |
| 1142 | `VAEDecodeAudio` | | -> 1151.audio [1257] |
| 1151 | `CreateVideo` | fps `24`, bit_depth `8` (color_space / codec not stored -> defaults `sRGB` / `none`) | -> 92.video [248] |
| 92 | `SaveVideo` | prefix `video/MiniMax_H3`, format `auto`, codec `auto` | |

Notes:

* No `ModelSamplingMiniMaxH3` (shift) node: shifts are the model-config defaults `shift 12.0`, `audio_shift 3.0`
  (`comfy/supported_models.py:970-973`).
* Frame count: `a = 5` -> `round(120) = 120`; `120 % 17 = 1`; `(5 - 1) % 17 = 4`; length = **124**. Then
  `temporal_shape(124)` -> `align_frame_count` (already `124 % 17 == 5`), `video_latent_t = ((124-5)//17)*5 + 2 = 37`,
  `audio_t = round(124/24 * 40) = round(206.67) = 207` (`comfy_extras/nodes_minimax_h3.py:37-50`).
* `ResolutionSelector`: `scale = sqrt(megapixels*1024*1024/(w_ratio*h_ratio))`, `width = round(w_ratio*scale/multiple)*multiple`
  (`comfy_extras/nodes_resolution.py:80-86`). 0.98 MP, 16:9: scale 84.47 -> 42.2 -> 1344, 23.8 -> 768.
* Functions called, in graph order:
  `comfy.sd.load_clip` (`nodes.py:1024-1033`, `comfy/sd.py:1755+`);
  `comfy.sd.load_diffusion_model`-equivalent inside `TF/tfvideo/comfy_nodes.py:build_model` (`comfy.model_detection.detect_unet_config`,
  `model_config_from_unet_config`, `model_config.set_inference_dtype(bf16, bf16)`, `model_config.get_model`, `CoreModelPatcher`,
  `load_model_weights`; `comfy_nodes.py:178-214`);
  `comfy.sd.load_lora_for_models` (hooked by TF, `comfy_nodes.py:125-167`, original `comfy/sd.py:106-138`);
  `BlockSparseAttention.execute` -> `apply_block_sparse_attention` (`comfy_extras/nodes_sparse_attention.py:353-430`, `323-350`);
  `MiniMaxH3ImageToVideo.execute` (`nodes_minimax_h3.py:138-162`) -> `clip.tokenize`, `clip.encode_from_tokens_scheduled`;
  `BasicScheduler.execute` -> `comfy.samplers.calculate_sigmas` (`nodes_custom_sampler.py:32-42`);
  `SamplerCustomAdvanced.execute` -> `Noise_RandomNoise.generate_noise` -> `comfy.sample.prepare_noise`, `Guider_Basic.sample`
  -> `CFGGuider.sample/outer_sample/inner_sample` -> `KSAMPLER.sample` -> `res_multistep` (`nodes_custom_sampler.py:1020-1075`);
  `VAEDecode.decode` -> `VAE.decode` (`nodes.py:333-341`, `comfy/sd.py:1251-1390`);
  `vae_decode_audio` (`comfy_extras/nodes_audio.py:98-112`);
  `CreateVideo.execute` -> `VideoFromComponents`; `SaveVideo.execute` -> `VideoFromComponents.save_to`
  (`comfy_extras/nodes_video.py:166-203, 248-261`, `comfy_api/latest/_input_impl/video_types.py:1116-1205`).
* The Image-to-Video workflow differs only by `LoadImage` + `ImageScaleToTotalPixels` (0.9 MP, multiple 32) -> `first_frame`
  of node `MiniMaxH3ImageToVideo`, and a `1:1` resolution selector; same sampler/LoRA/VAE/attention settings (nodes 1120-1141).

---

## 2. Text encoder

### 2.1 Tokenizer and "template"

There is **no chat template** and no special tokens for T2V (`comfy/text_encoders/minimax.py:3-20`).

* `CLIP.tokenize(prompt, images=[])` -> `MiniMaxH3Tokenizer.tokenize_with_weights` (`minimax.py:148-202`). With no images and no
  `minimax_ref_items` the entry list is exactly `add_text(prompt)`; if that produced nothing, `[(151643, 1.0)]`.
* `add_text` -> `Qwen3VLSDTokenizer` (`comfy/text_encoders/qwen3vl.py:147-151`): `transformers.Qwen2Tokenizer` from
  `comfy/text_encoders/qwen25_tokenizer/` (vocab.json, merges.txt, tokenizer_config.json), `has_start_token=False`,
  `has_end_token=False`, `pad_to_max_length=False`, `max_length=99999999`, `min_length=1`, `pad_token=151643`.
  The MiniMax subclass adds 7 extra special tokens `<d> 151669, </d> 151670, <|cutoff|> 151671, <|lyrics_start|> 151672,
  <|lyrics_end|> 151673, <|caption_start|> 151674, <|caption_end|> 151675` (`minimax.py:32-34, 129-133`); a prompt that does not
  contain those literal strings is unaffected.
* `SDTokenizer.tokenize_with_weights(..., disable_weights=True)` (`comfy/sd1_clip.py:572-674`): `escape_important` /
  `unescape_important` (identity for text without `\(` `\)`); weights are not parsed (`(...)` and `:1.2` are literal text);
  the text is split only at `(?<=\s)embedding:` boundaries; every segment is `tokenizer(word)["input_ids"][0:]` (no BOS/EOS,
  `add_bos_token False`, `tokenizer_config.json`). `embedding:name` is resolved from `models/embeddings` if present (the
  workflow prompt contains none). Result: one batch, a plain list of ints, weight 1.0 everywhere, **no padding, no truncation**.
  Text length `L` = number of Qwen2 BPE tokens of the raw prompt (not computed here; no tokenizer available offline).
* `NFC` normalization and the Qwen2 pretokenizer regex are those of `Qwen2Tokenizer` (slow python tokenizer, not a fast one).
* The returned tags: `token_tags_from_embeds_info` -> all ones for T2V (`minimax.py:75-82`), so every text row uses adaLN
  modality tag 1. They reach the DiT as cond key `minimax_token_tags` (`minimax.py:113-120` -> `comfy/sd.py:339-348, 442-446`).

### 2.2 Which hidden state

`MiniMaxH3ClipModel` (`minimax.py:106-120`): `layer="last"`, `layer_norm_hidden_state=False`, `enable_attention_masks=False`,
`return_attention_masks=False`, `special_tokens={"pad": 151643}`.

* `SDClipModel.forward` (`sd1_clip.py:260-299`): `process_tokens` embeds ids with `embed_tokens(..., out_dtype=torch.float32)`
  (fp32 `[1, L, 5120]`), no attention mask is passed, `z = outputs[0].float()`.
* `Llama2_.forward` (`comfy/text_encoders/llama.py:909-1034`): the stack runs all **50** layers; `Qwen3VL_32BConfig.final_norm = False`
  (`llama.py:358-366`) so `self.norm is None` and `outputs[0]` is the raw residual stream after layer index 49. No final norm, no
  `lm_head`, no projection. Deepstack/visual inputs are not used in T2V.
* Output of `encode_token_weights`: `cond` fp32 `[1, L, 5120]` on the intermediate (CPU) device, `pooled = None`, extra dict
  `{"minimax_token_tags": LongTensor[L]}` (`minimax.py:113-120`; `sd1_clip.py:27-67`).
* Conditioning list: `[[cond, {"pooled_output": None, "minimax_token_tags": tags}]]` (`sd.py:339-348, 442-446`).
* In `MiniMaxH3.extra_conds` (`model_base.py:2169-2218`) the cond is cast to bf16 and run **once per sampling run** through the DiT's
  `preprocess_text_embeds` (`condition_proj` 5120->5376 then the 2-layer token refiner, section 5.3); the result
  `[1, L, 5376]` bf16 is what every step receives as `context`.

### 2.3 Qwen3-VL-32B text config (`Qwen3VL_32BConfig`, `llama.py:342-366`, `Qwen3_8BConfig` `315-340`)

| field | value |
|---|---|
| layers | **50** (truncated from 64) |
| hidden_size | 5120 |
| intermediate_size | 25600 (SwiGLU: `down(silu(gate(x)) * up(x))`, separate gate/up; `merged_mlp False`) |
| heads / kv heads / head_dim | 64 / 8 (GQA) / 128 (`inner_size 8192`, `kv_size 1024`) |
| q/k norm | per-head RMSNorm on head_dim, weight not `+1`, eps 1e-6, applied before RoPE (`llama.py:640-643`) |
| rms_norm_eps | 1e-6 (input/post-attention layernorms: `RMSNorm` -> `F.rms_norm`, `comfy/rmsnorm.py:7-12`) |
| rope_theta | 5,000,000 |
| mrope | `rope_dims=[24,20,20]`, `interleaved_mrope=True` **but T2V passes `position_ids=None` -> `arange(L)` 1-D**, so the plain (non-mrope) branch of `precompute_freqs_cis` runs (`llama.py:532-552`): `inv_freq = 1/(theta ** (arange(0,128,2).float()/128))` (fp32, on device), `freqs = pos.float() * inv_freq`, `emb = cat(freqs, freqs)`, `cos/sin` fp32; split-half rotation of **all 128 dims** (pairs i, i+64) via `comfy_kitchen.apply_rope_split_half` (fp32 in/out) |
| qkv bias / attn bias | none; `o_proj` no bias |
| vocab | 151936, `embed_tokens` is an `ops.Embedding` |
| final_norm / lm_head | False / False |
| visual tower | built (`QWEN3VL_VISION["qwen3vl_32b"]`: 27 layers, 1152 wide, deepstack at 8/16/24) but unused for T2V |

Attention in the TE: `optimized_attention_for_device(device, mask=True, small_input=True)` -> `attention_pytorch` when PyTorch attention
is enabled (default on NVIDIA, `model_management.py:462-474`, `attention.py:906-911`). The mask is the additive fp32 causal mask
`full((L,L), finfo(fp32).min/4).triu_(1)` (`llama.py:936-941`); GQA is passed as `enable_gqa=True`
(`llama.py:700-701`), then `comfy.ops.scaled_dot_product_attention` (`comfy/ops.py:57-96`) with the sdpa priority
`[FLASH, CUDNN, EFFICIENT, MATH]` for `q.nelement() >= 128*1024`. With fp32 inputs and an arbitrary mask only EFFICIENT/MATH can run,
and whether native-GQA is accepted or `repeat_kv` is applied is a PyTorch-version decision (`ops.py:80-96`). **This is a pin item
(section 10).** Block order (`llama.py:739-766`): `x = x + attn(rms(x))`, `x = x + mlp(rms(x))`, all fp32.

### 2.4 The NVFP4 AWQ checkpoint: loading and compute

Loading:

1. `CLIPLoader` -> `load_clip` -> `load_text_encoder_state_dicts`. `detect_te_model` returns `QWEN3VL_32B` because the keys
   `visual.deepstack_merger_list.0.norm.weight` and `model.layers.49.self_attn.q_proj.weight` exist (`sd.py:1709-1711`);
   `lm_head.weight` is renamed if present (`sd.py:1767-1768`). Builder: `comfy.text_encoders.minimax.te(**llama_detect(clip_data))`
   (`sd.py:1968-1970`).
2. `llama_detect` (`comfy/text_encoders/hunyuan_video.py:11-23`): `dtype_llama` = dtype of `model.norm.weight` or, since the model is
   truncated, `model.layers.0.input_layernorm.weight`; `llama_quantization_metadata = {"mixed_ops": True}` if any key ends in
   `.comfy_quant` (`comfy/utils.py:1434-1439`). In `te()` (`minimax.py:208-217`) these override the constructor dtype. The dtype
   itself is **UNDETERMINED FROM CODE** (most likely bf16).
3. `SDClipModel.__init__` (`sd1_clip.py:109-117`): operations = `comfy.ops.mixed_precision_ops(quant_config, dtype, full_precision_mm=True)`.
4. Each linear (`q/k/v/o_proj`, `gate/up/down_proj`) loads through `_load_quantized_module` (`comfy/ops.py:1158-1325`): `comfy_quant` JSON
   `{"format":"nvfp4"}` -> `TensorCoreNVFP4Layout`; `weight` is `uint8` `[N, K/2]` (2 fp4 values per byte), `weight_scale_2` (fp32 per-tensor) ->
   `Params.scale`, `weight_scale` viewed as `float8_e4m3fn` (cuBLAS 128x4 swizzled block scales, group 16) -> `Params.block_scale`,
   `orig_dtype = compute_dtype`, `orig_shape = (N, K)` (`ops.py:1217-1222, 1270-1275`; `comfy/quant_ops.py:220-227`).
   Extra tensors `input_scale` and `pre_quant_scale` are registered as parameters when present (`ops.py:1276-1285`).
   `pre_quant_scale` is the ModelOpt **AWQ** per-input-channel smoothing vector (shape `[K]`); whether q/k/v share one is
   **UNDETERMINED FROM CODE**.

Compute (this is a **weight-only dequantize to fp32, then an fp32 matmul**, nothing runs in FP4):

* `MixedPrecisionOps.Linear.forward` (`ops.py:1419-1484`): (a) `input = input * cast(pre_quant_scale, input.dtype)` (fp32 multiply, `ops.py:1422-1425`);
  (b) `_full_precision_mm=True` (passed in step 3) forces `_use_quantized=False` (`ops.py:1432-1436`), so the activations are never quantized and
  `input_scale` is unused; (c) `forward_comfy_cast_weights` -> `CastBiasWeightContext` -> `cast_bias_weight(self, input)` with
  `dtype = input.dtype = float32`: `weight.dtype` (= `orig_dtype`) != fp32 -> `weight.to(dtype=float32)` sets `orig_dtype=fp32`, then
  `weight.dequantize()` (`ops.py:433-437`; `CK/tensor/base.py:439-460, 275+`; `CK/tensor/nvfp4.py:87-88`);
  (d) `torch.nn.functional.linear(input_fp32, weight_fp32, bias)` (`ops.py:1384-1385, 1402-1404`) = cuBLAS SGEMM (TF32 is not enabled
  anywhere in ComfyUI; default PyTorch `allow_tf32=False` for matmul).
* Dequantize math (`CK/backends/cuda/ops/quantize_nvfp4.cu:217-290`, eager reference `CK/backends/eager/quantization.py:175-220`):
  `decode_scale = float(block_scale_e4m3[row, k/16]) * tensor_scale_fp32` (fp32 multiply), `w = float(e2m1_value) * decode_scale`
  (fp32 multiply, **no intermediate rounding**; the e2m1 value set is {0, 0.5, 1, 1.5, 2, 3, 4, 6}); `HiFirst=True`: **the high nibble is the
  even-index element**, low nibble the odd one. The output type is the requested `orig_dtype` (fp32 here), written directly.
* The TE runs entirely in fp32 activations: embeddings fp32, RMSNorm fp32, rope fp32, attention fp32 (above), SwiGLU fp32.
* TE device/offload: `text_encoder_device()` is the GPU under aimdo or when `should_use_fp16(prioritize_performance=False)` holds
  (`model_management.py:1219-1230`); a CPU run would not be bit-identical (pin item).

---

## 3. Latent setup

### 3.1 Shapes (1344x768, 124 frames, 24 fps)

`_empty_av_latent(width, height, length)` (`nodes_minimax_h3.py:84-90`) builds a zero nested tensor on the intermediate device:

| tensor | shape | dtype | formula |
|---|---|---|---|
| video latent | `[1, 24, 37, 48, 84]` | fp32 zeros | `[B, 24, T, H/16, W/16]`, `T = 2 if frames<=5 else ((frames-5)//17)*5 + 2` |
| audio latent | `[1, 32, 2, 207]` | fp32 zeros | `[B, 32, 2 stereo, A]`, `A = round(frames/24 * 40)` (40 latent frames/s, 800 samples each at 32 kHz) |
| packed sampler state `x` | `[1, 1, 3,593,664]` | fp32 | `24*37*48*84 = 3,580,416` video + `32*2*207 = 13,248` audio, concatenated video-first (`comfy/utils.py:1412-1432`) |

`frames` snaps up to `17k+5` (`align_frame_count`, `nodes_minimax_h3.py:37-40`); `length` min 5. Video latent spatial size is
`H/16, W/16` (the VAE's total compression: `space_down` product 16, `vae.py:429, 417-419`); temporal `4x` with the first frame special
(1 frame -> 1 token, then 4 per token: `FRAME_PER_TOKEN = (1,4,4,4,4)`, `model.py:30`). Scaling: video tokens `T*(H/32)*(W/32)`, with
`T = 5k+2` for `17k+5` frames; audio latent frames grow linearly with duration; latent H/W must be multiples of 16 pixels
(`CANVAS_MULTIPLE = 32` keeps the DiT patch grid exact). The DiT pads H/W latents to even if needed (`pad_to_patch_size`, circular,
`model.py:606`), irrelevant at 48x84.

### 3.2 Latent formats

* `ModelSamplingAV`/`MiniMaxH3AV` (`comfy/latent_formats.py:622-675`): `scale_factor = 1.0`, **no shift, no per-channel mean/std in the
  latent format** (`LatentFormat.process_in = latent * scale_factor`, `process_out = latent / scale_factor`, `latent_formats.py:16-20`):
  identity, bit-exact.
* The DiT operates on **VAE-normalized latents**: the video VAE multiplies by std and adds mean at decode
  (`z = z * latents_std + latents_mean`, `vae.py:793-797`, constants `LATENTS_MEAN/LATENTS_STD` `vae.py:20-36`, 24 values; they are also
  persistent buffers `vae.py:465-466`, so the checkpoint's values override these defaults; **cast to fp16 with the module**, section 6).
  The audio VAE does the same per channel with checkpoint buffers `latents_mean/latents_std` (shape 32; values **UNDETERMINED FROM
  CODE**, `audio_vae.py:413-414, 420-422`).
* Audio carry scale: `MiniMaxH3.audio_scale()` = `model_sampling.audio_scale = shift / audio_shift = 12.0 / 3.0 = 4.0`
  (`model_sampling.py:353-357`, `model_base.py:2145-2149`). `process_latent_in` multiplies the audio slice by 4
  (`model_base.py:2151-2164`), `process_latent_out` by 0.25 (`:2166-2167`). For T2V `process_latent_in` is **skipped**:
  `inner_sample` only applies it when `count_nonzero(latent_image) > 0` (`samplers.py:1223-1224`) and the latent is all zeros;
  `process_latent_out` is always applied to the final packed fp32 state, `latent[..., n:] *= 0.25` with
  `n = prod(latent_shapes[0][1:]) = 3,580,416` (`model_base.py:2160`).

### 3.3 Initial noise

* `RandomNoise.execute(seed)` -> `Noise_RandomNoise(seed)` (`nodes_custom_sampler.py:1003-1017`, `samplers.py:724-731`).
  `generate_noise` -> `comfy.sample.prepare_noise(latent_image, seed, batch_inds=None)` (`comfy/sample.py:22-38`).
* **One CPU generator** `generator = torch.manual_seed(seed)` (`sample.py:27`); the nested latent is unbound and the noise drawn **in order: video
  first, then audio** (`sample.py:29-34`), each `torch.randn(shape, dtype=torch.float32, layout=..., generator=generator, device="cpu").to(latent.dtype)`
  (`sample.py:10-11`): video `[1,24,37,48,84]`, then audio `[1,32,2,207]`, fp32, row-major, then packed video-then-audio
  (`samplers.py:1276-1283`) and moved to the GPU as fp32 (`samplers.py:1254`).
* Seed value `757358688076805` is > 2^32. PyTorch's CPU `mt19937` is seeded with the **low 32 bits**: `757358688076805 & 0xFFFFFFFF = 1334969349`
  (from PyTorch's `at::mt19937`, not in ComfyUI; verify against your own torch build). The values are PyTorch CPU `randn` (Box-Muller in
  blocks of 16, with the AVX2/Sleef vector path on x86 hosts and the scalar path elsewhere); that algorithm is not in ComfyUI and is a pin
  item if you generate noise outside PyTorch.
* `noise_scale` is 1.0 (`model_sampling.py:295-306`). The first sampler `x`: `noise_scaling(sigma=1.0, noise, latent_image=0)` =
  `sigma*(1.0*noise) + (1.0-sigma)*latent_image` (`model_sampling.py:94-97`, called at `samplers.py:993`) = `noise` exactly (sigma0 = 1.0 exactly).
* The end of sampling applies `inverse_noise_scaling(sigmas[-1]=0, samples)` = `latent / (1 - 0)` = identity (`samplers.py:1006`).

### 3.4 Sigma schedule

* Model sampling class: `ModelSamplingAV(ModelSamplingDiscreteFlow)` + `CONST` (`model_base.py:149-154, 2142`), built from
  `sampling_settings = {shift: 12.0, audio_shift: 3.0}` (`supported_models.py:970-973`), `multiplier 1000`, `timesteps 1000`.
* `sigmas` buffer (fp32, length 1000): `set_parameters` (`model_sampling.py:308-312`)
  `ts = sigma((torch.arange(1, 1001) / 1000) * 1000)`, `sigma(t) = time_snr_shift(12.0, t / 1000)`,
  `time_snr_shift(alpha, t) = alpha * t / (1 + (alpha - 1) * t)` (`model_sampling.py:289-292, 328-329`), evaluated as fp32 tensor ops:
  `t = ((i/1000)*1000)/1000` (three fp32 ops), then `(12*t) / (1 + 11*t)` (`12*t` and `11*t` fp32, `1 + ...` fp32, one fp32 divide).
  `sigma_min = sigmas[0]`, `sigma_max = sigmas[-1] = 12/12 = 1.0` exactly.
* `BasicScheduler(simple, steps=8, denoise=1)` (`nodes_custom_sampler.py:32-42`) -> `calculate_sigmas` -> `simple_scheduler`
  (`samplers.py:645-652`):
  ```python
  ss = len(s.sigmas) / steps          # 1000/8 = 125.0
  for x in range(steps): sigs += [float(s.sigmas[-(1 + int(x * ss))])]
  sigs += [0.0]
  return torch.FloatTensor(sigs)
  ```
  Indices `-1, -126, -251, -376, -501, -626, -751, -876` (i.e. sigmas at t = 1.000, .875, ..., .125 of the discrete grid) then `0.0`.
  `sigmas[-(steps+1):]` keeps all 9 (`denoise=1`). Moved to the GPU as fp32 (`samplers.py:1256`).
* Values (fp32 emulation of the above, printed as the exact fp32 value):

| i | sigma_i | (sigma_i*1000)/1000 recomputed in the DiT | t_v = 1 - sigma_v | sigma_a = shift(sigma_v, 12->3) | t_a = 1 - sigma_a |
|---|---|---|---|---|---|
| 0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 |
| 1 | 0.9882352948188782 | same | 0.011764705181121826 | 0.9545455574989319 | 0.045454442501068115 |
| 2 | 0.9729729890823364 | same | 0.027027010917663574 | 0.8999999165534973 | 0.10000008344650269 |
| 3 | 0.9523809552192688 | same | 0.0476190447807312 | 0.833333432674408 | 0.16666656732559204 |
| 4 | 0.9230769276618958 | same | 0.07692307233810425 | 0.7499999403953552 | 0.2500000596046448 |
| 5 | 0.8780487775802612 | same | 0.12195122241973877 | 0.6428572535514832 | 0.35714274644851685 |
| 6 | 0.800000011920929 | same | 0.19999998807907104 | 0.5000000596046448 | 0.4999999403953552 |
| 7 | 0.6315789222717285 | same | 0.3684210777282715 | 0.2999999523162842 | 0.7000000476837158 |
| 8 | 0.0 | | | | |

  (My offline emulation of fp32 arithmetic with Python doubles + rounding to float32; verify with torch on a CPU before relying on the last digit.)
* The DiT recomputes the sigma it needs from `timestep = float32(sigma * 1000)`: `sigma_v = (timestep / 1000.0).float().clamp(min=1e-6)`
  (`model.py:566-570, 628`). For these 8 values the round trip is exact. The audio sigma is
  `time_shift_sigma(sigma_v, 12, 3) = 3*b / (1 + 2*b)` with `b = sigma_v / (12 + sigma_v * (1 - 12))` (`model.py:36-39`, fp32 tensor ops in that order).

---

## 4. Sampler: `res_multistep`, eta = 0

### 4.1 Plumbing (one model call per step)

`SamplerCustomAdvanced.execute` -> `Guider_Basic.sample` (`CFGGuider.sample`, `samplers.py:1276-1346`): nested latent unbound, packed with
`pack_latents` (`[1,1,N]`), `latent_shapes = [(1,24,37,48,84), (1,32,2,207)]`; `outer_sample` moves `noise`/`latent_image` to the device as fp32
and sigmas to the device; `inner_sample` (`samplers.py:1220-1238`) sets `inner_model.latent_shapes`, (skips `process_latent_in`), runs `process_conds`
(-> `MiniMaxH3.extra_conds`, which builds the `PackedLayout` once), stores `transformer_options["sample_sigmas"] = sigmas` (fp32, all 9),
calls `KSAMPLER.sample`, and finally `process_latent_out(samples.to(float32))`. Unpacking to a nested latent: `samplers.py:1344-1346`.

`KSAMPLER.sample` (`samplers.py:983-1007`): `extra_args["denoise_mask"] = None`, `model_k = KSamplerX0Inpaint(model_wrap, sigmas)` (no mask -> passthrough, `:634-643`),
`noise = noise_scaling(sigmas[0], noise, latent_image)` (section 3.3), `res_multistep(model_k, noise, sigmas, extra_args, ...)`,
`inverse_noise_scaling`. The `seed` is unused by the sampler (the ancestral noise sampler is created, `sampling.py:1465`, but eta=0 never calls it).

Each step: `denoised = model(x, sigmas[i] * s_in, **extra_args)` (`sampling.py:1486`; `s_in = x.new_ones([1])` so sigma is an fp32 `[1]` tensor)
-> `CFGGuider.predict_noise` -> `sampling_function` (`samplers.py:609-627`): `cond_scale == 1.0` and cfg1-optimization on, so `uncond_ = None` and
**exactly one `apply_model` call per step**; `calc_cond_batch` accumulates `out_conds[0] += output * mult` with `mult = 1.0`, `out_counts = 1e-37 + 1.0 = 1.0`, divides
(`samplers.py:235-236, 331-355`), and `cfg_function` returns `uncond_pred + (cond_pred - uncond_pred) * 1.0` with `uncond_pred = 0/1e-37 = 0`
(`samplers.py:592-605`); all of these are bit-exact identities.

`BaseModel._apply_model` (`comfy/model_base.py:224-262`):

```python
sigma = t                                   # fp32 [1]
xc = model_sampling.calculate_input(sigma, x)   # CONST: returns x (fp32 [1,1,N])
dtype = get_dtype_inference()               # bf16 (TF loader: set_inference_dtype(bf16, bf16))
xc = xc.to(dtype)                           # <-- bf16 rounding of the whole packed state
t = model_sampling.timestep(t).float()      # sigma * 1000, fp32
context = cast_to_device(context, device, dtype)    # already bf16 [1,L,5376]
extra conds: tensors cast to bf16 except int/long; non-tensor payload dict passes through untouched
xc = unpack_latents(xc, latent_shapes)      # [video bf16 [1,24,37,48,84], audio bf16 [1,32,2,207]]
model_output = diffusion_model(xc, t, context=context, ..., minimax_payload=payload)   # list of 2 bf16
model_output, _ = pack_latents(model_output)         # [1,1,N] bf16
return model_sampling.calculate_denoised(sigma, model_output.float(), x)   # CONST: x - out * sigma, fp32
```

`CONST.calculate_denoised` (`model_sampling.py:90-92`): `sigma` reshaped to `[1,1,1]`, `x - model_output.float() * sigma` in fp32 (two fp32 ops, mul then sub).

### 4.2 The per-step update (`comfy/k_diffusion/sampling.py:1462-1525`, eta=0, cfg_pp=False)

All tensors fp32. `sigmas[i]` is a 0-d fp32 tensor on the GPU (an element of the device sigma tensor).

```python
for i in range(len(sigmas) - 1):                      # 8 iterations
    denoised = model(x, sigmas[i] * s_in, **extra_args)
    sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i+1], eta=0.)   # -> (sigmas[i+1], 0.)
    if sigma_down == 0 or old_denoised is None:      # step 0 and the LAST step (sigmas[8] == 0) are Euler
        d  = (x - denoised) / sigmas[i]              # to_d: fp32 sub, fp32 div (sigma broadcast [1,1,1])
        dt = sigma_down - sigmas[i]                  # 0-d fp32
        x  = x + d * dt                              # fp32 mul, fp32 add
    else:                                            # steps 1..6: 2nd order multistep (RES, arXiv 2308.02157)
        t, t_old, t_next, t_prev = t_fn(sigmas[i]), t_fn(old_sigma_down), t_fn(sigma_down), t_fn(sigmas[i-1])   # t_fn = -log(sigma)
        h  = t_next - t
        c2 = (t_prev - t_old) / h
        phi1_val, phi2_val = phi1_fn(-h), phi2_fn(-h)           # phi1(t) = expm1(t)/t ; phi2(t) = (phi1(t) - 1.0)/t
        b1 = nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
        b2 = nan_to_num(phi2_val / c2, nan=0.0)
        x  = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)   # sigma_fn(h) = exp(-h)
    # (sigma_up == 0: no noise added)
    old_denoised = denoised
    old_sigma_down = sigma_down
```

Order-of-operations facts that matter for bits:

* `old_sigma_down` is the previous iteration's `sigma_down`, i.e. `sigmas[i]` (not `sigmas[i-1]`), so `t_old == t` and `c2 = (t_prev - t) / h`, `t_prev = -log(sigmas[i-1])`.
* Each elementwise op on the `[1,1,N]` tensors is a separate PyTorch kernel with an fp32 rounding (`b1*denoised`, `b2*old_denoised`, their sum, `h*(...)`,
  `sigma_fn(h)*x`, final add); the scalars `h, b1, b2, phi*, c2, exp(-h), log` are fp32 0-d CUDA tensor ops (`torch.log`, `torch.expm1`, `torch.exp`: CUDA libdevice, pin item).
* The last step is Euler with `dt = 0 - sigma_7`: `x + ((x - denoised) / sigma) * (-sigma)`, which is **not** bit-equal to `denoised` in fp32.
* `torch.nan_to_num` only matters if `c2` or `phi` is degenerate (does not happen on this schedule).
* `s_noise` is multiplied by `model_sampling.noise_scale` (1.0) at `sampling.py:1466`; unused.
* The per-step state kept: `x` (fp32), `old_denoised` (fp32 `[1,1,N]`), `old_sigma_down`.

---

## 5. The DiT (`comfy/ldm/minimax/model.py`, class `MiniMaxH3Model`)

Config (defaults, `model.py:474-481`; actual values come from checkpoint tensor shapes + header metadata `config.transformer`, `model_detection.py:390-415`):
hidden 5376, 50 layers, token refiner 2 layers, 56 heads x 128, ffn 14336, latents_dim 24, audio_latents_dim 32, patch (1,2,2), text_dim 5120,
timestep_input_dim 256, time_embed_hidden 5376, time_embed_dim 2688, rope_inv_freq_len 16, norm/qk/final eps 1e-5, shifts 12/3.
**UNDETERMINED FROM CODE** (checkpoint-only): whether the checkpoint is the "curve" form (`adaln_t_table` buffer `[grid, k]`, no `time_embedder`, `adaln_proj`
without SiLU in fp32; `model.py:492-500, 736-743`) -- the TF engine's docstring `tfvideo/minimax_h3.py:266` ("no SiLU on the curve form") and `docs/dev/RESULTS.md:12`
("small time-embedding curve") say it is -- and the PDD bank size `n` (`video_out.weight.shape[0] // 96`, `model.py:321`; `docs/dev/RESULTS.md:101` lists PDD heads as exact plumbing, so `n > 1` is
probable).

### 5.1 Outer `forward` (audio carry, `model.py:560-601`)

```python
scale = float(payload["audio_scale"])            # 4.0
audio_src = x[1]                                 # bf16, the sampler's carried audio
if scale != 1.0:
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    sigma_a = time_shift_sigma(sigma_v, 12.0, 3.0)
    carry   = (sigma_a / sigma_v).to(audio_src.dtype)        # <-- rounded to bf16
    x = [x[0], audio_src * carry]                            # bf16 multiply
out = _forward(...)                                          # [video_vel, audio_vel] bf16
# (denoise_mask None in T2V)
if scale != 1.0:
    out[1] = (1.0 - scale) * (audio_src * carry) + (1.0 + (scale - 1.0) * sigma_a).to(out[1].dtype) * out[1]
```

i.e. `out1 = -3.0 * a + bf16(1 + 3*sigma_a) * out1` with `a = bf16(audio_src * carry)`, all bf16 ops (`(1-scale) * tensor` is a bf16 multiply; `bf16(...) * out[1]` bf16; sum bf16).
Derivation: along the flow the sampler's audio variable is `audio_src = sigma_v*n + scale*(1 - sigma_v)*c`, so `d(audio_src)/d(sigma_v) = (1-scale)*x_a + (1 + (scale-1)*sigma_a) * v_a`.
At step 0 `carry == 1.0` and `sigma_a == 1.0` exactly.

### 5.2 Packing: `[text | audio | video]` (T2V)

`PackedLayout` (`model.py:350-470`) is built once in `extra_conds` (`model_base.py:2206-2216`) with `text_len = L`, `(latent_t, lat_h, lat_w) = (37, 48, 84)` (rounded up to even),
`audio_t = 207`; no keyframes/refs, so the segment table is exactly three segments, in this order (`model.py:440-452`):

| segment | rows | kind | timestep class | modality tag |
|---|---|---|---|---|
| text | `[0, L)` | `text` | video t (`t_v`) | 1 (from `minimax_token_tags`, one run) |
| audio | `[L, L+414)` | `audio` | audio t (`t_a`) | 2 |
| video | `[L+414, L+414+37296)` | `video` | video t (`t_v`) | 0 |

Total `S = L + 37,710`. (The docstring `[text | cond rows | audio | video]` matches; with no conditioning there are no cond rows.)

Patchify / packing of the latents (`_forward`, `model.py:696-714`):

* `video_x = pad_to_patch_size(video_x, (1,2,2))` (circular pad, no-op here), `video_rows = patchify_video(video_x.to(float32))`:
  `reshape(1,24,37,1,24,2,42,2)`, `einsum("nctrhpwq->nthwcrpq")`, `reshape(37*24*42 = 37296, 96)`; token order t-major then h then w (`h` fastest-but-one),
  channel order inside a row `c*4 + p*2 + q` (`model.py:42-49`). fp32 `[37296, 96]`.
* `audio_rows = pack_audio(audio_x.to(float32))`: `latent[0].permute(1,2,0).reshape(2*207, 32)`: `[414, 32]` fp32, **channel-major** (rows 0..206 = stereo ch 0, 207..413 = ch 1) (`model.py:59-62`).
* `video_embed = video_patch_proj(video_rows).to(bf16)` (`Linear(96->5376)`, **fp32 weights**, fp32 matmul, then round to bf16): `[37296, 5376]`;
  `audio_embed = audio_patch_proj(audio_rows).to(bf16)` (`Linear(32->5376)` fp32): `[414, 5376]` (`model.py:496-497, 714-715`).
* `h = empty(S, 5376, bf16)`; slice-assigned `text_states` (`context[0]`, `[L,5376]`), then `audio_embed`, then `video_embed` (`model.py:721-733`).

### 5.3 Text path (`condition_proj` + token refiner), run once in `extra_conds`

`preprocess_text_embeds` (`model.py:517-521`): `token_refiner(condition_proj(text_states[0])).unsqueeze(0)` with
`text_states` the bf16 TE output `[1, L, 5120]`. `condition_proj = Linear(5120->5376, bias)` in model dtype (bf16). `TokenRefiner` (`model.py:263-275`):
2 x `RefinerBlock` (`249-260`): `x = attn(norm1(x)) + x` (`add_` in place), `x = mlp(norm2(x)) + x`, with `attn` = the same `Attention` class but
**without rope** (`rope_freqs=None`: `q = q_norm(q.view(L,56,128))`, `k = k_norm(...)`, plain `RMSNorm` per head, eps 1e-5, dense attention over the L text rows via
`optimized_attention(..., mask=None)`; **called without `transformer_options`**, so no sparse-attention override applies), then `final_norm` (RMSNorm 5376, eps 1e-5).
Dtypes: all bf16 (ops are the checkpoint's ops; whether the refiner linears are quantized **UNDETERMINED FROM CODE**). Output `[1, L, 5376]` bf16.

### 5.4 Timestep embedding and modulation (per token class)

In `_forward` (`model.py:626-693, 735-746`):

```python
sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
t_v = float(1.0 - sigma_v);  t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
seg_t = {"text": t_v, "video": t_v, "audio": t_a}
unique_t = sorted({t_v, t_a} | {seg_t[k] ...})            # step 0: one value (0.0); steps >=1: [t_v, t_a] since t_v < t_a
t_row = {t: i for i, t in enumerate(unique_t)}
mod_row(segment) = t_row[seg_t[kind]] * 3 + tag            # tag: video 0, text 1 (per-token tags), audio 2
```

`mod_segments = [(0, L, row*3+1), (L, L+414, row_a*3+2), (L+414, S, row_v*3+0)]` (the text span is split into runs of equal tag; all-ones -> one run).
`t_vals = torch.tensor(unique_t, float32)`:

* Non-curve checkpoint: `t_emb = time_embedder(t_vals).to(bf16)`; `TimeEmbedder` (`model.py:133-146`), **fp32 end to end, cos before sin**:
  `freqs = exp(-log(10000.0) * arange(128, fp32) / 128)`, `args = t[:,None] * freqs[None]`, `emb = cat([cos(args), sin(args)])` (256), `proj_out(silu(proj_in(emb)))` (fp32 Linear 256->5376->2688).
  `AdalnProj` (`model.py:213-227`): `Linear(t_dim -> 6*5376*3)` on `silu(t_emb)` in **bf16**, `.view(M*3, 6*5376)`, `.chunk(6, -1)` into
  `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`, each `[M*3, 5376]`; the `modalities=3` index is the fastest-varying index of the first dim (`row*3 + tag`).
* Curve checkpoint (probable, see above): `table = adaln_t_table` `[grid, k]` fp32; `pos = t_vals.clamp(0,1) * (grid-1)`; `i0 = pos.floor().long().clamp(max=grid-2)`;
  `t_emb = torch.lerp(table[i0], table[i0+1], (pos - i0).unsqueeze(1))` fp32 (`model.py:736-741`); `AdalnProj.linear` runs in **fp32 with no SiLU** (`apply_silu=False`, `adaln_dtype=float32`, `model.py:492-495, 223-227`).

Modulation application per block (`DiTBlock.forward`, `model.py:291-297`; helpers `230-246`), ComfyUI/bf16 version:

```python
h = norm1(x)                                         # RMSNorm(5376, eps 1e-5, weight), bf16 (F.rms_norm)
for a,b,row in segs: h[a:b].mul_(1.0 + vec[row].to(bf16)).add_(vec[row2].to(bf16))     # scale: (1.0 + scale) in bf16
x[a:b].addcmul_(attn_out[a:b], gate[row].to(bf16))   # x += out * gate  (bf16 tensor, fp32 opmath, one rounding)
```

### 5.5 RoPE (`model.py:70-130, 149-155, 523-530, 745-746`)

* Axes `(t, h, w)`, float64 positions built in `PackedLayout` (CPU), then `position_ids.to(float32)`:
  * text rows: `t = 0..L-1`, `h = w = 0` (`model.py:358-361`).
  * audio rows (`_audio_grid(cursor, 207, w_low, w_high)`, `model.py:117-123`): `cursor = float(L)`; `t = cursor + [0..206]` repeated for both stereo channels (rows 0..206, 207..413),
    `h = 0`, `w = w_low` for the first channel's rows and `w_high` for the second, where `(w_low, w_high)` = first/last value of the video w axis (`model.py:366`).
  * video rows (`_video_grid`, `model.py:126-130`): per latent frame `t = cursor + exclusive-cumsum of spans`, `spans[k] = (5/3) * (1,4,4,4,4)[k%5]`
    (`model.py:95-102`: frame 0 -> `L+0`, 1 -> `L+5/3`, 2 -> `L+25/3`, 3 -> `L+15`, ..., 36 -> `L+200`), with the **same `cursor = L`** as the audio;
    `(h, w)` = the area-normalised per-patch grid below, repeated for every latent frame (row order t, h, w).
  * Spatial axes (`_axis_from_sqrt_area`, `model.py:70-74, 88-92`): `area = sqrt(lat_h*lat_w) = sqrt(48*84) = 63.498...`; for an axis of latent size `dim` (48 or 84) with `n = dim//2` patches:
    `coord[i] = (i * ((dim/area)/n) + (1 - dim/area)/2) * 32` (float64). For 48: `3.905...` .. `27.087...`; for 84: `-5.166...` .. `36.158...`.
* `rope_freqs` (`model.py:523-530`): `pos = position_ids.to(float32)`; `per_axis = pos.unsqueeze(-1) * inv_freq.view(1,1,16)` -> `[S,3,16]` fp32 (`inv_freq` is the checkpoint buffer `rope.inv_freq`, 16 values, **UNDETERMINED FROM CODE**);
  `half = cat(t_f, h_f, w_f)` `[S,48]`; angles `[S,96] = cat(half, half)`.
* `rope_rotation_table(angles, bf16)` (`model.py:149-155`): `ang = angles[:, :48]`, `c, s = cos(ang), sin(ang)` (**fp32**), `table = stack([c, -s, s, c])` -> `[1, S, 1, 48, 2, 2]` and **rounded to bf16**.
* Application (`Attention.forward`, `model.py:177-190`): per block, `ck.rms_rope_split_half_(q, k, rope_freqs, q_norm.weight, k_norm.weight, epsilon=1e-5, rot_dim=96)` in place on the qkv buffer.
  Kernel semantics (`CK/backends/cuda/ops/rms_rope.cu:30-230`, `rope_device.cuh:65-98`): per (token, head) row: `sum = fmaf-chain of x^2 over 128 dims (lane-strided, then warp shuffle-down tree 16,8,4,2,1)`, `rrms = rsqrtf(sum/128 + eps)`;
  each element `x' = bf16( float(x) * rrms * float(scale[d]) )` (rounded to bf16 **before** rotation); the first 96 dims are rotated as split-half pairs `(i, i+48)`:
  `y0 = f00*x0 + f01*x1`, `y1 = f10*x0 + f11*x1` in **fp32** (matrix `[[c,-s],[s,c]]` from the bf16 table), each rounded to bf16; dims 96..127 are only normalised.
  `v` is never normalised or rotated.

### 5.6 One block, op order (50 blocks; stock ComfyUI, `model.py:158-246, 278-297`)

Shapes at S = L + 37,710 (write `S` for it); `x` bf16 `[S, 5376]` throughout:

| step | op | out shape / dtype |
|---|---|---|
| adaln | `adaln_proj(t_emb)` -> 6 x `[M*3, 5376]` | bf16 (curve: fp32) |
| 1 | `h = norm1(x)` (RMSNorm eps 1e-5) then per-segment `h *= (1 + scale_msa[row])`, `h += shift_msa[row]` | `[S, 5376]` bf16 |
| 2 | `qkv = qkv_proj(h)` (`Linear 5376 -> 3*7168`, no bias), `split(7168)` | q,k `[S,7168]`, v `[S,7168]` bf16 |
| 3 | `v.view(S,56,128)`; `q,k .view(1,S,56,128)`; fused per-head RMSNorm(q_norm/k_norm, eps 1e-5) + partial RoPE (rot_dim 96) in place | bf16 |
| 4 | `q,k,v -> [1,56,S,128]` (`transpose(0,1).unsqueeze(0)`), `optimized_attention(q,k,v, heads=56, mask=None, skip_reshape=True)` full non-causal softmax(QK^T/sqrt(128))V, scale default `128**-0.5` | `[1, S, 7168]` bf16 |
| 5 | `out_proj` (`Linear 7168 -> 5376`, no bias) | `[S, 5376]` bf16 |
| 6 | gated residual `x[a:b].addcmul_(out[a:b], gate_msa[row])` per segment | bf16 |
| 7 | `h = norm2(x)`, modulate with `scale_mlp/shift_mlp` | bf16 |
| 8 | `fc1` (`Linear 5376 -> 28672`, no bias) -> `[S, 28672]`; `linear_input_act(fc2, ., "swiglu")`: `gate, up = x.chunk(2, -1)`, `silu(gate) * up` (`ops.py:949-951`, **first half = gate**), `fc2` (`Linear 14336 -> 5376`) | `[S, 5376]` bf16 |
| 9 | `x[a:b].addcmul_(mlp_out[a:b], gate_mlp[row])` | bf16 |

Nothing masks attention (mask is `None`): every token (text, audio, video) attends to every token.

### 5.7 Final layer, PDD heads, unpatchify (`FinalLayer`, `model.py:300-347`; `_forward` `767-784`)

```python
shift, scale = adaln_proj(t_emb)                              # modalities=1 -> 2 x [M, 5376]
mod(seg) = ( norm(x[a:b]) * (1.0 + scale[row].to(scale.dtype)) + shift[row].to(shift.dtype) ).to(float32)
```

(`norm` is RMSNorm eps 1e-5; for a bf16 `adaln_proj` the multiply-add is bf16 before the cast to fp32, for the curve form `scale` is fp32 so the product is fp32.)
`video_seg = (L+414, S, t_row[t_v])`, `audio_seg = (L, L+414, t_row[t_a])`. Heads are the "fp32 island": `video_out` `Linear(5376 -> 96)`, `audio_out` `Linear(5376 -> 32)`, fp32 weights/bias.

* `n == 1`: `video_out(mod(video_seg))` `[37296, 96]`, `audio_out(mod(audio_seg))` `[414, 32]` fp32.
* `n > 1` (PDD): `i = argmin(|sample_sigmas - sigma_v|)`; `sigma_next = sample_sigmas[min(i+1, 8)]`;
  `start, stop = (round(float(1.0 - time_shift_sigma(s, shift_v, 1.0)) * n) for s in (sigma_v, sigma_next))` (Python `round`, half-to-even;
  `time_shift_sigma(s, 12, 1)` is the **unshifted base-grid sigma** `s / (12 + s*(1-12))`), `start = min(start, n-1)`, `stop = max(stop, start+1)`.
  For this 8-step schedule `1 - sigma_base` is, in fp32: step 0: `0.0 -> 0.12499994039535522`; 1: `.. -> 0.25000011920928955`; 2: `.. -> 0.3749999403953552`; 3: `.. -> 0.5000001192092896`;
  4: `.. -> 0.6249999403953552`; 5: `.. -> 0.75`; 6: `.. -> 0.875`; 7: `.. -> 1.0` (each step's `stop` value is the next step's `start`). These sit within 1e-7 of exact eighths,
  so for any `n` with `k*n/8` half-integral the rounding direction is decided by fp32 noise: replicate the fp32 arithmetic exactly (`1.0 - tensor` and the tensor ops in `time_shift_sigma`, then `float(...)*n`).
  `_pdd_head` (`model.py:338-347`): `grid = linspace(1.0, 0.0, n+1, float64)`; `dt = (1.0 - flow_shift*grid/(1 + (flow_shift-1)*grid)).diff()[start:stop]`
  (flow_shift = 12 for video, 3 for audio); `w = (dt / dt.sum()).to(fp32)`; `rows = weight.reshape(n, -1, 5376)`, `brows = bias.reshape(n, -1)`;
  `first = max(start, 1)`; effective head `W = rows[0] + einsum("n,noi->oi", w[first-start:], rows[first:stop])`, `b = brows[0] + einsum("n,no->o", w[first-start:], brows[first:stop])`
  (row block 0 is the full head, later blocks are offsets from it; note that block 0 gets coefficient 1, not `w[0]`); then `F.linear(h_fp32, W, b)`.
  All fp32. The weight is cast with `CastBiasWeightContext(head, h)` to `h.dtype = fp32`.
* `unpatchify_video(v, 37, 24, 42, 24)` (`model.py:52-56`): `reshape(-1,37,24,42,24,1,2,2)`, `einsum("nthwcrpq->nctrhpwq")`, `reshape(-1,24,37,48,84)`, crop to the original `[:, :, :37, :48, :84]`.
  `unpack_audio(a)`: `a.reshape(2, 207, 32).permute(2, 0, 1).unsqueeze(0)` -> `[1, 32, 2, 207]`.
* Return: `[-video_out.to(bf16), -audio_out.to(bf16)]` (`model.py:784`; **the model returns the negated velocity** so that `denoised = x - out*sigma`); i.e. fp32 head output negated then **rounded to bf16**.
  Then section 5.1's audio un-carry, in bf16.

### 5.8 What `tfvideo/minimax_h3.py` replaces, and what stays ComfyUI

The TF loader builds ComfyUI's `MiniMaxH3Model` with **no blocks** and swaps `model.diffusion_model.blocks` for 50 `EngineBlock`s (`TF/tfvideo/comfy_nodes.py:178-214`, `minimax_h3.py:367-382`).

Stays ComfyUI (exact code above): `forward` audio carry, `_forward` packing / patchify / `video_patch_proj` / `audio_patch_proj`, `condition_proj` + token refiner (`extra_conds`),
`TimeEmbedder` or curve lerp (`t_emb`), rope table, the unique-timestep / modulation-row bookkeeping, `FinalLayer` + PDD heads, unpatchify, sign/dtype of the outputs, the sampler, `ModelSamplingAV`, LoRA key mapping,
`comfy_kitchen.rms_rope_split_half_` (the engine calls the **same kernel** with the same `rot_dim`/eps, `minimax_h3.py:275-312`).

Replaced by the engine (`minimax_h3.py:265-325`):

| ComfyUI op | engine |
|---|---|
| `adaln_proj` linear | `torch.addmm(ada_b, t_emb.float(), ada_w.t())` in **fp32**, viewed `[M*3, 6, 5376]` (`modulation`, `minimax_h3.py:265-270`); no SiLU (curve form) |
| `norm1/norm2` + `(1+scale)*h+shift` (3 bf16 roundings) | `K.norm_mod`: fp32 `y = (x * rsqrt(mean(x^2)+eps) * w) * (1 + scale) + shift`, one bf16 rounding; norm weights stored fp32 (`tfvideo/kernels.py:20-45`, `store.py:75-81`) |
| gated residual `addcmul_` | `K.gate_add`: fp32 `x + y*g`, bf16 store (`kernels.py:48-66`) |
| `qkv/out/fc1/fc2` linears | NVFP4 W4A4 TensorFold GEMMs (`linear.py`), activations quantized per call or by delayed absmax x 2 (`update_delayed`, `DELAY_MARGIN`); LoRA merged into the weights before quantization |
| SwiGLU | `K.swiglu`: fp32 `g/(1+exp(-g)) * u`, bf16 store (`kernels.py:69-86`), MLP processed in 8192-row chunks |
| attention (dense) | `engine.attn` = `A.get("auto")` = `comfy_kitchen.sage_attention.int8_attention` (INT8, SageAttention-style) (`attention.py:44-54`) |
| attention (when the Model Sparse Attention patch is eligible) | the patch's `attention` callable (section 5.9) driving `AttnFacade` -> the engine's NVFP4 `qkv`/`out` |

Engine block order (`minimax_h3.py:275-325`): `mod = addmm(...)`; `h = norm_mod(x, norm1, mod, part 0)`; `qkv = lin_qkv(h)`; `rms_rope_split_half_`; attention; `out = lin_out(o)` written into `h`'s buffer; `gate_add(x, h, mod, part 0)`;
`norm_mod(x, norm2, ..., out=h)`; per 8192-row chunk `swiglu(lin_fc1(h))` -> `lin_fc2`; `gate_add(x, h, mod, part 1)`.

### 5.9 Sparse attention (`BlockSparseAttention`, `comfy_extras/nodes_sparse_attention.py`)

* Patch registration (`apply_block_sparse_attention`, `:323-350`): `percent_to_sigma(start 0.2) = time_snr_shift(12, 0.8) = 0.9795918367346939` (python float) and `percent_to_sigma(1.0) = 0.0`; a per-block "dit"/"double_block" patch is installed on every block (`:345`).
* Per block call (`make_h3_block_patch`, `:310-320`; `h3_eligible` `:229-245`; `dense_reason` `:68-82`): sparse iff `x.dtype == bf16`, CUDA, head_dim 128, `rope_freqs is not None`, **`sigma_end <= float(transformer_options["sigmas"][0]) <= sigma_start`**,
  `S >= min_tokens (12288)`, block not in `dense_blocks`, and `ck.sol_attn_is_available`. With the 8-step schedule: **steps 0 (sigma 1.0) and 1 (0.98824) are dense; steps 2..7 are sparse** (`0.97297 <= 0.97959`).
  Dense calls fall through to the block's original attention (engine int8 attention in the TF workflow; `optimized_attention` + the installed override that also declines -> stock `optimized_attention` in a pure-ComfyUI run).
* Sparse call (`h3_sparse_attention`, `:248-307`): `qkv` is produced in chunks of **4096 rows** (`PRODUCER_CHUNK`), each `attn.qkv_proj(x[i:i+4096])` (so the engine's NVFP4 qkv linear sees 4096-row slices),
  then `ck.sol_attn_chunked(chunks, n, heads=56, freqs, (q_norm_w, k_norm_w), kmean, vscale, tau=1.3, topk_ratio=0.0, token_aug=256, sink_blocks, sink_q, rope_eps=1e-5)` which fuses the same RMSNorm+RoPE into the int8-carrier producer,
  and `attn.out_proj(out)` on the result. Statistics `(kmean, vscale)` `[56,128]` fp32 per `(block_index, n_tokens, uuids)` are **carried from the previous sparse step** (`first` call passes `None`, `:248-307`); they are reset in ON_CLEANUP.
* Sinks (`SparseAttnPatch.sinks`, `:84-100`) with `exact_kv_and_rows` and 64-row blocks: exact-KV blocks `[0, ceil((L+414)/64))` (the whole text+audio prefix, the video starts at row `L+414`),
  dense-query blocks `[floor(L/64), ceil((L+414)/64))` (the target-audio query rows run dense).
* Kernel semantics (training-free Sol-Attn, arXiv 2607.24027, 64-token blocks) are in `CK/__init__.py:144-208` (docstring) and `CK/backends/cuda`; not reproduced here.

---

## 6. Video VAE decode (`comfy/ldm/minimax/vae.py`, `MiniMaxH3VideoVAE`)

Only the **decoder** is used in T2V (the 3D-CNN encoder, `quant_conv` and `post_quant_conv`'s encoder-side counterpart are never run; `post_quant_conv` *is* used).

### 6.1 Loader / dtype (`comfy/sd.py:1023-1056, 1085-1110`)

* `VAE(sd)` detects `decoder.transformer_blocks.0.scale1` + `encoder.down.5.block.0.conv1.weight`. If any key ends `.comfy_quant` (`detect_layer_quantization`) the ops are
  `comfy.ops.mixed_precision_ops(quant, dtype or float16)` (**the workflow's int8_convrot file takes this path**), else `disable_weight_init`.
* `working_dtypes = [fp16, fp32]`: `vae_dtype()` returns fp16 whenever `should_use_fp16(device)` (true for compute capability >= 8, `model_management.py:1290-1305, 1867-1904`), so
  **the decoder runs in fp16** and `first_stage_model.to(fp16)` converts every parameter **and floating buffer** (`LATENTS_MEAN/STD`, `pixel_mean`, `pixel_std`, `qk_norm_scale`, rope `inv_freq`) to fp16.
  `process_output` is the identity, `handles_tiling = True`, `comfy_has_chunked_io = True`.
* `VAEDecode.decode` (`nodes.py:333-341`): nested latent -> `latent.unbind()[0]` (the video stream, fp32 CPU `[1,24,37,48,84]` after `process_latent_out`); `VAE.decode` (`sd.py:1251-1325`) moves it to the GPU **as fp16**
  (`samples_in[x:x+batch].to(device, dtype=vae_dtype)`), pre-allocates `pixel_samples = empty(decode_output_shape, fp32, CPU)` and calls `first_stage_model.decode(samples, output_buffer=...)`, then
  `.movedim(1, -1)` -> `[1,124,768,1344,3]` fp32, and `VAEDecode` reshapes to `[124, 768, 1344, 3]`.

### 6.2 `decode(z)` (`vae.py:793-805`) and temporal chunking (`704-764`)

```python
z = z * latents_std + latents_mean            # fp16 buffers (fp16 rounded copies of the constants), fp16 mul + add
# z.shape[2] == 37 > 1 -> decode_temporal
```

Constants (`vae.py:429-438`): `vae_ratio = 16`, `vae_ratio_t = 4`, `clip_length 17`, `token_drop 3`, `tokens_chunk_size = ceil(17/4) = 5`, `token_overlap = (-3) % 5 = 2`,
`frame_pre_padding = (-17) % 4 = 3`, `frame_overlap = max(2*4 - 3, 0) = 5`.
For `z_len = 37`: `pseudo_total = 37 + 3 = 40`, `pad_tokens = 0`, **`num_chunks = 40//5 - 1 = 7`**; decode clip `i` takes latent frames `[5i, 5i + 7)` (7 frames: `0:7, 5:12, ..., 30:37`),
each decoded to `7*4 = 28` frames; `chunk_dec = 20`, `split_count = 2`:

* `j = 0`: frames `[0, 20)` of the clip, drop the first `frame_pre_padding = 3` -> 17 frames; if a carried overlap exists, **blend** it with the first 5 frames (`blend(dec_overlap, chunk, 5, dim=-3)`), then `write_part` (finalise + copy into the output buffer).
* `j = 1`: frames `[20, 28)`, drop 3 -> 5 frames, kept as `dec_overlap` (`.contiguous()`); on the last clip it is written out directly.
* Total frames `7*17 + 5 = 124` (`_decode_temporal_frame_plan`, `:668-690`, and the 17k+5 <-> 5k+2 mapping, `sd.py:1032-1034`).
* `write_part` -> `_finalize_pixels` (`:481-484`): `part * pixel_std.to(float32)` (fp16 decoder output promoted to fp32 by the multiply; **`pixel_std`/`pixel_mean` are the fp16-rounded ImageNet constants cast back to fp32**),
  `.add_(pixel_mean)`, `.clamp_(0, 1)` -> fp32 in [0,1]; then copied into the fp32 output buffer. Pixel std/mean: `(0.229, 0.224, 0.225)`, `(0.485, 0.456, 0.406)` (`vae.py:17-18`) before fp16 rounding.
* `blend(a, b, extent, dim)` (`:531-554`): `extent = min(a.shape[dim], b.shape[dim], extent)`; `positions = arange(extent, dtype=b.dtype)`; `w_a = 1 - positions/extent`, `w_b = positions/extent` (**computed in b's dtype = fp16**);
  `blended = a[-extent:] * w_a + b[:extent] * w_b` (fp16 mul, mul, add); the rest of `b` is concatenated unchanged.

### 6.3 Spatial tiling per clip (`tiled_decode`, `:597-635`; `split_tiles`, `:507-529`; `tile_size 256`, `tile_overlap_min 64`, in pixels)

`split_tiles(len)`: `N = ceil(len/256)`, grow `N` until `256*N - 64*(N-1) - len >= 0`, distribute `remaining // 16` units of 16 px round-robin over the `N-1` overlaps, starts = cumulative `256 - overlap`.

* Height 768 -> 4 tiles, starts `[0, 160, 336, 512]`, overlaps `[96, 80, 80]` (px).
* Width 1344 -> 7 tiles, starts `[0, 176, 352, 528, 704, 896, 1088]`, overlaps `[80, 80, 80, 80, 64, 64]` (px).
* Each tile is `[1, 24, 7, 16, 16]` latent (256 px); 28 tiles per clip, decoded `batch` at a time where `batch = int(max(1, min(4, free_vram // (128 MiB))))` (`_decode_tile_row`, `:588-595`) -> **the number of tiles per decoder call depends on free VRAM** (pin item: a different `M` may change cuBLAS/cuDNN kernel choice; the math is per-tile).
* Rows are decoded top to bottom; for each tile the **y blend** (`blend(row_tails[j], tile, y_overlap[i-1], dim=-2)`) happens first, then the **x blend** (`blend(left_tail, tile, x_overlap[j-1], dim=-1)`),
  where the tails are `.clone()`s of the *un-blended* decoded tile's last `y_overlap[i]` rows / `x_overlap[j]` columns (`:616-622`); then the tile is cropped by the trailing overlap (`:623-626`) and copied into a pre-allocated canvas (`:627-629`). All in fp16.

### 6.4 The ViT3D decoder, one tile (`ViT3DDecoder`, `vae.py:344-402`)

Config (defaults): `patch_size 16`, `patch_size_t 4`, `in_channels 24`, `out_channels 3`, `num_layers 36`, `heads 32`, `dim_head 64` -> `dim 2048`, `rope_theta 100`, `rope_dim_ratio 0.75`, `eps 1e-5`, `num_register_tokens 4`.

```text
x            [1, 24, 7, 16, 16]  fp16
post_quant_conv (Conv3d 24->24, k=1)                               -> [1,24,7,16,16]
h = x_embedder(x.flatten(2).transpose(1,2))        Linear 24->2048 -> [1, 1792, 2048]
h = cat([h, register_tokens(4), zeros(1 token)], dim=1)            -> [1, 1797, 2048]   (4 register + 1 zero suffix)
ids = create_token_ids((7,16,16))  # per axis: (arange(0.5, n, dtype=fp16) / n) * 2 - 1, computed in fp16 (x.dtype); meshgrid 'ij', flattened t-major; suffix ids = 0
rotary = RotaryEmbeddingND(48, 100, 3)(ids):  inv_freq = 1 / 100 ** arange(0, 1, 0.125) (8 values, fp32 buffer -> fp16 after .to(fp16)),
         angles = (2*pi) * ids.float()[..., None] * inv_freq[...] -> flatten(2,3) (3 axes x 8 = 24 pairs = 48 dims),
         table = stack([cos, -sin, sin, cos]) -> [1, 1797, 1, 24, 2, 2] fp16
for each of 36 blocks:
   qkv = linear_input_act(to_qkv, x, "rms_norm", norm1.weight, 1e-5)  # = Linear(2048->6144)(rms_norm(x))   [fp16]
   qkv.view(1, S, -1, 3*64) -> [1, S, 32, 192]; q, k, v = chunk(3, -1)
   q, k = ck.rms_rope_split_half_(q, k, rotary, qk_norm_scale (ones), eps 1e-5, rot_dim 48)   # per-head RMSNorm (non-affine) + rope on the first 48 of 64 dims
   q,k,v -> [1, 32, S, 64];  out = optimized_attention(q, k, v, 32, skip_reshape=True)  # SDPA, non-causal, [1, S, 2048]
   x = x + scale1 * Linear(2048->2048)(nan_to_num(out))                                  # addcmul(residual, out, scale1)
   h = Linear(2048->16384)(rms_norm(x, norm2.weight, 1e-5))                              # w1, bias
   x = x + scale2 * Linear(8192->2048)(silu(h[:, :, :8192]) * h[:, :, 8192:])            # w2, bias, swiglu: first half = gate
out = proj_out(norm_out(x))   # LayerNorm(2048, eps 1e-5, affine) then Linear 2048 -> 3*4*16*16 = 3072
out = out[:, :1792].view(1, 7, 16, 16, 3, 4, 16, 16).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 3, 28, 256, 256)
```

* `qk_norm_scale` is a non-persistent ones buffer (`vae.py:286`), fp16 after the module cast; `norm_q/norm_k` (`ops.RMSNorm(elementwise_affine=False)`) are only used if `rotary_pos_emb is None` (not the case).
* The fp16 (non-quantized VAE) path: `linear_input_act` with a plain `Linear` weight runs `linear(_eager_input_act(...))` = `F.linear(F.rms_norm(x, (2048,), weight, eps), W, b)` (`ops.py:1003-1014`) unless `--fast fp16_accumulation` is set, in which case kitchen's `fp16_linear` GEMM is used (`ops.py:1003-1013, 968-972`;
  off by default, pin item). The `addcmul` is `torch.addcmul(residual, out, residual_scale)` fp16 (`ops.py:991-994`).
* **int8_convrot VAE (the workflow's file)**: the `to_qkv/to_out/w1/w2` weights that carry `comfy_quant` are `TensorWiseINT8Layout` with ConvRot; `linear_input_act` goes to `comfy_kitchen.int8_linear` (rms_norm / swiglu folded into the activation quantizer; activations Hadamard-rotated per 256 and quantized per row; int8 GEMM; fp32 weight scale epilogue; residual addcmul fused in the epilogue), and attention goes through
  `ck.int8_attention(query, key, value)` instead of SDPA (`vae.py:310-313`). Their exact arithmetic is in `CK/backends/cuda/ops/int8_linear.cu`, `cutlass_gemm_int8.cu`, `convrot_w4a4.cu`/`per_tensor_quantize.cu`, `CK/sage_attention.py` and is **not restated here**; which layers are quantized is
  **UNDETERMINED FROM CODE** (checkpoint `comfy_quant` keys). Layers without `comfy_quant` (`x_embedder`, `proj_out`, probably the norms) stay fp16.
* RoPE table is rounded to fp16 (`vae.py:259`) and used by the same kitchen kernel (HasRms path: bf16 -> here fp16 input, fp32 math, fp16 stores).
* `cudnn`/SDPA backend for the fp16 SDPA (head_dim 64, S = 1797, `[1,32,S,64]`): `FLASH` if available, else `CUDNN`, else `EFFICIENT` (`ops.py:69-96`; pin item).
* `NVIDIA_MEMORY_CONV_BUG_WORKAROUND` (`ops.py:102-111, 616-620`) only changes which conv call `post_quant_conv` uses on torch 2.9-2.10 + cuDNN 9.10.2..9.14 (`torch.cudnn_convolution(..., allow_tf32=True)`); a 1x1x1 fp16 conv is a GEMM either way.

### 6.5 Output frame tensor and conversion to uint8

* `VAE.decode` returns fp32 `[1, 124, 768, 1344, 3]` in [0, 1] (`sd.py:1325`), `VAEDecode` reshapes to `[124, 768, 1344, 3]` (`nodes.py:339-341`), `CreateVideo` keeps it as `components.images`.
* `save_to` converts each frame (`video_types.py:1176`): **`img = (frame * 255).clamp(0, 255).byte().cpu().numpy()`**: fp32 multiply by 255, clamp, then `.byte()` = **truncation toward zero** (not rounding). (Only when `bit_depth == 8`, as in this workflow; the 10-bit path uses `rgb48le` and is not taken, `:1171-1175`.)

---

## 7. Audio VAE decode (`comfy/ldm/minimax/audio_vae.py`, `MiniMaxH3AudioVAE`)

* Loader (`comfy/sd.py:1058-1082`): key `pre_block.attn.zero_k_bias` -> `MiniMaxH3AudioVAE()` (plain `disable_weight_init` ops, weight-norm already folded into plain conv weights), `latent_channels 32`, `output_channels 2`,
  `upscale_ratio 800`, `audio_sample_rate = 32000`, `working_dtypes = [fp32]` -> **fp32 everywhere** (`vae_dtype()` finds no fp16/bf16 entry in the list and falls through to `torch.float32`, `model_management.py:1290-1305`).
* `VAEDecodeAudio` (`nodes_audio.py:98-112`): nested latent -> `latent.unbind()[-1]` = audio `[1, 32, 2, 207]` fp32 (already divided by 4 by `process_latent_out`); `vae.decode(latent)` (`sd.py:1251-1325`, `latent_dim 2`, 4-D input so no squeeze) then `.movedim(1, -1)`, back to `[B, 2, L]` by `.movedim(-1, 1)`.

`decode(z)` (`audio_vae.py:416-425`), `z` `[1, 32, 2, 207]`:

1. `z = z.permute(0, 2, 1, 3).reshape(b*s = 2, 32, 207)` (stereo channels become batch items; channel 0 first).
2. `z = z * latents_std.view(1,-1,1) + latents_mean.view(1,-1,1)` (fp32, per channel; checkpoint buffers).
3. `x = dec_in_proj(z)`: `Conv1d(32 -> 2048, k=1)`.
4. `BigVGAN` (`audio_vae.py:308-369`, `num_mels 2048`, `upsample_initial_channel 1024`):
   `conv_pre` `Conv1d(2048 -> 1024, k7, pad 3)`; for `i in 0..6`: `ConvTranspose1d(1024/2^i -> 1024/2^(i+1), k, u, pad (k-u)//2)` with `(u, k)` = `(5,9), (5,9), (2,4), (2,4), (2,4), (2,4), (2,4)` (total x800);
   then the **average of 3 `AMPBlock1`** (kernels 3, 7, 11, dilations `(1,3,5)`): `xs = rb0(x); xs += rb1(x); xs += rb2(x); x = xs.div_(3)` (`:360-366`).
   `AMPBlock1.forward` (`:297-305`): for each of 3 `(c1, c2, a1, a2)`: `xt = a1(x); xt = c1(xt); xt = a2(xt); xt = c2(xt); x = xt.add_(x)`; `c1` dilated k, `c2` dilation 1; activations `Activation1d(SnakeBeta)`.
   `Activation1d` (`:137-150`): `UpSample1d(2, kernel 12)` -> `SnakeBeta` -> `DownSample1d(2, kernel 12)`;
   `UpSample1d` (`:86-104`): `pad = 12//2 - 1 = 5`, `F.pad(x, (5, 5), "replicate")`, `F.conv_transpose1d(x, filter.expand(C,-1,-1), stride=2, groups=C).mul_(2)`, crop `[pad_left : -pad_right]` with `pad_left = 5*2 + (12-2)//2 = 15`, `pad_right = 5*2 + (12-2+1)//2 = 15`;
   `DownSample1d` (`:107-134`): `F.pad(x, (5, 6), "replicate")`, `F.conv1d(x, filter.expand(C,-1,-1), stride=2, groups=C)`;
   the 12-tap Kaiser-sinc filters are buffers (`kaiser_sinc_filter1d`, `:58-83`: `cutoff 0.5/ratio`, `half_width 0.6/ratio`, `torch.kaiser_window(12, beta, periodic=False)`, `torch.sinc`, normalised to sum 1) and are persistent buffers, so the checkpoint's values (if present) win.
   `SnakeBeta` (`:42-53, 24-27`): `alpha = exp(alpha_param).view(1,-1,1)`, `beta = exp(beta_param)...`; `t = sin(alpha*x); t.mul_(t).mul_(1/(beta + 1e-9)).add_(x)` (the order: `(beta + 1e-9).reciprocal()`).
   Tail: `activation_post = Activation1d(SnakeBeta(8))`, `conv_post` `Conv1d(8 -> 1, k7, pad 3, bias False)`, `.clamp_(-1, 1)`. `[2, 1, L]` with `L = 207 * 800 = 165,600`.
5. `x.reshape(b, s, -1)` -> `[1, 2, 165600]` (channel 0 = first stereo channel).

Post-processing in `vae_decode_audio` (`nodes_audio.py:108-112`), fp32:

```python
std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0     # unbiased std over (channels, samples)
std[std < 1.0] = 1.0
audio /= std                                               # only attenuates when 5*std > 1
return {"waveform": audio, "sample_rate": 32000}           # [1, 2, 165600] fp32, CPU; "sample_rate" not in the latent dict -> VAE rate 32000
```

The encoder (`Encoder`, `AttnProjection`) is not used for T2V.
Backend caveat: all convs are cuDNN fp32 convs with `torch.backends.cudnn.allow_tf32` at its PyTorch default (True for convolutions; ComfyUI never changes it): **TF32 may be used inside the fp32 audio VAE** (pin item; verify on the target torch build). The first-time `torch.kaiser_window`/`sinc` filter construction is on CPU/fp32 and only matters if you do not load them from the checkpoint.

---

## 8. mp4 writing

`CreateVideo.execute` (`nodes_video.py:248-261`): `VideoFromComponents(VideoComponents(images=[124,768,1344,3] fp32, audio={"waveform":[1,2,165600] fp32, "sample_rate":32000}, frame_rate=Fraction(24)), bit_depth=8, color_space="sRGB")`;
`codec` stays `none`. `SaveVideo.execute` (`:166-203`): `format "auto"`, codec `auto`, `encoding` none -> `crf=None`; output name `{prefix}_{counter:05}_.mp4`; metadata `{"prompt": ..., "workflow": ...}` (JSON-dumped per key into the container, unless `--disable-metadata`).
`save_to(path, format=MP4, codec=AUTO, metadata, crf=None)` (`video_types.py:1116-1205`), **library: PyAV (`av`)**:

* Container: `av.open(path, mode="w", format="mp4", options={"movflags": "use_metadata_tags+faststart"})` (`video_output_config`, `:164-186`; `AUTO` + `.mp4` -> MP4, `AUTO` codec -> `H264` for non-webm).
* Video stream: `output.add_stream("h264", rate=Fraction(round(24*1000), 1000) = 24)` (`VIDEO_ENCODERS[H264] = "h264"`, `:21-24`), `width = 1344`, `height = 768`, `pix_fmt = "yuv420p"`, `video_stream.options = {}`
  (`video_encoder_options(H264, crf=None, preset=None)` -> empty, `:210-221`; so **no `-crf`, no `-preset`: the encoder's own defaults**; PyAV resolves `"h264"` to `libx264` when the build has it (default CRF 23, preset medium)).
  Colour properties set on the codec context and on each frame: primaries BT.709, trc BT.709 (`VIDEO_COLOR_TRANSFERS["sRGB"]`, `:41+`), colorspace BT.709 NCL (`=1`), range MPEG (limited) (`set_video_color_properties`, `:189-194`).
* Per frame: `frame = av.VideoFrame.from_ndarray(uint8 [H,W,3], format="rgb24")` (uint8 from section 6.5), `frame = frame.reformat(format="yuv420p", dst_colorspace=BT709_NCL)` (swscale RGB->YUV, BT.709, default interpolation), `packet = video_stream.encode(frame)`, `output.mux(packet)`; then flush `encode(None)` (`:1170-1191`).
* Audio stream (added before the video loop, written after it): `waveform = waveform[0, :, :ceil(32000/24 * 124) = 165334]` (`:1162`; the 165,600-sample decode is trimmed to the video length 5.1667 s), `layout = "stereo"` (2 channels),
  `output.add_stream("aac", rate=32000, layout="stereo")` (mp4: no resample, `audio_sample_rate = source rate`, `:1160, 1163-1166`), `AudioFrame.from_ndarray(waveform.float().cpu().contiguous().numpy(), format="fltp", layout)`, `sample_rate = 32000`, `pts = 0`, a **single frame holding the whole waveform**
  (PyAV re-chunks it to the AAC frame size), `output.mux(audio_stream.encode(frame))`, flush `encode(None)` (`:1193-1205`). **No audio bit rate is set anywhere**: AAC bitrate is FFmpeg's native-AAC default for the stream (**UNDETERMINED FROM CODE**; check with `ffprobe` on a produced file), and x264/AAC output depends on the FFmpeg build and the x264 thread count.
* Frame rate is exactly 24/1; the reported duration is `124/24 = 5.1667 s` for video and `165334/32000 = 5.1667 s` (rounded up to an AAC frame boundary by the encoder) for audio.

---

## 9. LoRA (Turbo, 8 steps)

* File: `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` (lightx2v `Minimax-h3-Turbo`, listed in the workflow's note), in `models/loras`. Not present on this machine: **tensor names/ranks/alpha UNDETERMINED FROM CODE.**
* Node: `LoraLoaderModelOnly` strength_model **1.0** -> `load_lora(model, None, lora_name, 1.0, 0)` -> `comfy.sd.load_lora_for_models(model, None, lora, 1.0, 0)` (`nodes.py:756-769`), which TF replaces by its own hook for engine models (`TF/tfvideo/comfy_nodes.py:125-167`):
  * `key_map = comfy.lora.model_lora_keys_unet(model.model, {})` (includes, for `MiniMaxH3`, `key_map[<name without "diffusion_model.">] = <weight key>`, `comfy/lora.py:389-394`; plus `lora_unet_<path_with_underscores>` and the `diffusion_model.<path>` forms, `lora.py:187-198`),
    extended with `diffusion_model.blocks.{i}.{attn.qkv_proj|attn.out_proj|mlp.fc1|mlp.fc2}.weight` forms (`comfy_nodes.py:107-114`);
  * `comfy.lora.load_lora(convert_lora(lora), key_map)` (`lora.py:37-`; adapters in `comfy/weight_adapter/lora.py:LoRAAdapter.load`, keys `.lora_up.weight/.lora_down.weight` (also `lora_B/lora_A`, `lora.up/lora.down`, ...), optional `.alpha`, `.lora_mid.weight`, `.dora_scale`);
  * patches for the 200 block linears go to the engine (`transformer_options["tfvideo_loras"] = [(fingerprint, {key: adapter}, 1.0)]`), the rest (token refiner, `condition_proj`, ...) to ComfyUI's patcher via `add_patches(rest, 1.0)`.
* Merge for the block linears (TF engine, `comfy_nodes.py:_sync_loras` 95-123 and `store.py:convert` 84-125): the **source weight is rebuilt to fp32 `[N, K]` in the model basis** (`Checkpoint.linear`, `TF/tfvideo/source.py:95-111`: int8 `w = q.float() * weight_scale.float()`, then per 256-group `w = (w.view(n, k//256, 256) @ H256).view(n, k)` with the Hadamard `H = kron(H4,...)/16`, `H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`),
  then `merge(name, w) = comfy.lora.calculate_weight([(strength=1.0, adapter, 1.0, None, None)], w, name)` (`comfy/lora.py:~452-516`, `weight_adapter/lora.py:224-286`): all in **fp32**:
  ```python
  mat1 = up.to(fp32); mat2 = down.to(fp32)                         # cast_to_device(..., intermediate_dtype=float32)
  alpha = v[2] / mat2.shape[0] if v[2] is not None else 1.0        # alpha / rank
  lora_diff = torch.mm(mat1.flatten(1), mat2.flatten(1)).reshape(weight.shape)     # fp32 matmul (no TF32)
  weight += function(((strength * alpha) * lora_diff).type(weight.dtype))          # weight is fp32 here; strength = 1.0
  ```
  (`if strength_model != 1.0: weight *= strength_model` is skipped, `p[2] == 1.0`.) Then `w.to(bfloat16)` and quantization to NVFP4 (`store.py:100-105`, `linear.py:nvfp4_weight`: one global scale `amax/(6*448)`, `quant4`), cached per LoRA fingerprint.
  Mid-weights (`lora_mid`) or DoRA would take other branches (`weight_adapter/lora.py:253-266, 272-281`).
* Stock ComfyUI (no TF engine) would instead patch through `ModelPatcher.patch_weight_to_device` (`comfy/model_patcher.py:900-929`): weight cast to `lora_compute_dtype` (fp16 on GPUs where `should_use_fp16`, `model_management.py:2075-2086`), `calculate_weight`, then the layout's `set_func`
  (re-quantize with `scale="recalculate"` and a CRC32-of-key seed for int8/fp8 layouts) or plain `.to(bf16)` for bf16 weights (`comfy/float.py:65-71`: `bfloat16`/`float16` targets are plain round-to-nearest-even).
  For non-block weights (token refiner etc.) in the TF workflow this stock path applies: fp16 intermediate weight, adapter diff in fp32 -> fp16 add -> cast to the parameter dtype.
* Turbo changes the schedule to 8 steps (workflow switch), nothing else (shift stays 12/3, no CFG).

---

## 10. Backend-dependent numerics a bit-exact twin must pin

Grouped by stage; "ComfyUI default" = what the code does without flags.

**Global**
1. `torch.inference_mode()` for execution (`execution.py:751`); TF32: ComfyUI never sets `allow_tf32` (matmul default off, **cuDNN conv default on**); `--fast fp16_accumulation` is off by default (`model_management.py:553-559`, `cli_args.py:195-201`) but switches fp16 linears/convs to kitchen's fp16-accumulate GEMM (`ops.py:968-972, 1003-1013`, `vae.py:64-69`). Pin: no TF32 for matmul, no fp16 accumulation.
2. `torch.backends.cudnn.benchmark` stays False unless `--fast autotune` (`model_management.py:563-565`).
3. cuBLAS: fp32 SGEMM in the TE and in the DiT's patch/PDD heads, bf16 GEMMs elsewhere; `cublas workspace`/algorithm selection and split-K choices depend on shape; the TE fp32 shapes depend on `L`.

**Text encoder**
4. Device (GPU vs CPU, `text_encoder_device`), dequantize kernel (`dequantize_nvfp4` CUDA vs eager LUT; same formula, fp32), AWQ `pre_quant_scale` dtype handling, fp32 SGEMM, `F.rms_norm` fp32 (torch's fused kernel; `comfy/rmsnorm.py:5-6` notes it differs from a manual rsqrt), kitchen `apply_rope_split_half` fp32 (`y0 = f00*x0 + f01*x1` likely FMA-contracted by nvcc), `inv_freq = 1/(5e6 ** (arange(0,128,2)/128))` via CUDA `pow` fp32 and `cos/sin` fp32 on device.
5. TE attention: SDPA fp32 with a float mask and GQA; backend = EFFICIENT or MATH (flash/cudnn cannot take fp32), native-GQA vs `repeat_kv` decided by `torch.backends.cuda.can_use_*` for the installed torch; `SDP_BATCH_LIMIT`.

**Sampling**
6. Noise: CPU `torch.randn` fp32 from `mt19937(seed & 0xFFFFFFFF)`, video then audio from one generator; vector (AVX2/Sleef) vs scalar Box-Muller differ by hardware.
7. fp32 sampler arithmetic as in section 4.2 with CUDA `log`, `expm1`, `exp` (non-fast-math libdevice) and **separate kernels per elementwise op** (no fusion); the sigma tensor arithmetic of section 3.4 (fp32 on CPU for the buffer, fp32 CUDA in the DiT).
8. `xc.to(bf16)` of the packed state, bf16 negated velocity, `calculate_denoised` in fp32 (`x - out.float()*sigma`).

**DiT**
9. bf16 vs fp32 islands: fp32: `video_patch_proj`, `audio_patch_proj`, `TimeEmbedder` (or the curve lerp), `rope` angles (cos/sin), PDD/final heads and their weight combination, engine modulation; bf16: everything in the blocks, the rope table (bf16 cos/sin), `t_emb` for non-curve, the audio carry arithmetic, the model outputs.
10. Attention backend per step: dense steps 0-1 = comfy-kitchen INT8 attention (`attention="auto"`, `tfvideo/attention.py:44`), steps 2-7 = `sol_attn_chunked` with tau 1.3, `extra_tokens` 256, sinks as in 5.9, 4096-row qkv producer chunks, **cross-step carried `(kmean, vscale)` statistics**. In a pure ComfyUI run the dense backend would be the SDPA priority list FLASH -> CUDNN -> EFFICIENT -> MATH (`ops.py:69-96`; the README says flash is not built for sm_120 on Windows, so cuDNN).
11. NVFP4 activation scales in the engine: per-call exact absmax on the first forward of a run, then **delayed**: previous forward's absmax x 2 (margin doubles after a clipped forward, max 16) (`linear.py:update_delayed`, `minimax_h3.py:begin_forward/end_run`, `TFVIDEO_DELAYED_SCALES`); reset at ON_CLEANUP. The qkv chunking in sparse mode must use the same scales as the unchunked path.
12. The PDD `round()` ties (section 5.7) and `argmin` over the fp32 sigma list; `unique_t` float equality (python floats of fp32 values) deciding whether step 0 has 1 or 2 timestep rows.
13. In-place ops: `h[a:b].mul_().add_()` and `addcmul_` aliasing on the stream tensor (stock path) vs the engine's fp32 fused kernels (different rounding); `x` is a persistent, in-place-updated buffer.
14. LoRA merge: fp32 `mm` then `.type(fp32)` add, then bf16 round, then NVFP4 quantization (global scale from `amax`), ordering of multiple LoRA patches (one here).

**Video VAE**
15. fp16 weights/buffers (mean/std/pixel mean/std rounded to fp16, `qk_norm_scale`, rope table fp16), fp16 `blend` weights, `_finalize_pixels` mixing fp16 x fp32; SDPA backend for fp16 hd 64 (flash/cudnn) or kitchen INT8 attention + INT8 ConvRot linears for the int8 VAE; number of tiles per decoder call (`min(4, free_vram // 128 MiB)`); cuDNN conv algorithm for `post_quant_conv` (1x1x1).
16. Output conversion: fp32 `* 255`, `.clamp`, `.byte()` truncation.

**Audio VAE**
17. fp32 cuDNN conv1d / conv_transpose1d (grouped depthwise filters, k=1 convs): TF32 allowed by default for cuDNN; `exp`, `sin`, reciprocal in SnakeBeta; replicate padding; `torch.std` (unbiased) normalisation and the `<1.0 -> 1.0` floor.

**mp4**
18. PyAV/FFmpeg versions, libx264 build and thread count, swscale RGB->YUV420 BT.709 limited-range conversion and chroma downsampling filter, AAC encoder defaults (no bitrate set), single-frame audio feed, container metadata (JSON of workflow/prompt unless disabled).
