# Kandinsky-6 text conditioning

The text embedder of Kandinsky-6.0 Pro-distill 5 s (Hugging Face `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers`,
revision `1018d5828716e8bea3de99609f45c924de9c5564`), ported from upstream kandinsky-6 at
`eac06e39b904940baff2c408e925cdb0c4529269` (`kandinsky/core/components/text_embedder.py`) and transformers v5.18.0
(`models/qwen2_5_vl/modeling_qwen2_5_vl.py`, `models/clip/modeling_clip.py`, `masking_utils.py`,
`integrations/sdpa_attention.py`, `utils/output_capturing.py`, `activations.py`, `models/{qwen2,clip}/tokenization_*.py`).
No Python at run time.

| File | Contents |
| --- | --- |
| `text.cu` | `k6t_rope_table`, `k6t_rms_norm` (Qwen); `k6t_clip_embed`, `k6t_layer_norm_f32`, `k6t_linear_f32`, `k6t_quick_gelu`, `k6t_add_f32`, `k6t_attention_f32`, `k6t_cast_bf16` (CLIP) |
| `src/engines/kandinsky6/text/prompt.zig` | the Qwen template, truncation and right padding to 1153 ids + mask; the 77 CLIP ids, mask count, pooled row |
| `src/engines/kandinsky6/text/clip_tokenizer.zig` | CLIPTokenizer on TensorFold's tokenizer plus the raw `<\|endoftext\|>` split and Lowercase |
| `src/engines/kandinsky6/text/qwen25.zig` | `Kernels` (both encoders), the Qwen manifest, weights binding, scratch plan, `Encoder.forward` |
| `src/engines/kandinsky6/text/clip.zig` | the CLIP manifest, weights binding, scratch plan, `Encoder.forward`, `castPooled` |

## What upstream computes (Pro-distill, `text_token_padding = True`)

- Qwen: `_PROMPT_TEMPLATE.format(prompt)` (text_embedder.py:19-29, its spellings "promt" / "scren" kept), processor
  tokenizer with `max_length = 1153`, truncation, `padding="max_length"` (right, pad `<|endoftext|>` 151643), forward
  with `input_ids` and `attention_mask` only, `hidden_states[-1][:, 129:]` (the final RMSNorm's output:
  output_capturing.py:292-305) -> `[1024, 3584]` bf16 and the mask's crop as bool. The template prefix is 129 tokens
  exactly (host test).
- Positions: `arange(1153)` on all three MRoPE axes, padding included (no `mm_token_type_ids` reaches
  `compute_3d_position_ids`, modeling_qwen2_5_vl.py:1155-1189, :821-824); the [16, 24, 24] recomposition is then plain
  RoPE with theta 1e6 (configuration_qwen2_5_vl.py:154-155: legacy `mrope` = `default`).
- Attention mask: causal AND key padding as a 4D bool mask (masking_utils.py:170-180, 502-511), so SDPA runs a masked
  kernel with `repeat_kv`; only a prompt that fills all 1153 tokens skips the mask (`is_causal`, GQA in SDPA).
  Padded query rows see the valid keys; their outputs are part of the 1024 rows.
- CLIP: `CLIPTextModel` fp32 from its config (text_embedder.py:79-103), ids + attention mask of the 77-token call
  (:205-213); `create_causal_mask` gives causal AND padding (is_causal only when nothing is padded); pooled =
  `final_layer_norm` output at `argmax(input_ids)` (modeling_clip.py:564-576, `eos_token_id == 2`), i.e. the first
  `<|endoftext|>`. pipeline.py:467 casts both outputs to bf16 for the DiT.
- `torch.set_float32_matmul_precision("high")` (factory.py:162) is global when the embedder runs: every fp32 matmul
  (CLIP's Linears, Qwen's RoPE frequency product) may run TF32.

## Precision receipt

Both layers are bit-equality gates, as `kernels/cuda/kandinsky6/dit/README.md` sets out: production vs the twin (same
fatbins), and native upstream on SM121 (torch 2.14.1+cu130, transformers 5.18.0, `sdpa`, `high` matmul precision,
never forced to `highest`). A row marked UNQUALIFIED is not equal, not waived and not claimed until its native kernel
and parameters are captured on SM121 (`CUBLASLT_LOG_LEVEL=5`, the torch profiler) and this kernel matched to them or
the op declared unsupported.

### Qwen2.5-VL-7B text stack (28 layers, bf16)

| Op | Upstream (torch 2.14.1 CUDA) | Here | Native status |
| --- | --- | --- | --- |
| Tokenize, template, truncate, pad | tokenizers (Rust) Qwen2 BPE | TensorFold's tokenizer.json BPE + `prompt.qwenIds` | equal on the host tests (below); not exhaustive |
| `embed_tokens` | gather | `stk_embed` (te.cu) | exact |
| RoPE frequencies `inv_freq @ positions` (:557) | fp32 bmm, K = 1, under `high` (TF32 permitted) | `k6t_rope_table`, form = `Encoder.rope_freq` (no default): fp32 = one rounding; TF32 forms round the operands, product exact | UNQUALIFIED: algorithm uncaptured. `inv_freq` itself is an input: native's buffer bytes must be captured (`invFreqCandidate` is a host candidate, not qualified) |
| cos / sin (:558-559), `* attention_scaling` (1.0) | `cosf` / `sinf` fp32 kernels | same in `k6t_rope_table` (scaling by 1.0 omitted: exact) | elementwise: expected equal given the same frequencies |
| RMSNorm (input, post-attention, final) (:71-76) | `pow(2)`, vectorized `mean`, `+ eps`, `rsqrt`, `*`, `.to(bf16)`, `weight *` in bf16 | `k6t_rms_norm`: four accumulators per lane, 32 lanes, vector width 4; left-to-right combine and shuffle 16/8/4/2/1, then the same seven roundings | Seed-42 fixture: both observed layers, all 28 layer boundaries and final embeddings exact after matching ATen's reduction order |
| q / k / v Linear + bias | cuBLASLt bf16, fp32 compute, bias epilogue, heuristic algorithm | `stk_gemm_bf16` (gemm.cu): K ascending in tiles of 32, no split-K, bias added to the fp32 sum, one rounding | UNQUALIFIED: algorithm uncaptured |
| o_proj, gate_proj, up_proj, down_proj | cuBLAS(Lt) bf16 | `stk_gemm_bf16`, no bias | UNQUALIFIED: as above |
| RoPE apply (:590-613) | bf16 `q*cos + rotate_half(q)*sin`, each op rounded | `stk_rope_half` (te.cu): the same three roundings, cos / sin rounded to bf16 | elementwise: expected equal |
| Attention (sdpa) | Captured EFFICIENT_ATTENTION specialization `fmha_cutlassF_bf16_aligned_64x128_rf_sm80`, scale `128**-0.5` | `text/attention.cu`: pinned source specialization, full BF16 causal/padding bias, 64x128 tiles, scale 0x3db504f3 | Integrated Qwen seed-42 fixture: 49 comparisons exact, including all 28 layer boundaries and final embeddings; whole pipeline gates remain pending |
| Residual adds | bf16 add | `stk_add` | elementwise: expected equal |
| SiLU(gate) * up (:584-587) | `F.silu` bf16 (`x / (1 + exp(-x))` in fp32), bf16 product | `stk_silu_mul` | elementwise: expected equal |
| Crop rows 129.., mask crop | views | final norm written for rows 129..1152 only (rows independent) | exact |

### CLIP ViT-L/14 text tower (12 layers, fp32)

| Op | Upstream | Here | Native status |
| --- | --- | --- | --- |
| Tokenize (77, specials, truncation, padding) | tokenizers CLIP BPE (`</w>`), NFC, `\s+`, Lowercase | `clip_tokenizer.zig` + `prompt.clipIds` | equal on the host tests; Lowercase table Unicode 16.0.0 (Rust std's version must match), applied before NFC (see the file's comment) |
| Token + position embedding | gather, fp32 add | `k6t_clip_embed` | exact / elementwise equal |
| LayerNorm (layer_norm1/2, final) | `vectorized_layer_norm_kernel<float>` (block (32, 4), Welford) | `k6t_layer_norm_f32`: torch's expressions and launch shape, default FMA contraction retained | First-layer SM121 isolated fixture byte exact with contraction; disabling it gives 32,535 unequal bytes. Full encoder and all-layer gates remain pending |
| Linear + bias (q/k/v/out_proj, fc1, fc2) | Captured `cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4`, under `high` | `clip_linear.cu`, explicit `tf32_rn`: RN-even TF32 operands, pinned CUTLASS 64x64x16 tile, six stages, bias epilogue; other candidate forms retained explicitly | First-layer q/k/v exact after integration; full encoder gate remains pending |
| Attention (sdpa, causal + padding bool mask) | Captured `fmha_cutlassF_f32_aligned_64x64_rf_sm80`, scale 0.125 | `attention.cu`: pinned FP32 specialization with full 77-key causal/padding bias | Isolated reference backend output exact; integrated encoder gate remains pending |
| quick_gelu | `x * sigmoid(1.702 * x)` as three fp32 ops (`1/(1+exp(-t))`) | `k6t_quick_gelu` | elementwise: expected equal |
| Residual adds | fp32 add | `k6t_add_f32` | exact |
| Pooled row | `argmax(input_ids)` gather | host-side index, final LayerNorm on that row only | exact |
| `.to(bf16)` (pipeline.py:467) | cast | `k6t_cast_bf16` | exact |

## Tokenizer exactness evidence

TensorFold's tokenizer reproduces `tokenizers.encode(add_special_tokens=False)` for the Qwen2 pipeline as published
(NFC, isolated Split with the Qwen regex, ByteLevel, BPE). For CLIP it lacks the Lowercase normalizer, so
`clip_tokenizer.zig` checks the file's normalizer / post-processor / added tokens verbatim, runs the raw
`<|endoftext|>` split and the per-character lowercase itself, and hands TensorFold the rest. The expected ids in
`prompt.zig`'s tests come from an independent reference implementation of both tokenizer.json pipelines (ASCII
inputs, written from the JSON files, not shipped); the 129-token template prefix and well-known ids (151644, 8948,
198, 872; 320 "a</w>", 2368 "cat</w>", 1125 "photo</w>", 539 "of</w>") agree. The snapshot tests run with
`K6_SNAPSHOT_ROOT=<dir holding text_encoder/ and tokenizer_2/>` and skip without it. The `tokenizers` package was not
available on the development host, so no direct comparison against it was made; a fixture of HF ids for a prompt set
(including non-ASCII, whitespace runs and truncation) is part of the GPU-window capture.

## ABI

```zig
const qwen25 = @import("text/qwen25.zig");
const clip = @import("text/clip.zig");
var k = try qwen25.Kernels.load(&driver, .{ .text = k6_kernels.text, .qwen_te = qwen_image.kernels.te, .qwen_gemm = qwen_image.kernels.gemm, .text_attention = k6_kernels.text_attention });
const qw = try qwen25.bindWeights(&store); // store.get(name) ?u64, BF16 as stored
const cw = try clip.bindWeights(&store); // F32 as stored
var ids: [1153]u32 = undefined;
var mask: [1153]u8 = undefined;
const valid = try prompt.qwenIds(&qwen_tok, gpa, text, &ids, &mask); // DiT mask: mask[129..]
const q: qwen25.Encoder = .{ .k = &k, .s = stream, .w = &qw, .base = scratch_q, .inv_freq = captured_inv_freq, .rope_freq = math };
try q.forward(&ids, valid, text_embeds); // [1024, 3584] bf16
var cids: [77]u32 = undefined;
const c = try prompt.clipIds(&clip_tok, gpa, text, &cids);
const e: clip.Encoder = .{ .k = &k, .s = stream, .w = &cw, .base = scratch_c, .linear = math };
try e.forward(&cids, c.valid, c.eot, pooled_f32); // [768] f32
try e.castPooled(pooled_f32, pooled_bf16);
```

Scratch: `qwen25.scratchBytes()` = 123,385,344 bytes, `clip.scratchBytes()` = 2,602,496 bytes (256-byte aligned
bases, caller-owned). `Hook` sees every op's output by upstream module path for captures and fixture injection.

## Build and integration

`build/kandinsky6.zig` already compiles `text.cu` as `k6_fatbin_text` (torch's nvcc flags, `-O3 --fmad=false`,
`-gencode` sm_121a / sm_120a); `kernels/cuda/kandinsky6.zig` must expose it as `text`. The Qwen side also loads the
image engine's `te`, `gemm`, `attention` fatbins (`qwen_image.kernels`), compiled with that engine's flags. The
`kandinsky6` module root should reference `text/prompt.zig`, `text/clip.zig` (which pulls `text/qwen25.zig`) in its
tests; they need the `cuda` and `tokenizer` imports only.

## Provenance and licenses

- `text.cu`'s Welford statistics port PyTorch v2.14.1 `aten/src/ATen/native/cuda/layer_norm_kernel.cu` (BSD-3-Clause),
  copied from `kernels/cuda/kandinsky6/dit/ops.cu`; the rest of `text.cu` is LocalRouter's (Apache-2.0).
- Reused kernels: `kernels/cuda/qwen_image/{te,gemm}.cu` (LocalRouter) and `attention.cu` (derived from TensorFold's
  prefill attention, Apache-2.0, see NOTICE).
- `clip_tokenizer.zig`'s lowercase table is derived from the Unicode Character Database 16.0.0 (Unicode License v3);
  the tokenizer files themselves are read from the model snapshot and not distributed.

## Remaining for GPU fixtures (approved SM121 window)

1. Native captures at the oracle's settings for one prompt set: Qwen `input_ids` / mask, `rotary_emb.inv_freq` bytes,
   cos / sin, every layer's op outputs (hook names), `hidden_states[-1][:, 129:]`; CLIP ids / mask, per-layer outputs,
   `pooler_output`; plus HF tokenizer ids for non-ASCII and truncation prompts.
2. Algorithm evidence: cuBLASLt logs for the bf16 Linears (with / without bias), the fp32 Linears and the K = 1 RoPE
   bmm; the SDPA backend chosen for the Qwen bool mask, the unpadded `is_causal` case and the CLIP fp32 mask; the
   torch reduce config for the RMSNorm `mean`; SASS of `vectorized_layer_norm_kernel<float>`.
3. Per op: zig vs twin bit-equality, then vs native; each UNQUALIFIED row either matched or declared unsupported.
