# MiniMax H3 text encoder (fp32 twin): what runs, on which kernels

ComfyUI 0.37.0's Qwen3-VL 32B text path (`VIDEO-PORT.md` section 2), NVFP4 AWQ checkpoint, 50 layers, fp32 activations, output =
the raw residual stream after layer index 49 (`[L, 5120]` fp32, no final norm). Kernels: `kernels/cuda/minimax/te32.cu`. Twin:
`tools/twin/stk_twin/h3/te32.py` (`TextEncoder32`, `tokenize`). Test (GPU pod): `python -m stk_twin.h3.test_te32 --ckpt <file>` with
`$COMFY` set. Build without `--use_fast_math` (`expf`, `rsqrtf`, divisions are used).

## Checkpoint (`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, from its header)

Per layer `model.layers.{0..49}.` (the file holds layers 0-49 plus `visual.*`, which is never read):

| tensor | dtype | shape |
|---|---|---|
| `input_layernorm.weight`, `post_attention_layernorm.weight` | BF16 | [5120] |
| `self_attn.q_norm.weight`, `self_attn.k_norm.weight` | BF16 | [128] |
| `self_attn.q_proj.{weight,weight_scale,weight_scale_2,comfy_quant}` | U8 / F8_E4M3 / F32 / U8[55] | [8192, 2560] / [8192, 320] / [] |
| `self_attn.k_proj`, `self_attn.v_proj` | same | [1024, 2560] / [1024, 320] |
| `self_attn.o_proj` + `pre_quant_scale` BF16 [8192] | same | [5120, 4096] / [5120, 512] |
| `mlp.gate_proj`, `mlp.up_proj` | same | [25600, 2560] / [25600, 320] |
| `mlp.down_proj` + `pre_quant_scale` BF16 [25600] | same | [5120, 12800] / [5120, 1600] |

* `pre_quant_scale` exists ONLY on `o_proj` and `down_proj`. q/k/v share nothing (they have none; the AWQ smoothing of their input is
  presumably folded into `input_layernorm.weight`); gate/up have none either. There is no `input_scale` anywhere (it would be unused:
  `full_precision_mm`).
* Norm weights: BF16 (so ComfyUI's compute dtype `dtype_llama` is bf16); cast to fp32 exactly.
* `model.embed_tokens`: `weight` **I8** [151936, 5120], `weight_scale` **F32** [151936, 1] (per row), `comfy_quant` U8[29]
  (`{"format": "int8_tensorwise"}`). Not an NVFP4 table, not bf16.
* `comfy_quant` of a linear is JSON `{"format": "nvfp4", ...}` (the loader asserts the format).
* No bias anywhere. Per-layer sizes are all multiples of 128 rows and 64 K, so the swizzled block-scale arrays have no padding.

## Ops (all fp32, deterministic; "rn" = one IEEE rounding)

| twin op name (`te.{i}.` prefix) | kernel | arithmetic |
|---|---|---|
| `te.embed` | `te32_embed_i8` | `v = float(int8) * scale[id]` (rn); then rounded to **bf16** (rne) and widened: ComfyUI's `dequantize_embedding` returns `orig_dtype` = the compute dtype bf16, and `out_dtype=fp32` only widens afterwards (`quant_ops`/`ops.py:1693`, `kitchen int8.py:180`). `te32_embed_bf16` for an unquantized table. |
| `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm` | `te32_rms_norm` | `r = rsqrtf(sum(x^2)/D + eps)`; `y = (x * r) * w`; sum = 256 thread-strided partials (each ascending) + shared tree; eps = fp32(1e-6); D = 5120 or 128 (per head row) |
| `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | `te32_linear_nvfp4` | `a = x * pqs[k]` (rn, only o/down); `w = e2m1(nibble) * (e4m3(scale) * tscale)` (two rn products, sign applied after); `y = fmaf chain over k ascending from 0` |
| `rope` | `te32_rope_split_half` | pairs (i, i+64): `y0 = f00*x0 + f01*x1`, `y1 = f10*x0 + f11*x1`, `f00=f11=cos`, `f01=-sin`, `f10=sin`; FMA contraction by `mode` (below) |
| `attention` | `te32_attention` | `logit = (fmaf chain over d of q.k) * 128^-0.5` (scale = fp32(1/sqrt(128))), keys j <= i only; `e = expf(logit - max)`; `sum` = per-thread ascending partials + tree; `out = (fmaf chain over j of e*v) / sum` |
| `attn_residual`, `mlp_residual` | `te32_add` | `h + y` (rn) |
| `silu_mul` | `te32_silu_mul` | `s = g / (1 + expf(-g))` (rn each), `out = s * u` (rn) |

Block order is ComfyUI's: `h += o_proj(attn(rope(norm_k(k), norm_q(q)))( rms(h)))`, `h += down(silu(gate(rms(h))) * up(rms(h)))`.
GQA: query head h uses KV head `h / 8`. RoPE is 1-D (`position_ids = arange(L)`), all 128 dims, theta 5e6.

### RoPE tables
`inv_freq[j] = 1 / powf(5e6, 2j/128)` (`2j/128` exact): `exp(e * log(5e6))` in f64 by `smath` (fdlibm, bit-equal in Zig), rounded to
fp32, then fp32 `1/p`. `freqs = f32(inv * pos)`; `cos`/`sin` = `smath.sincos` of that angle in f64, rounded to fp32. A frozen
`kernels/minimax/te32_inv_freq.json` (`{"inv_freq": [64 floats]}`) overrides the computation; `test_te32 --write-inv-freq` writes it from
ComfyUI's own GPU values if they differ.

### NVFP4 layouts (as `ck.dequantize_nvfp4` reads them)
* `weight` uint8 `[N, K/2]`: byte `b` of row `n` = elements `2b` (HIGH nibble) and `2b+1` (low nibble). Nibble: bit 3 sign, bits 0-2 e2m1
  index into `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`.
* `weight_scale` e4m3fn, one per 16 elements of a row, the cuBLAS 128x4 swizzled layout. For row `n`, block `c` (`0 <= c < K/16`), with
  `cbc = ceil((K/16) / 4)`:
  ```
  rb = n / 128;  rem = n % 128;  d4 = rem / 32;  d3 = rem % 32;  cbg = c / 4;  d5 = c % 4
  byte offset = ((rb * cbc + cbg) * 32 + d3) * 16 + d4 * 4 + d5
  ```
  (kitchen `scale_factor_swizzled_offset`; equals the inverse of `from_blocked`: flat -> `[t, d3, 16]` -> `[t, d3, d4, d5]` ->
  transpose -> `[rb, cbg, d4, d3, d5]` -> row `rb*128 + d4*32 + d3`, col `cbg*4 + d5`. Verified by exhaustive index check.) The array holds
  `ceil(N/128)*128 * cbc*4` bytes; the safetensors shape `[N, K/16]` is only a reshape of it.
* `weight_scale_2` fp32 scalar `tscale`. `decode_scale = float(e4m3) * tscale` (fp32), `w = float(e2m1) * decode_scale` (fp32).

## What differs from ComfyUI, and why

1. **Linear**: ComfyUI dequantizes the whole weight to fp32 and calls cuBLAS SGEMM (shape-dependent algorithm and split-K, not
   reproducible). Ours dequantizes tile by tile inside the GEMM and sums each output as one fmaf chain over k ascending: not
   bit-equal to cuBLAS (relative difference ~1e-7 per op), but independent of M and tiling, and the Zig engine runs the same kernel.
2. **RMSNorm**: torch's `F.rms_norm` reduction order (and, depending on the torch version, a fused kernel using `mean = sum * (1/D)`) is not
   ours; the order is fixed and documented here.
3. **Attention**: SDPA EFFICIENT/MATH fp32 replaced by `te32_attention` (any correct deterministic design was allowed). Masked logits
   contribute exactly zero, as ComfyUI's additive `finfo.min/4` mask does.
4. **cos/sin/inv_freq**: portable f64 math instead of CUDA `powf`/`cosf`/`sinf` (see above).
5. **Embedding** bf16 rounding is reproduced (derived from code, see `te.embed`); verify with the pod test's end to end numbers.
6. Tokenizer: `transformers.Qwen2Tokenizer` over the Qwen2.5 tokenizer files vendored in `stk_twin/h3/tokenizer/` (ComfyUI 0.37.0's `qwen25_tokenizer` bytes, sha256-checked on first use) + the 7 extra special tokens (`<d>` 151669 ...
   `<|caption_end|>` 151675, asserted), `SDTokenizer` splitting at `(?<=\s)embedding:` and `\(` `\)` unescaping, no embeddings
   directory (ComfyUI would resolve `embedding:name` files if present), empty result -> `[151643]`.

## Risky / undetermined for bit-exactness against ComfyUI itself (the twin and the Zig engine agree by construction)

* **RoPE FMA contraction** (`mode`): which product is fused differs by backend (fp32 q/k are not accepted by kitchen's CUDA backend; Triton
  or the eager path runs). `mode 0` = `fmaf(f_a, x_a, f_b*x_b)` (compiler default), `mode 1` = eager's `x0*f00` then `addcmul_` (second
  fused), `mode 2` unfused. `test_te32` prints which one equals kitchen's output on the pod; `TextEncoder32(rope_mode=...)` pins it.
* inv_freq / cos / sin may differ by 1 ulp from CUDA's `powf`/`cosf`/`sinf` in some entries (reported by the test).
* The bf16 rounding of the embeddings and the compute dtype (bf16 from the norm weights' dtype) come from reading `ops.py`/`sd.py`; confirm.
* The whole stream is a 50-layer fp32 recurrence with different (but fixed) summation orders: cosine/rel-L2 versus ComfyUI is
  expected near 1e-6 / 1e-5, not zero. The thresholds in the test (cos >= 0.999999, rel L2 <= 1e-3) are heuristics, not measured.
* `te32_attention` holds `L` logits in shared memory (`(L + 256) * 4` bytes; the launcher raises the limit above 48 KB). Prompts of several
  thousand tokens need > 100 KB and would need a tiled variant.
* Nothing here was compiled or run (no GPU/nvcc on the authoring machine): Python syntax only.
