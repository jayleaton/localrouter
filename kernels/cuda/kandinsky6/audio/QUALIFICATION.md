# Kandinsky-6 audio: SM121 qualification record

The native audio chain (latent → mel → wave → int16 PCM) is bit-equal to the pinned native Python reference on the
target GPU. This is the evidence that qualifies it (CONTRIBUTING, "Engine correctness"). No threshold is applied: every
compared tensor is byte-identical.

This records arithmetic, not admission. `admission.zig` still lists the audio contracts as `gpu_qualification_pending`,
and moving them is a separate decision.

## Result

| Op (`k6.audio.*`, chained) | dtype, shape | sha256 (native = reference 2) |
|---|---|---|
| `latent_in` | f32 [1, 218, 40] | `47ebf1d4e82a493b2e2bc62f0668de5bb2fe5ed6e5c7a195832b6c2a79a93e90` |
| `mel` | f32 [1, 128, 436] | `15f922a484defbc22ec0baa456aec18e485c24089d0296f265b97eb53003216a` |
| `wave` | f32 [1, 1, 223232], 44.1 kHz | `e2e5fe6448e0913ca3eabb7bc3e577af2889891834facbf5e2e122b44d24d031` |
| `pcm16` | int16 [223232] | `f1aed5063dd98130223f305fe25551656a055edef687e26d0b4107f0cbac5fc5` |

`stk_twin.kandinsky6.compare --ops k6.audio.` reported 4 pairs, 0 failed, 0 unequal values, and "audio chain: qualified".
That means each op's input is the previous op's output on both sides, and every role is bit-equal.

### Supporting gates (same build)

| Gate | Scope | Result |
|---|---|---|
| Decoder compare vs reference 12 | every decoder hook, `latent_unscale` through `mel` | 91 / 91 equal |
| Vocoder compare vs vocoder reference v1 (reference 2's mel) | `conv_pre`, `upsample0..5`, `stage{0..5}_amp{0..2}`, `stage{0..5}_avg`, `post_act`, `conv_post`, `wave` | 34 / 34 equal |
| Integrated kernels on per-convolution fixtures | every vocoder convolution on a TF32 or gathered-GEMM path (97), plus the up0 shortcut | 97 / 97 and 1 / 1 all bytes equal |
| Stage merge, final clamp, pcm16 (CPU, from the references' own tensors) | 6 stages; clamp; pcm16 | `avg` `.reciprocal` 0 bytes unequal on every stage (`.true_div` 895,583 on stage 0); clamp 0; pcm16 0 |

## Reference

| Item | Value |
|---|---|
| Reference | reference 2, `gpu-e2e-reference-v2/reference` (immutable; `SHA256SUMS` checked by every job) |
| Request | `--stages e2e --seed 42 --e2e-prompt 0`, prompt "A small red boat crosses a quiet lake." (prompts.txt sha256 `7b001586…549d`), audio seed = seed + 1 per the pipeline's noise contract |
| Upstream | kandinskylab/kandinsky-6 `eac06e39b904940baff2c408e925cdb0c4529269`, config `pro-distill`; checkpoint HF `Kandinsky-6.0-Pro-distill-5s-Diffusers` `1018d582` |
| Stack | Python 3.13.16, torch 2.14.1+cu130 (CUDA 13), numpy 2.5.4: the `k6-oracle-full-v2` prefix, archive sha256 `106e608b…bc85`, freeze `18f5e02f…4451` |
| Precision (reference) | fp32 tensors; `float32_matmul_precision = high`, `cudnn.allow_tf32 = true`, `cuda.matmul.allow_tf32 = true`, `cudnn.benchmark = false`, `cudnn.deterministic = false`, SDPA backends default |

## Native run

| Item | Value |
|---|---|
| Hardware | NVIDIA GB10 (DGX Spark), compute capability 12.1, 48 SMs, 1536 threads / SM, integrated, 130,661,117,952 bytes unified memory reported |
| Binary | `localrouter` sha256 `c51dbb89ce507c8694f6b33569ab957bb499e53405c8749a74722eb5179c728c` (check v78 and hardware v74 passed on that tree) |
| Build | `kernels/cuda/kandinsky6/audio/audio.cu`: `-O3 --fmad=false`, sm_121a / sm_120a. `conv_tf32.cu`: `-O3 --fmad=false`, compute_80 PTX (JIT on SM121), vendored CUTLASS `e05f953`. Both add LocalRouter's torch flags. |
| Command | `localrouter check k6-components --model <models> --options options.json --out <dir> --stages audio --oracle gpu-e2e-reference-v2/reference`: the full audio stage, no diagnostic hooks, from reference 2's own latent |
| Options | `options.json` sha256 `974aa03f43223aca7c55e2fb2fc2d8023798181f7f9d7df515dd6e1985fb6f50` (below) |
| Compare | `python -m stk_twin.kandinsky6.compare <reference 2> <native> --ops k6.audio. --json audio-compare.json` (exit 0) |

## The policies exact audio needs

**Paths: none to choose.** Every qualified path is unconditional at the qualified geometry (`launch.zig`
`qualifiedDecoder` / `qualifiedVocoder`: batch 1, 218 latent / 436 mel frames).
- **Decoder:** the fixed T218 kernels by graph position, the T436 convolutions by `tf32_contracts`, both nin shortcuts and the attention's QKV / SDPA / projection.
- **Vocoder:** every convolution in `tf32_contracts` / `gather_contracts`.
- **Other geometry:** runs the direct paths and is not qualified.
- **The recorded run:** it used the earlier opt-in form of the same routing, all 33 path flags true.

**Arithmetic policies: required, no defaults.** `pipeline.parseOptions` refuses a missing, unknown or extra field. The
tool config's `options.audio` must be exactly:

```json
{
  "prepare": {"eps": 0.0001, "reduce": "lanes4_fused", "den": "fma"},
  "decode": {
    "acc": "fma", "unscale": "reciprocal", "silu": "reciprocal", "residual": "reciprocal",
    "norm_residual": {"eps": 0.0001, "reduce": {"torch_reduce": {"sm_count": 48, "max_threads_per_sm": 1536}}, "den": "fma"},
    "norm_qkv": {"eps": 0.0001, "reduce": {"torch_reduce": {"sm_count": 48, "max_threads_per_sm": 1536}}, "den": "fma"},
    "attention": "math_candidate"
  },
  "vocode": {"conv_acc": "fma", "aa_acc": "fma", "avg": "reciprocal"}
}
```

What the fields mean:
- **`prepare`:** host weight preparation. `lanes4_fused` is the aarch64 (Grace) reduction.
- **`torch_reduce` limits:** the running device's (GB10: 48 SMs, 1536 threads / SM). `checkDevice` refuses another device.
- **`decode.acc` / `vocode.conv_acc`:** the convolutions with no qualified path. On the native geometry that is conv_in, and the vocoder's 19 legacy-sgemm convolutions including conv_post.
- **`attention`:** only the non-native geometry's comparison candidate. The native decoder always runs the TF32 SDPA.
- **`reciprocal`:** ATen turns each of these CPU-scalar divisions into a multiply by the reciprocal. `true_div` is not exact.

### How the 116 vocoder convolutions are reproduced

Each convolution follows its original kernel's summation order, proven per instance on reference fixtures (`launch.zig`
`tf32_contracts` / `gather_contracts`):

| Path | Convolutions | Original | Order |
|---|---|---|---|
| CUTLASS fprop tile (16x10 / 128x64 / 32x5) | 37 | cuDNN's CUTLASS `s1688fprop_optimized_tf32_*` | the same CUTLASS kernel type |
| gathered GEMM, chunked | 42 | xmma fprop 128x128x16 | 16-channel chunks, taps inside |
| gathered GEMM, split | 6 + ups.0 | xmma fprop / dgrad indexed (grid z 17 / 10) | chunks round-robin over z slices, partials summed in the measured order |
| gathered GEMM, flat | 6 | xmma indexed_wo_smem 128x32x64 | flat KRSC k16 blocks, two MMAs each (f mod 4 ∈ {0,1}, then {2,3}) |
| gathered GEMM, dgrad | 5 | CUTLASS strided dgrad | phase-decomposed, channel chunk = the K tile |
| direct `.fma` | 19 | cuDNN legacy `implicit_convolve_sgemm` (FP32 FFMA) | channel outer, taps inner, then the bias add |

The split and flat orders were measured on the original kernels with a cancellation oracle (±2^30 / 1 queries per
output channel). They were then confirmed exact-or-reject on every instance.

## Scope

- **Shapes:** batch 1 only, 218 latent frames (5 s at 44.1 kHz), the `1018d582` checkpoint. Every TF32 / gather path refuses any other shape.
- **Target:** this result covers SM121 (GB10). sm_120a is built but not compared.
- **Device limits:** `torch_reduce` and `checkDevice` pin the device limits above.
