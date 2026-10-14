# Text attention source headers

These 281 unmodified headers are the compiler dependency closure for the reference's
`fmha_cutlassF_bf16_aligned_64x128_rf_sm80` attention, CLIP's
`fmha_cutlassF_f32_aligned_64x64_rf_sm80` attention, and its 64x64 TF32 GEMMs
(K tiles 16 and 32, with two serial K partitions for the latter). They compile into our CUDA
fatbin; the production path neither imports Python nor links Torch, ATen or cuBLAS.

- PyTorch source: commit `5c4886908584029761b579af026dcfb627c84070`, matching the
  pinned oracle's `torch.version.git_version`. Headers retain their original relative
  paths beneath `torch/include`. License: `PYTORCH-LICENSE.txt`.
- NVIDIA CUTLASS source: commit `e05f953a5b3d38adc240df2ff928e0421c2abba3`, the
  PyTorch commit's `third_party/cutlass` submodule. Headers retain paths beneath
  `cutlass/include`. License: `CUTLASS-LICENSE.txt`.
- CUTLASS source archive SHA256:
  `30d038c2aabb65b70ef4f3152e03a7ba85f8a26e5243ce6ff8fbe4c2f24d7a85`.
- `SHA256SUMS` records every header's bytes; manifest SHA256:
  `a6b102cd5b53a4de5fdbd8c5d57caa8792390b3a10f9fdb06641bd4d47c24a66`.

The standalone CUDA driver specialization matched all 8,264,704 bytes of the
captured first Qwen attention output on sm_121a. The integrated Qwen encoder then
passed 49 comparisons, including all 28 layer boundaries and final embeddings, for
the captured seed-42 prompt. The integrated CLIP encoder passed 10 comparisons,
including the first layer's projections, attention and MLP and the final pooled
output, using RN-even TF32 operands and the captured serial split-K behavior.
Full generation gates remain mandatory; these fixtures do not establish arbitrary
prompts or the whole pipeline.
