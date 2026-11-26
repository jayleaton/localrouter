# MiniMax H3-Image on gfx1151: qualification record

**Status: not run.** Every gate is pending, and every memory and time figure below is an *estimate* or a blank until a
gfx1151 run fills it in. This file is the skeleton ROCM.md section 5 describes. The gates and what must hold are in
ROCM.md section 5; what the engine copies from ComfyUI e9afbccc is in H3-IMAGE.md. The tool entry is `minimax-h3-image`
in `deploy/strix-halo/config.json.in`, whose resident figure (`H3I_RESIDENT_MB` in `env.sh`) is the engine's estimate.

Nothing here passes on a cosine or a tolerance unless its row says so. G1 to G4 are byte equality against the twin, run
on the same machine and the same code objects.

## The run

Fill this in once for the whole record. A gate run on a different setup gets its own copy of this table.

| Field | Value |
| --- | --- |
| Machine, CPU and GPU (`rocminfo` name, gfx target) | |
| ROCm version (the `-Dhip` tree), kernel, `amdgpu.gttsize`, GTT total | |
| LocalRouter commit, and the `zig build -Dgpu=hip -Dhip=<ROCm>` command | |
| `sha256` of each `zig-out/hsaco/*.hsaco` | |
| Twin commit, torch and ROCm wheel versions | |
| Pack (`stk_twin.h3i.build` manifest sha256), text encoder file, VAE file | |
| Precision | bf16 DiT, the 32B TE as packed, fp16 VAE |
| Seeds | |
| Date, operator | |

## Gates

Run them in order; a gate runs only after every gate above it passes. The command names are the ones proposed in ROCM.md
section 5. Use them, or record the name actually used.

| Gate | Command | What must hold | Result | Evidence |
| --- | --- | --- | --- | --- |
| G0 | `stk_twin` op tests (`test_ops`, `test_kitchen`-style, `test_wmma`) on the engine's code objects | each HIP kernel against its torch reference op within the twin's stated tolerance, and bit-equal across runs and grid shapes (row independence) | not run | |
| G1 | `localrouter check h3i-replay`, `h3-te`, `h3i-vae` | every op alone, then chained, equal to the gfx1151 twin capture | not run | |
| G2 | one DiT step chained, fast schedule vs reference schedule | velocities equal; the fused and eager schedules byte for byte equal | not run | |
| G3 | `localrouter check h3i-e2e` | prompt to decoded uint8 pixels equal to the twin; a warm rerun identical | not run | |
| G4 | `localrouter check h3i-edit-e2e`, masked and whole scene | reference encode, every step and the final pixels (after the composite) equal | not run | |
| Baseline | twin bf16 vs ComfyUI bf16 on gfx1151; LPIPS/DINO vs #157 at the same seeds | cosine a step (thresholds as ROCM.md "Baseline"); time and GTT peak against #157's table | not run | |

G0 covers every kernel the image engine loads on HIP:
- the bf16 and f16 WMMA GEMMs (`stk_gemm_bf16`, `stk_gemm_f16`, `_ref`, `_s`, `_n`);
- the bf16 attention (`stk_attention_bf16`);
- kitchen's `rms_rope` copy (`kitchen_rms_rope_bf16`);
- the image kernels (`h3i_head_mix`, `vi_rope_table`, `vi_rms_rope`, `vi_unshuffle`, `vi_blend`, `vi_copy`);
- the single-source kernels of PR 1.3 that the image path calls.

List each one's result under G0's evidence.

## Memory: estimates to replace

Measure each figure as the GTT peak (`gtt_used`) over the request, as `default_resident_mb` was measured on GB10, with
nothing else loaded. Then replace the estimate in `env.sh` (`H3I_RESIDENT_MB`) and in the engine's byte functions
(`minimax_h3_image_tool.zig`) in the same PR as this record.

| Figure | Estimate (engine, max_pixels 1024²) | Measured |
| --- | --- | --- |
| Resident, loaded and idle (edits on or off) | 55.2 GiB | |
| Working, text to image 1024² | 0.1 GiB | |
| Working, masked edit, 1 reference | 6.9 GiB | |
| Working, whole-scene edit, 5 references | 11.9 GiB | |
| Peak the tool can book (`validate`) | 67.1 GiB | |

## Times

| Request | #157 (ComfyUI, gfx1151) | LocalRouter | Notes |
| --- | --- | --- | --- |
| Text to image 1024², 6 steps, cold | | | |
| Text to image 1024², 6 steps, warm | | | |
| Masked edit, 1 reference, 6 steps | | | |
| Whole-scene edit, 30 steps | | | |

## Findings

Record anything the run shows that the docs did not predict, and the PR that answers it. Two items are open:
- H3-IMAGE.md section 3's attention question;
- the image VAE encoder's tensor names.
