Original compiler dependency closure for Torch 5c488690 Flash Attention submodule 14c377950125c70b7a9dabf9c561fca53715ac7d and CUTLASS e05f953a5b3d38adc240df2ff928e0421c2abba3. Headers are unmodified; upstream licenses are retained. Flash_fwd_kernel_traits<128,128,64,4,false,false,bfloat16_t>, UNFUSE_FMA, no fast math. Captured 50220-token standalone proof: 411402240 bytes identical. Integrated qualification remains pending.

Audio-head WMMA uses the same pinned CUTLASS commit's GEMM dependency closure,
with a BF16 arch bridge to CUDA's native 16x16x16 WMMA primitive. The captured
218x2048 / 400x2048 projection uses tile32x32x128, warp16x16x128, one stage,
align8, FP32 accumulation and eight serial K partitions with BF16 epilogues.
Both device-GEMM and native-wrapper standalone replays matched all 174400 bytes;
split1 reproduced the generic projection mismatch. This qualifies that fixture
only; integrated and full-generation qualification are separate gates.
