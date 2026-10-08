# H3 DiT step speed on a GB10, old schedule against new, bits unchanged: a fragment SOURCED by the window's inner script
# after its checks (tools/spark/h3-inner.sh: it defines run, sel, drop, $R, $STK, $S the stk binary, $P the pack, $VMODELS,
# $CAP, WITH_COMFY; add
#   [[ ${H3_SPEED_DIT:-0} != 1 ]] || source $STK/repo/tools/spark/h3-speed-dit.sh
# before its final "done" line, start the window with H3_SPEED_DIT=1, and ONLY="h3sd-replay h3sd-bench-56 h3sd-bench-124"
# (with SKIP_SETUP=1 SKIP_PACK=1 and the zig build of this tree done) for just these steps). Needs the zig build ($S), the
# pack ($P) and the weights; the tests need the twin's stack (ComfyUI, comfy-kitchen: WITH_COMFY=1, the inner default).
# Steps (names are the log names; each step's last JSON line goes to summary.jsonl):
#   h3sd-test-ops      LocalRouter's ops against torch (ops.cu's norm kernels share a refactored row reduction: still
#                      within the old bounds, and norm_mod deterministic)
#   h3sd-test-fuse     the fused kernels against the unfused ones, torch.equal: gate_add + norm_mod (both parts, one and
#                      two modulations, 5967 rows) and the INT8 attention stored as rows against its transpose
#   h3sd-test-kitchen  LocalRouter's kitchen kernels against the wheel, bitwise: kitchen_launch.cu was refactored (the rows
#                      entry point shares the launch code), so the wheel equality must be shown again
#   h3sd-capture       a fresh capture of a step (the twin runs the reference schedule while recording: every op is recorded)
#   h3sd-replay        h3-replay on it: the probed passes (reference schedule, every op alone and chained) and the fast
#                      passes (ref, fuse, graph, fuse_graph: velocities equal to the capture's, byte for byte; graph_error null)
#   h3sd-bench-56      h3-step-bench at 768x448, 56 frames (5,967 tokens): ref / fuse / graph / fuse_graph step ms (GPU
#                      events, median of H3SD_REPS) with the host's issue time, equality to ref, and the per-op profile
#                      JSON (ms per op class over the 50 blocks, launches, time between ops) of ref and fuse
#   h3sd-bench-124     the same at 768x448, 124 frames (the 5 s clip, 12,915 tokens; the capture's text states, synthetic latents)
# The old path as a switch: STK_H3_REF=1 in the environment of any stk process (the engine) or twin process turns the fusions
# and the step graph off; h3-step-bench runs both in one process (its `ref` configuration), so no switch is needed there.
# Env: H3SD_REPS (5; the 124-frame bench uses 3). Each bench JSON is also kept as $R/h3sd-bench-*.json. About 10 minutes.
: "${R:?h3-speed-dit.sh is sourced by h3-inner.sh}"
type run > /dev/null 2>&1 || { echo "h3-speed-dit: no run helper (source it from h3-inner.sh)" >&2; return 1; }
STK=${STK:-/workspace/stk}
S=${S:-$STK/h3-out/bin/stk}
P=${P:-$STK/packs/h3-turbo-nvfp4}
VMODELS=${VMODELS:-$STK/models/minimax-h3}
CS=${CAP:-$STK/captures}/h3-speed-dit
H3SD_REPS=${H3SD_REPS:-5}

# a bench step: the run helper, then the JSON line kept beside the log
h3sd_bench() { local name=$1; shift; sel $name || return 0
    run $name "$@"; local rc=$?; tail -n 1 $R/$name.log | grep '^{' > $R/${name}.json; return $rc; }

if [[ ${WITH_COMFY:-1} == 1 ]]; then
    run h3sd-test-ops python -m stk_twin.h3.test_ops
    run h3sd-test-fuse python -m stk_twin.h3.test_fuse
    run h3sd-test-kitchen python -m stk_twin.h3.test_kitchen
else
    run h3sd-test-fuse python -m stk_twin.h3.test_fuse   # needs no ComfyUI
fi

sel h3sd-capture && rm -rf $CS
run h3sd-capture python -m stk_twin.h3.capture --models $VMODELS --pack $P --out $CS
run h3sd-replay $S check h3-replay $P $CS
h3sd_bench h3sd-bench-56 $S check h3-step-bench $P $CS --reps $H3SD_REPS
h3sd_bench h3sd-bench-124 $S check h3-step-bench $P $CS --frames 124 --reps 3
[[ ${KEEP_CAPTURES:-0} == 1 || ${SKIP_CAPTURE:-0} == 1 || -n ${ONLY:-} ]] || rm -rf $CS
