# H3 speed and memory of the engine alone on a GB10: a fragment SOURCED by the window's inner script after its checks
# (tools/spark/h3-inner.sh: it defines run, sel, $R, $STK, $S the stk binary, $P the pack, $TE, $VA, $VV; add
#   [[ ${H3_SPEED_MEM:-0} != 1 ]] || source $STK/repo/tools/spark/h3-speed-mem.sh
# before its final "done" line, and start the window with H3_SPEED_MEM=1; ONLY="h3sm-bench-124" runs one step). Needs the
# zig build ($S) and the weights and the pack; no capture, no twin.
# ENGINE ONLY: nothing else runs on the machine while a step does (no twin, no ComfyUI, no other GPU process); the fragment
# refuses to start otherwise (h3sm-idle), because MemAvailable on a GB10 is the whole system's, device memory included.
# Steps (names are the log names; each is `stk check h3-bench ... --runs 2`: the first run, then a warm one, 8 steps):
#   h3sm-idle            the machine before: MemAvailable, GPU processes, python processes; fails when a twin is up
#   h3sm-ffmpeg          ffmpeg on the PATH (the NGC image may lack it; apt installs it): only the MP4 needs it
#   h3sm-bench-56        768x448, 56 frames (2.3 s), Limits sized for 56 frames, encoder resident
#   h3sm-bench-56-unload the same with --unload-te (encoder freed after each encode, reloaded by the next request)
#   h3sm-bench-124       768x448, 124 frames (5.2 s), Limits for 124 (the tool's own: max_seconds 5), writes the MP4
#                        $R/h3-124f-768x448.mp4, which stays in the results (the clip to look at)
#   h3sm-bench-124-unload the same with --unload-te (no MP4)
# Each step is its own process, so memory is back to the idle level before the next (the idle markers show it).
# Output: each step's bench JSON line to summary.jsonl (its own sampler: start / min / peak drop of MemAvailable, 50 ms), and
# per step the HOST sampler's line {"h3sm_memavail": {...}} (this fragment's, 0.2 s, outside the process: the cross-check,
# and the idle level before and after); $R/h3sm-memavail.log has the samples and step markers.
# Env: H3SM_RUNS (2), H3SM_STEPS (8), H3SM_SIZE (768x448). About 12 minutes.
: "${R:?h3-speed-mem.sh is sourced by h3-inner.sh}"
type run > /dev/null 2>&1 || { echo "h3-speed-mem: no run helper (source it from h3-inner.sh)" >&2; return 1; }
STK=${STK:-/workspace/stk}
S=${S:-$STK/h3-out/bin/stk}
P=${P:-$STK/packs/h3-turbo-nvfp4}
VMODELS=${VMODELS:-$STK/models/minimax-h3}
TE=${TE:-$VMODELS/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors}
VA=${VA:-$VMODELS/vae/minimax_h3_audio_vae_fp32.safetensors}
VV=${VV:-$VMODELS/vae/minimax_h3_video_vae_fp16.safetensors}
H3SM_RUNS=${H3SM_RUNS:-2}; H3SM_STEPS=${H3SM_STEPS:-8}; H3SM_SIZE=${H3SM_SIZE:-768x448}

# the host-side sampler: MemAvailable (kB) every 0.2 s with the step markers in one append-only file (the sampler and the
# markers both append, so neither overwrites the other); pure bash reads, no fork but the sleep
HM=$R/h3sm-memavail.log; : > $HM
( while :; do while read -r k v _; do [[ $k == MemAvailable: ]] && { printf '%s %s\n' "$EPOCHREALTIME" "$v"; break; }; done < /proc/meminfo
    sleep 0.2; done ) >> $HM &
HMS=$!
h3sm_mark() { echo "$1 $2" >> $HM; }
# a step under a marker pair, a pause after so the memory settles before the next process
h3sm_run() { local name=$1; shift; sel $name || return 0
    sync; { echo 3 > /proc/sys/vm/drop_caches; } 2> /dev/null || true   # a cold page cache; ignored where the container may not
    h3sm_mark "#B" $name; run $name "$@"; local rc=$?; h3sm_mark "#E" $name; sleep 5; return $rc; }

# the machine before: diagnostics first, the JSON line last (run keeps a log's last line); fails when anything else is up
h3sm_idle() {
    local n t c
    echo "gpu processes:"; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    t=0; for c in /proc/[0-9]*/comm; do case $(< $c) in python | python3 | comfyui) t=$((t + 1)); echo "process: $c $(< $c)";; esac; done 2> /dev/null
    [[ $n == 0 && $t == 0 ]] || echo "NOT ENGINE ONLY: $n GPU process(es), $t python process(es)" >&2
    awk '/MemAvailable/ {printf "{\"h3sm_idle\": {\"memavail_gib\": %.1f}}\n", $2 / 1048576}' /proc/meminfo
    [[ $n == 0 && $t == 0 ]]
}
h3sm_mark "#B" idle-before
run h3sm-idle h3sm_idle
idle_rc=$?
sleep 3; h3sm_mark "#E" idle-before
if [[ $idle_rc != 0 ]]; then
    echo "h3-speed-mem: another process is up; not an engine-only measurement, stopped" | tee -a $R/h3sm-idle.log
    kill $HMS 2> /dev/null; return 0
fi

sel h3sm-ffmpeg && run h3sm-ffmpeg bash -c 'command -v ffmpeg || { apt-get update -qq && apt-get install -y -qq ffmpeg; }; ffmpeg -version | head -1'

BENCH="$S check h3-bench $P $TE $VA $VV --size $H3SM_SIZE --steps $H3SM_STEPS --runs $H3SM_RUNS"
h3sm_run h3sm-bench-56 $BENCH --frames 56
h3sm_run h3sm-bench-56-unload $BENCH --frames 56 --unload-te
h3sm_run h3sm-bench-124 $BENCH --frames 124 --out $R/h3-124f-768x448.mp4
h3sm_run h3sm-bench-124-unload $BENCH --frames 124 --unload-te

h3sm_mark "#B" idle-after; sleep 3; h3sm_mark "#E" idle-after
kill $HMS 2> /dev/null
# per marked interval: the level at its start, its minimum, the peak drop and the level at its end (MiB); the idle intervals
# show whether the memory came back after the last process (end_mib of idle-after against idle-before)
awk '$1 == "#B" {cur = $2; n = 0; next}
     $1 == "#E" {if (cur != "" && n > 0) printf "{\"h3sm_memavail\": {\"step\": \"%s\", \"start_mib\": %.0f, \"min_mib\": %.0f, \"peak_drop_mib\": %.0f, \"end_mib\": %.0f, \"samples\": %d}}\n", cur, s / 1024, m / 1024, (s - m) / 1024, last / 1024, n; cur = ""; next}
     cur != "" {if (n == 0) {s = $2; m = $2} if ($2 < m) m = $2; last = $2; n++}' $HM >> $R/summary.jsonl
ls -l $R/*.mp4 2> /dev/null || echo "h3-speed-mem: no MP4 written (see h3sm-bench-124.log, mp4_error)"
