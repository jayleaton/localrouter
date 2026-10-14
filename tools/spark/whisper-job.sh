#!/usr/bin/env bash
# Spark lane job for the speech work (see the owner's lane protocol): run in the extracted input directory, offline,
# no containers, no package install; every step's log and exit go to $LANE_OUT, and LANE_OUT/job.json sums them up.
#   cd <extracted input> && LANE_OUT=<dir> STEPS="..." bash run.sh        (run.sh is this script)
# Two jobs use it (whisper-kit.sh packs their inputs):
#   checks (CPU)  STEPS="setup checks"   the project's checks on COMMIT, natively on aarch64: zig build, zig build
#                 test --summary all, zig fmt --check identity against the base's findings (the same files, and in
#                 each the same formatter changes, line for line), py_compile, bash -n, and git diff --check
#                 BASE..COMMIT. Run under a CPU cap: zig takes -j from the lane's cpus (LANE_CPUS, default 8)
#   gpu           STEPS="setup verify build venv loadcheck twin bench serve"   Whisper large-v3-turbo on the GPU: the input
#                 digests, a ReleaseSafe localrouter, the pinned venv, twin.py (worker against the inline model-card
#                 reference, bit for bit, for timestamps none, segment and word, with WER/CER), bench.py (cold, warm,
#                 RTF, memory, SIGKILL cancel and reclaim) and qualify.py through that localrouter over HTTP and MCP
# Input: COMMIT (a candidate commit: the exact working tree, not on any branch), TREE (its git tree id), PARENT (its
# parent, the branch head), BASE (the PR's base, origin/main),
# repo.bundle, zig-aarch64-linux-0.17.0.tar.xz, zig-pkg/ (build.zig.zon's dependencies, hash-checked by zig), ffmpeg
# (BtbN's arm64 GPL build, pinned below: a PIE linking the host's glibc and libgcc_s, not fully static), basefmt/; the
# gpu job also the python-build-standalone CPython 3.12 aarch64 tarball (pinned below), wheels/ + requirements.txt,
# model/<revision>/, vad/silero_vad.jit and fixtures/. Exit 0 only if every step passed.
set -uo pipefail
here=$(pwd)
out=${LANE_OUT:?LANE_OUT is required}
STEPS=${STEPS:-setup checks}
mkdir -p "$out"
commit=$(cat COMMIT) tree=$(cat TREE) base=$(cat BASE) parent=$(cat PARENT)
zig_sha=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
ffmpeg_tar=ffmpeg-n9.0.2-27-g7e93014c5b-linuxarm64-gpl-9.0.tar.xz # BtbN/FFmpeg-Builds autobuild-2026-10-12-13-12
ffmpeg_sha=84163454c37bcaab22c5e30cd504d1d9d5e4a760b701d7d74a9e97ad241418ba
ffmpeg_bin_sha=0782dba06c50b2faebe17357681d70f123a38d329ef008a1b03e1e161a0a8de2 # its bin/ffmpeg
python_tar=cpython-3.12.15+20261003-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz # astral-sh/python-build-standalone 20261003
python_sha=6541297dd1798dec8b98c3ad7492808a5b9d1c126801ceb2011e7754cd20d1ce # the release's SHA256SUMS
steps=()
step() { # step NAME COMMAND...: logs to out/NAME.log, records exit and seconds
    local name=$1 t0 rc=0
    shift
    t0=$(date +%s)
    { echo "cmd: $*"; "$@"; } > "$out/$name.log" 2>&1 || rc=$?
    echo "exit $rc" >> "$out/$name.log"
    steps+=("$name $rc $(($(date +%s) - t0))")
    echo "[lane $(date -u +%T)] $name exit $rc"
    return $rc
}
setup() {
    [[ $(uname -m) == aarch64 ]] || { echo "not an arm64 host: $(uname -m)"; return 1; }
    echo "$zig_sha  zig-aarch64-linux-0.17.0.tar.xz" | sha256sum -c - && echo "$ffmpeg_sha  $ffmpeg_tar" | sha256sum -c - || return 1
    tar -xJf zig-aarch64-linux-0.17.0.tar.xz && mkdir -p ffmpeg && tar -xJf "$ffmpeg_tar" -C ffmpeg --strip-components=1 || return 1
    echo "$ffmpeg_bin_sha  ffmpeg/bin/ffmpeg" | sha256sum -c - || return 1
    mkdir -p zigcache tmp pyc
    # The bundle's refs are refs/lane/*, which a clone does not fetch: fetch each one explicitly, then pin all three.
    git init -q repo && git -C repo bundle verify ../repo.bundle && git bundle list-heads repo.bundle || return 1
    git -C repo fetch -q ../repo.bundle \
        refs/lane/whisper-candidate:refs/lane/whisper-candidate refs/lane/whisper-parent:refs/lane/whisper-parent \
        refs/lane/whisper-base:refs/lane/whisper-base || return 1
    local r want
    for r in candidate:$commit parent:$parent base:$base; do
        want=${r#*:}
        [[ $(git -C repo rev-parse "refs/lane/whisper-${r%%:*}^{commit}") == "$want" ]] || { echo "refs/lane/whisper-${r%%:*} is not $want"; return 1; }
    done
    [[ $(git -C repo rev-parse "$commit^") == "$parent" ]] || { echo "the candidate's parent is not $parent"; return 1; }
    git -C repo merge-base --is-ancestor "$base" "$commit" || { echo "$base is not an ancestor of the candidate"; return 1; }
    git -C repo -c advice.detachedHead=false checkout -q --detach "$commit" || return 1
    cp -a zig-pkg repo/zig-pkg || return 1
    local got; got=$(git -C repo rev-parse "$commit^{tree}")
    echo "commit $commit tree $got (expected $tree), parent $parent, base $base"
    [[ $got == "$tree" ]] || return 1
    local v; zig version && v=$(ffmpeg -hide_banner -version) && printf '%s\n' "${v%%$'\n'*}" || return 1
    echo "host python3 (verify, checks, job.json): $(python3 --version)"
    if [[ -e $python_tar ]]; then # the gpu job: its own Python, for the venv
        echo "$python_sha  $python_tar" | sha256sum -c - && rm -rf python && tar -xzf "$python_tar" || return 1
        echo "kit python (venv): $(python/bin/python3.12 --version)"
    fi
    echo "cpu: nproc $(nproc) (affinity and OMP_* env), online $(getconf _NPROCESSORS_ONLN), cgroup cpu.max $(cpu_max), LANE_CPUS ${LANE_CPUS:-8}"
}
cpu_max() { cat "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/cpu.max" 2> /dev/null || echo unknown; }
verify() {
    local rev; rev=$(ls model)
    python3 -I repo/tools/speech/whisper/fetch_model.py --verify "model/$rev" || return 1
    local vad_sha; vad_sha=$(sed -n 's/^VAD_JIT_SHA256 = "\([0-9a-f]*\)".*/\1/p' repo/tools/speech/whisper/whisper_core.py)
    echo "$vad_sha  vad/silero_vad.jit" | sha256sum -c - || return 1
    python3 -I - <<'PY'
import hashlib, json
m = json.load(open("fixtures/manifest.json"))["fixtures"]
bad = [f["file"] for f in m.values() if hashlib.sha256(open("fixtures/" + f["file"], "rb").read()).hexdigest() != f["sha256"]]
print(len(m), "fixtures, mismatched:", bad)
raise SystemExit(1 if bad else 0)
PY
}
venv() { # wheel_tags.py: the locked cuSPARSELt wheel's WHEEL-tag label, corrected only once proven (see the file).
    # pip runs isolated (-I: PYTHON* variables ignored), so the job's PYTHONPYCACHEPREFIX cannot move the bytecode
    # pip compiles out of the venv; the self-test lays out its scratch copy the same way.
    local W=repo/tools/speech/whisper
    rm -rf venv && python/bin/python3.12 -m venv venv &&
        venv/bin/python -I -m pip install --no-index --require-hashes --find-links wheels -r requirements.txt &&
        venv/bin/python -I $W/wheel_tags_selftest.py --wheels wheels --requirements requirements.txt \
            --report "$out/wheel-tags-selftest.json" &&
        venv/bin/python -I $W/wheel_tags.py --site venv/lib/python3.12/site-packages --wheels wheels \
            --requirements requirements.txt --audit "$out/wheel-tags" &&
        venv/bin/python -I -m pip freeze --all && venv/bin/python -I -m pip check &&
        venv/bin/python -c 'import sys, torch, transformers; print(sys.version.split()[0], torch.__version__, transformers.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
}
serve() {
    local rev port=18191; rev=$(ls model)
    cat > tools.json << JSON
{"reserve_bytes": 8589934592, "tools": [
 {"id": "whisper-large-v3-turbo", "kind": "transcription", "cmd": ["$here/venv/bin/python", "$here/repo/tools/speech/whisper/worker.py"],
  "weights": "$here/model/$rev", "resident_bytes": 10737418240, "working_bytes": 2147483648, "idle_ttl_s": 600,
  "options": {"device": "cuda", "dtype": "float16", "attn": "sdpa", "threads": ${LANE_CPUS:-8},
              "vad": "silero", "vad_model": "$here/vad/silero_vad.jit"}}]}
JSON
    bin/bin/localrouter serve --config tools.json --host 127.0.0.1 --port $port --data "$here/data" > "$out/serve-daemon.log" 2>&1 &
    local pid=$!
    for _ in $(seq 50); do curl -fs http://127.0.0.1:$port/health > /dev/null && break; sleep 0.2; done
    venv/bin/python repo/tools/speech/whisper/qualify.py --url http://127.0.0.1:$port --model whisper-large-v3-turbo --fixtures fixtures --report "$out/qualify.json"
    local rc=$?
    kill $pid; wait $pid 2> /dev/null
    cp data/logs/whisper-large-v3-turbo.log "$out/worker.log" 2> /dev/null
    return $rc
}
export PATH=$here/zig-aarch64-linux-0.17.0:$here/ffmpeg/bin:$PATH ZIG_GLOBAL_CACHE_DIR=$here/zigcache TMPDIR_BUILD=$here/tmp
export PYTHONPYCACHEPREFIX=$here/pyc HF_HUB_OFFLINE=1
FX=fixtures W=repo/tools/speech/whisper
SHORT="$FX/en_librispeech_1272-128104-0000.wav $FX/en_librispeech_1272-135031-0001.wav $FX/de_fleurs_01.wav $FX/fr_fleurs_01.wav $FX/es_fleurs_01.wav $FX/ja_fleurs_01.wav $FX/zh_fleurs_01.wav $FX/silence_10s.wav $FX/noise_10s.wav $FX/speech_in_pink_noise_10db.wav"
LONG="$FX/long_en_librispeech_60s.wav $FX/long_en_librispeech_270s.wav"
# A failed prerequisite (setup, verify, build, venv: the environment the native steps run in, which venv may have
# changed) stops the job: every later step is recorded as not run, and no native step launches on it.
stopped=""
for s in $STEPS; do
    if [[ -n $stopped ]]; then
        steps+=("$s not-run 0")
        echo "[lane $(date -u +%T)] $s not run: $stopped failed"
        continue
    fi
    case $s in
    setup) step setup setup ;;
    checks) step checks bash repo/tools/spark/whisper-checks.sh "$here/repo" "$out/checks" "$here/basefmt"
        step diff-check git -C repo diff --check "$base" "$commit" ;;
    verify) step verify verify ;;
    build) step build bash -c "cd repo && zig build -j${LANE_CPUS:-8} -Doptimize=ReleaseSafe --prefix '$here/bin'" ;;
    venv) step venv venv ;;
    loadcheck) step loadcheck venv/bin/python $W/loadcheck.py --weights model/$(ls model) --vad-model vad/silero_vad.jit \
        --audio $FX/en_librispeech_1272-128104-0000.wav --silence $FX/silence_10s.wav --device cuda --dtype float16 \
        --threads ${LANE_CPUS:-8} --report "$out/loadcheck.json" ;;
    twin) step twin venv/bin/python $W/twin.py $SHORT $LONG --weights model/$(ls model) --device cuda --dtype float16 \
        --threads ${LANE_CPUS:-8} --vad-model vad/silero_vad.jit --timestamps none,segment,word --wer $FX/refs.json \
        --report "$out/twin-report.json" ;;
    bench) step bench venv/bin/python $W/bench.py $FX/en_librispeech_1272-135031-0001.wav $FX/long_en_librispeech_60s.wav \
        $FX/long_en_librispeech_270s.wav --weights model/$(ls model) --device cuda --dtype float16 --warm 3 \
        --threads ${LANE_CPUS:-8} --vad-model vad/silero_vad.jit \
        --cancel $FX/long_en_librispeech_270s.wav --report "$out/bench-report.json" ;;
    serve) step serve serve ;;
    *) echo "unknown step $s"; steps+=("$s 2 0"); false ;;
    esac || case $s in loadcheck | twin | bench | serve | checks) ;; *) stopped=$s ;; esac
done
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader > "$out/gpu.txt" 2>&1
python3 -I - "$out/job.json" "$commit" "$tree" "$base" "$(uname -srm)" "$(nproc)" "$(getconf _NPROCESSORS_ONLN)" "$(cpu_max)" "${LANE_CPUS:-8}" "${steps[@]}" << 'PY'
import json, sys
path, commit, tree, base, host, nproc, online, cpu_max, lane_cpus, *steps = sys.argv[1:]
rows = [{"step": n, "status": "not-run", "exit": None, "seconds": 0} if e == "not-run"
        else {"step": n, "status": "ran", "exit": int(e), "seconds": int(t)} for n, e, t in (s.split() for s in steps)]
ok = bool(rows) and all(r["status"] == "ran" and r["exit"] == 0 for r in rows)
json.dump({"commit": commit, "tree": tree, "base": base, "host": host,
           "cpu": {"nproc_affinity": int(nproc), "online": int(online), "cgroup_cpu_max": cpu_max, "lane_cpus": int(lane_cpus)},
           "steps": rows, "passed": ok},
          open(path, "w"), indent=1)
print("job", "PASSED" if ok else "FAILED", rows)
raise SystemExit(0 if ok else 1)
PY
