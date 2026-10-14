#!/usr/bin/env bash
# The project's checks on one tree, each with its command and exit code in OUT/<check>.txt and a summary in
# OUT/checks.json; exits 1 if any fails. Run by whisper-job.sh on the node (zig, python3 and ffmpeg on PATH):
#   bash whisper-checks.sh <tree> <out dir> <basefmt dir>
# basefmt holds the base commit's `zig fmt --check` file list (base-fmt.txt) and each file's formatter diff
# (<path with / as _>.diff, its +/- lines without positions): the tree must show the same files and, in each, the same
# formatter changes line for line (per finding, not just per file), so no new or changed finding passes. Exit 2 from
# a check means a tool failed or stopped early (not a finding). py_compile writes under PYTHONPYCACHEPREFIX (the job sets it).
set -u
SRC=$1 OUT=$2 BASE=$3
mkdir -p "$OUT"
cd "$SRC" || exit 1
declare -A rc
run() { # run <name> <command...>
    local name=$1; shift
    { echo "cmd: $*"; "$@" 2>&1; echo "exit $?"; } > "$OUT/$name.txt"
    rc[$name]=$(tail -1 "$OUT/$name.txt" | cut -d' ' -f2)
    echo "$name: exit ${rc[$name]}"
}
# Every producer's exit status is read on its own (no pipes): a formatter, find or compiler that fails or stops early
# fails the check instead of looking like "no findings". Expected statuses are explicit: zig fmt --check 0 (clean) or
# 1 (files listed), zig fmt --stdin 0, diff 0 or 1, grep 0 or 1, cmp 0 (same) or 1 (changed).
fmt_identity() {
    local t=$OUT/fmt.work rc bad=0 f key
    mkdir -p "$t"
    zig fmt --check build.zig build src tests > "$t/check.out" 2> "$t/check.err"; rc=$?
    if [[ -s $t/check.err ]] || { [[ $rc != 0 ]] && [[ $rc != 1 ]]; } || { [[ $rc == 1 ]] && [[ ! -s $t/check.out ]]; } \
        || { [[ $rc == 0 ]] && [[ -s $t/check.out ]]; }; then
        echo "zig fmt --check did not finish cleanly: exit $rc"; cat "$t/check.err"; return 2
    fi
    sort "$t/check.out" > "$OUT/fmt-files.txt" || { echo "sort failed"; return 2; }
    grep -v '^exit' "$BASE/base-fmt.txt" > "$t/base.unsorted"; rc=$?
    [[ $rc == 0 ]] || { echo "the base finding list is unreadable or empty: grep exit $rc"; return 2; }
    sort "$t/base.unsorted" > "$t/base-files.txt" || { echo "sort failed"; return 2; }
    diff "$OUT/fmt-files.txt" "$t/base-files.txt" > "$t/files.diff"; rc=$?
    case $rc in 0) ;; 1) echo "file set differs from the base:"; cat "$t/files.diff"; bad=1 ;; *) echo "diff failed: exit $rc"; return 2 ;; esac
    while read -r f; do
        key=$(printf '%s' "$f" | tr / _)
        zig fmt --stdin < "$f" > "$t/formatted" 2> "$t/stdin.err"; rc=$?
        [[ $rc == 0 && ! -s $t/stdin.err && -s $t/formatted ]] || { echo "zig fmt --stdin failed on $f: exit $rc"; cat "$t/stdin.err"; return 2; }
        diff "$f" "$t/formatted" > "$t/file.diff"; rc=$?
        [[ $rc == 0 || $rc == 1 ]] || { echo "diff failed on $f: exit $rc"; return 2; }
        grep '^[<>]' "$t/file.diff" > "$t/lines"; rc=$?
        [[ $rc == 0 || $rc == 1 ]] || { echo "grep failed on $f: exit $rc"; return 2; }
        [[ -f $BASE/$key.diff ]] || { echo "identity changed: $f has no base finding"; bad=1; continue; }
        cmp -s "$t/lines" "$BASE/$key.diff"; rc=$?
        case $rc in 0) ;; 1) echo "identity changed: $f"; bad=1 ;; *) echo "cmp failed on $f: exit $rc"; return 2 ;; esac
    done < "$OUT/fmt-files.txt"
    echo "$(wc -l < "$OUT/fmt-files.txt") files with inherited findings; identity changes: $bad"
    rm -rf "$t"
    return $bad
}
py_compile() {
    local list=$OUT/py-files rc
    find tools tests -name '*.py' ! -path '*/zig-pkg/*' -print0 > "$list"; rc=$?
    [[ $rc == 0 && -s $list ]] || { echo "could not list the Python files: find exit $rc"; return 2; }
    python3 -c 'import py_compile, sys; fs = [f for f in open(sys.argv[1], "rb").read().decode().split("\0") if f]
[py_compile.compile(f, doraise=True) for f in fs]; print(len(fs), "files")' "$list"
}
shell_syntax() {
    local f n=0
    for f in tools/spark/*.sh; do
        [[ -f $f ]] || { echo "no scripts found"; return 2; }
        bash -n "$f" || return 1
        n=$((n + 1))
    done
    echo "$n scripts"
}
# whisper-job.sh's stop rule, on the CPU: in a scratch dir with no Python, the venv step must fail and the native
# steps after it must be recorded as not run and leave no log, report or venv behind (nothing was launched).
job_gate() {
    local d rc
    d=$(mktemp -d) || return 2
    mkdir "$d/out" && for f in COMMIT TREE PARENT BASE; do echo x > "$d/$f"; done && cp tools/spark/whisper-job.sh "$d/run.sh" || { rm -rf "$d"; return 2; }
    (cd "$d" && LANE_OUT="$d/out" STEPS="venv loadcheck twin bench serve" bash run.sh) > "$d/job.out" 2>&1; rc=$?
    python3 -I - "$d" "$rc" <<'PY'
import json, os, sys
d, rc = sys.argv[1], int(sys.argv[2])
j = json.load(open(os.path.join(d, "out", "job.json")))
st = {r["step"]: (r["status"], r["exit"]) for r in j["steps"]}
native = ["loadcheck", "twin", "bench", "serve"]
left = [f for f in ("loadcheck.log", "twin.log", "bench.log", "serve.log", "loadcheck.json", "twin-report.json",
                    "bench-report.json", "qualify.json") if os.path.exists(os.path.join(d, "out", f))]
ok = rc != 0 and not j["passed"] and st["venv"][0] == "ran" and st["venv"][1] != 0 \
    and all(st[n] == ("not-run", None) for n in native) and not left and not os.path.exists(os.path.join(d, "venv", "bin", "python"))
print("job rc", rc, "steps", st, "native outputs", left, "->", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
PY
    rc=$?
    rm -rf "$d" || { echo "scratch cleanup failed: $d"; return 2; }
    return $rc
}
run zig-version zig version
run build zig build -j"${LANE_CPUS:-8}"
run test zig build test -j"${LANE_CPUS:-8}" --summary all
run fmt fmt_identity
run py-compile py_compile
run bash-n shell_syntax
run job-gate job_gate
python3 - "$OUT" "${rc[@]}" <<'PY'
import json, os, sys
out = sys.argv[1]
names = ["zig-version", "build", "test", "fmt", "py-compile", "bash-n", "job-gate"]
res = {n: int(open(os.path.join(out, n + ".txt")).read().rstrip().rsplit(" ", 1)[1]) for n in names}
json.dump({"exit": res, "passed": all(v == 0 for v in res.values())}, open(os.path.join(out, "checks.json"), "w"), indent=1)
print("checks", "PASSED" if all(v == 0 for v in res.values()) else "FAILED", res)
sys.exit(0 if all(v == 0 for v in res.values()) else 1)
PY
