#!/usr/bin/env bash
# Kandinsky-6 audio GPU qualification job runner (one job per node).
#
#   run_job.sh A|B WORKSPACE PREFIX UPSTREAM MODELS FATBIN OUT [extra jobs.py args, e.g. --weights synthetic]
#
# PREFIX is the sealed portable Python prefix (staging's oracle closure, e.g. /opt/k6-oracle mounted read-only):
# PREFIX/bin/python3.13 with PREFIX/lib/python3.13/site-packages. Its integrity is pinned by two required environment
# values, with no defaults: ORACLE_MANIFEST (a sha256sum-format file listing the prefix's files relative to PREFIX;
# absolute, or relative to PREFIX) and ORACLE_MANIFEST_SHA256 (that file's sha256). UPSTREAM, MODELS, FATBIN and OUT
# may be relative to WORKSPACE. The prefix, upstream, models and kit are only read (python -I -B writes no bytecode);
# OUT is the only write target. No network, no image builds; a container wrapper (if any) belongs to the caller.
# Exit status: jobs.py's (0 = job ran, mismatches included; 2/3 = infrastructure failure), 124 = 28 min timeout,
# 64 = usage, 65 = precondition failed.
set -euo pipefail

usage() { echo "usage: $0 A|B WORKSPACE PREFIX UPSTREAM MODELS FATBIN OUT [jobs.py args...]" >&2; exit 64; }
[ $# -ge 7 ] || usage
JOB=$1; WORKSPACE=$2; shift 2
case "$JOB" in A|B) ;; *) usage ;; esac
[ -d "$WORKSPACE" ] || { echo "WORKSPACE $WORKSPACE is not a directory" >&2; exit 65; }
WORKSPACE=$(cd "$WORKSPACE" && pwd)
abs() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$WORKSPACE/$1" ;; esac; }
PREFIX=$(abs "$1"); UPSTREAM=$(abs "$2"); MODELS=$(abs "$3"); FATBIN=$(abs "$4"); OUT=$(abs "$5"); shift 5
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY="$PREFIX/bin/python3.13"
SITE="$PREFIX/lib/python3.13/site-packages"

fail() { echo "precondition: $*" >&2; exit 65; }
[[ "${ORACLE_MANIFEST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || fail "ORACLE_MANIFEST_SHA256 must be the sealed prefix manifest's sha256"
[ -n "${ORACLE_MANIFEST:-}" ] || fail "ORACLE_MANIFEST must name the sealed prefix manifest"
case "$ORACLE_MANIFEST" in /*) MANIFEST=$ORACLE_MANIFEST ;; *) MANIFEST=$PREFIX/$ORACLE_MANIFEST ;; esac
[ -f "$MANIFEST" ] || fail "no prefix manifest at $MANIFEST"
[ -x "$PY" ] || fail "no interpreter at $PY"
[ -d "$SITE" ] || fail "no $SITE"
[ -d "$UPSTREAM/kandinsky" ] || fail "no Kandinsky-6 checkout at $UPSTREAM"
[ -f "$FATBIN" ] || fail "no fatbin at $FATBIN"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then fail "OUT $OUT exists and is not empty"; fi
mkdir -p "$OUT"
LOG="$OUT/job-$JOB.log"

# Resource sanity: CUDA reserves a large virtual address space, so a finite `ulimit -v` breaks cuInit; the host RSS
# (100 GiB) and GPU (40 GiB) limits are enforced by jobs.py itself. Output files are capped at 20 GiB each, no cores.
VMEM=$(ulimit -v)
[ "$VMEM" = unlimited ] || fail "ulimit -v is $VMEM KiB; CUDA needs it unlimited"
ulimit -c 0
ulimit -f $((20 * 1024 * 1024))
AVAIL_KB=$(df -Pk "$OUT" | awk 'NR==2 {print $4}')
[ "$AVAIL_KB" -ge $((20 * 1024 * 1024)) ] || fail "less than 20 GiB free under $OUT"

{
  echo "== k6a job $JOB $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "workspace=$WORKSPACE prefix=$PREFIX manifest=$MANIFEST upstream=$UPSTREAM models=$MODELS fatbin=$FATBIN out=$OUT"
  echo "extra args: $*"
  if [ -f "$HERE/SHA256SUMS" ]; then (cd "$HERE" && sha256sum --strict -c SHA256SUMS) || fail "kit checksum mismatch"; fi
  echo "$ORACLE_MANIFEST_SHA256  $MANIFEST" | sha256sum --strict -c - || fail "prefix manifest sha256"
  (cd "$PREFIX" && sha256sum --quiet --strict -c "$MANIFEST") || fail "prefix content differs from its manifest"
  echo "prefix verified"
  sha256sum "$FATBIN" "$HERE"/*.py
} 2>&1 | tee "$LOG"
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 65

export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export no_proxy='*' NO_PROXY='*'
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

set +e
timeout --signal=TERM --kill-after=60s 28m \
  "$PY" -I -B "$HERE/jobs.py" "$JOB" --fatbin "$FATBIN" --native-src "$UPSTREAM" --models "$MODELS" --out "$OUT" \
  --site-packages "$SITE" --time-limit-min 25 "$@" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
echo "== exit status $STATUS $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
exit "$STATUS"
