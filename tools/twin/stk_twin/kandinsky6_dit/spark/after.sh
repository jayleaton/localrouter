#!/usr/bin/env bash
# after.sh OUT_DIR T0 PY NODE VMSTAT_BEFORE WORK UNIT... -- RESULT_JSON...
# Run after a node's Kandinsky-6 DiT jobs (node-job.sh) finish, always, outside every job scope. Port of the D1
# uma-probe-after.sh (sha256 5501b24c411bd5784ab6cb621b4ccd6c942f788452d34fc6f072bcc36f261b87).
# Covers what a job cannot: it was killed (scope OOM, SIGKILL) and its result says nothing of it; something outside
# the run's scopes was killed. It changes nothing persistent: it only kills/stops this run's transient scope units.
#   1. SIGKILLs and stops each UNIT (k6-dit-*.scope), resets a failed one, verifies none is still active;
#   2. verifies no process of this run remains (argv containing $WORK/kit-, node-job.sh or spark/supervise.py);
#   3. captures /proc/vmstat oom_kill and both journals since T0 (kernel; earlyoom + systemd-oomd), each with its exit
#      status, and proves readability (a journal that returns nothing at all counts as unreadable);
#   4. classifies with classify.py (next to this script) into OUT_DIR/after.json (PASS / FAIL_LOCAL / STOP_ALL) from
#      the RESULT_JSON files node-job.sh wrote, and writes OUT_DIR/after.txt.
#      The classifier is stdlib-only: if PY cannot run it (e.g. a broken venv), the system /usr/bin/python3 is tried
#      (K6_FALLBACK_PY overrides it for fixtures only).
# Exit: 0 PASS, 1 FAIL_LOCAL, 4 STOP_ALL (also when classification fails with both interpreters), 2 usage.
set -u
usage() { echo "usage: $0 OUT_DIR T0 PY NODE VMSTAT_BEFORE WORK UNIT... -- RESULT_JSON..." >&2; exit 2; }
[ $# -ge 9 ] || usage
OUT=$1 T0=$2 PY=$3 NODE=$4 VB=$5 WORK=$6
shift 6
UNITS=() RESULTS=()
while [ $# -gt 0 ] && [ "$1" != -- ]; do UNITS+=("$1"); shift; done
[ $# -gt 0 ] || usage
shift
RESULTS=("$@")
[ ${#UNITS[@]} -ge 1 ] && [ ${#RESULTS[@]} -ge 1 ] || usage
[[ $T0 =~ ^[0-9]+$ ]] && [[ $NODE =~ ^[a-z0-9-]+$ ]] || { echo "$0: bad T0 or NODE" >&2; exit 2; }
[[ $WORK == /* ]] || { echo "$0: WORK must be an absolute path" >&2; exit 2; }
for u in "${UNITS[@]}"; do
  [[ $u =~ ^k6-dit-[a-z0-9-]+\.scope$ ]] || { echo "$0: bad UNIT $u" >&2; exit 2; }
done
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT" || exit 2
U=(); [ "$(id -u)" -eq 0 ] || U=(--user)
# processes of this run: WORK is matched literally inside the extended regex pgrep takes
WRE=$(printf '%s' "$WORK" | sed 's/[][\.*^$+?(){}|]/\\&/g')
TXT=$OUT/after.txt TMP=$OUT/after.txt.tmp.$$
clean=1
{
  echo "k6-dit-after T0=$T0 node=$NODE OUT=$OUT WORK=$WORK at $(date -u +%FT%TZ) uid=$(id -u)"
  for u in "${UNITS[@]}"; do
    echo "unit=$u before: $(systemctl "${U[@]}" show -p LoadState -p ActiveState -p Result "$u" 2>&1 | tr '\n' ' ')"
    systemctl "${U[@]}" kill -s SIGKILL "$u" >/dev/null 2>&1; echo "unit=$u kill_rc=$?"
    systemctl "${U[@]}" stop "$u" >/dev/null 2>&1; echo "unit=$u stop_rc=$?"
    systemctl "${U[@]}" reset-failed "$u" >/dev/null 2>&1
    st=$(systemctl "${U[@]}" is-active "$u" 2>/dev/null)
    echo "unit=$u after: ${st:-unknown}"
    case $st in active|activating|deactivating|reloading|refreshing) clean=0 ;; esac
  done
  # pgrep never lists itself; this script's own shells carry after.sh in their argv. pgrep rc 1 = none, > 1 = error.
  left=$(pgrep -af -- "$WRE/kit-|node-job\.sh|spark/supervise\.py"); prc=$?
  left=$(printf '%s\n' "$left" | grep -vE -- '(^|[ /])after\.sh( |$)' | grep -v '^$' || true)
  echo "remaining_processes=${left:-none} pgrep_rc=$prc"
  [ -z "$left" ] || clean=0
  [ $prc -le 1 ] || clean=0
  printf '%s %s\n' "$(date +%s)" "$(awk '$1=="oom_kill"{print $2}' /proc/vmstat)" > "$OUT/vmstat.after"
  echo "vmstat_before=$(cat "$VB" 2>&1) vmstat_after=$(cat "$OUT/vmstat.after")"
  journalctl -k -q --no-pager -o short-unix --since "@$T0" > "$OUT/kernel.log" 2> "$OUT/kernel.err"; krc=$?
  [ -n "$(journalctl -k -b -q --no-pager -n 1 2>/dev/null)" ] || { [ $krc -eq 0 ] && krc=98; }
  journalctl --system -q --no-pager -o short-unix --since "@$T0" -u earlyoom -u systemd-oomd \
    > "$OUT/system.log" 2> "$OUT/system.err"; src=$?
  [ -n "$(journalctl --system -q --no-pager -n 1 2>/dev/null)" ] || { [ $src -eq 0 ] && src=98; }
  echo "journal_rc kernel=$krc system=$src (98 = query returned nothing: unreadable, not clean)"
  echo "cleanup_verified=$clean"
} > "$TMP" 2>&1
CARGS=()
for u in "${UNITS[@]}"; do CARGS+=(--unit "$u"); done
for r in "${RESULTS[@]}"; do CARGS+=(--job-result "$r"); done
rc=none
for py in "$PY" "${K6_FALLBACK_PY:-/usr/bin/python3}"; do
  rm -f "$OUT/after.json"
  "$py" -I "$HERE/classify.py" --node "$NODE" --t0 "$T0" --vmstat-before "$VB" \
    --vmstat-after "$OUT/vmstat.after" --kernel-log "$OUT/kernel.log" --kernel-rc "$krc" \
    --system-log "$OUT/system.log" --system-rc "$src" --cleanup-verified "$clean" "${CARGS[@]}" \
    --out "$OUT/after.json" >> "$TMP" 2>&1
  rc=$?
  echo "classifier_python=$py exit=$rc" >> "$TMP"
  case $rc in 0|1|4) [ -s "$OUT/after.json" ] && break ;; esac
done
if [ $rc -ne 0 ] && [ $rc -ne 1 ] && [ $rc -ne 4 ] || [ ! -s "$OUT/after.json" ]; then
  printf '{"schema": "k6-dit-job-classification-v1", "node": "%s", "t0": %s, "class": "STOP_ALL", "sticky": true, "stop_reasons": ["CLASSIFIER_ERROR: exit %s"], "local_reasons": [], "cleanup_verified": %s}\n' \
    "$NODE" "$T0" "$rc" "$([ $clean -eq 1 ] && echo true || echo false)" > "$OUT/after.json.tmp.$$"
  mv -f "$OUT/after.json.tmp.$$" "$OUT/after.json"
  echo "classifier failed (exit $rc): STOP_ALL" >> "$TMP"
  rc=4
fi
echo "after_class_exit=$rc" >> "$TMP"
sync "$TMP" "$OUT/after.json" 2>/dev/null
mv -f "$TMP" "$TXT" && sync "$OUT" 2>/dev/null
exit $rc
