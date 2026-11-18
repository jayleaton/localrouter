#!/usr/bin/env bash
# serve.sh without a GPU: a stub podman whose "container" runs outside the caller's process group, as conmon's does.
# A SIGKILL of serve.sh's group (the daemon's, on abort or timeout) must not leave the container running; SIGTERM
# stops it through the trap; a later run of the same name is not removed by an earlier run's watcher.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
t=$(mktemp -d); trap 'pkill -KILL -f "stub-container $t" 2> /dev/null; rm -rf "$t"' EXIT
pass=0 fail=0
check() { if eval "$2"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL: $1" >&2; fi; }
mkdir -p "$t/bin" "$t/state"
cat > "$t/bin/podman" <<STUB
#!/usr/bin/env bash
# run --rm --replace --cidfile F --name N ... : a "container" (sleep) in a session of its own, its pid under the id
S=$t/state
case "\$1" in
  run)
    shift
    printf '%s\n' "\$@" > "\$S/last-args"
    cid=\$(date +%s%N); name=; cidfile=
    while [ \$# -gt 0 ]; do case "\$1" in --cidfile) cidfile=\$2; shift ;; --name) name=\$2; shift ;; esac; shift; done
    setsid bash -c 'exec -a "stub-container $t \$0" sleep 300' "\$cid" < /dev/null > /dev/null 2>&1 &
    echo \$! > "\$S/\$cid"; echo "\$cid" > "\$S/name-\$name"; echo "\$cid" > "\$cidfile"
    wait ;;  # the client waits; it dies with the group, the container does not
  stop) for a in "\$@"; do f=\$S/\$(cat "\$S/name-\$a" 2> /dev/null); [ -f "\$f" ] && kill "\$(cat "\$f")" 2> /dev/null; done; exit 0 ;;
  rm) shift; for a in "\$@"; do [ -f "\$S/\$a" ] && kill "\$(cat "\$S/\$a")" 2> /dev/null; echo "\$a" >> "\$S/removed"; done; exit 0 ;;
esac
STUB
chmod +x "$t/bin/podman"
export PATH="$t/bin:$PATH" GUFO_IMAGE="x@sha256:$(printf '0%.0s' {1..64})" GUFO_STOP_TIMEOUT=1 TMPDIR="$t"
container() { local c; c=$(cat "$t/state/name-lr-gufo-$1" 2> /dev/null) && cat "$t/state/$c" 2> /dev/null; }
alive() { [ -n "$1" ] && kill -0 "$1" 2> /dev/null; }
start() { local name=$1; shift; setsid bash "$here/serve.sh" "$name" 18080 "$@" -- gufo serve llm > /dev/null 2>&1 & echo $!; }  # its own group, as a worker's
waitfor() { local i; for i in $(seq 100); do eval "$1" && return 0; sleep 0.1; done; return 1; }

# a hard kill of the group: the container goes too
s=$(start a); waitfor '[ -n "$(container a)" ]'; c=$(container a)
check "the container runs" 'alive "$c"'
kill -KILL -- "-$s"
check "SIGKILL of serve.sh's group: the container is removed" 'waitfor "! alive $c"'
check "by its id" 'grep -q "$(cat "$t/state/name-lr-gufo-a")" "$t/state/removed"'
check "the watcher cleans up after itself" 'waitfor "[ -z \"\$(ls -d "$t"/lr-gufo-a.* 2> /dev/null)\" ]"'

# SIGTERM (an unload): the trap stops the container and the script exits
s=$(start b); waitfor '[ -n "$(container b)" ]'; c=$(container b)
kill -TERM "$s"
check "SIGTERM: the container stops" 'waitfor "! alive $c"'
check "SIGTERM: serve.sh exits" 'waitfor "! alive $s"'

# a run killed, then the same name started at once: the first run's watcher leaves the new container alone
s=$(start c); waitfor '[ -n "$(container c)" ]'; c1=$(container c)
kill -KILL -- "-$s"; s2=$(start c); waitfor '[ "$(container c)" != "$c1" ] && [ -n "$(container c)" ]'; c2=$(container c)
check "the first container is removed" 'waitfor "! alive $c1"'
sleep 1.5
check "the second is not" 'alive "$c2"'
kill -KILL -- "-$s2"; waitfor "! alive $c2"

# the host zone is an argument of podman run, so container logs use the host clock
export TZ=Europe/Helsinki
s=$(start tz); waitfor '[ -n "$(container tz)" ]'
check "podman run receives the host TZ" 'grep -qx "TZ=Europe/Helsinki" "$t/state/last-args"'
if [ -d /usr/share/zoneinfo ]; then
  check "the host zoneinfo is mounted read-only" 'grep -qx "/usr/share/zoneinfo:/usr/share/zoneinfo:ro" "$t/state/last-args"'
fi
kill -KILL -- "-$s"; waitfor '! alive "$(container tz)"'

# TZ cannot resolve without the zone database. A directory that exists is mounted; one that does not is left off.
mkdir -p "$t/zoneinfo"
export GUFO_ZONEINFO="$t/zoneinfo"
rm -f "$t/state/last-args"
s=$(start zi); waitfor '[ -n "$(container zi)" ]'
check "a present zoneinfo directory is mounted read-only" "grep -qx '$t/zoneinfo:/usr/share/zoneinfo:ro' '$t/state/last-args'"
kill -KILL -- "-$s"; waitfor '! alive "$(container zi)"'
export GUFO_ZONEINFO="$t/no-such-zoneinfo"
rm -f "$t/state/last-args"
s=$(start zn); waitfor '[ -n "$(container zn)" ]'
check "a missing zoneinfo directory is not mounted" '! grep -q zoneinfo "$t/state/last-args"'
kill -KILL -- "-$s"; waitfor '! alive "$(container zn)"'
unset GUFO_ZONEINFO

rm -f "$t/state/last-args"
unset TZ
s=$(start zu); waitfor 'grep -q "^TZ=" "$t/state/last-args" 2>/dev/null'
check "without TZ, a zone name is still passed" 'grep -Eq "^TZ=[A-Za-z0-9_+/-]+$" "$t/state/last-args"'
kill -KILL -- "-$s"; waitfor '! alive "$s"'

# the container's output goes to the attached stdout/stderr only: no journald copy at error priority
rm -f "$t/state/last-args"
s=$(start ld); waitfor '[ -s "$t/state/last-args" ]'
check "podman run gets --log-driver=none" 'grep -qx -- "--log-driver=none" "$t/state/last-args"'
kill -KILL -- "-$s"; waitfor '! alive "$s"'
rm -f "$t/state/last-args"
s=$(start lo --log-driver=k8s-file); waitfor '[ -s "$t/state/last-args" ]'
check "a caller's --log-driver comes after it, so it wins" '[ "$(grep -n -- "^--log-driver" "$t/state/last-args" | tail -1 | cut -d: -f2)" = "--log-driver=k8s-file" ]'
kill -KILL -- "-$s"; waitfor '! alive "$s"'

echo "gufo serve.sh: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
