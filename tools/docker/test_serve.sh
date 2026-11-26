#!/usr/bin/env bash
# serve.sh without a GPU: a stub docker whose "container" runs outside the caller's process group, as dockerd's does.
# A SIGKILL of serve.sh's group (the daemon's, on abort or timeout) must not leave the container running; SIGTERM
# stops it through the trap; a later run of the same name is not removed by an earlier run's watcher; a leftover of
# the name is removed before a run; an unpinned image is refused.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
t=$(mktemp -d); trap 'pkill -KILL -f "stub-container $t" 2> /dev/null; rm -rf "$t"' EXIT
pass=0 fail=0
check() { if eval "$2"; then pass=$((pass + 1)); else fail=$((fail + 1)); echo "FAIL: $1" >&2; fi; }
mkdir -p "$t/bin" "$t/state"
cat > "$t/bin/docker" <<STUB
#!/usr/bin/env bash
# run --rm --cidfile F --name N ... : a "container" (sleep) in a session of its own, its pid under the id
S=$t/state
kill_ref() { local c=\$1; [ -f "\$S/name-\$c" ] && c=\$(cat "\$S/name-\$c"); [ -f "\$S/\$c" ] && kill "\$(cat "\$S/\$c")" 2> /dev/null; echo "\$c" >> "\$S/removed"; }
case "\$1" in
  run)
    shift
    printf '%s\n' "\$@" > "\$S/last-args"
    cid=\$(date +%s%N); name=; cidfile=
    while [ \$# -gt 0 ]; do case "\$1" in --cidfile) cidfile=\$2; shift ;; --name) name=\$2; shift ;; esac; shift; done
    setsid bash -c 'exec -a "stub-container $t \$0" sleep 300' "\$cid" < /dev/null > /dev/null 2>&1 &
    echo \$! > "\$S/\$cid"; echo "\$cid" > "\$S/name-\$name"; echo "\$cid" > "\$cidfile"
    wait ;;  # the client waits; it dies with the group, the container does not
  stop) shift; while [ \$# -gt 0 ]; do case "\$1" in -t) shift ;; *) kill_ref "\$1" ;; esac; shift; done; exit 0 ;;
  rm) shift; for a in "\$@"; do [ "\$a" = -f ] || kill_ref "\$a"; done; exit 0 ;;
esac
STUB
chmod +x "$t/bin/docker"
export PATH="$t/bin:$PATH" DOCKER_STOP_TIMEOUT=1 TMPDIR="$t"
img="sha256:$(printf '0%.0s' {1..64})"
container() { local c; c=$(cat "$t/state/name-lr-$1" 2> /dev/null) && cat "$t/state/$c" 2> /dev/null; }
alive() { [ -n "$1" ] && kill -0 "$1" 2> /dev/null; }
start() { setsid bash "$here/serve.sh" "$1" --network host -- "$img" serve --port 18080 > /dev/null 2>&1 & echo $!; }  # its own group, as a worker's
waitfor() { local i; for i in $(seq 100); do eval "$1" && return 0; sleep 0.1; done; return 1; }

# a hard kill of the group: the container goes too
s=$(start a); waitfor '[ -n "$(container a)" ]'; c=$(container a)
check "the container runs" 'alive "$c"'
check "docker run gets the GPUs, the options and the command" 'grep -qx -- "--gpus" "$t/state/last-args" && grep -qx host "$t/state/last-args" && [ "$(tail -3 "$t/state/last-args" | tr "\n" " ")" = "serve --port 18080 " ]'
kill -KILL -- "-$s"
check "SIGKILL of serve.sh's group: the container is removed" 'waitfor "! alive $c"'
check "by its id" 'grep -q "$(cat "$t/state/name-lr-a")" "$t/state/removed"'
check "the watcher cleans up after itself" 'waitfor "[ -z \"\$(ls -d "$t"/lr-a.* 2> /dev/null)\" ]"'

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

# a container of the name left over from anything else is removed before the run
setsid bash -c "exec -a 'stub-container $t old' sleep 300" < /dev/null > /dev/null 2>&1 & old=$!
echo old > "$t/state/name-lr-d"; echo "$old" > "$t/state/old"
s=$(start d); waitfor '[ "$(cat "$t/state/name-lr-d")" != old ]'
check "a leftover of the name is removed first" 'waitfor "! alive $old"'
kill -KILL -- "-$s"; waitfor '! alive "$(container d)"'

# a name Docker refuses: its characters become _
s=$(start fishaudio/drama-3); waitfor '[ -n "$(container fishaudio_drama-3)" ]'
check "a / in the name becomes _" 'alive "$(container fishaudio_drama-3)"'
kill -KILL -- "-$s"; waitfor '! alive "$(container fishaudio_drama-3)"'

# an image by tag is refused, unless asked for
out=$(bash "$here/serve.sh" e -- tensorfold-inkling2:latest serve 2>&1); r=$?
check "a tag is refused" '[ $r = 2 ] && grep -q "not pinned" <<< "$out" && [ ! -f "$t/state/name-lr-e" ]'

echo "docker serve.sh: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
