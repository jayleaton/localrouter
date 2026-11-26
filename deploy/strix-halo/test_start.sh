#!/usr/bin/env bash
# Tests that the Gufo image start.sh chooses reaches the workers' containers (no GPU, no systemd, no podman):
#   bash deploy/strix-halo/test_start.sh
# A stub systemd-run runs the daemon the way a transient unit does, with an empty environment plus its --setenv
# values; the stub daemon starts tools/gufo/serve.sh as a worker would, and a stub podman records the image it is given.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$here/../.." && pwd)
pinned=$(cat "$REPO/tools/gufo/IMAGE")
other="registry.invalid/gufo/runtime:9.9@sha256:$(printf 'ab%.0s' $(seq 32))"
fails=0 n=0
scratch=$(mktemp -d); trap 'rm -rf "$scratch"' EXIT
check() { n=$((n + 1)); if [ "$2" = "$3" ]; then echo "PASS $1"; else echo "FAIL $1: got '$2', want '$3'"; fails=$((fails + 1)); fi; }

# start [VAR=VALUE...]: start.sh in a fresh prefix with these variables exported; sets $t (the scratch directory)
start() {
  t=$(mktemp -d "$scratch/run.XXXXXX"); mkdir -p "$t/bin"
  cat > "$t/bin/systemd-run" <<SH
#!/usr/bin/env bash
# a transient unit: an empty environment plus --setenv; the command runs to completion here
envs=()
while [ \$# -gt 0 ]; do
  case \$1 in --setenv=*) envs+=("\${1#--setenv=}") ;; --user|--collect|--unit=*) ;; -p) shift ;; *) break ;; esac
  shift
done
env -i "\${envs[@]}" "\$@" > "$t/unit.log" 2>&1
SH
  printf '#!/bin/sh\n[ "$2" = is-active ] && exit 3\nexit 0\n' > "$t/bin/systemctl"
  printf '#!/bin/sh\nexit 0\n' > "$t/bin/curl"
  printf '#!/bin/sh\nfor a in "$@"; do case $a in *@sha256:*|*:latest|*/*:*) echo "$a" > %q ;; esac; done\n' "$t/podman-image" > "$t/bin/podman"
  cat > "$t/lr" <<SH
#!/usr/bin/env bash
# the daemon: record the unit's environment, then start a Gufo worker as the openai_audio worker does
env > "$t/unit-env"
"$REPO/tools/gufo/serve.sh" asr 18092 -- gufo serve asr --model /m 2> "$t/serve.err"
echo \$? > "$t/serve-rc"
SH
  chmod +x "$t/bin/"* "$t/lr"
  ( export PATH="$t/bin:$PATH" J="$t/j" LR_BIN="$t/lr" "$@"; bash "$here/start.sh" > "$t/start.out" 2>&1 )
}
unit() { sed -n "s/^$1=//p" "$t/unit-env"; }
seen() { cat "$t/podman-image" 2> /dev/null; }

start
check "default: the unit gets the pinned image" "$(unit GUFO_IMAGE)" "$pinned"
check "default: the worker's container runs it" "$(seen)" "$pinned"
check "the pin is a digest" "$(grep -c '@sha256:[0-9a-f]\{64\}$' "$REPO/tools/gufo/IMAGE")" 1
start GUFO_IMAGE="$other" UNRELATED=leak
check "exported override: reaches the unit" "$(unit GUFO_IMAGE)" "$other"
check "exported override: reaches the container" "$(seen)" "$other"
check "a variable start.sh does not pass stays out (the unit's environment is empty)" "$(unit UNRELATED)" ""
site=$scratch/site.env; echo "GUFO_IMAGE=$other" > "$site"
start SITE="$site"
check "site file override: reaches the container" "$(seen)" "$other"
start GUFO_IMAGE=ghcr.io/gufo-org/toolboxes/gufo-runtime:latest
check "a tag without a digest: serve.sh refuses (exit 2)" "$(cat "$t/serve-rc")" 2
check "a tag without a digest: no container" "$(seen)" ""
check "a tag without a digest: it says why" "$(grep -c 'not pinned by digest' "$t/serve.err")" 1
start GUFO_IMAGE=ghcr.io/gufo-org/toolboxes/gufo-runtime:latest GUFO_ALLOW_UNPINNED=1
check "GUFO_ALLOW_UNPINNED=1 reaches the unit and runs the tag" "$(unit GUFO_ALLOW_UNPINNED) $(seen)" "1 ghcr.io/gufo-org/toolboxes/gufo-runtime:latest"
echo "$((n - fails))/$n passed"
[ $fails = 0 ]
