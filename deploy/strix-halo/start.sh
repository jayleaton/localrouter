#!/usr/bin/env bash
# Starts the daemon as a transient systemd user unit ($UNIT; no unit file, no sudo), after rendering the config, and
# waits for /health. Logs: journalctl --user -u $UNIT. SITE=path/to/site.env selects a site file.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$here/../.." && pwd)
source "$here/env.sh"; [ -z "${SITE:-}" ] || source "$SITE"
[ -x "$LR_BIN" ] || { echo "start.sh: no $LR_BIN (cd $REPO && zig build -Doptimize=ReleaseSafe)" >&2; exit 1; }
cfg=$(bash "$here/render.sh")
systemctl --user is-active -q "$UNIT" && { echo "start.sh: $UNIT is already running (stop.sh first)" >&2; exit 1; }
systemctl --user reset-failed "$UNIT" 2> /dev/null || true
systemd-run --user --unit="$UNIT" --collect -p KillMode=mixed -p TimeoutStopSec=90 \
  --setenv=PATH="$PATH" --setenv=HOME="$HOME" \
  "$LR_BIN" serve --config "$cfg" --host 127.0.0.1 --port "$PORT" --data "$DEPLOY/data" > /dev/null
for _ in $(seq 1 60); do curl -fs -m 2 "http://127.0.0.1:$PORT/health" > /dev/null && { echo "localrouter up on http://127.0.0.1:$PORT ($UNIT)"; exit 0; }; sleep 1; done
echo "start.sh: no /health after 60 s; journalctl --user -u $UNIT -n 50" >&2; exit 1
