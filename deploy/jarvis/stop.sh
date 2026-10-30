#!/usr/bin/env bash
# Stops the Jarvis service and every model with it (they run in its cgroup). It stays enabled, so it starts again at
# boot; --disable keeps it off.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
systemctl --user stop "$UNIT"
[ "${1:-}" != --disable ] || systemctl --user disable "$UNIT"
echo "stopped $UNIT; MemAvailable $(mem_gib) GiB"
