#!/usr/bin/env bash
# Starts the Jarvis service (installed by install.sh) and waits until chat, Drama 3, Whisper and the image model are warm.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
[ -f "$HOME/.config/systemd/user/$UNIT.service" ] || die "not installed: deploy/jarvis/install.sh first"
systemctl --user start "$UNIT"
for _ in $(seq 150); do curl -fs -m 2 "$URL/health" > /dev/null && break; sleep 0.2; done
curl -fs -m 2 "$URL/health" > /dev/null || die "the daemon did not answer on $URL: journalctl --user -u $UNIT"
wait_warm "${WARM_TIMEOUT_S:-1200}"
echo "MemAvailable $(mem_gib) GiB; $(public_url || true) (loopback $URL)"
