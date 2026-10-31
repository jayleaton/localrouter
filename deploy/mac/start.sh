#!/usr/bin/env bash
# Starts LocalRouter's launchd agent (installed by install.sh) and waits until the warm models (chat, Drama 3,
# Whisper) are ready. --no-wait: return once the daemon answers.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env.sh"
[ -f "$PLIST" ] || die "not installed: deploy/mac/install.sh first"
if [ "$(agent_state)" = running ]; then echo "$LABEL is running (pid $(agent_pid))"; else
  agent_unload   # a loaded agent that is not running (it exited): load it afresh
  agent_load || die "launchctl bootstrap $DOMAIN $PLIST failed"
fi
for _ in $(seq 150); do curl -fs -m 2 "$URL/health" > /dev/null && break; sleep 0.2; done
curl -fs -m 2 "$URL/health" > /dev/null || die "the daemon did not answer on $URL: tail $LOG"
echo "$LABEL answers on $URL (pid $(agent_pid)); log $LOG"
[ "${1:-}" = --no-wait ] || wait_warm "${WARM_TIMEOUT_S:-1200}"
pub=$(public_url || true)
echo "URL: ${pub:+$pub, }loopback $URL"
