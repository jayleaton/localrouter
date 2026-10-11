#!/usr/bin/env bash
# LocalRouter on one DGX Spark, in one command, from a checkout of this repo:
#   bash tools/spark/install.sh
# 1. checks the machine (GPU, Docker with the NVIDIA runtime and compose, disk);
# 2. makes the weight packs from the official checkpoints inside the NGC PyTorch container (prepare.sh; Qwen-Image 2.1
#    about 30 minutes the first time, MiniMax H3 about 1 hour with its 61 GB download and the 15 minute pack build;
#    each skipped once done);
# 3. builds LocalRouter's image natively and starts it (compose, restarts with the machine);
# 4. checks it through MCP (tools/list, then one small image: the cold load and a generation; then, with video, a
#    1 second clip at 768x448 with its wall time);
# 5. prints how an agent connects (http://<this machine>:PORT/mcp).
# Models load on the first call and unload after their idle TTL, so the machine's memory is only held while in use.
# Re-running updates the image and keeps the packs. Env: LOCALROUTER_HOME (/data/localrouter), LOCALROUTER_PORT (8190),
# LOCALROUTER_BIND (who can reach it: "localhost" (the default: this machine only), "tailscale" (the tailnet IP, from
# `tailscale ip -4`), "all" (every interface) or an IPv4 address; asked once on a terminal when unset, and remembered in
# $LOCALROUTER_HOME/.env), MODELS ("image video"; "image" or "video" installs one; about 200 GB free for both, 110 GB for
# images only), PRECISIONS (the image model: "fp8s", the default and primary; "fp8s nvfp4" adds the faster NVFP4 as
# qwen-image-2.1-nvfp4), IMAGE_MODELS (the image checkpoints: "qwen-image-2.1 qwen-image-2.1-turbo", the default, or
# either alone; Turbo adds about 25 GB), IMAGE_DEFAULT (the model a request without one gets: qwen-image-2.1-turbo when
# installed, else qwen-image-2.1; it is also the one kept warm), LOCALROUTER_SKIP_SMOKE=1.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LOCALROUTER_HOME=${LOCALROUTER_HOME:-/data/localrouter}
ENVFILE=$LOCALROUTER_HOME/.env   # compose's variables, kept so re-runs and restarts bind the same way
envsaved() { { [[ -f $ENVFILE ]] && sed -n "s/^$1=//p" "$ENVFILE" | tail -1; } || true; }
LOCALROUTER_PORT=${LOCALROUTER_PORT:-$(envsaved LOCALROUTER_PORT)}; LOCALROUTER_PORT=${LOCALROUTER_PORT:-8190}
PRECISIONS=${PRECISIONS:-fp8s}
MODELS=${MODELS:-image video}
IMAGE_MODELS=${IMAGE_MODELS:-qwen-image-2.1 qwen-image-2.1-turbo}
NGC=nvcr.io/nvidia/pytorch:26.07-py3
step() { echo "[install $(date +%T)] $*"; }
die() { echo "install: $*" >&2; exit 1; }

want() { [[ " $MODELS " == *" $1 "* ]]; }
[[ -n ${PRECISIONS//[[:space:]]/} ]] || die "PRECISIONS is empty: use \"fp8s\" (default) or \"fp8s nvfp4\""
for p in $PRECISIONS; do [[ $p == fp8s || $p == nvfp4 ]] || die "PRECISIONS has '$p': use \"fp8s\", \"nvfp4\" or \"fp8s nvfp4\""; done
for m in $IMAGE_MODELS; do [[ $m == qwen-image-2.1 || $m == qwen-image-2.1-turbo ]] || die "IMAGE_MODELS has '$m': use qwen-image-2.1 and/or qwen-image-2.1-turbo"; done
[[ " $IMAGE_MODELS " == *" qwen-image-2.1-turbo "* ]] && IMAGE_DEFAULT=${IMAGE_DEFAULT:-qwen-image-2.1-turbo}
IMAGE_DEFAULT=${IMAGE_DEFAULT:-qwen-image-2.1}
[[ " $IMAGE_MODELS " == *" $IMAGE_DEFAULT "* ]] || die "IMAGE_DEFAULT '$IMAGE_DEFAULT' is not in IMAGE_MODELS"

# ---- who may connect: LOCALROUTER_BIND, else what an earlier run saved, else (on a terminal) ask, else this machine only
BIND_MODE=${LOCALROUTER_BIND:-$(envsaved LOCALROUTER_BIND_MODE)}
if [[ -z $BIND_MODE ]]; then
    BIND_MODE=localhost
    if [[ -t 0 && -t 1 ]]; then
        read -r -p "Make LocalRouter reachable from your tailnet? [y/N] " answer || answer=
        [[ $answer == [yY]* ]] && BIND_MODE=tailscale
    fi
fi
BIND_MODE=${BIND_MODE,,}
case $BIND_MODE in
    localhost) BIND_IP=127.0.0.1 ;;
    all) BIND_IP=0.0.0.0 ;;
    tailscale)
        # The tailnet address is found here only to check the service and to print it; the daemon looks for it itself
        # (--host tailscale) every time it starts, which is what makes boot order a non-issue (see compose.tailscale.yaml).
        command -v tailscale > /dev/null || die "LOCALROUTER_BIND=tailscale, but tailscale is not installed (https://tailscale.com/download)"
        BIND_IP=$(tailscale ip -4 2> /dev/null | head -n1 || true)
        [[ $BIND_IP =~ ^100\.([0-9]+)\.[0-9]+\.[0-9]+$ ]] && (( BASH_REMATCH[1] >= 64 && BASH_REMATCH[1] <= 127 )) ||
            die "LOCALROUTER_BIND=tailscale, but 'tailscale ip -4' gave no tailnet address: is tailscale up? (tailscale up)" ;;
    *)
        [[ $BIND_MODE =~ ^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]] &&
            (( BASH_REMATCH[1] < 256 && BASH_REMATCH[2] < 256 && BASH_REMATCH[3] < 256 && BASH_REMATCH[4] < 256 )) ||
            die "LOCALROUTER_BIND='$BIND_MODE': use localhost, tailscale, all or an IPv4 address"
        BIND_IP=$BIND_MODE ;;
esac
export LOCALROUTER_PORT LOCALROUTER_BIND=$BIND_IP
[[ -n ${MODELS//[[:space:]]/} ]] || die "MODELS is empty: use \"image\", \"video\" or \"image video\""
for m in $MODELS; do [[ $m == image || $m == video ]] || die "MODELS has '$m': use \"image\", \"video\" or \"image video\""; done

# ---- 1. the machine
step "checks"
command -v nvidia-smi > /dev/null || die "no nvidia-smi: this needs an NVIDIA GPU and its driver"
command -v docker > /dev/null || die "no docker"
docker info --format '{{json .Runtimes}}' | grep -q nvidia || die "Docker has no nvidia runtime (install the NVIDIA Container Toolkit)"
docker compose version > /dev/null 2>&1 || die "no 'docker compose' (the compose plugin)"
mkdir -p "$LOCALROUTER_HOME" 2> /dev/null || sudo install -d -o "$(id -u)" -g "$(id -g)" "$LOCALROUTER_HOME"
free_gb=$(df -BG --output=avail "$LOCALROUTER_HOME" | tail -1 | tr -dc 0-9)
need=0   # what is still to make: images about 110 GB (checkpoint, packs, caches), video about 90 GB more (61 GB of checkpoints, the pack, scratch)
if want image && [[ ! -f $LOCALROUTER_HOME/weights/qwen-image-2.1/te/manifest.json ]]; then need=$(( need + 110 )); fi
if want image && [[ " $IMAGE_MODELS " == *" qwen-image-2.1-turbo "* && ! -d $LOCALROUTER_HOME/weights/qwen-image-2.1/turbo-fp8s ]]; then need=$(( need + 25 )); fi
if want video && [[ ! -f $LOCALROUTER_HOME/weights/minimax-h3/h3-turbo-nvfp4/manifest.json ]]; then need=$(( need + 90 )); fi
(( free_gb >= need )) ||
    die "$LOCALROUTER_HOME has ${free_gb} GB free; installing [$MODELS] needs about ${need} GB more (images 110, video 90; both 200)"
others=$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | grep -v "^$" || true)
[[ -z $others ]] || echo "install: warning: other GPU processes are running (they share the memory):"$'\n'"$others" >&2

# ---- 2. the weights
step "weights (inside $NGC)"
mkdir -p "$LOCALROUTER_HOME/repo" "$LOCALROUTER_HOME/twin" "$LOCALROUTER_HOME/kernels"
for d in tools/twin:twin kernels:kernels .:repo; do   # the container's copy of the tree (prepare.sh runs from it)
    src=${d%%:*} dst=${d##*:}
    tar -C "$REPO/$src" --exclude=.zig-cache --exclude=zig-out --exclude=zig-pkg --exclude=results --exclude=__pycache__ \
        --exclude=.git -cf - . | tar -C "$LOCALROUTER_HOME/$dst" -xf -
done
docker run --rm --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 -e PRECISIONS="$PRECISIONS" -e MODELS="$MODELS" -e IMAGE_MODELS="$IMAGE_MODELS" \
    -v "$LOCALROUTER_HOME":/workspace/localrouter "$NGC" bash /workspace/localrouter/repo/tools/spark/prepare.sh

# ---- 3. the service
step "config"
mkdir -p "$LOCALROUTER_HOME/data" && chmod 777 "$LOCALROUTER_HOME/data"   # the image runs as uid 10001
python3 - "$LOCALROUTER_HOME/tools.json" "$PRECISIONS" "$(hostname)" "$MODELS" "$IMAGE_MODELS" "$IMAGE_DEFAULT" "$REPO/tools/twin/packs" <<'PY'
import json, os, sys
tools = []
models = sys.argv[4].split()
defaults = {}
if "image" in models:
    precisions = sys.argv[2].split()
    primary = "fp8s" if "fp8s" in precisions else precisions[0]
    image_default = sys.argv[6]
    for model in sys.argv[5].split():
        turbo = model.endswith("-turbo")
        for p in precisions:
            if turbo and not os.path.exists(f"{sys.argv[7]}/acts-{p.removesuffix('s')}-turbo.json"):
                continue  # no committed Turbo scales for this precision: prepare.sh made no pack
            # FP8 is each checkpoint's primary model (closest to the original); NVFP4 (faster) gets the -nvfp4 id. If
            # only NVFP4 was asked for, it takes the primary id. The image default is kept warm and kept past the others
            # when memory is needed; the other image models load on demand.
            tid = model if p == primary else f"{model}-{p.replace('fp8s', 'fp8')}"
            label = "NVFP4" if p == "nvfp4" else "FP8"
            tools.append({"id": tid, "kind": "image", "engine": "qwen_image_turbo" if turbo else "qwen_image",
                          "name": f"Qwen-Image 2.1 Turbo ({label}, 8 steps)" if turbo else f"Qwen-Image 2.1 ({label})",
                          "weights": "/models/qwen-image-2.1", "idle_ttl_s": 300, "priority": 10 if tid == image_default else 5,
                          "keep_loaded": tid == image_default, "options": {"precision": p, "max_side": 1664}})
    if any(t["id"] == image_default for t in tools):
        defaults["text_to_image"] = image_default
if "video" in models:
    # the video tool loads on demand (priority 0): it makes room by unloading idle tools, the image model last
    tools.append({"id": "minimax-h3", "kind": "video", "name": "MiniMax H3 (video with audio, Turbo 8 steps)",
                  "engine": "minimax_h3", "weights": "/models/minimax-h3", "idle_ttl_s": 300,
                  "options": {"max_width": 768, "max_height": 448, "max_seconds": 5}})
# machine: this host's name for list_models (inside the container the hostname is the container's id)
json.dump({"machine": sys.argv[3], "reserve_bytes": 8 << 30, "defaults": defaults, "tools": tools}, open(sys.argv[1], "w"), indent=1)
PY
step "image and service"
export LOCALROUTER_MODELS=$LOCALROUTER_HOME/weights LOCALROUTER_DATA=$LOCALROUTER_HOME/data LOCALROUTER_CONFIG=$LOCALROUTER_HOME/tools.json
# the variables compose reads, kept beside the packs: docker compose --env-file $LOCALROUTER_HOME/.env ... sees the same bind
# (LOCALROUTER_BIND_MODE is what was asked for, LOCALROUTER_BIND the address it resolved to when this was written)
printf '%s\n' "# written by tools/spark/install.sh" "LOCALROUTER_BIND=$BIND_IP" "LOCALROUTER_BIND_MODE=$BIND_MODE" \
    "LOCALROUTER_PORT=$LOCALROUTER_PORT" "LOCALROUTER_MODELS=$LOCALROUTER_MODELS" "LOCALROUTER_DATA=$LOCALROUTER_DATA" \
    "LOCALROUTER_CONFIG=$LOCALROUTER_CONFIG" > "$ENVFILE.new"
mv "$ENVFILE.new" "$ENVFILE"
# Bind: localhost / all / an IP is the published port (LOCALROUTER_BIND:PORT:8190, compose.yaml). tailscale runs the
# container on the host network with the daemon's own --host tailscale (compose.tailscale.yaml): the daemon listens on
# the tailnet IP only, and when Docker starts before tailscale has an address it exits and Docker's restart policy
# (unless-stopped) starts it again until tailscale is up. A port published on the tailnet IP would instead fail to
# bind at boot and need a systemd ordering drop-in; this needs none.
COMPOSE=(docker compose --env-file "$ENVFILE" -f "$REPO/docker/compose.yaml")
[[ $BIND_MODE != tailscale ]] || COMPOSE+=(-f "$REPO/docker/compose.tailscale.yaml")
"${COMPOSE[@]}" -p localrouter up -d --build --force-recreate   # re-reads tools.json
url=http://127.0.0.1:$LOCALROUTER_PORT
[[ $BIND_IP == 0.0.0.0 || $BIND_IP == 127.0.0.1 ]] || url=http://$BIND_IP:$LOCALROUTER_PORT
for _ in $(seq 1 60); do curl -sf "$url/health" > /dev/null && break; sleep 1; done
curl -sf "$url/health" > /dev/null || die "the service did not come up: docker compose -p localrouter logs"

# ---- 4. a check through MCP: what an agent will do
if [[ ${LOCALROUTER_SKIP_SMOKE:-0} != 1 ]]; then
    rpc() { curl -sS -m 1000 "$url/mcp" -H 'content-type: application/json' -d "$1"; }
    step "MCP check: tools/list"
    rpc '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python3 -c 'import json,sys; t=[x["name"] for x in json.load(sys.stdin)["result"]["tools"]]; print("  tools:", ", ".join(t))'
    if want image; then
        step "MCP check: a 512x512 image (includes the model's cold load)"
        t0=$(date +%s)
        rpc '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"generate_image","arguments":{"prompt":"a red fox in fresh snow","size":"512x512","steps":8,"seed":1,"inline_images":false}}}' |
            python3 -c 'import json,sys; r=json.load(sys.stdin)["result"]; t=r["content"][0]["text"]; print("  image:", t); sys.exit(1 if r["isError"] else 0)' ||
            die "the MCP image check failed: docker compose -p localrouter logs; $LOCALROUTER_HOME/data/logs/"
        echo "  $(( $(date +%s) - t0 )) s including the load"
    fi
    if want video; then
        step "MCP check: a 1 second 768x448 clip, 8 steps (includes the model's cold load, and unloading idle models if memory is short)"
        t0=$(date +%s)
        rpc '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"generate_video","arguments":{"prompt":"waves at dusk","model":"minimax-h3","size":"768x448","seconds":1,"steps":8,"seed":1,"wait_s":900}}}' |
            python3 -c 'import json,sys; r=json.load(sys.stdin)["result"]; t=r["content"][0]["text"]; print("  video:", t); sys.exit(1 if r["isError"] or json.loads(t).get("status") != "completed" else 0)' ||
            die "the MCP video check failed: docker compose -p localrouter logs; $LOCALROUTER_HOME/data/logs/"
        echo "  video check: $(( $(date +%s) - t0 )) s wall time including the load"
    fi
fi

# ---- 5. how agents connect
name=$BIND_IP
if [[ $BIND_MODE == all || $BIND_MODE == tailscale ]]; then   # the tailnet name resolves to the tailnet IP
    name=$(tailscale status --json 2> /dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' 2> /dev/null || true)
    [[ -n $name ]] || { name=$BIND_IP; [[ $BIND_IP != 0.0.0.0 ]] || name=$(hostname); }
fi
case $BIND_MODE in
    localhost) reach="It listens on this machine only. To reach it from your tailnet: LOCALROUTER_BIND=tailscale bash tools/spark/install.sh" ;;
    tailscale) reach="It listens on the tailnet address $BIND_IP only." ;;
    all) reach="It listens on every interface and has no authentication: keep it on a private network." ;;
    *) reach="It listens on $BIND_IP only." ;;
esac
cat <<EOF

LocalRouter is running: http://$name:$LOCALROUTER_PORT (API) and http://$name:$LOCALROUTER_PORT/mcp (MCP).
$reach
Connect an agent on any machine that reaches this one:
  stdio bridge: localrouter mcp-stdio --url http://$name:$LOCALROUTER_PORT
  JSON config:   {"mcpServers": {"localrouter": {"type": "http", "url": "http://$name:$LOCALROUTER_PORT/mcp"}}}
  stdio only:    npx -y mcp-remote http://$name:$LOCALROUTER_PORT/mcp --allow-http
Stop: docker compose -p localrouter down    Logs: docker compose -p localrouter logs; $LOCALROUTER_HOME/data/logs/
Other compose commands: ${COMPOSE[*]} -p localrouter ...
EOF
