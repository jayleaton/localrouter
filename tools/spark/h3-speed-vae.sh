# The video VAE's speed steps for a GB10 window. A fragment, not a script: tools/spark/h3-inner.sh sources it (or a shell that
# has its helpers does: run, cap, sel, drop, and $S the stk binary, $VV the video VAE checkpoint, $VMODELS, $CAP, $R) after
# its zig-build and h3-vvae steps, with the twin's venv active (video-env.sh) so `python -m stk_twin...` works:
#   [[ -f $STK/repo/tools/spark/h3-speed-vae.sh ]] && source $STK/repo/tools/spark/h3-speed-vae.sh
# What changed (kernels/cuda/minimax/gemm_f16.cu, vae_video_tile.zig, stk_twin/h3/vae_video.py), every bit of the output kept:
# new GEMM kernels (256 x 128 x 32 tiles with a 3-stage cp.async pipeline, the m tile fastest in the grid so the weights stream
# from memory once) against the reference stk_gemm_f16_ref, and the attention run two heads at a time so the scores stay in the L2.
# Knobs of the Zig engine and the twin (env, read at start): STK_VVAE_REF=1 (the old path: reference GEMM, all heads at
# once), STK_VVAE_HEADS=n (heads a group runs, default 2; 32 = all at once), STK_VVAE_PROF=1 (GPU ms of a decode by class
# to stderr: "vvae_phase_ms gemm=.. attn_gemm=.. softmax=.. norm=.. rope=.. swiglu=.. other=.. place=.. sum=..").
# Steps (the names ONLY and the logs use; a failing step does not stop the others):
#   vvae-gemm-test     twin GPU test, the new tests only: every GEMM kernel == stk_gemm_f16_ref bit for bit over every shape the
#                      decoder launches and odd edges, the attention by groups == all heads; the timing table (ms, TFLOP/s,
#                      speedup) is in the log
#   vvae-capture       twin capture of the 22-frame 768x448 clip on the new kernels (one tile, blocks 0 and 35 recorded)
#   vvae-check         stk check h3-vvae: Zig vs twin, both on the new kernels: every recorded op alone and chained, the frames' hash
#   vvae-check-ref     the same capture checked with STK_VVAE_REF=1: the Zig engine's old path against the new twin's frames
#   vvae-capture-ref   the twin on the old path (STK_VVAE_REF=1), vvae-hash-eq: its frames' sha256 equals the new twin's
#   for F in 56 124 (768x448: 24 and 56 tile decodes):  vvae-capture-F, then the check's decode_ms (the last, plain decode):
#     vvae-new-F (default), vvae-ref-F (STK_VVAE_REF=1: the old path), vvae-prof-F (STK_VVAE_PROF=1: where the time goes);
#     and at 56 the group sweep vvae-heads-G-56 for G in 32 8 4 2 1 (the L2 residency of the scores; 32 = one launch of all heads)
#   vvae-speed-table   decode_ms, pass and frames_equal of every vvae-* step, the phase lines, one JSON line
# Every check also compares the decoded video's sha256 with the capture's ("frames_equal"), so old and new are proven equal at
# the whole-video level, not only per op. The new default is a win when vvae-new-F decode_ms < vvae-ref-F decode_ms with every
# frames_equal true; the best G of the sweep replaces the default of STK_VVAE_HEADS (Ops.load, vae_video.py `groups`).
declare -F run > /dev/null && declare -F cap > /dev/null && [[ -n ${S:-} && -n ${VV:-} && -n ${CAP:-} ]] ||
    { echo "h3-speed-vae: source it from h3-inner.sh (run, cap, \$S, \$VV, \$CAP)" >&2; return 1 2> /dev/null || exit 1; }

VS=$CAP/h3-vvae-speed
# vv_capture NAME DIR FRAMES [VAR=value ...]: the twin's capture of a 768x448 clip (small: one tile, two blocks)
vv_capture() { local name=$1 dir=$2 frames=$3; shift 3
    cap $name $dir env "$@" python -m stk_twin.h3.capture_vvae --models $VMODELS --out $dir --size 768x448 --frames $frames \
        --tiles 0:1:1 --layers 0,35; }
# vv_check NAME DIR [VAR=value ...]: stk check h3-vvae on a capture
vv_check() { local name=$1 dir=$2; shift 2
    run $name env "$@" $S check h3-vvae $VV $dir; }

run vvae-gemm-test python -m stk_twin.h3.test_vae_video --gemm

# the check on the standard clip: new kernels on both sides, then the Zig old path against the same capture
vv_capture vvae-capture $VS-22 22
vv_check vvae-check $VS-22
vv_check vvae-check-ref $VS-22 STK_VVAE_REF=1
# the twin on the old path makes the same frames: the sha256 of the two captures' videos
vv_capture vvae-capture-ref $VS-22r 22 STK_VVAE_REF=1
run vvae-hash-eq python - $R/vvae-capture.log $R/vvae-capture-ref.log <<'PY'
import json, sys
def sha(p):
    for l in reversed(open(p, errors="replace").read().splitlines()):
        if l.startswith("{"):
            return json.loads(l).get("sha256")
a, b = sha(sys.argv[1]), sha(sys.argv[2])
print(json.dumps({"vvae_hash_eq": {"new": a, "ref": b, "equal": a is not None and a == b}}))
sys.exit(0 if a is not None and a == b else 1)
PY
drop $VS-22 $VS-22r

# decode timing at 768x448: the old path, the new one, the phases; the plain decode's time is the check's "decode_ms"
for F in 56 124; do
    vv_capture vvae-capture-$F $VS-$F $F
    vv_check vvae-new-$F $VS-$F
    vv_check vvae-ref-$F $VS-$F STK_VVAE_REF=1
    vv_check vvae-prof-$F $VS-$F STK_VVAE_PROF=1
    if [[ $F == 56 ]]; then for G in 32 8 4 2 1; do vv_check vvae-heads-$G-56 $VS-56 STK_VVAE_HEADS=$G; done; fi
    drop $VS-$F
done

run vvae-speed-table python - $R <<'PY'
import glob, json, os, sys
rows, out = [], {}
for f in sorted(glob.glob(sys.argv[1] + "/vvae-*.log")):
    name = os.path.basename(f)[:-4]
    lines = open(f, errors="replace").read().splitlines()
    last = next((l for l in reversed(lines) if l.startswith("{")), "")
    try:
        v = json.loads(last).get("h3_vvae") or {}
    except ValueError:
        v = {}
    ph = [l for l in lines if l.startswith("vvae_phase_ms")]
    d = {k: v.get(k) for k in ("frames", "decode_ms", "frames_equal", "pass") if k in v}
    if ph:
        d["phase_ms"] = dict(kv.split("=") for kv in ph[-1].split()[1:])
    if d:
        out[name] = d
        rows.append((name, d.get("frames"), d.get("decode_ms"), d.get("frames_equal"), d.get("pass")))
print(f"{'step':<22}{'frames':>8}{'decode_ms':>11}{'frames_eq':>11}{'pass':>7}")
for r in rows:
    print(f"{r[0]:<22}{str(r[1]):>8}{str(r[2]):>11}{str(r[3]):>11}{str(r[4]):>7}")
for n, d in out.items():
    if "phase_ms" in d:
        print(n, d["phase_ms"])
print(json.dumps({"vvae_speed": out}))
PY
