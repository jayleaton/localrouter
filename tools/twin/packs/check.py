"""python tools/twin/packs/check.py PACK_DIR DIGESTS.json: every tensor of a pack's manifest against the reference
digests (a pack made from the official checkpoint with the committed scales is byte-identical on any machine)."""
import json
import sys

man = json.load(open(f"{sys.argv[1]}/manifest.json"))["tensors"]
ref = json.load(open(sys.argv[2]))["tensors"]
bad = sorted(k for k in ref if man.get(k, {}).get("sha256") != ref[k])
extra = sorted(set(man) - set(ref))
print(json.dumps({"pack": sys.argv[1], "tensors": len(ref), "differ_or_missing": bad[:20], "extra": extra[:20],
                  "pass": not bad and not extra}))
sys.exit(0 if not bad and not extra else 1)
