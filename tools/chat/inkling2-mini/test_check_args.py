#!/usr/bin/env python3
"""check_args.py against the recipe's renders (no GPU, no network, no kit): the argv `recipe serve PROFILE ... --dry-run`
prints at the pinned recipe, recorded below, must accept the Mac service's rendered Inkling-2-Mini entry
(deploy/mac/install.sh RENDER_ONLY=1, which needs jq), the Spark's config.example.json and deploy/jarvis/config.json.in
entry, and every change but the daemon's own must be refused.
    python3 -I tools/chat/inkling2-mini/test_check_args.py"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)  # -I leaves the script's directory off sys.path
import check_args  # noqa: E402

# `recipe serve PROFILE ENGINE MODEL --dry-run` at the recipe pins.json pins, after BIN serve MODEL --name NAME.
FLAGS = {
    "mac": ["--context", "131072", "--parallel", "2", "--no-update-check", "--max-tokens", "32768", "--thinking",
            "--temperature", "1.0", "--top-p", "1.0", "--top-k", "0", "--drafter", "incoai/Inkling-2-Mini-DFlash2",
            "--vision"],
    "spark": ["--context", "131072", "--parallel", "2", "--no-update-check", "--max-tokens", "32768", "--thinking",
              "--temperature", "1.0", "--top-p", "1.0", "--top-k", "0", "--prompt-cache-disk", ".recipe/sessions",
              "--drafter", "{model}", "--vision"],
}
# and the environment the launch sets (the profile's product switches)
ENV = {"mac": {}, "spark": {"TF_INKLING2_COMPUTE_LANE": "1"}}


def launch(profile, binary, model):
    flags = [model if f == "{model}" else f for f in FLAGS[profile]]
    return {"rank": 0, "argv": [binary, "serve", model, "--name", "Inkling-2-Mini", *flags, "--port", "8888"],
            "env": dict(ENV[profile])}


def tool(binary, model, flags, env=None):
    return {"id": "Inkling-2-Mini", "weights": model, "options": {
        "server": [binary, "serve", "{weights}", "--name", "{id}", "--host", "127.0.0.1", "--port", "{port}", *flags],
        "env": env or {}}}


class CheckArgs(unittest.TestCase):
    bin, model = "/r/engine/zig-out/native/bin/tensorfold-native", "/m/Inkling-2-Mini"

    def problems(self, flags, profile="mac", env=None, want_env=None):
        want = launch(profile, self.bin, self.model)
        want["env"] = want_env or {}
        return check_args.check(tool(self.bin, self.model, flags, env), want)[0]

    def test_the_recipes_flags_pass(self):
        self.assertEqual(self.problems(FLAGS["mac"]), [])

    def test_a_changed_missing_extra_or_moved_flag_fails(self):
        f = FLAGS["mac"]
        for flags in ([x.replace("131072", "65536") for x in f], f[:-1], f + ["--prompt-cache-gib", "8"],
                      f[2:4] + f[:2] + f[4:]):
            self.assertTrue(self.problems(flags), flags)

    def test_only_the_daemons_own_flags_may_differ(self):
        t = tool(self.bin, self.model, FLAGS["mac"])
        for i, v in ((4, "Inkling"), (6, "0.0.0.0"), (8, "8888")):
            bad = json.loads(json.dumps(t))
            bad["options"]["server"][i] = v
            self.assertTrue(check_args.check(bad, launch("mac", self.bin, self.model))[0], v)
        bad = json.loads(json.dumps(t))
        bad["options"]["server"][0] = "/elsewhere/tensorfold-native"
        self.assertTrue(check_args.check(bad, launch("mac", self.bin, self.model))[0])

    def test_spark_needs_an_absolute_disk_tier(self):
        flags = [self.model if f == "{model}" else f for f in FLAGS["spark"]]
        at = flags.index("--prompt-cache-disk") + 1
        self.assertTrue(self.problems(flags, "spark"))  # .recipe/sessions as the recipe has it: relative
        flags[at] = "/r/sessions"
        self.assertEqual(self.problems(flags, "spark"), [])
        flags[at + 2] = "{weights}"  # --drafter {weights}: the checkpoint, as the recipe's {model}
        self.assertEqual(self.problems(flags, "spark"), [])

    def test_the_launchs_env_must_be_set(self):
        self.assertTrue(self.problems(FLAGS["mac"], want_env={"TF_X": "1"}))
        self.assertEqual(self.problems(FLAGS["mac"], env={"TF_X": "1"}, want_env={"TF_X": "1"}), [])

    def spark(self, cfg, root, models):
        serve = launch("spark", f"{root}/engine/zig-out/native/bin/tensorfold-native", f"{models}/Inkling-2-Mini")
        t = next(t for t in cfg["tools"] if t["id"] == "Inkling-2-Mini")
        return check_args.check(t, serve)[0]

    def test_the_spark_example_serves_the_recipes_spark_argv(self):
        with open(os.path.join(HERE, "config.example.json")) as f:
            text = f.read()  # as setup.sh fills it in
        text = text.replace("/opt/localrouter/inkling/models", "/j/ink/models").replace("/opt/localrouter/inkling", "/j/ink")
        self.assertEqual(self.spark(json.loads(text), "/j/ink", "/j/ink/models"), [])

    def test_the_jarvis_service_serves_the_recipes_spark_argv(self):
        with open(os.path.join(REPO, "deploy", "jarvis", "config.json.in")) as f:
            text = f.read()  # as deploy/jarvis/install.sh renders it, every other placeholder a stand-in
        text = text.replace("@INKLING_ROOT@", "/j/ink").replace("@INKLING_MODELS@", "/j/ink/models")
        text = text.replace("@DRAMA_PORT@", "0").replace("@DRAMA_RESIDENT@", "0")
        text = re.sub(r"@[A-Z_]+@", "/x", text)
        self.assertEqual(self.spark(json.loads(text), "/j/ink", "/j/ink/models"), [])

    @unittest.skipUnless(shutil.which("jq"), "needs jq")
    def test_the_mac_service_serves_the_recipes_mac_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            j, home = os.path.join(tmp, "jarvis"), os.path.join(tmp, "home")
            env = {k: v for k, v in os.environ.items() if not k.startswith("INKLING_")}
            env.update(JARVIS=j, HOME=home, RENDER_ONLY="1", TOOLS="Inkling-2-Mini")
            cfg = subprocess.run(["bash", os.path.join(REPO, "deploy", "mac", "install.sh")], env=env, check=True,
                                 capture_output=True, text=True).stdout
            serve = os.path.join(tmp, "serve-mac.json")
            with open(serve, "w") as f:
                json.dump([launch("mac", f"{j}/inkling2-mini/engine/zig-out/native/bin/tensorfold-native",
                                  f"{home}/models/hf/Inkling-2-Mini")], f)
            r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "check_args.py"), "-", serve], input=cfg,
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
