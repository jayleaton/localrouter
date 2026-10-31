"""LocalRouter's launcher for KnowLine-4B's /v1/systemone server on Apple silicon (MLX on Metal).

The macOS counterpart of ../serve.py: it runs the model repository's own front end, `knowline_server.py` with its
`knowline_layout.py` (Apache-2.0), so the prompt layout, the read-twice rule, the label tokens and the readout are
exactly the authors'. Only the backend's forward pass changes, to mlx-lm's Qwen3.5 text model: one prefill of the
whole prompt, the hidden state of the last position only through the tied head, then a float32 log-softmax over the
vocabulary, the row the front end reads its labels from (as ../serve.py does on CUDA). Text only: the vision tower is
not loaded, and a request with pictures is refused (422) as by ../serve.py.

The weights are the shipped bf16 by default (the reviewed baseline, docs in ../README.md#macos); `--weights` may name
an MLX conversion instead (the 5-bit one in pins-macos.json, for comparison), while the tokenizer, chat template and
front end always come from the original repository, `front_end`. Both front-end files are checked against the hashes
reviewed for this launcher before anything is imported from that directory, so a changed file there is refused.

    serve_mlx.py FRONT_END_DIR --port N [--served-model-name ID] [--weights DIR]   (the openai_server engine's `server`)
"""

import argparse
import hashlib
import json
import os
import signal
import sys

# The front end reviewed for this launcher (the same files as ../serve.py): PelaAI/KnowLine-4B-Gen5.5 at
# 779944d2cc8ffa19cef386aee83c83bcb967ade2 (recorded in pins-macos.json too).
REVIEWED = {
    "knowline_server.py": "1b4c73419ce597cbf5635a8171b2fe981b577a9cf4856fb9387ce261e81a3729",
    "knowline_layout.py": "aa0fb35619f4883edd6c3bd1cc6dc259f41f082cfe0471e586c60d783da2af9d",
}


def verify(front_end):
    for name, want in REVIEWED.items():
        try:
            with open(os.path.join(front_end, name), "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
        except OSError as e:
            sys.exit(f"{name} is missing from {front_end}: {e.strerror}")
        if got != want:
            sys.exit(f"{name} in {front_end} is not the reviewed file (sha256 {got}); refusing to run it")


def text_model(weights):
    """mlx-lm's Qwen3.5 model for `weights`, text only (mlx-lm's own sanitizer drops the vision tower and the MTP head,
    which no decision reads). Raises when the checkpoint is not that model."""
    from pathlib import Path
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5
    from mlx_lm.utils import load_model
    with open(os.path.join(weights, "config.json")) as f:
        kind = json.load(f).get("model_type")
    if kind != "qwen3_5":
        raise ValueError(f"{weights} is a {kind!r} checkpoint, not KnowLine's qwen3_5")

    # Qwen3.5 stores its RMSNorm gains as w and applies 1 + w in float32 (transformers' Qwen3_5RMSNorm, the reference).
    # mlx-lm's sanitizer folds the 1 into the bf16 weight, a step of 1/128 near 1, which doubled the probability error
    # against a float32 run (README.md#macos). The gains of the original checkpoint are kept in float32 here, and the
    # norm's output is cast back to the activations' dtype, as in transformers. A converted (already folded) checkpoint
    # such as the 5-bit one is loaded as it is.
    sanitize = qwen3_5.TextModel.sanitize

    def precise(self, w):
        if any("conv1d.weight" in k and v.shape[-1] != 1 for k, v in w.items()):  # the original, unconverted layout
            w = {k: v.astype(mx.float32) if k.endswith(NORM_GAINS) and v.ndim == 1 else v for k, v in w.items()}
        return sanitize(self, w)

    class CastRMSNorm(nn.RMSNorm):
        def __call__(self, x):
            return mx.fast.rms_norm(x, self["weight"], self.eps).astype(x.dtype)

    qwen3_5.TextModel.sanitize = precise
    try:
        model, _ = load_model(Path(weights), lazy=False, strict=True)
    finally:
        qwen3_5.TextModel.sanitize = sanitize
    for _, m in model.named_modules():
        if type(m) is nn.RMSNorm and m.weight.dtype == mx.float32:
            m.__class__ = CastRMSNorm
    return model.language_model


# The RMSNorm gains mlx-lm's Qwen3.5 sanitizer adds 1 to (mlx_lm/models/qwen3_5.py, TextModel.sanitize).
NORM_GAINS = (".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight", ".q_norm.weight",
              ".k_norm.weight")


class MLX:
    """A knowline_server backend (warm, score) on MLX: the HF backend's contract, the last position's log-softmax."""

    def __init__(self, weights, tokenizer):
        import threading
        import mlx.core as mx
        self.mx = mx
        lm = text_model(weights)
        self.body = lm.model
        # KnowLine ties its head to the embeddings (the config's tie_word_embeddings); a conversion may keep its own.
        self.head = lm.model.embed_tokens.as_linear if lm.args.tie_word_embeddings else lm.lm_head
        self.tok = tokenizer
        self.lock = threading.Lock()
        self.resident = mx.get_active_memory()
        # One forward pass at load, so the first decision does not pay for building the Metal kernels (2.4 s).
        self.score("warm up " * 64, [], [0])

    def warm(self, prefix, media):
        pass

    def score(self, text, media, ids):
        if media:
            raise ValueError("text only: pictures are not served on macOS")
        mx = self.mx
        tokens = self.tok.encode(text, add_special_tokens=False)
        with self.lock:
            hidden = self.body(mx.array([tokens]))[:, -1:, :]
            row = self.head(hidden)[0, -1].astype(mx.float32)
            got = (row - mx.logsumexp(row))[mx.array(ids)].tolist()
            del hidden, row
            mx.clear_cache()  # the prefill's working memory goes back to the system, not into MLX's buffer cache
        return [float(x) for x in got], len(tokens)


def bind(self):
    """HTTPServer.server_bind without its socket.getfqdn(host): on macOS the reverse lookup of 127.0.0.1 goes to mDNS
    and took 35 s of the 37 s load on the M5 Ultra. The name is only the server's own label."""
    import socketserver
    socketserver.TCPServer.server_bind(self)
    self.server_name, self.server_port = self.server_address[:2]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("front_end", help="PelaAI/KnowLine-4B-Gen5.5 at the pinned revision: front end, tokenizer, template")
    p.add_argument("--weights", help="the MLX-loadable checkpoint (default: front_end, the shipped bf16)")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--served-model-name", default="knowline-4b")
    a = p.parse_args()
    front_end = os.path.abspath(a.front_end)
    weights = os.path.abspath(a.weights or a.front_end)
    verify(front_end)
    sys.dont_write_bytecode = True  # nothing written into the weights directory (its files are checked, not changed)
    sys.path.insert(0, front_end)
    # The daemon stops the server with SIGTERM: leave at once, a decision in flight included (nothing to save).
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))

    import knowline_server as ks
    from transformers import AutoTokenizer

    backend = MLX(weights, AutoTokenizer.from_pretrained(front_end))
    # knowline_server.main() builds its backend from --backend; this one replaces the HF backend's construction.
    ks.HF = lambda *_a, **_k: backend
    ks.Server.server_bind = bind
    print(f"knowline-mlx: {weights} resident {backend.resident} B", file=sys.stderr, flush=True)
    sys.argv = ["knowline_server.py", "--model", front_end, "--backend", "hf", "--host", "127.0.0.1",
                "--port", str(a.port), "--served-model-name", a.served_model_name, "--workers", "1"]
    ks.main()


if __name__ == "__main__":
    main()
