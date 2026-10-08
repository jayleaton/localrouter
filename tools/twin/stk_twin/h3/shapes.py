"""The H3 twin's request defaults and latent geometry, shared by build / capture / generate / the ComfyUI reference gate
(no ComfyUI import anywhere on this path)."""

from __future__ import annotations

PROMPT = ("Cinematic handheld shot in a rainy neon-lit night market. A street cook in a white apron flips noodles in a "
          "flaming wok, steam and sparks rising, customers laughing in the background. Rain drips from red paper "
          "lanterns. Sound: sizzling wok, crackling fire, rain on tarp, distant chatter.")


def latent_shapes(width: int, height: int, frames: int) -> tuple[tuple, tuple]:
    t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
    return (1, 24, t, height // 16, width // 16), (1, 32, 2, round(frames / 24 * 40))
