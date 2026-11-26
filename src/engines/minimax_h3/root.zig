//! The MiniMax H3 engine: text to video + audio. The DiT (`dit.zig`) with its launchers and host-side math, the 32B text
//! encoder, the two VAE decoders, the prompt tokenizer, and the whole run (`pipeline.zig`, the twin's `h3/generate.py`).
//! The gates: `localrouter check h3-replay`, `h3-te`, `h3-avae`, `h3-vvae` per part, `h3-e2e` for the chain.
//! MiniMax H3-Image's text to image shares them (`pipeline_image.zig`: the DiT's image variant, the PDD sampler
//! `pdd.zig`, the image VAE decoder `vae_image.zig`, its kernels `launch_image.zig`).

pub const launch = @import("launch.zig");
pub const layout = @import("layout.zig");
pub const profile = @import("profile.zig");
pub const dit = @import("dit.zig");
pub const sampler = @import("sampler.zig");
pub const pdd = @import("pdd.zig");
pub const launch_image = @import("launch_image.zig");
pub const te32 = @import("te32.zig");
pub const vae_audio = @import("vae_audio.zig");
pub const vae_video = @import("vae_video.zig");
pub const vae_image = @import("vae_image.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const image = @import("image.zig");
pub const vae_video_enc = @import("vae_video_enc.zig");
pub const te_vision = @import("te_vision.zig");
pub const pipeline = @import("pipeline.zig");
pub const pipeline_image = @import("pipeline_image.zig");
pub const kernels = @import("minimax_kernels");

test {
    _ = profile;
    _ = dit;
    _ = layout;
    _ = sampler;
    _ = pdd;
    _ = te32;
    _ = vae_audio;
    _ = vae_video;
    _ = vae_image;
    _ = tokenizer;
    _ = image;
    _ = vae_video_enc;
    _ = te_vision;
    _ = pipeline;
    _ = pipeline_image;
}
