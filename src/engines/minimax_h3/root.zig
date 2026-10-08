//! The MiniMax H3 engine: text to video + audio. The DiT (`dit.zig`) with its launchers and host-side math, the 32B text
//! encoder, the two VAE decoders, the prompt tokenizer, and the whole run (`pipeline.zig`, the twin's `h3/generate.py`).
//! The gates: `localrouter check h3-replay`, `h3-te`, `h3-avae`, `h3-vvae` per part, `h3-e2e` for the chain.

pub const launch = @import("launch.zig");
pub const layout = @import("layout.zig");
pub const dit = @import("dit.zig");
pub const sampler = @import("sampler.zig");
pub const te32 = @import("te32.zig");
pub const vae_audio = @import("vae_audio.zig");
pub const vae_video = @import("vae_video.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const pipeline = @import("pipeline.zig");
pub const kernels = @import("minimax_kernels");

test {
    _ = layout;
    _ = sampler;
    _ = te32;
    _ = vae_audio;
    _ = vae_video;
    _ = tokenizer;
    _ = pipeline;
}
