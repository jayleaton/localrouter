//! Kandinsky-6 Pro distilled on LocalRouter: the engine module (`kandinsky6`). Components keep their own directories
//! (dit/, video_vae/, audio/: each with its own owner, receipts and gates); this module adds what joins them: the
//! geometry, weights from the published snapshot, the text encoders, the starting noise, the PiFlow sampler, the position
//! tables, the video VAE's GPU host and the pipeline the tool drives.

pub const shapes = @import("shapes.zig");
pub const weights = @import("weights.zig");
pub const rope = @import("rope.zig");
pub const noise = @import("noise.zig");
pub const piflow = @import("piflow.zig");
pub const central = @import("central.zig");
pub const vae_gpu = @import("vae_gpu.zig");
pub const admission = @import("admission.zig");
pub const pipeline = @import("pipeline.zig");
pub const text = struct {
    pub const prompt = @import("text/prompt.zig");
    pub const qwen25 = @import("text/qwen25.zig");
    pub const clip = @import("text/clip.zig");
};
pub const dit = @import("dit/block.zig");
pub const video_vae = struct {
    pub const launch = @import("video_vae/launch.zig");
    pub const tiles = @import("video_vae/tiles.zig");
    pub const decode = @import("video_vae/decode.zig");
    pub const encode = @import("video_vae/encode.zig");
};
pub const audio = struct {
    pub const vae = @import("audio/vae.zig");
    pub const vocoder = @import("audio/vocoder.zig");
    pub const launch = @import("audio/launch.zig");
};
pub const kernels = @import("kandinsky6_kernels");
pub const dit_kernels = @import("kandinsky6_dit_kernels");

test {
    _ = shapes;
    _ = weights;
    _ = rope;
    _ = noise;
    _ = piflow;
    _ = central;
    _ = vae_gpu;
    _ = admission;
    _ = pipeline;
    _ = text.prompt;
    _ = text.qwen25;
    _ = text.clip;
    _ = dit;
    _ = video_vae.launch;
    _ = video_vae.tiles;
    _ = video_vae.decode;
    _ = video_vae.encode;
    _ = audio.vae;
    _ = audio.vocoder;
    _ = audio.launch;
}
