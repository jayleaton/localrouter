//! Kandinsky-6 Pro-distill audio: the host side of the mel VAE decoder (MMAudio 44k, `VAE(data_dim=128, embed_dim=40,
//! hidden_dim=512)`, decoder only) and the audio ABI the vocoder (vocoder.zig) and the launches (launch.zig) share:
//! fp32 tensor views with explicit shape, element strides, base, offset, ownership, device and capture name; the operator
//! contracts with their qualification state; prepared versus stored convolution weights; the decoder graph resolved from
//! the source receipts (Kandinsky-6 `eac06e39`, MIT graph code `vae.py` / `vae_modules.py`; HF `1018d582` config);
//! its memory plan; and fp32 host references of the ready operators, written op for op like
//! kernels/cuda/kandinsky6/audio/audio.cu.
//!
//! Activation normalization and the stored -> prepared weight transform are independent compositions of public
//! primitives (vector norm, scaled-norm-plus-epsilon denominator in one rounding, true division) with an explicit
//! reduction order and an explicit epsilon (no default). On CPU they reproduce every oracle fixture with zero unequal
//! values (`NormReduce`); their target bits (SM121 for normalization, the native twin's aarch64 host for the weight
//! transform, which the native loader runs on CPU before `.to(device)`) are not yet compared. No operator in this file is
//! qualified on the target (SM121, native pinned Python twin, zero unequal values), so production is refused; see
//! `contract`.

const std = @import("std");

// ---------------------------------------------------------------------------------------------------------------- ABI

pub const Device = enum { host, cuda };
pub const Ownership = enum { owned, borrowed };
pub const DType = enum { f32 };
pub const max_rank = 5;

pub const Error = error{
    /// The operator's arithmetic contract is not independently specified and validated; it never runs.
    UnresolvedContract,
    /// The operator is implemented but not yet bit-qualified on the target against the native twin.
    GpuQualificationPending,
    ShapeMismatch,
    LayoutUnsupported,
    OutOfBounds,
    DeviceMismatch,
    /// A weight still in checkpoint state was given to an operator that consumes prepared weights.
    WeightNotPrepared,
    /// A prepared weight was offered for preparation again (the transform is not idempotent).
    AlreadyPrepared,
    /// A weight prepared by the synthetic diagnostic oracle was offered for production use.
    DiagnosticWeight,
    /// The operator exists in the source graph but the native load path for the selected checkpoint never runs it.
    NotOnNativePath,
    InvalidArgument,
};

/// An fp32 tensor: `base` is the allocation's address (device pointer or host address), `offset` and `strides` are in
/// elements, `alloc_bytes` bounds every reachable element. A view never frees a borrowed allocation.
pub const View = struct {
    name: []const u8,
    device: Device,
    dtype: DType = .f32,
    base: u64,
    alloc_bytes: u64,
    offset: u64 = 0,
    ownership: Ownership,
    rank: u8,
    shape: [max_rank]u64 = .{ 0, 0, 0, 0, 0 },
    strides: [max_rank]u64 = .{ 0, 0, 0, 0, 0 },

    /// A contiguous row-major view of `shape` at the start of an allocation.
    pub fn contiguous(name: []const u8, device: Device, base: u64, alloc_bytes: u64, ownership: Ownership, shape: []const u64) View {
        std.debug.assert(shape.len > 0 and shape.len <= max_rank);
        var v: View = .{ .name = name, .device = device, .base = base, .alloc_bytes = alloc_bytes, .ownership = ownership, .rank = @intCast(shape.len) };
        var stride: u64 = 1;
        var i = shape.len;
        while (i > 0) {
            i -= 1;
            v.shape[i] = shape[i];
            v.strides[i] = stride;
            stride *= shape[i];
        }
        return v;
    }

    pub fn numel(v: View) u64 {
        var n: u64 = 1;
        for (v.shape[0..v.rank]) |d| n *= d;
        return n;
    }

    /// Logical bytes of the elements (not of the allocation).
    pub fn bytes(v: View) u64 {
        return v.numel() * 4;
    }

    /// Device address of the first element.
    pub fn addr(v: View) u64 {
        return v.base + v.offset * 4;
    }

    pub fn isContiguous(v: View) bool {
        var stride: u64 = 1;
        var i: usize = v.rank;
        while (i > 0) {
            i -= 1;
            if (v.shape[i] != 1 and v.strides[i] != stride) return false;
            stride *= v.shape[i];
        }
        return true;
    }

    /// Every element in bounds and a nonzero shape.
    pub fn validate(v: View) Error!void {
        if (v.rank == 0 or v.rank > max_rank) return error.InvalidArgument;
        var last: u64 = v.offset;
        for (v.shape[0..v.rank], v.strides[0..v.rank]) |d, s| {
            if (d == 0) return error.ShapeMismatch;
            last += (d - 1) * s;
        }
        if ((last + 1) * 4 > v.alloc_bytes) return error.OutOfBounds;
    }

    pub fn expectShape(v: View, shape: []const u64) Error!void {
        if (v.rank != shape.len) return error.ShapeMismatch;
        for (v.shape[0..v.rank], shape) |a, b| if (a != b) return error.ShapeMismatch;
    }

    /// What a kernel taking this view needs: in bounds, contiguous, on `device`, of `shape`. No operator relayouts.
    pub fn expect(v: View, device: Device, shape: []const u64) Error!void {
        try v.validate();
        if (v.device != device) return error.DeviceMismatch;
        try v.expectShape(shape);
        if (!v.isContiguous()) return error.LayoutUnsupported;
    }
};

// ---------------------------------------------------------------------------------------------- numeric policies

/// The reduction order of one output (audio.cu K6A_ACC_*): always input channel ascending, then tap ascending; this
/// names the per-term rounding. There is no default: the target's choice is unresolved until native fixtures fix it.
pub const Acc = enum(i32) { fma = 0, mul_add = 1 };

/// How a division by a host scalar is executed (audio.cu K6A_DIV_*): a true division, or a multiplication by the fp32
/// reciprocal `1.0f / f32(divisor)` (PyTorch CUDA's handling of a CPU scalar divisor). Named per call site.
pub const Div = enum(i32) { true_div = 0, reciprocal = 1 };

pub const ScalarDiv = struct {
    div: Div,
    /// The divisor (true_div) or its fp32 reciprocal (reciprocal), the value the kernel receives.
    c: f32,

    /// From a Python-float divisor: rounded to fp32 first, then (reciprocal) 1.0f / that in fp32.
    pub fn of(divisor: f64, div: Div) ScalarDiv {
        const d: f32 = @floatCast(divisor);
        return .{ .div = div, .c = if (div == .true_div) d else 1.0 / d };
    }

    pub fn apply(s: ScalarDiv, v: f32) f32 {
        return if (s.div == .true_div) v / s.c else v * s.c;
    }
};

/// The magnitude-preserving residual's constants (formula receipt, MIT expression `a * (1. - t) + b * t` over
/// `sqrt((1 - t) ** 2 + t ** 2)`): the Python-double scalars rounded to fp32, the division explicit.
pub const MpResidual = struct {
    c0: f32,
    c1: f32,
    den: ScalarDiv,

    pub fn of(t: f64, div: Div) MpResidual {
        const u = 1.0 - t;
        return .{ .c0 = @floatCast(u), .c1 = @floatCast(t), .den = .of(@sqrt(u * u + t * t), div) };
    }

    pub fn apply(m: MpResidual, a: f32, b: f32) f32 {
        return m.den.apply(a * m.c0 + b * m.c1);
    }
};

/// The native decoder's residual and attention blend (`mp_sum(x, h, t=0.3)`).
pub const residual_t: f64 = 0.3;
/// Scaled SiLU's divisor (formula receipt: silu(x) / 0.596; do not substitute the quadrature value 0.59647).
pub const silu_divisor: f64 = 0.596;
/// Decoder clamp after the middle and after every up-level residual block.
pub const clip_act: f32 = 256.0;

// ---------------------------------------------------------------------------------------------------- contracts

pub const Op = enum {
    latent_unscale,
    mel_affine,
    clamp,
    upsample_nearest_exact2,
    scaled_silu,
    mp_residual,
    gain_weight_scale,
    conv1d_prepared,
    conv_transpose1d_prepared,
    qkv_split,
    attn_merge,
    snake_beta_log,
    aa_up2,
    aa_down2,
    vocoder_add,
    vocoder_avg3,
    /// MPConv1D-style activation normalization (residual axis 1, packed QKV axis 2): `normalizeHost`.
    activation_normalize,
    /// The decoder's stored -> prepared effective-weight transform (`prepare`), once, on the host at load.
    mp_weight_prepare,
    /// The vocoder's weight-norm parametrization fold (PyTorch `weight_norm`). The selected checkpoint stores the folded
    /// `weight` tensors and the native loader uses them as stored, so this never runs (`Weight.checkpointEffective`).
    vocoder_weight_norm_fold,
    /// The middle attention's native SDPA (fp32, one head, head dimension 2048). The native backend choice is not yet
    /// recorded; `k6a_sdpa_math` is a candidate for comparison only. Strict FLASH is not an accepted baseline for this
    /// shape. No automatic backend or precision fallback.
    mid_attention_sdpa,
};

pub const Status = enum {
    /// Implemented to a written contract; target operator / chain / mel / PCM gates still open.
    gpu_qualification_pending,
    /// An independent composition reproduces every CPU oracle fixture bit for bit; the target bits are not yet compared.
    /// Diagnostic only, like `gpu_qualification_pending`.
    cpu_bits_matched,
    /// Implemented as a written candidate with no matching evidence yet: diagnostic comparison runs only.
    candidate_unverified,
    /// The native load path for the selected checkpoint never runs it; refused for every use.
    not_on_native_path,
    /// No independently specified and validated arithmetic; rejected before allocation.
    unresolved,
};

pub const Contract = struct { status: Status, provenance: []const u8 };

pub fn contract(op: Op) Contract {
    return switch (op) {
        .latent_unscale => .{ .status = .gpu_qualification_pending, .provenance = "postprocess_audio.py L27-64 (MIT): audio / scaling_factor + mean_value, [B,T,40] -> [B,40,T]" },
        .mel_affine => .{ .status = .gpu_qualification_pending, .provenance = "vae.py L56-103 (MIT): x * data_std + data_mean, fp32 buffers [1,128,1]" },
        .clamp => .{ .status = .gpu_qualification_pending, .provenance = "vae.py L237-314 (MIT) clamp(-256, 256); bigvgan.py (MIT) clamp(-1, 1); PyTorch clamp NaN propagation" },
        .upsample_nearest_exact2 => .{ .status = .gpu_qualification_pending, .provenance = "vae_modules.py L81-93 (MIT): F.interpolate(scale_factor=2.0, mode='nearest-exact')" },
        .scaled_silu => .{ .status = .gpu_qualification_pending, .provenance = "formula receipt 1: silu(x) / 0.596 (MIT lucidrains MPSiLU expression); native mp_silu bits unverified" },
        .mp_residual => .{ .status = .gpu_qualification_pending, .provenance = "formula receipt 2: ((1-t)a + tb) / sqrt((1-t)^2 + t^2), t = 0.3 (MIT lucidrains MPAdd expression); native mp_sum bits unverified" },
        .gain_weight_scale => .{ .status = .gpu_qualification_pending, .provenance = "CPU oracle: gain multiplies the prepared weight before convolution (25/25 calls zero unequal; output-side gain differs)" },
        .conv1d_prepared => .{ .status = .gpu_qualification_pending, .provenance = "PyTorch conv1d semantics (BSD): zero padding K//2, stride 1, dilation 1, groups 1, no bias (CPU oracle boundaries); target accumulation unresolved" },
        .conv_transpose1d_prepared => .{ .status = .gpu_qualification_pending, .provenance = "PyTorch conv_transpose1d semantics (BSD); BigVGAN ups (MIT) padding (k - u) // 2" },
        .qkv_split => .{ .status = .gpu_qualification_pending, .provenance = "vae_modules.py L55-78 (MIT): reshape [B,1,2048,3,T], unbind(3), permute to [B,1,T,2048]" },
        .attn_merge => .{ .status = .gpu_qualification_pending, .provenance = "vae_modules.py L55-78 (MIT): permute(0,1,3,2).reshape(B,-1,T)" },
        .snake_beta_log => .{ .status = .gpu_qualification_pending, .provenance = "BigVGAN v2 activations.py SnakeBeta (MIT, NVIDIA 2024), alpha_logscale, no_div_by_zero 1e-9" },
        .aa_up2 => .{ .status = .gpu_qualification_pending, .provenance = "BigVGAN alias_free_activation/torch/resample.py UpSample1d (Apache-2.0, alias-free-torch); loaded filter buffer" },
        .aa_down2 => .{ .status = .gpu_qualification_pending, .provenance = "BigVGAN alias_free_activation/torch/filter.py LowPassFilter1d (Apache-2.0, alias-free-torch); loaded filter buffer" },
        .vocoder_add => .{ .status = .gpu_qualification_pending, .provenance = "BigVGAN AMPBlock1 x = xt + x (MIT)" },
        .vocoder_avg3 => .{ .status = .gpu_qualification_pending, .provenance = "BigVGAN forward xs += ...; x = xs / num_kernels (MIT)" },
        .activation_normalize => .{ .status = .cpu_bits_matched, .provenance = "x / (eps + ||x|| * sqrt(n/N)), denominator one rounding, true division, serial reduction over the strided axis: 15/15 CPU oracle cases (40462 values) zero unequal; SM121 reduction order open (k6a_l2norm orders)" },
        .mp_weight_prepare => .{ .status = .cpu_bits_matched, .provenance = "w / (eps + ||w_row|| * sqrt(1/fan)) / sqrt(fan) on the host CPU at load (autoencoder.py MIT: load_state_dict, remove_weight_norm, .to(device)); x86 AVX2 8-lane reduction: 5/5 raw->baked fixtures zero unequal; native aarch64 host bits open" },
        .vocoder_weight_norm_fold => .{ .status = .not_on_native_path, .provenance = "HF 1018d582 vocoder stores 667 fp32 tensors with plain `weight` (no g / v / parametrizations); load_bigvgan_v2 (MIT) removes weight norm before loading them as stored" },
        .mid_attention_sdpa => .{ .status = .candidate_unverified, .provenance = "bare F.scaled_dot_product_attention, fp32 head 2048, no mask, default scale; native backend selection to be recorded on SM121; k6a_sdpa_math is a comparison candidate; strict FLASH unaccepted" },
    };
}

/// Diagnostic runs (fixture plumbing, CPU comparisons) admit implemented operators; production admits only
/// target-qualified ones, which none is yet. An unresolved operator is refused for both.
pub const Use = enum { diagnostic, production };

pub fn admitOp(op: Op, use: Use) Error!void {
    switch (contract(op).status) {
        .unresolved => return error.UnresolvedContract,
        .not_on_native_path => return error.NotOnNativePath,
        .gpu_qualification_pending, .cpu_bits_matched, .candidate_unverified => if (use == .production) return error.GpuQualificationPending,
    }
}

// --------------------------------------------------------------------------------------------- weight states

pub const WeightState = enum {
    stored_checkpoint,
    prepared_native_reference,
    /// The checkpoint already holds the effective weight and the native loader uses it as stored (the vocoder: its
    /// weight-norm fold is not on the native path). Never prepared.
    checkpoint_effective,
};

pub const WeightOrigin = enum {
    /// Read from the selected checkpoint (stored) or produced from it by an accepted transform (prepared).
    checkpoint,
    /// Produced by the CPU diagnostic oracle from synthetic weights: exercises ordinary convolution only and
    /// certifies neither the asset loader nor the stored -> prepared transform.
    synthetic_oracle,
};

/// A convolution weight with its preparation state and manifest. `preparations` is 0 for stored, exactly 1 for
/// prepared; a second preparation is refused (`prepare`).
pub const Weight = struct {
    view: View,
    state: WeightState,
    origin: WeightOrigin,
    preparations: u8,
    /// The checkpoint tensor name (or the fixture file) the bytes came from.
    source: []const u8,
    source_sha256: [32]u8,
    /// sha256 of the prepared bytes (zero while stored).
    prepared_sha256: [32]u8 = @splat(0),

    pub fn stored(view: View, source: []const u8, source_sha256: [32]u8) Weight {
        return .{ .view = view, .state = .stored_checkpoint, .origin = .checkpoint, .preparations = 0, .source = source, .source_sha256 = source_sha256 };
    }

    /// A checkpoint tensor the native loader uses as stored (`WeightState.checkpoint_effective`).
    pub fn checkpointEffective(view: View, source: []const u8, source_sha256: [32]u8) Weight {
        return .{ .view = view, .state = .checkpoint_effective, .origin = .checkpoint, .preparations = 0, .source = source, .source_sha256 = source_sha256, .prepared_sha256 = source_sha256 };
    }

    /// Bytes the diagnostic oracle already prepared (synthetic fixtures only).
    pub fn oraclePrepared(view: View, source: []const u8, sha256: [32]u8) Weight {
        return .{ .view = view, .state = .prepared_native_reference, .origin = .synthetic_oracle, .preparations = 1, .source = source, .source_sha256 = sha256, .prepared_sha256 = sha256 };
    }

    /// What a ConvPrepared1D-style consumer requires.
    pub fn requirePrepared(w: Weight, use: Use) Error!void {
        switch (w.state) {
            .stored_checkpoint => return error.WeightNotPrepared,
            .prepared_native_reference => if (w.preparations != 1) return error.AlreadyPrepared,
            .checkpoint_effective => if (w.preparations != 0) return error.AlreadyPrepared,
        }
        if (use == .production and w.origin == .synthetic_oracle) return error.DiagnosticWeight;
    }
};

/// The order a sum of squares is accumulated in. Each product and each sum is rounded to f32 (no fused steps). Named by
/// the CPU behaviour it reproduces; none is a default, and none is claimed for a target it was not compared on.
pub const NormReduce = enum {
    /// s = s + x*x along the axis in index order. torch 2.14.1 CPU when the reduced axis is strided in memory (the
    /// reduction then vectorizes across the outputs): all 15 activation normalization oracle cases.
    serial,
    /// Eight lane sums, lane j over elements j, j+8, ...; the lanes then added in order 0..7. torch 2.14.1 CPU (AVX2) for
    /// a contiguous innermost reduction whose length is a multiple of 8, which every decoder weight row is (Cin * K).
    /// Other lengths are refused.
    lanes8,
    /// `lanes8`, then the length % 8 tail added serially. Reproduces the five synthetic weight fixtures (rows of 2, 6 and
    /// 15) but not every random tail row (2 of 252 differ), so it is diagnostic only.
    lanes8_tail_diagnostic,
};

fn sumSquares(x: []const f32, start: usize, n: usize, stride: usize, reduce: NormReduce) Error!f32 {
    switch (reduce) {
        .serial => {
            var s: f32 = 0;
            for (0..n) |i| {
                const v = x[start + i * stride];
                s += v * v;
            }
            return s;
        },
        .lanes8, .lanes8_tail_diagnostic => {
            if (stride != 1) return error.LayoutUnsupported;
            const full = n - n % 8;
            if (full != n and reduce == .lanes8) return error.UnresolvedContract;
            var lanes: [8]f32 = @splat(0);
            var i: usize = 0;
            while (i < full) : (i += 8) for (&lanes, 0..) |*l, j| {
                const v = x[start + i + j];
                l.* += v * v;
            };
            var s: f32 = 0;
            for (lanes) |l| s += l;
            for (full..n) |k| {
                const v = x[start + k];
                s += v * v;
            }
            return s;
        },
    }
}

/// The normalization denominator eps + norm * scale: `.fma` is one rounding (what the CPU oracle shows), `.mul_add`
/// two (the discriminator).
fn normDen(norm: f32, scale: f32, eps: f32, den: Acc) f32 {
    return switch (den) {
        .fma => @mulAdd(f32, norm, scale, eps),
        .mul_add => norm * scale + eps,
    };
}

pub const PrepareParams = struct {
    /// Added to the scaled row norm. No default: the oracle observed 1e-4 (its default equals an explicit 1e-4).
    eps: f32,
    reduce: NormReduce,
    den: Acc,
};

/// The decoder's stored -> prepared transform (`mp_weight_prepare`), run once on the host as the native loader does
/// (load_state_dict, remove_weight_norm on CPU, then `.to(device)`): for each output row r of a [Cout, Cin, K] weight,
/// w' = w / (eps + ||w_r|| * f32(sqrt(1/fan))) / f32(sqrt(fan)), fan = Cin * K, true divisions. `out` receives the
/// prepared bytes and `out_view` describes them. A weight that is not stored (prepared, or used as stored) is refused
/// first, since a second preparation changes the data; `vocoder_weight_norm_fold` is refused as off the native path.
pub fn prepare(w: Weight, op: Op, stored_data: []const f32, out_view: View, out: []f32, p: PrepareParams, use: Use) Error!Weight {
    if (op != .mp_weight_prepare and op != .vocoder_weight_norm_fold) return error.InvalidArgument;
    if (w.state != .stored_checkpoint or w.preparations != 0) return error.AlreadyPrepared;
    try admitOp(op, use);
    if (use == .production and p.reduce == .lanes8_tail_diagnostic) return error.DiagnosticWeight;
    for ([_]View{ w.view, out_view }) |v| {
        try v.validate();
        if (v.device != .host) return error.DeviceMismatch;
        if (!v.isContiguous()) return error.LayoutUnsupported;
    }
    if (w.view.rank != 3) return error.ShapeMismatch;
    try out_view.expectShape(w.view.shape[0..3]);
    const rows: usize = @intCast(w.view.shape[0]);
    const fan: usize = @intCast(w.view.shape[1] * w.view.shape[2]);
    if (stored_data.len != rows * fan or out.len != rows * fan) return error.ShapeMismatch;
    const fan_f: f64 = @floatFromInt(fan);
    const scale: f32 = @floatCast(@sqrt(1.0 / fan_f));
    const root: f32 = @floatCast(@sqrt(fan_f));
    for (0..rows) |r| {
        const den = normDen(@sqrt(try sumSquares(stored_data, r * fan, fan, 1, p.reduce)), scale, p.eps, p.den);
        for (r * fan..(r + 1) * fan) |e| out[e] = stored_data[e] / den / root;
    }
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(out), &sha, .{});
    return .{ .view = out_view, .state = .prepared_native_reference, .origin = w.origin, .preparations = 1, .source = w.source, .source_sha256 = w.source_sha256, .prepared_sha256 = sha };
}

/// One normalization as outer x n x inner, n reduced, with `scale` = f32(sqrt(1/n)).
pub const NormGeometry = struct { outer: u64, n: u64, inner: u64, scale: f32 };

/// Activation normalization ABI: the residual tensor [B, C, T] reduces axis 1; the packed attention tensor
/// [B, H, D, 3, T] reduces axis 2 (Q, K and V all normalized). Validates the views and the axis, admits the operator for
/// diagnostic use (target bits are not compared) and returns the geometry. The epsilon, the reduction order and the
/// denominator rounding are the caller's explicit arguments (`normalizeHost`, launch.zig); none has a default.
pub fn normalizeActivation(input: View, output: View, axis: u8) Error!NormGeometry {
    try input.validate();
    try output.validate();
    try output.expectShape(input.shape[0..input.rank]);
    const ok = (input.rank == 3 and axis == 1) or (input.rank == 5 and axis == 2 and input.shape[3] == 3);
    if (!ok) return error.InvalidArgument;
    if (!input.isContiguous() or !output.isContiguous()) return error.LayoutUnsupported;
    try admitOp(.activation_normalize, .diagnostic);
    var g: NormGeometry = .{ .outer = 1, .n = input.shape[axis], .inner = 1, .scale = undefined };
    for (input.shape[0..axis]) |d| g.outer *= d;
    for (input.shape[axis + 1 .. input.rank]) |d| g.inner *= d;
    g.scale = @floatCast(@sqrt(1.0 / @as(f64, @floatFromInt(g.n))));
    return g;
}

/// y = x / (eps + ||x||_axis * scale) over a contiguous outer x n x inner tensor, true division.
pub fn normalizeHost(x: []const f32, y: []f32, g: NormGeometry, eps: f32, reduce: NormReduce, den: Acc) Error!void {
    const n: usize = @intCast(g.n);
    const inner: usize = @intCast(g.inner);
    const outer: usize = @intCast(g.outer);
    if (x.len != outer * n * inner or y.len != x.len) return error.ShapeMismatch;
    for (0..outer) |o| for (0..inner) |i| {
        const start = o * n * inner + i;
        const d = normDen(@sqrt(try sumSquares(x, start, n, inner, reduce)), g.scale, eps, den);
        for (0..n) |k| y[start + k * inner] = x[start + k * inner] / d;
    };
}

// ------------------------------------------------------------------------------------------------- the graph

pub const latent_channels = 40;
pub const mel_channels = 128;
pub const hidden_dim = 512;
pub const ch_mult = [3]u32{ 1, 2, 4 };
pub const sample_rate = 44100;
/// Waveform samples per latent frame (FeaturesUtils.downsample_factor; = 2 mel frames x the vocoder's 512 hop).
pub const downsample_factor = 1024;
/// The selected Pro-distill audio_vae/config.json scaling_factor (standard Pro's 0.5302 is out of scope).
pub const scaling_factor: f64 = 0.417;
/// postprocess_audio adds `mean_value` if the VAE has one, else 0.0 (the selected config has none).
pub const mean_value: f32 = 0.0;
/// Native 5 s distill audio latent length.
pub const native_latent_frames = 218;

pub const Gain = enum {
    /// forward(x, gain=1): the weight times 1.0f is bitwise the weight, so no scaled copy is made.
    one,
    /// conv_out: learnable_gain + 1, the loaded scalar (initialized 0, checkpoint value authoritative).
    loaded_plus_one,
};

pub const Conv = struct { path: []const u8, cin: u32, cout: u32, k: u32, gain: Gain = .one };

pub const Node = union(enum) {
    conv: Conv,
    /// ResnetBlock1D: x = normalize(x, dim=1); h = conv2(silu(conv1(silu(x)))); x = nin(x) if Cin != Cout;
    /// mp_sum(x, h, 0.3).
    resblock: struct { path: []const u8, cin: u32, cout: u32 },
    /// AttnBlock1D: qkv K1 conv, reshape, normalize dim 2, SDPA, merge, proj K1 conv, mp_sum(x, h, 0.3).
    attention: struct { path: []const u8, c: u32, heads: u32 },
    clamp,
    /// Upsample1D with_conv: nearest-exact x2, then the K3 conv `conv`.
    upsample: Conv,
    silu,
    unnormalize,
};

fn up(comptime level: u32, comptime block: u32, comptime cin: u32, comptime cout: u32) [2]Node {
    return .{ .{ .resblock = .{ .path = std.fmt.comptimePrint("decoder.up.{d}.block.{d}", .{ level, block }), .cin = cin, .cout = cout } }, .clamp };
}

/// Decoder1D(dim=512, ch_mult=(1,2,4), num_res_blocks=2, attn_layers=[3] (none at levels 0..2), down_layers=[0] (an
/// upsample at level 1)) as its forward runs, then VAE.unnormalize. Eleven residual blocks, one attention.
pub const decoder_graph = [_]Node{
    .{ .conv = .{ .path = "decoder.conv_in", .cin = latent_channels, .cout = 2048, .k = 3 } },
    .{ .resblock = .{ .path = "decoder.mid.block_1", .cin = 2048, .cout = 2048 } },
    .{ .attention = .{ .path = "decoder.mid.attn_1", .c = 2048, .heads = 1 } },
    .{ .resblock = .{ .path = "decoder.mid.block_2", .cin = 2048, .cout = 2048 } },
    .clamp,
} ++ up(2, 0, 2048, 2048) ++ up(2, 1, 2048, 2048) ++ up(2, 2, 2048, 2048) //
++ up(1, 0, 2048, 1024) ++ up(1, 1, 1024, 1024) ++ up(1, 2, 1024, 1024) //
++ [_]Node{.{ .upsample = .{ .path = "decoder.up.1.upsample.conv", .cin = 1024, .cout = 1024, .k = 3 } }} //
++ up(0, 0, 1024, 512) ++ up(0, 1, 512, 512) ++ up(0, 2, 512, 512) ++ [_]Node{
    .silu,
    .{ .conv = .{ .path = "decoder.conv_out", .cin = 512, .cout = mel_channels, .k = 3, .gain = .loaded_plus_one } },
    .unnormalize,
};

/// The operators one node runs, in order.
pub fn nodeOps(n: Node, buf: *[8]Op) []const Op {
    const list: []const Op = switch (n) {
        .conv => |c| if (c.gain == .one) &.{.conv1d_prepared} else &.{ .gain_weight_scale, .conv1d_prepared },
        .resblock => |r| if (r.cin == r.cout)
            &.{ .activation_normalize, .scaled_silu, .conv1d_prepared, .scaled_silu, .conv1d_prepared, .mp_residual }
        else
            &.{ .activation_normalize, .scaled_silu, .conv1d_prepared, .scaled_silu, .conv1d_prepared, .conv1d_prepared, .mp_residual },
        .attention => &.{ .conv1d_prepared, .activation_normalize, .qkv_split, .mid_attention_sdpa, .attn_merge, .conv1d_prepared, .mp_residual },
        .clamp => &.{.clamp},
        .upsample => &.{ .upsample_nearest_exact2, .conv1d_prepared },
        .silu => &.{.scaled_silu},
        .unnormalize => &.{.mel_affine},
    };
    @memcpy(buf[0..list.len], list);
    return buf[0..list.len];
}

/// A decoder's weights, in `decoder_graph` order: every MPConv1D weight [Cout, Cin, K] (conv1, conv2, nin for a block;
/// qkv, proj_out for attention), plus the conv_out gain scalar and data_mean / data_std [128].
pub const ParamCounts = struct { conv_weights: u64, convs: u32, scalars: u64 };

pub fn decoderParams() ParamCounts {
    var p: ParamCounts = .{ .conv_weights = 0, .convs = 0, .scalars = 1 + 2 * mel_channels };
    for (decoder_graph) |n| switch (n) {
        .conv, .upsample => |c| {
            p.conv_weights += @as(u64, c.cin) * c.cout * c.k;
            p.convs += 1;
        },
        .resblock => |r| {
            p.conv_weights += @as(u64, r.cin) * r.cout * 3 + @as(u64, r.cout) * r.cout * 3;
            p.convs += 2;
            if (r.cin != r.cout) {
                p.conv_weights += @as(u64, r.cin) * r.cout;
                p.convs += 1;
            }
        },
        .attention => |a| {
            p.conv_weights += @as(u64, a.c) * a.c * 3 + @as(u64, a.c) * a.c;
            p.convs += 2;
        },
        else => {},
    };
    return p;
}

/// Bytes the decoder needs, from shapes alone (no I/O). Resident: prepared fp32 weights, conv_out's gain-scaled copy,
/// the scalars. Working (per request): six [B, Cmax*T] activation slots (block input, normalized input, two branch
/// buffers, shortcut, output) at the largest C*T of the graph, plus the middle attention's qkv [B,6144,T], q/k/v
/// [B,T,2048] each, the output [B,T,2048] and one [B,T,T] score matrix.
pub const Memory = struct { resident_bytes: u64, working_bytes: u64 };

pub const Decoder = struct {
    batch: u32,
    /// Latent frames T (the decoder's input [B, 40, T]); the mel has 2T frames.
    frames: u32,

    pub fn init(batch: u32, frames: u32) Error!Decoder {
        if (batch == 0 or frames == 0) return error.InvalidArgument;
        return .{ .batch = batch, .frames = frames };
    }

    pub fn inputShape(d: Decoder) [3]u64 {
        return .{ d.batch, latent_channels, d.frames };
    }

    /// The sampler's per-sample audio latent rows, [B, T, 40], before `latent_unscale`.
    pub fn latentRowsShape(d: Decoder) [3]u64 {
        return .{ d.batch, d.frames, latent_channels };
    }

    pub fn melShape(d: Decoder) [3]u64 {
        return .{ d.batch, mel_channels, 2 * @as(u64, d.frames) };
    }

    pub fn samples(d: Decoder) u64 {
        return @as(u64, d.frames) * downsample_factor;
    }

    pub fn memory(d: Decoder) Memory {
        const p = decoderParams();
        const conv_out_gain_copy: u64 = 512 * mel_channels * 3;
        const t: u64 = d.frames;
        var max_ct: u64 = 0;
        var len = t;
        for (decoder_graph) |n| switch (n) {
            .conv => |c| max_ct = @max(max_ct, @max(c.cin, c.cout) * len),
            .resblock => |r| max_ct = @max(max_ct, @max(r.cin, r.cout) * len),
            .attention => |a| max_ct = @max(max_ct, a.c * len),
            .upsample => |c| {
                len *= 2;
                max_ct = @max(max_ct, c.cout * len);
            },
            else => {},
        };
        const attn = 6144 * t + 3 * 2048 * t + 2048 * t + t * t;
        return .{
            .resident_bytes = (p.conv_weights + conv_out_gain_copy + p.scalars) * 4,
            .working_bytes = (6 * max_ct + attn) * @as(u64, d.batch) * 4,
        };
    }

    /// Whether the whole decoder may run under `use`, with weights in `state`. Checks every operator before anything is
    /// allocated (stored weights also need `mp_weight_prepare`). Diagnostic runs are admitted for comparison against the
    /// native twin; production is refused until every operator is target-qualified. It never runs a substitute.
    pub fn admit(d: Decoder, use: Use, state: WeightState) Error!void {
        _ = d;
        // refused operators first, so the refusal names the real blocker rather than a pending qualification
        if (state == .checkpoint_effective) return error.InvalidArgument; // decoder weights are always prepared at load
        if (state == .stored_checkpoint) try admitOp(.mp_weight_prepare, .diagnostic);
        var buf: [8]Op = undefined;
        for (decoder_graph) |n| for (nodeOps(n, &buf)) |op| try admitOp(op, .diagnostic);
        if (use == .production) for (decoder_graph) |n| for (nodeOps(n, &buf)) |op| try admitOp(op, .production);
    }

    /// Every operator the decoder needs whose contract is not yet at `gpu_qualification_pending` (CPU-matched,
    /// candidate or unresolved), with the load-time weight preparation last: the honest list for a status report.
    pub fn openContracts(out: *[8]Op) []const Op {
        var n: usize = 0;
        var buf: [8]Op = undefined;
        for (decoder_graph) |node| for (nodeOps(node, &buf)) |op| {
            if (contract(op).status == .gpu_qualification_pending) continue;
            if (std.mem.indexOfScalar(Op, out[0..n], op) == null) {
                out[n] = op;
                n += 1;
            }
        };
        out[n] = .mp_weight_prepare;
        return out[0 .. n + 1];
    }
};

// -------------------------------------------------------------------------------------- host references (fp32)
//
// Op for op like audio.cu, for fixture plumbing and as the CPU mirror the device kernels are compared with. Zig never
// contracts f32 arithmetic into fma unless `@mulAdd` is written, so each expression below is one rounding. The two that
// call transcendental functions (scaledSilu, and the vocoder's snake) use Zig's libm, not CUDA's: their host results
// are structural references only.

pub fn latentUnscale(rows: []const f32, z: []f32, b: usize, t: usize, c: usize, scale: ScalarDiv, mean: f32) void {
    for (0..b) |bi| for (0..c) |ci| for (0..t) |ti| {
        z[(bi * c + ci) * t + ti] = scale.apply(rows[(bi * t + ti) * c + ci]) + mean;
    };
}

pub fn melAffine(x: []const f32, stdv: []const f32, mean: []const f32, y: []f32, b: usize, c: usize, t: usize) void {
    for (0..b * c * t) |e| {
        const ci = (e / t) % c;
        y[e] = x[e] * stdv[ci] + mean[ci];
    }
}

pub fn clamp(x: []const f32, y: []f32, lo: f32, hi: f32) void {
    for (x, y) |v, *o| o.* = if (std.math.isNan(v)) v else if (v < lo) lo else if (v > hi) hi else v;
}

pub fn upsampleNearestExact2(x: []const f32, y: []f32, rows: usize, t: usize) void {
    for (0..rows) |r| for (0..2 * t) |o| {
        y[r * 2 * t + o] = x[r * t + (o >> 1)];
    };
}

pub fn scaledSilu(x: []const f32, y: []f32, c: ScalarDiv) void {
    for (x, y) |v, *o| o.* = c.apply(v / (1.0 + @exp(-v)));
}

pub fn mpResidual(a: []const f32, b: []const f32, y: []f32, m: MpResidual) void {
    for (a, b, y) |av, bv, *o| o.* = m.apply(av, bv);
}

pub fn scaleWeights(w: []const f32, out: []f32, gain: f32) void {
    for (w, out) |v, *o| o.* = v * gain;
}

inline fn accum(acc: f32, w: f32, x: f32, mode: Acc) f32 {
    return if (mode == .fma) @mulAdd(f32, w, x, acc) else acc + w * x;
}

pub const ConvDims = struct { b: usize, cin: usize, cout: usize, lin: usize, k: usize, stride: usize = 1, pad: usize, dil: usize = 1 };

pub fn convOutLen(d: ConvDims) usize {
    return (d.lin + 2 * d.pad - d.dil * (d.k - 1) - 1) / d.stride + 1;
}

/// audio.cu k6a_conv1d: w [Cout, Cin, K] prepared, zero padding, chain c ascending then k ascending from +0.0, padded
/// taps contribute w * 0.0, bias added after the chain.
pub fn conv1d(x: []const f32, w: []const f32, bias: ?[]const f32, y: []f32, d: ConvDims, mode: Acc) void {
    const lout = convOutLen(d);
    for (0..d.b) |b| for (0..d.cout) |o| for (0..lout) |t| {
        var acc: f32 = 0.0;
        for (0..d.cin) |c| for (0..d.k) |k| {
            const ti = @as(isize, @intCast(t * d.stride + k * d.dil)) - @as(isize, @intCast(d.pad));
            const xv: f32 = if (ti >= 0 and ti < d.lin) x[(b * d.cin + c) * d.lin + @as(usize, @intCast(ti))] else 0.0;
            acc = accum(acc, w[(o * d.cin + c) * d.k + k], xv, mode);
        };
        y[(b * d.cout + o) * lout + t] = if (bias) |bs| acc + bs[o] else acc;
    };
}

pub fn convTransposeOutLen(lin: usize, k: usize, stride: usize, pad: usize) usize {
    return (lin - 1) * stride + k - 2 * pad;
}

/// audio.cu k6a_conv_transpose1d: w [Cin, Cout, K] prepared, chain c ascending then k ascending over the taps that
/// exist, bias after the chain.
pub fn convTranspose1d(x: []const f32, w: []const f32, bias: ?[]const f32, y: []f32, b_n: usize, cin: usize, cout: usize, lin: usize, k_n: usize, stride: usize, pad: usize, mode: Acc) void {
    const lout = convTransposeOutLen(lin, k_n, stride, pad);
    for (0..b_n) |b| for (0..cout) |o| for (0..lout) |t| {
        var acc: f32 = 0.0;
        for (0..cin) |c| for (0..k_n) |k| {
            const q = @as(isize, @intCast(t + pad)) - @as(isize, @intCast(k));
            if (q < 0 or @mod(q, @as(isize, @intCast(stride))) != 0) continue;
            const i: usize = @intCast(@divExact(q, @as(isize, @intCast(stride))));
            if (i >= lin) continue;
            acc = accum(acc, w[(c * cout + o) * k_n + k], x[(b * cin + c) * lin + i], mode);
        };
        y[(b * cout + o) * lout + t] = if (bias) |bs| acc + bs[o] else acc;
    };
}

/// audio.cu k6a_qkv_split: [B, H, D, 3, T] -> q, k, v [B, H, T, D].
pub fn qkvSplit(qkv: []const f32, q: []f32, k: []f32, v: []f32, bh: usize, d: usize, t: usize) void {
    for (0..bh) |h| for (0..t) |ti| for (0..d) |di| {
        const dst = (h * t + ti) * d + di;
        const src = ((h * d + di) * 3) * t + ti;
        q[dst] = qkv[src];
        k[dst] = qkv[src + t];
        v[dst] = qkv[src + 2 * t];
    };
}

/// audio.cu k6a_attn_merge: [B, H, T, D] -> [B, H*D, T].
pub fn attnMerge(h_in: []const f32, y: []f32, bh: usize, d: usize, t: usize) void {
    for (0..bh) |h| for (0..d) |di| for (0..t) |ti| {
        y[(h * d + di) * t + ti] = h_in[(h * t + ti) * d + di];
    };
}

// ------------------------------------------------------------------------------------------------------ tests

const testing = std.testing;

fn bits(v: f32) u32 {
    return @bitCast(v);
}

fn unequal(a: []const f32, b: []const f32) usize {
    var n: usize = 0;
    for (a, b) |x, y| n += @intFromBool(bits(x) != bits(y));
    return n;
}

fn f32le(comptime file: []const u8) [@embedFile("fixtures/" ++ file).len / 4]f32 {
    const raw = @embedFile("fixtures/" ++ file);
    var out: [raw.len / 4]f32 = undefined;
    for (&out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
    return out;
}

fn expectSha(comptime file: []const u8, comptime hex: []const u8) !void {
    var got: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(@embedFile("fixtures/" ++ file), &got, .{});
    var want: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&want, hex);
    try testing.expectEqualSlices(u8, &want, &got);
}

const GainCase = struct { gain: f32, file: []const u8, sha256: []const u8 };
const ConvCase = struct { name: []const u8, cin: usize, cout: usize, k: usize, weight: []const u8, weight_sha256: []const u8, input: []const u8, input_sha256: []const u8, gains: [5]GainCase };
const BoundaryCase = struct { impulse: usize, input: []const u8, input_sha256: []const u8, output: []const u8, output_sha256: []const u8 };

// Generated from cpu-oracle-observations.json (fixtures/manifest.json): B = 1, L = 7, padding K // 2.
const conv_cases = [_]ConvCase{
    .{ .name = "2-1-1-ordinary", .cin = 2, .cout = 1, .k = 1, .weight = "conv-2-1-1-ordinary-baked.f32le", .weight_sha256 = "ab7b8e77f6105a5f3977d09f2222163649f523c05faf7939972bef6c39da1e04", .input = "conv-2-1-1-ordinary-input.f32le", .input_sha256 = "70669dd298afbbfaa9448cebdb780f19e2f8bbc8f68c70d914cd560efb9ea4b4", .gains = .{ .{ .gain = 0.0, .file = "conv-2-1-1-ordinary-gain-0.0.f32le", .sha256 = "3addfb141cd7c9c4c6543a82191a3707ac29c7a041217782e61d4d91c691aee8" }, .{ .gain = 1.0, .file = "conv-2-1-1-ordinary-gain-1.0.f32le", .sha256 = "7c26818bfe53d06f5817ce6d514bc098c270cabe5e70b07954e8c61051cfb0cc" }, .{ .gain = 0.25, .file = "conv-2-1-1-ordinary-gain-0.25.f32le", .sha256 = "aa421773f7534c5193ae45340f40d0da399a5be6330734fdc65c5d14a7281605" }, .{ .gain = 1.3, .file = "conv-2-1-1-ordinary-gain-1.3.f32le", .sha256 = "a4d9300fff5b48557d2ed8a48546de12a4bd653a8b67bca5f50c5eeaf33e0830" }, .{ .gain = -1.0, .file = "conv-2-1-1-ordinary-gain--1.0.f32le", .sha256 = "64233caf6f77c482677aed6b2a08b757741e1d4761b96def9667cb09b1c83f96" } } },
    .{ .name = "2-2-3-ordinary", .cin = 2, .cout = 2, .k = 3, .weight = "conv-2-2-3-ordinary-baked.f32le", .weight_sha256 = "68965513ae2fb1be79d27c5f9b4356411bdf504857317dfea111e8361fb417a4", .input = "conv-2-2-3-ordinary-input.f32le", .input_sha256 = "70669dd298afbbfaa9448cebdb780f19e2f8bbc8f68c70d914cd560efb9ea4b4", .gains = .{ .{ .gain = 0.0, .file = "conv-2-2-3-ordinary-gain-0.0.f32le", .sha256 = "d4817aa5497628e7c77e6b606107042bbba3130888c5f47a375e6179be789fbb" }, .{ .gain = 1.0, .file = "conv-2-2-3-ordinary-gain-1.0.f32le", .sha256 = "711c11e4d3b9e5070401e5544f4cca0b1984680cd943af7d16dd779c21ed80c2" }, .{ .gain = 0.25, .file = "conv-2-2-3-ordinary-gain-0.25.f32le", .sha256 = "5b24bf75c82ba0a0bc8b2ec5454a9e36fb8909cec1434dac92bc9a1b1b8d6f11" }, .{ .gain = 1.3, .file = "conv-2-2-3-ordinary-gain-1.3.f32le", .sha256 = "00d44d6a0a83086cc2332e0234f6cdf2cc7f57d80759b5e6853a61b353005900" }, .{ .gain = -1.0, .file = "conv-2-2-3-ordinary-gain--1.0.f32le", .sha256 = "38515ba2c753d81c435e5ec41dad724b7d3817ad3714b6aaa172692ef1b0a85d" } } },
    .{ .name = "3-2-5-ordinary", .cin = 3, .cout = 2, .k = 5, .weight = "conv-3-2-5-ordinary-baked.f32le", .weight_sha256 = "7fcc3d6d48a197010c76715c80df5b96a6ff11ffea8f859bd58d454bddd2acfd", .input = "conv-3-2-5-ordinary-input.f32le", .input_sha256 = "5d6b92e5c56866dc50d2632a9d4156d1a0bb2170b6eb7ef8efa6c70651b22df1", .gains = .{ .{ .gain = 0.0, .file = "conv-3-2-5-ordinary-gain-0.0.f32le", .sha256 = "d4817aa5497628e7c77e6b606107042bbba3130888c5f47a375e6179be789fbb" }, .{ .gain = 1.0, .file = "conv-3-2-5-ordinary-gain-1.0.f32le", .sha256 = "46d6cd11c6b34ae6eed9ca5475f250cfa60d03d7a24a42fff8b68c81b23256c9" }, .{ .gain = 0.25, .file = "conv-3-2-5-ordinary-gain-0.25.f32le", .sha256 = "f960793d05b6c2c0540f8986946f218a748c2d6642f212ff01a3ee7320a68d69" }, .{ .gain = 1.3, .file = "conv-3-2-5-ordinary-gain-1.3.f32le", .sha256 = "24128f07bc0c602dd2af10fc10ccf3ff06ce5e6350e634df37f67804702dc098" }, .{ .gain = -1.0, .file = "conv-3-2-5-ordinary-gain--1.0.f32le", .sha256 = "162ad55658bda37c0d69f91aafcd4187a085b6cf0ffb0722ab1b5ba08c9ca3f7" } } },
    .{ .name = "2-1-3-tiny", .cin = 2, .cout = 1, .k = 3, .weight = "conv-2-1-3-tiny-baked.f32le", .weight_sha256 = "d7f7814f8712363911bc99a156094cf0988a5b2b1798c40e002f09bce4880547", .input = "conv-2-1-3-tiny-input.f32le", .input_sha256 = "70669dd298afbbfaa9448cebdb780f19e2f8bbc8f68c70d914cd560efb9ea4b4", .gains = .{ .{ .gain = 0.0, .file = "conv-2-1-3-tiny-gain-0.0.f32le", .sha256 = "3addfb141cd7c9c4c6543a82191a3707ac29c7a041217782e61d4d91c691aee8" }, .{ .gain = 1.0, .file = "conv-2-1-3-tiny-gain-1.0.f32le", .sha256 = "b4968a50192310723c22d277b37409ca3189c19b121d9d102c53bea545dd2d54" }, .{ .gain = 0.25, .file = "conv-2-1-3-tiny-gain-0.25.f32le", .sha256 = "7869f5b8f0c5577d371cdb974ec384bca6c7b78b8623f62274e03ec3c65e34b5" }, .{ .gain = 1.3, .file = "conv-2-1-3-tiny-gain-1.3.f32le", .sha256 = "02040608281dfb0f836a92b519ba8984d0db436641d0450a993e3fd539230ad5" }, .{ .gain = -1.0, .file = "conv-2-1-3-tiny-gain--1.0.f32le", .sha256 = "b5bc153b4fdb2d158b4f770943075f7579ba363b81208573a8080069238b58a4" } } },
    .{ .name = "2-1-3-zero", .cin = 2, .cout = 1, .k = 3, .weight = "conv-2-1-3-zero-baked.f32le", .weight_sha256 = "4bfc74e95ac3e7eff044ba21fa5e0dd106a880b56c696184f029bb1cc7acdac5", .input = "conv-2-1-3-zero-input.f32le", .input_sha256 = "70669dd298afbbfaa9448cebdb780f19e2f8bbc8f68c70d914cd560efb9ea4b4", .gains = .{ .{ .gain = 0.0, .file = "conv-2-1-3-zero-gain-0.0.f32le", .sha256 = "3addfb141cd7c9c4c6543a82191a3707ac29c7a041217782e61d4d91c691aee8" }, .{ .gain = 1.0, .file = "conv-2-1-3-zero-gain-1.0.f32le", .sha256 = "3addfb141cd7c9c4c6543a82191a3707ac29c7a041217782e61d4d91c691aee8" }, .{ .gain = 0.25, .file = "conv-2-1-3-zero-gain-0.25.f32le", .sha256 = "3addfb141cd7c9c4c6543a82191a3707ac29c7a041217782e61d4d91c691aee8" }, .{ .gain = 1.3, .file = "conv-2-1-3-zero-gain-1.3.f32le", .sha256 = "3addfb141cd7c9c4c6543a82191a3707ac29c7a041217782e61d4d91c691aee8" }, .{ .gain = -1.0, .file = "conv-2-1-3-zero-gain--1.0.f32le", .sha256 = "3addfb141cd7c9c4c6543a82191a3707ac29c7a041217782e61d4d91c691aee8" } } },
};
const boundary_weight = "boundary-prepared-weight.f32le";
const boundary_weight_sha256 = "ecdb3f24a6cc52a0f19c761d397267c28156c35f9198b204ae361667534e4a50";
const boundary_cases = [_]BoundaryCase{
    .{ .impulse = 0, .input = "boundary-0-input.f32le", .input_sha256 = "5d9fd5e0959822cd94105e518fc2e03b75685231d689fddc26dcedda8d8be06a", .output = "boundary-0-output.f32le", .output_sha256 = "9131102058c7a88bd53258e62f772424bf55aa007591111e909d95eee8c9bdb0" },
    .{ .impulse = 1, .input = "boundary-1-input.f32le", .input_sha256 = "0c028287c274bf97517c3e10900f4a1d774ec08d7aa6350eb5d76b45b69811d5", .output = "boundary-1-output.f32le", .output_sha256 = "b41d680cbec3b952ad4c59f6da1c069c25179510a1fa3e2467e312a547d2d486" },
    .{ .impulse = 3, .input = "boundary-3-input.f32le", .input_sha256 = "b85992acf28be2e336558ae9a321bb850494f94d83e006924dec0b8302934ab6", .output = "boundary-3-output.f32le", .output_sha256 = "cfb527de9eca16a3e98bc23bfe77c9f391bd127fac38e9c2c450b5751e7115a6" },
    .{ .impulse = 6, .input = "boundary-6-input.f32le", .input_sha256 = "5b9b333da9aef69a1da1e68744424578726c887215c97076e3529c2a6bc0b019", .output = "boundary-6-output.f32le", .output_sha256 = "addd8c135deaf3ba1cf646c9532158dd64720ae79db7f2bd10dab73b9c65d94c" },
};

test "decoder graph resolves the receipt shapes" {
    var res: usize = 0;
    var attn: usize = 0;
    var clamps: usize = 0;
    for (decoder_graph) |n| switch (n) {
        .resblock => res += 1,
        .attention => attn += 1,
        .clamp => clamps += 1,
        else => {},
    };
    try testing.expectEqual(11, res);
    try testing.expectEqual(1, attn);
    try testing.expectEqual(10, clamps); // after the middle and after each of the 9 up-level blocks
    const d = try Decoder.init(1, native_latent_frames);
    try testing.expectEqualSlices(u64, &.{ 1, 40, 218 }, &d.inputShape());
    try testing.expectEqualSlices(u64, &.{ 1, 128, 436 }, &d.melShape());
    try testing.expectEqual(223232, d.samples()); // 436 mel frames x the vocoder's 512 hop
    const p = decoderParams();
    try testing.expectEqual(29, p.convs); // conv_in, 22 block convs, 2 nin shortcuts, qkv + proj, upsample, conv_out
    // conv_in 40*2048*3; 5 blocks 2048->2048 (2 x 2048^2*3); attention 4 x 2048^2; up.1.block.0 2048*1024*3 +
    // 1024^2*3 + 2048*1024; 2 blocks 1024 (2 x 1024^2*3); upsample 1024^2*3; up.0.block.0 1024*512*3 + 512^2*3 +
    // 1024*512; 2 blocks 512 (2 x 512^2*3); conv_out 512*128*3
    const want: u64 = 40 * 2048 * 3 + 5 * 2 * 2048 * 2048 * 3 + 4 * 2048 * 2048 + (2048 * 1024 * 3 + 1024 * 1024 * 3 + 2048 * 1024) +
        2 * 2 * 1024 * 1024 * 3 + 1024 * 1024 * 3 + (1024 * 512 * 3 + 512 * 512 * 3 + 1024 * 512) + 2 * 2 * 512 * 512 * 3 + 512 * 128 * 3;
    try testing.expectEqual(want, p.conv_weights);
    const m = d.memory();
    try testing.expectEqual((want + 512 * 128 * 3 + 257) * 4, m.resident_bytes);
    // largest C*T: 2048 x 218 = 446464 (= 1024 x 436 after the upsample)
    try testing.expectEqual((6 * 446464 + (6144 + 6144 + 2048) * 218 + 218 * 218) * 4, m.working_bytes);
}

test "the full decoder: diagnostic admitted for comparison, production refused before allocation" {
    const d = try Decoder.init(1, native_latent_frames);
    try d.admit(.diagnostic, .prepared_native_reference);
    try d.admit(.diagnostic, .stored_checkpoint);
    try testing.expectError(error.GpuQualificationPending, d.admit(.production, .prepared_native_reference));
    try testing.expectError(error.GpuQualificationPending, d.admit(.production, .stored_checkpoint));
    try testing.expectError(error.InvalidArgument, d.admit(.diagnostic, .checkpoint_effective));
    var buf: [8]Op = undefined;
    try testing.expectEqualSlices(Op, &.{ .activation_normalize, .mid_attention_sdpa, .mp_weight_prepare }, Decoder.openContracts(&buf));
    try admitOp(.conv1d_prepared, .diagnostic);
    try testing.expectError(error.GpuQualificationPending, admitOp(.conv1d_prepared, .production));
    try testing.expectError(error.GpuQualificationPending, admitOp(.activation_normalize, .production));
    try testing.expectError(error.GpuQualificationPending, admitOp(.mid_attention_sdpa, .production));
    try testing.expectError(error.NotOnNativePath, admitOp(.vocoder_weight_norm_fold, .diagnostic));
    // normalization: the axis and layout are checked; the geometry carries the scale, never an epsilon
    var store: [2 * 3 * 4]f32 = undefined;
    const base: u64 = @intFromPtr(&store);
    const x = View.contiguous("x", .host, base, @sizeOf(@TypeOf(store)), .borrowed, &.{ 1, 3, 4 });
    const g = try normalizeActivation(x, x, 1);
    try testing.expectEqual(NormGeometry{ .outer = 1, .n = 3, .inner = 4, .scale = @floatCast(@sqrt(@as(f64, 1.0) / 3.0)) }, g);
    try testing.expectError(error.InvalidArgument, normalizeActivation(x, x, 2));
    const qkv = View.contiguous("qkv", .host, base, @sizeOf(@TypeOf(store)), .borrowed, &.{ 1, 1, 2, 3, 4 });
    const gq = try normalizeActivation(qkv, qkv, 2);
    try testing.expectEqual(@as(u64, 1), gq.outer);
    try testing.expectEqual(@as(u64, 2), gq.n);
    try testing.expectEqual(@as(u64, 12), gq.inner);
}

test "weights: stored is not prepared, prepared is never prepared twice, oracle weights are diagnostic only" {
    var store: [6]f32 = .{ 1, -2, 3, 0.5, 0, 4 };
    var out: [6]f32 = undefined;
    const v = View.contiguous("w", .host, @intFromPtr(&store), @sizeOf(@TypeOf(store)), .borrowed, &.{ 1, 2, 3 });
    const ov = View.contiguous("w'", .host, @intFromPtr(&out), @sizeOf(@TypeOf(out)), .borrowed, &.{ 1, 2, 3 });
    const params: PrepareParams = .{ .eps = 1e-4, .reduce = .lanes8_tail_diagnostic, .den = .fma };
    const s = Weight.stored(v, "decoder.conv_in.weight", @splat(1));
    try testing.expectError(error.WeightNotPrepared, s.requirePrepared(.diagnostic));
    try testing.expectError(error.GpuQualificationPending, prepare(s, .mp_weight_prepare, &store, ov, &out, params, .production));
    try testing.expectError(error.NotOnNativePath, prepare(s, .vocoder_weight_norm_fold, &store, ov, &out, params, .diagnostic));
    // a row of 6 is not a multiple of 8: the AVX2 contract refuses it rather than guess the tail
    try testing.expectError(error.UnresolvedContract, prepare(s, .mp_weight_prepare, &store, ov, &out, .{ .eps = 1e-4, .reduce = .lanes8, .den = .fma }, .diagnostic));
    const once = try prepare(s, .mp_weight_prepare, &store, ov, &out, params, .diagnostic);
    try testing.expectEqual(WeightState.prepared_native_reference, once.state);
    try testing.expectEqual(@as(u8, 1), once.preparations);
    try once.requirePrepared(.diagnostic);
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(&out), &sha, .{});
    try testing.expectEqualSlices(u8, &sha, &once.prepared_sha256);
    try testing.expectError(error.AlreadyPrepared, prepare(once, .mp_weight_prepare, &out, ov, &out, params, .diagnostic));
    const p = Weight.oraclePrepared(v, "fixture", @splat(2));
    try p.requirePrepared(.diagnostic);
    try testing.expectError(error.DiagnosticWeight, p.requirePrepared(.production));
    try testing.expectError(error.AlreadyPrepared, prepare(p, .mp_weight_prepare, &store, ov, &out, params, .diagnostic));
    var twice = p;
    twice.preparations = 2;
    try testing.expectError(error.AlreadyPrepared, twice.requirePrepared(.diagnostic));
    // the vocoder's checkpoint weights are used as stored: accepted by consumers, never prepared
    const e = Weight.checkpointEffective(v, "ups.0.0.weight", @splat(3));
    try e.requirePrepared(.diagnostic);
    try e.requirePrepared(.production);
    try testing.expectError(error.AlreadyPrepared, prepare(e, .vocoder_weight_norm_fold, &store, ov, &out, params, .diagnostic));
}

const RawCase = struct { name: []const u8, raw: []const u8, raw_sha256: []const u8, baked: []const u8, baked_sha256: []const u8, cout: u64, cin: u64, k: u64 };

// fixtures/manifest.json: synthetic stored weights and the oracle's prepared ("baked") result for the same tensor.
const raw_cases = [_]RawCase{
    .{ .name = "2-1-1-ordinary", .raw = "conv-2-1-1-ordinary-raw.f32le", .raw_sha256 = "716707780862263735b493ae2b383eb7f5900a3357cb0b2a18009ea6557a85ec", .baked = "conv-2-1-1-ordinary-baked.f32le", .baked_sha256 = "ab7b8e77f6105a5f3977d09f2222163649f523c05faf7939972bef6c39da1e04", .cout = 1, .cin = 2, .k = 1 },
    .{ .name = "2-2-3-ordinary", .raw = "conv-2-2-3-ordinary-raw.f32le", .raw_sha256 = "5f61408149026253954af141f8c3d007bcd06c2d45d0d5aee5fe7fd15f52b98c", .baked = "conv-2-2-3-ordinary-baked.f32le", .baked_sha256 = "68965513ae2fb1be79d27c5f9b4356411bdf504857317dfea111e8361fb417a4", .cout = 2, .cin = 2, .k = 3 },
    .{ .name = "3-2-5-ordinary", .raw = "conv-3-2-5-ordinary-raw.f32le", .raw_sha256 = "4fc1a7ad1bd1d9ade3c7b709820d723b84243355a324e336534e5ebfe76a64ce", .baked = "conv-3-2-5-ordinary-baked.f32le", .baked_sha256 = "7fcc3d6d48a197010c76715c80df5b96a6ff11ffea8f859bd58d454bddd2acfd", .cout = 2, .cin = 3, .k = 5 },
    .{ .name = "2-1-3-tiny", .raw = "conv-2-1-3-tiny-raw.f32le", .raw_sha256 = "331d13acafab09f0a18a783ebd389e28c96f2406779598056cec0c2956d674ad", .baked = "conv-2-1-3-tiny-baked.f32le", .baked_sha256 = "d7f7814f8712363911bc99a156094cf0988a5b2b1798c40e002f09bce4880547", .cout = 1, .cin = 2, .k = 3 },
    .{ .name = "2-1-3-zero", .raw = "conv-2-1-3-zero-raw.f32le", .raw_sha256 = "4bfc74e95ac3e7eff044ba21fa5e0dd106a880b56c696184f029bb1cc7acdac5", .baked = "conv-2-1-3-zero-baked.f32le", .baked_sha256 = "4bfc74e95ac3e7eff044ba21fa5e0dd106a880b56c696184f029bb1cc7acdac5", .cout = 1, .cin = 2, .k = 3 },
};

fn prepareFixtureUnequal(den: Acc, values: *usize) !usize {
    var bad: usize = 0;
    inline for (raw_cases) |rc| {
        try expectSha(rc.raw, rc.raw_sha256);
        try expectSha(rc.baked, rc.baked_sha256);
        const raw = f32le(rc.raw);
        const want = f32le(rc.baked);
        var got: @TypeOf(want) = undefined;
        const shape = [_]u64{ rc.cout, rc.cin, rc.k };
        const wv = View.contiguous(rc.raw, .host, @intFromPtr(&raw), @sizeOf(@TypeOf(raw)), .borrowed, &shape);
        const ov = View.contiguous(rc.baked, .host, @intFromPtr(&got), @sizeOf(@TypeOf(got)), .borrowed, &shape);
        var sha: [32]u8 = undefined;
        _ = try std.fmt.hexToBytes(&sha, rc.raw_sha256);
        const p = try prepare(Weight.stored(wv, rc.raw, sha), .mp_weight_prepare, &raw, ov, &got, .{ .eps = 1e-4, .reduce = .lanes8_tail_diagnostic, .den = den }, .diagnostic);
        try p.requirePrepared(.diagnostic);
        bad += unequal(&got, &want);
        values.* += want.len;
    }
    return bad;
}

test "CPU diagnostic: stored -> prepared weights reproduce the oracle (not target parity)" {
    var values: usize = 0;
    try testing.expectEqual(0, try prepareFixtureUnequal(.fma, &values));
    try testing.expectEqual(56, values);
    // discriminator: the denominator in two roundings misses 2 of the 56
    values = 0;
    try testing.expectEqual(2, try prepareFixtureUnequal(.mul_add, &values));
}

test "lanes8 reduction: eight lane sums added in order, multiples of 8 only" {
    var x: [16]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = @floatFromInt(i + 1);
    // lane j sums (j+1)^2 + (j+9)^2, exact in f32; the lane total is the plain sum of squares 1496
    try testing.expectEqual(@as(f32, 1496), try sumSquares(&x, 0, 16, 1, .lanes8));
    try testing.expectError(error.UnresolvedContract, sumSquares(&x, 0, 15, 1, .lanes8));
    try testing.expectError(error.LayoutUnsupported, sumSquares(&x, 0, 8, 2, .lanes8));
    // order matters in f32: 2^24 then eight 1s is lost serially but kept by a lane that only sums 1s
    var y: [16]f32 = @splat(1);
    y[0] = 4096; // 4096^2 = 2^24
    try testing.expectEqual(@as(f32, 16777216), try sumSquares(&y, 0, 16, 1, .serial));
    try testing.expectEqual(@as(f32, 16777230), try sumSquares(&y, 0, 16, 1, .lanes8));
}

const NormCase = struct { name: []const u8, input: []const u8, output: []const u8, input_sha256: []const u8, output_sha256: []const u8, shape: []const u64, axis: u8, eps: f32 };

// fixtures/manifest.json (normalization cases); files hold the logical tensor in contiguous order.
const norm_cases = [_]NormCase{
    .{ .name = "zero", .input = "norm-zero-input.f32le", .output = "norm-zero-output.f32le", .input_sha256 = "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0", .output_sha256 = "9d908ecfb6b256def8b49a7c504e6c889c4b0e41fe6ce3e01863dd7b61a20aa0", .shape = &.{ 1, 2, 3 }, .axis = 1, .eps = 0.0001 },
    .{ .name = "tiny", .input = "norm-tiny-input.f32le", .output = "norm-tiny-output.f32le", .input_sha256 = "8e8365e205d882699acf9625452cff9a5ea7c000d7bcf3ec7e6d1c8fa4b095ff", .output_sha256 = "bb2c05788a7d0207a429f0782351dc262a2db494f52e9f12f1ca9add442a80aa", .shape = &.{ 1, 2, 1 }, .axis = 1, .eps = 0.0001 },
    .{ .name = "signed_zero", .input = "norm-signed_zero-input.f32le", .output = "norm-signed_zero-output.f32le", .input_sha256 = "a9765c4805658a968e5abdccd437e25907681a1cff6375e363239c53b125fcd4", .output_sha256 = "a9765c4805658a968e5abdccd437e25907681a1cff6375e363239c53b125fcd4", .shape = &.{ 1, 2, 1 }, .axis = 1, .eps = 0.0001 },
    .{ .name = "ordinary", .input = "norm-ordinary-input.f32le", .output = "norm-ordinary-output.f32le", .input_sha256 = "1cb2f8fcaee102d6e073bede544b324fe2fdd71d3c49fec37d46f883735c868a", .output_sha256 = "4fbef97e3e2d70e2454f20a4df24e853924ccf0d445b26ad622a859702d439fc", .shape = &.{ 1, 2, 3 }, .axis = 1, .eps = 0.0001 },
    .{ .name = "epsilon_override", .input = "norm-epsilon_override-input.f32le", .output = "norm-epsilon_override-output.f32le", .input_sha256 = "8e8365e205d882699acf9625452cff9a5ea7c000d7bcf3ec7e6d1c8fa4b095ff", .output_sha256 = "a7f43ace0026010bd74dc1aa2a3a4cdd06149332770ac81c36f0a9131a28d30c", .shape = &.{ 1, 2, 1 }, .axis = 1, .eps = 0.001 },
    .{ .name = "overflow", .input = "norm-overflow-input.f32le", .output = "norm-overflow-output.f32le", .input_sha256 = "68f411145913035e8676b890c02d84fa8e90a396baabb7bd0e919d60a139ffd2", .output_sha256 = "af5570f5a1810b7af78caf4bc70a660f0df51e42baf91d4de5b2328de0e83dfc", .shape = &.{ 1, 2, 1 }, .axis = 1, .eps = 0.0001 },
    .{ .name = "underflow", .input = "norm-underflow-input.f32le", .output = "norm-underflow-output.f32le", .input_sha256 = "25720ccd8f4bbe8bba7d91ebd77db938c76c34315aac7609f9da513c7be6361a", .output_sha256 = "8ba48586058f38e1c55e8aa9c00335165cf5609dcb27746441a677e0ecd59487", .shape = &.{ 1, 2, 1 }, .axis = 1, .eps = 0.0001 },
    .{ .name = "channels_40", .input = "norm-channels_40-input.f32le", .output = "norm-channels_40-output.f32le", .input_sha256 = "23ea3104360254fb5b1f48cac2b51c883537f1b1246fac07bd1860e288f6ec8a", .output_sha256 = "156fe3900555aae233cfc76f22a618e1716aebb1d52b1a29261a9b44b0da8d1e", .shape = &.{ 1, 40, 3 }, .axis = 1, .eps = 0.0001 },
    .{ .name = "channels_128", .input = "norm-channels_128-input.f32le", .output = "norm-channels_128-output.f32le", .input_sha256 = "f0ea7a2da17da46c65aee2d82733f866eb0f68bcb1460a9c8b6e2157064f7327", .output_sha256 = "d50d67a88c2e3c12d27aad2f4b23424d6b758adcb32e0441f9c63db3ae2bc300", .shape = &.{ 1, 128, 3 }, .axis = 1, .eps = 0.0001 },
    .{ .name = "channels_512", .input = "norm-channels_512-input.f32le", .output = "norm-channels_512-output.f32le", .input_sha256 = "f48b78807a1327c82369f0a427596cbdbcb9278fb048bea3f477b693d7687872", .output_sha256 = "dc5854ab908b68d3086e50cb0fd63fe70d4255e9c56c91ee41f1d3a55c04617c", .shape = &.{ 1, 512, 3 }, .axis = 1, .eps = 0.0001 },
    .{ .name = "channels_1024", .input = "norm-channels_1024-input.f32le", .output = "norm-channels_1024-output.f32le", .input_sha256 = "e3b30cbbf44ef8af8caf4bed572b80898434e1be8d6044c7676344bda817f102", .output_sha256 = "ff7004266d952abe727f0359353adeb76a2a8020f38f126db285ba932c23e764", .shape = &.{ 1, 1024, 3 }, .axis = 1, .eps = 0.0001 },
    .{ .name = "channels_2048", .input = "norm-channels_2048-input.f32le", .output = "norm-channels_2048-output.f32le", .input_sha256 = "59f11fa0349479f6d0495c497d8fe1e74ed4637e8b1f01be42fb0af4db270bc4", .output_sha256 = "75c5e6b92a38abf0908dcfdf83da098fedb1b2651b54371316a4ddbbcae5eaee", .shape = &.{ 1, 2048, 3 }, .axis = 1, .eps = 0.0001 },
    .{ .name = "packed_qkv", .input = "norm-packed_qkv-input.f32le", .output = "norm-packed_qkv-output.f32le", .input_sha256 = "cf46e8550196db2feab79a72feaf0860326d1db4408b371ee484f8a0bd5b7f4a", .output_sha256 = "6f42b629451ea17008ed064417812663f65361ee84d9a021096c1e6bc248257c", .shape = &.{ 2, 1, 128, 3, 7 }, .axis = 2, .eps = 0.0001 },
    .{ .name = "packed_qkv_native_2048", .input = "norm-packed_qkv_native_2048-input.f32le", .output = "norm-packed_qkv_native_2048-output.f32le", .input_sha256 = "6fc20feebe35b2b49df3a9b6cf3f8f9ebb4904d6be9c4afe57b24f8f0a16b330", .output_sha256 = "a8d4302ae332ae390e207948c8222fd12c6a741e4db32063b040b17a984e1749", .shape = &.{ 1, 1, 2048, 3, 3 }, .axis = 2, .eps = 0.0001 },
    .{ .name = "noncontiguous_channels", .input = "norm-noncontiguous_channels-input.f32le", .output = "norm-noncontiguous_channels-output.f32le", .input_sha256 = "b8990e2f95c530d40945e3b14b2cfba4d997443e83f687571500febc176ff674", .output_sha256 = "ef3933b3fc0f005da7df54631c7b2c9eaaffcb88e037bb75bc825915f516a61e", .shape = &.{ 2, 1, 7, 3, 128 }, .axis = 4, .eps = 0.0001 },
};

fn normFixtureUnequal(den: Acc, values: *usize) !usize {
    var bad: usize = 0;
    inline for (norm_cases) |nc| {
        try expectSha(nc.input, nc.input_sha256);
        try expectSha(nc.output, nc.output_sha256);
        const x = f32le(nc.input);
        const want = f32le(nc.output);
        var y: @TypeOf(want) = undefined;
        var g: NormGeometry = .{ .outer = 1, .n = nc.shape[nc.axis], .inner = 1, .scale = @floatCast(@sqrt(1.0 / @as(f64, @floatFromInt(nc.shape[nc.axis])))) };
        for (nc.shape[0..nc.axis]) |d| g.outer *= d;
        for (nc.shape[nc.axis + 1 ..]) |d| g.inner *= d;
        try normalizeHost(&x, &y, g, nc.eps, .serial, den);
        bad += unequal(&y, &want);
        values.* += want.len;
    }
    return bad;
}

test "CPU diagnostic: activation normalization reproduces all 15 oracle cases (not target parity)" {
    var values: usize = 0;
    try testing.expectEqual(0, try normFixtureUnequal(.fma, &values));
    try testing.expectEqual(40462, values);
    // discriminator: the denominator in two roundings
    values = 0;
    try testing.expectEqual(7171, try normFixtureUnequal(.mul_add, &values));
}

test "views: bounds, contiguity, device" {
    var store: [24]f32 = undefined;
    const base: u64 = @intFromPtr(&store);
    var v = View.contiguous("x", .host, base, 96, .borrowed, &.{ 2, 3, 4 });
    try v.expect(.host, &.{ 2, 3, 4 });
    try testing.expectEqual(96, v.bytes());
    try testing.expectError(error.DeviceMismatch, v.expect(.cuda, &.{ 2, 3, 4 }));
    try testing.expectError(error.ShapeMismatch, v.expect(.host, &.{ 2, 4, 3 }));
    v.alloc_bytes = 92;
    try testing.expectError(error.OutOfBounds, v.validate());
    v.alloc_bytes = 96;
    v.strides = .{ 12, 1, 3, 0, 0 }; // a transposed view: in bounds, not contiguous
    try v.validate();
    try testing.expectError(error.LayoutUnsupported, v.expect(.host, &.{ 2, 3, 4 }));
}

test "elementwise references: latent unscale, clamp, nearest-exact, residual constants" {
    // latent: [1, 2, 2] rows -> [1, 2, 2] channel-first; -0.0 / s + 0.0 = +0.0
    const rows = [_]f32{ 1.0, -0.0, 0.417, 2.0 };
    var z: [4]f32 = undefined;
    latentUnscale(&rows, &z, 1, 2, 2, .of(scaling_factor, .true_div), mean_value);
    try testing.expectEqual(bits(1.0 / @as(f32, 0.417)), bits(z[0]));
    try testing.expectEqual(bits(@as(f32, 1.0)), bits(z[1])); // (t=1, c=0) = 0.417 / 0.417
    try testing.expectEqual(@as(u32, 0), bits(z[2])); // +0.0, not -0.0
    const r = ScalarDiv.of(scaling_factor, .reciprocal);
    try testing.expectEqual(bits(1.0 / @as(f32, 0.417)), bits(r.c));
    // clamp keeps NaN
    const cx = [_]f32{ std.math.nan(f32), -300, 300, 5 };
    var cy: [4]f32 = undefined;
    clamp(&cx, &cy, -clip_act, clip_act);
    try testing.expect(std.math.isNan(cy[0]));
    try testing.expectEqualSlices(f32, &.{ -256, 256, 5 }, cy[1..]);
    const ux = [_]f32{ 1, 2, 3 };
    var uy: [6]f32 = undefined;
    upsampleNearestExact2(&ux, &uy, 1, 3);
    try testing.expectEqualSlices(f32, &.{ 1, 1, 2, 2, 3, 3 }, &uy);
    const m = MpResidual.of(residual_t, .true_div);
    try testing.expectEqual(bits(@as(f32, 0.7)), bits(m.c0));
    try testing.expectEqual(bits(@as(f32, 0.3)), bits(m.c1));
    try testing.expectEqual(bits(@as(f32, @floatCast(@as(f64, 0.7615773105863908)))), bits(m.den.c));
}

test "qkv split and merge follow the reshape [B,1,2048,3,T] channel order" {
    // H = 1, D = 2, T = 3: qkv channel (d * 3 + j)
    var qkv: [2 * 3 * 3]f32 = undefined;
    for (&qkv, 0..) |*v, i| v.* = @floatFromInt(i);
    var q: [6]f32 = undefined;
    var k: [6]f32 = undefined;
    var v: [6]f32 = undefined;
    qkvSplit(&qkv, &q, &k, &v, 1, 2, 3);
    // q[t, d] = channel 3d, time t
    try testing.expectEqualSlices(f32, &.{ 0, 9, 1, 10, 2, 11 }, &q);
    try testing.expectEqualSlices(f32, &.{ 3, 12, 4, 13, 5, 14 }, &k);
    try testing.expectEqualSlices(f32, &.{ 6, 15, 7, 16, 8, 17 }, &v);
    var merged: [6]f32 = undefined;
    attnMerge(&q, &merged, 1, 2, 3);
    try testing.expectEqualSlices(f32, &.{ 0, 1, 2, 9, 10, 11 }, &merged);
}

/// Unequal values of ConvPrepared1D over the oracle-prepared fixture weights: gain applied to the weight before the
/// convolution (fma and mul_add chains) and, as the discriminator, to the output afterwards.
const FixtureCounts = struct { weight_gain_fma: usize = 0, weight_gain_mul_add: usize = 0, output_gain: usize = 0, values: usize = 0 };

fn convFixtureCounts() !FixtureCounts {
    var n: FixtureCounts = .{};
    inline for (conv_cases) |cc| {
        try expectSha(cc.weight, cc.weight_sha256);
        try expectSha(cc.input, cc.input_sha256);
        const w = f32le(cc.weight);
        const x = f32le(cc.input);
        const d: ConvDims = .{ .b = 1, .cin = cc.cin, .cout = cc.cout, .lin = 7, .k = cc.k, .pad = cc.k / 2 };
        // the views the device path would take, and the state it requires
        const wv = View.contiguous(cc.weight, .host, @intFromPtr(&w), @sizeOf(@TypeOf(w)), .borrowed, &.{ cc.cout, cc.cin, cc.k });
        var sha: [32]u8 = undefined;
        _ = try std.fmt.hexToBytes(&sha, cc.weight_sha256);
        try Weight.oraclePrepared(wv, cc.weight, sha).requirePrepared(.diagnostic);
        inline for (cc.gains) |g| {
            try expectSha(g.file, g.sha256);
            const want = f32le(g.file);
            var ws: @TypeOf(w) = undefined;
            scaleWeights(&w, &ws, g.gain);
            var y: @TypeOf(want) = undefined;
            conv1d(&x, &ws, null, &y, d, .fma);
            n.weight_gain_fma += unequal(&y, &want);
            conv1d(&x, &ws, null, &y, d, .mul_add);
            n.weight_gain_mul_add += unequal(&y, &want);
            conv1d(&x, &w, null, &y, d, .mul_add);
            for (&y) |*v| v.* *= g.gain;
            n.output_gain += unequal(&y, &want);
            n.values += want.len;
        }
    }
    return n;
}

test "CPU diagnostic: ConvPrepared1D over oracle-prepared synthetic weights (not target parity)" {
    const n = try convFixtureCounts();
    try testing.expectEqual(245, n.values); // 5 configurations x 5 gains
    // The oracle's own evaluator reports weight-side gain 0 and output-side gain 51 unequal values over these calls;
    // the independent host reference reproduces both. Only the two-rounding chain reproduces the CPU bits; a fused
    // chain does not. This fixes nothing for the target: the SM121 accumulation stays an explicit open choice.
    try testing.expectEqual(0, n.weight_gain_mul_add);
    try testing.expectEqual(51, n.output_gain);
    try testing.expectEqual(68, n.weight_gain_fma);
}

test "CPU diagnostic: boundary impulses fix zero padding K//2" {
    try expectSha(boundary_weight, boundary_weight_sha256);
    const w = f32le(boundary_weight);
    inline for (boundary_cases) |bc| {
        try expectSha(bc.input, bc.input_sha256);
        try expectSha(bc.output, bc.output_sha256);
        const x = f32le(bc.input);
        const want = f32le(bc.output);
        var y: [7]f32 = undefined;
        inline for (.{ Acc.fma, Acc.mul_add }) |mode| {
            conv1d(&x, &w, null, &y, .{ .b = 1, .cin = 1, .cout = 1, .lin = 7, .k = 3, .pad = 1 }, mode);
            try testing.expectEqual(0, unequal(&y, &want));
        }
    }
}
