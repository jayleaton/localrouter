//! cuBLASLt with an explicit, recorded algorithm, from a pinned library: no heuristic query, no preference, no soname
//! search and nothing to fall back to. FP32 only (R_32F everywhere, CUBLAS_COMPUTE_32F), column order, alpha 1, beta 0.
//! Plans are built and checked at load; the hot path binds pointers only.
//!
//! TensorFold's own cuBLASLt (`cuda.cublaslt`) is not used: it opens by soname and takes the first heuristic. Only its
//! ABI types and the enum values the daemon already relies on are shared; no TensorFold change.

const std = @import("std");
const cuda = @import("cuda");
const pinned = @import("pinned.zig");
const tf = cuda.cublaslt;
const log = @import("report.zig"); // .err at run time, .warn in tests

pub const Status = tf.Status;
pub const Handle = tf.Handle;
pub const Algo = tf.Algo;
pub const Op = tf.Op;
const Desc = tf.MatmulDesc;
const Layout = tf.Layout;
/// cublasLtMatmulHeuristicResult_t, as AlgoCheck fills it.
const CheckResult = tf.Heuristic;
const S = Status;
const CPtr = ?*const anyopaque;

/// Exactly the entry points this path calls. There is no cublasLtMatmulAlgoGetHeuristic and no preference.
pub const Api = struct {
    cublasLtCreate: *const fn (*Handle) callconv(.c) S,
    cublasLtDestroy: *const fn (Handle) callconv(.c) S,
    cublasLtGetVersion: *const fn () callconv(.c) usize,
    cublasLtGetStatusName: *const fn (S) callconv(.c) ?[*:0]const u8,
    cublasLtMatmulDescCreate: *const fn (*Desc, c_int, tf.DataType) callconv(.c) S,
    cublasLtMatmulDescDestroy: *const fn (Desc) callconv(.c) S,
    cublasLtMatmulDescSetAttribute: *const fn (Desc, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatrixLayoutCreate: *const fn (*Layout, tf.DataType, u64, u64, i64) callconv(.c) S,
    cublasLtMatrixLayoutDestroy: *const fn (Layout) callconv(.c) S,
    cublasLtMatrixLayoutSetAttribute: *const fn (Layout, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmulAlgoInit: *const fn (Handle, c_int, tf.DataType, tf.DataType, tf.DataType, tf.DataType, tf.DataType, c_int, *Algo) callconv(.c) S,
    cublasLtMatmulAlgoConfigSetAttribute: *const fn (*Algo, c_int, CPtr, usize) callconv(.c) S,
    cublasLtMatmulAlgoConfigGetAttribute: *const fn (*const Algo, c_int, ?*anyopaque, usize, *usize) callconv(.c) S,
    cublasLtMatmulAlgoCheck: *const fn (Handle, Desc, Layout, Layout, Layout, Layout, *const Algo, *CheckResult) callconv(.c) S,
    cublasLtMatmul: *const fn (Handle, Desc, CPtr, cuda.abi.DevicePtr, Layout, cuda.abi.DevicePtr, Layout, CPtr, cuda.abi.DevicePtr, Layout, cuda.abi.DevicePtr, Layout, ?*const Algo, cuda.abi.DevicePtr, usize, cuda.abi.Stream) callconv(.c) S,
};

/// The enum values this path adds to TensorFold's. Gap G1: each is copied, with its citation, from the cublasLt.h of
/// the exact pinned cuBLASLt build, read read-only; none is written from memory. Null refuses every plan.
pub const Constants = struct {
    desc_sm_count_target: ?c_int = null,
    algo_id: ?c_int = null,
    algo_tile_id: ?c_int = null,
    algo_stages_id: ?c_int = null,
    algo_splitk_num: ?c_int = null,
    algo_reduction_scheme: ?c_int = null,
    algo_cta_swizzling: ?c_int = null,
    algo_custom_option: ?c_int = null,
    algo_inner_shape_id: ?c_int = null,
    algo_cluster_shape_id: ?c_int = null,

    fn complete(self: Constants) bool {
        inline for (@typeInfo(Constants).@"struct".field_names) |name| if (@field(self, name) == null) return false;
        return true;
    }
};

/// The pinned build's values: none cited yet (G1).
pub const constants: Constants = .{};

/// One recorded algorithm. Every field comes from a native record (gap G2 for those still null); AlgoInit's defaults
/// are not a record, so a null refuses. Widths as cublasLt.h documents the attributes, to be confirmed with G1.
pub const AlgoSpec = struct {
    algo_id: c_int,
    tile: ?u32,
    stages: ?u32,
    splitk: ?u32,
    reduction: ?u32,
    cta_swizzling: ?u32,
    custom_option: ?u32,
    inner_shape: ?u16,
    cluster_shape: ?u16,
    /// AlgoCheck's workspace size must equal this.
    workspace: usize,
};

pub const GemmSpec = struct {
    m: u64,
    n: u64,
    k: u64,
    transa: Op,
    transb: Op,
    lda: i64,
    ldb: i64,
    /// D shares C's layout.
    ldc: i64,
    /// CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET; null if the record does not show it was set explicitly (G2).
    sm_count_target: ?i32,
    algo: AlgoSpec,
};

pub const Error = error{
    LtVersionMismatch,
    LtFailed,
    LtConstantUnpinned,
    LtAlgoUnrecorded,
    LtReadBackMismatch,
    LtWorkspaceMismatch,
    LtBadSpec,
    LtNotQuiesced,
    /// A destroy call failed: its raw status was logged, the object stays owned and held, and the plan or handle refuses
    /// every later use and release.
    LtReleaseFailed,
    LtReleased,
} || pinned.Error;

pub const State = enum { live, released, release_failed };
pub const PlanState = State;

/// One cuBLASLt handle and its lifecycle, from `Lt.create`. It holds the set until its one successful destroy; keep it
/// at one address (plans point at it) and never copy it.
pub const OwnedHandle = struct {
    raw: Handle,
    state: State = .live,
    /// The failed destroy's raw status.
    release_status: S = 0,

    fn usable(self: *const OwnedHandle) Error!void {
        switch (self.state) {
            .live => {},
            .released => return error.LtReleased,
            .release_failed => return error.LtReleaseFailed,
        }
    }
};

/// A refused `Plan.init` whose own cleanup failed: `held` is the plan with state `release_failed`, its failed
/// descriptors or layouts still set, and its hold on the set kept.
pub const InitRetained = pinned.InitRetained(Error, Plan);

pub const Lt = struct {
    api: Api,
    /// The pinned set's dependent count (`pinned.Set.hold` / `drop`): each live handle and plan holds it, and only a
    /// successful destroy releases it, so the libraries cannot be unloaded under a handle or plan that failed to go.
    deps: *u32,

    /// Resolves the API from the pinned set's `name` library and refuses unless cublasLtGetVersion() equals
    /// `expect_version` (the P4 record, e.g. 130101). The set must not move while this `Lt` lives.
    pub fn bind(set: *pinned.Set, name: []const u8, expect_version: usize) Error!Lt {
        const lib = set.get(name) orelse return error.PinMissingSymbol;
        return fromApi(try lib.resolve(Api), expect_version, &set.dependents);
    }

    pub fn fromApi(api: Api, expect_version: usize, deps: *u32) Error!Lt {
        const v = api.cublasLtGetVersion();
        if (v != expect_version) {
            log.err("cuBLASLt version {d}, pinned {d}", .{ v, expect_version });
            return error.LtVersionMismatch;
        }
        return .{ .api = api, .deps = deps };
    }

    pub fn create(self: *const Lt) Error!OwnedHandle {
        var h: Handle = null;
        try self.check(self.api.cublasLtCreate(&h), "cublasLtCreate");
        self.deps.* += 1;
        return .{ .raw = h };
    }

    /// After every plan on it is destroyed, with a valid quiescence proof. Attempted once: a released handle refuses
    /// (`LtReleased`) and a failed one refuses (`LtReleaseFailed`) without another vendor call; a failure keeps its raw
    /// status and the handle held on the set.
    pub fn destroy(self: *const Lt, h: *OwnedHandle, q: pinned.Quiesced) Error!void {
        try h.usable();
        if (!q.valid()) return error.LtNotQuiesced;
        const s = self.api.cublasLtDestroy(h.raw);
        if (s != 0) {
            self.check(s, "cublasLtDestroy") catch {};
            h.state = .release_failed;
            h.release_status = s;
            return error.LtReleaseFailed;
        }
        h.state = .released;
        self.deps.* -= 1;
    }

    fn check(self: *const Lt, s: S, what: []const u8) Error!void {
        if (s == 0) return;
        const name = if (self.api.cublasLtGetStatusName(s)) |p| std.mem.span(p) else "?";
        log.err("{s}: {s} ({d})", .{ what, name, s });
        return error.LtFailed;
    }
};

pub const Plan = struct {
    lt: *const Lt,
    handle: *const OwnedHandle,
    desc: Desc = null,
    a: Layout = null,
    b: Layout = null,
    c: Layout = null,
    algo: Algo = .{ .data = @splat(0) },
    workspace: usize,
    state: PlanState = .live,
    /// The first failed destroy's raw status.
    release_status: S = 0,

    /// At load: the descriptor (COMPUTE_32F, R_32F scale, transa / transb, SM count target), the layouts, AlgoInit with
    /// the recorded id and every config attribute set, read back and compared, then AlgoCheck. A failed status, a
    /// read-back difference, a workspace other than the record's, or any unpinned constant or unrecorded field refuses.
    /// A refused plan destroys what it created, each once (nothing was queued with it), and returns its own error; if
    /// that destroy fails, `retained` receives the plan with the failed objects and its hold on the set. A handle that
    /// is released or failed to release refuses.
    pub fn init(lt: *const Lt, owned: *const OwnedHandle, spec: GemmSpec, k: Constants, retained: *?InitRetained) Error!Plan {
        retained.* = null;
        try owned.usable();
        if (!k.complete()) return error.LtConstantUnpinned;
        const al = spec.algo;
        if (spec.sm_count_target == null or al.tile == null or al.stages == null or al.splitk == null or al.reduction == null or
            al.cta_swizzling == null or al.custom_option == null or al.inner_shape == null or al.cluster_shape == null)
            return error.LtAlgoUnrecorded;
        if (spec.m == 0 or spec.n == 0 or spec.k == 0) return error.LtBadSpec;

        var p: Plan = .{ .lt = lt, .handle = owned, .workspace = al.workspace };
        buildDescriptors(&p, spec, k) catch |primary| {
            const s = p.destroyDescriptors();
            if (s != 0) {
                lt.deps.* += 1;
                p.state = .release_failed;
                p.release_status = s;
                retained.* = .{ .primary = primary, .cleanup = .{ .what = "cuBLASLt plan descriptors", .rc = s }, .held = p };
            }
            return primary;
        };
        lt.deps.* += 1;
        return p;
    }

    fn buildDescriptors(p: *Plan, spec: GemmSpec, k: Constants) Error!void {
        const lt = p.lt;
        const handle = p.handle.raw;
        const al = spec.algo;
        const api = lt.api;
        try lt.check(api.cublasLtMatmulDescCreate(&p.desc, tf.compute_32f, .f32), "cublasLtMatmulDescCreate");
        try setDesc(lt, p.desc, tf.desc_transa, spec.transa);
        try setDesc(lt, p.desc, tf.desc_transb, spec.transb);
        try setDesc(lt, p.desc, k.desc_sm_count_target.?, @as(i32, spec.sm_count_target.?));

        // op(A) is m x k and op(B) is k x n; the layouts describe A and B as stored
        if (spec.transa == .n) try layout(lt, &p.a, spec.m, spec.k, spec.lda) else try layout(lt, &p.a, spec.k, spec.m, spec.lda);
        if (spec.transb == .n) try layout(lt, &p.b, spec.k, spec.n, spec.ldb) else try layout(lt, &p.b, spec.n, spec.k, spec.ldb);
        try layout(lt, &p.c, spec.m, spec.n, spec.ldc);

        try lt.check(api.cublasLtMatmulAlgoInit(handle, tf.compute_32f, .f32, .f32, .f32, .f32, .f32, al.algo_id, &p.algo), "cublasLtMatmulAlgoInit");
        try setAlgo(lt, &p.algo, k.algo_tile_id.?, al.tile.?);
        try setAlgo(lt, &p.algo, k.algo_stages_id.?, al.stages.?);
        try setAlgo(lt, &p.algo, k.algo_splitk_num.?, al.splitk.?);
        try setAlgo(lt, &p.algo, k.algo_reduction_scheme.?, al.reduction.?);
        try setAlgo(lt, &p.algo, k.algo_cta_swizzling.?, al.cta_swizzling.?);
        try setAlgo(lt, &p.algo, k.algo_custom_option.?, al.custom_option.?);
        try setAlgo(lt, &p.algo, k.algo_inner_shape_id.?, al.inner_shape.?);
        try setAlgo(lt, &p.algo, k.algo_cluster_shape_id.?, al.cluster_shape.?);
        try readBack(lt, &p.algo, k.algo_id.?, @as(i32, al.algo_id));

        var r: CheckResult = std.mem.zeroes(CheckResult);
        try lt.check(api.cublasLtMatmulAlgoCheck(handle, p.desc, p.a, p.b, p.c, p.c, &p.algo, &r), "cublasLtMatmulAlgoCheck");
        if (r.workspace_size != al.workspace) {
            log.err("cuBLASLt algo {d}: AlgoCheck workspace {d}, recorded {d}", .{ al.algo_id, r.workspace_size, al.workspace });
            return error.LtWorkspaceMismatch;
        }
    }

    /// The hot path: D = A·B on the caller's stream. `workspace_len` must equal the plan's; the pinned set is checked
    /// by the caller before and after. A plan or handle that is released or failed to release refuses.
    pub fn run(self: *const Plan, a: cuda.abi.DevicePtr, b: cuda.abi.DevicePtr, d: cuda.abi.DevicePtr, workspace: cuda.abi.DevicePtr, workspace_len: usize, stream: cuda.abi.Stream) Error!void {
        switch (self.state) {
            .live => {},
            .released => return error.LtReleased,
            .release_failed => return error.LtReleaseFailed,
        }
        try self.handle.usable();
        if (workspace_len != self.workspace) return error.LtWorkspaceMismatch;
        const alpha: f32 = 1;
        const beta: f32 = 0;
        try self.lt.check(self.lt.api.cublasLtMatmul(self.handle.raw, self.desc, &alpha, a, self.a, b, self.b, &beta, d, self.c, d, self.c, &self.algo, workspace, workspace_len, stream), "cublasLtMatmul");
    }

    /// With a valid quiescence proof only (no matmul queued with this plan can still run). Every destroy is attempted
    /// once; the first failed status is kept, the failed objects stay owned, the plan stays held on the set, and the
    /// plan refuses every later run and release.
    pub fn deinit(self: *Plan, q: pinned.Quiesced) Error!void {
        switch (self.state) {
            .live => {},
            .released => return error.LtReleased,
            .release_failed => return error.LtReleaseFailed,
        }
        if (!q.valid()) return error.LtNotQuiesced;
        const s = self.destroyDescriptors();
        if (s != 0) {
            self.state = .release_failed;
            self.release_status = s;
            return error.LtReleaseFailed;
        }
        self.state = .released;
        self.lt.deps.* -= 1;
    }

    /// Destroys each descriptor once; a failed one stays set. The first failed raw status, or 0.
    fn destroyDescriptors(self: *Plan) S {
        const api = self.lt.api;
        var first: S = 0;
        for ([_]*Layout{ &self.c, &self.b, &self.a }) |l| if (l.* != null) {
            const s = api.cublasLtMatrixLayoutDestroy(l.*);
            if (s == 0) {
                l.* = null;
            } else {
                log.err("cublasLtMatrixLayoutDestroy: {d}", .{s});
                if (first == 0) first = s;
            }
        };
        if (self.desc != null) {
            const s = api.cublasLtMatmulDescDestroy(self.desc);
            if (s == 0) {
                self.desc = null;
            } else {
                log.err("cublasLtMatmulDescDestroy: {d}", .{s});
                if (first == 0) first = s;
            }
        }
        return first;
    }
};

fn setDesc(lt: *const Lt, desc: Desc, attr: c_int, value: anytype) Error!void {
    const v = value;
    try lt.check(lt.api.cublasLtMatmulDescSetAttribute(desc, attr, &v, @sizeOf(@TypeOf(v))), "cublasLtMatmulDescSetAttribute");
}

/// Creates the layout into the plan's own slot, so a refusal after the create leaves it for the plan's one cleanup.
fn layout(lt: *const Lt, out: *Layout, rows: u64, cols: u64, ld: i64) Error!void {
    if (ld < 1 or @as(u64, @intCast(ld)) < rows) return error.LtBadSpec;
    try lt.check(lt.api.cublasLtMatrixLayoutCreate(out, .f32, rows, cols, ld), "cublasLtMatrixLayoutCreate");
    const order: tf.Order = .col;
    try lt.check(lt.api.cublasLtMatrixLayoutSetAttribute(out.*, tf.layout_order, &order, @sizeOf(tf.Order)), "cublasLtMatrixLayoutSetAttribute");
}

/// Sets one config attribute and reads it back.
fn setAlgo(lt: *const Lt, algo: *Algo, attr: c_int, value: anytype) Error!void {
    const v = value;
    try lt.check(lt.api.cublasLtMatmulAlgoConfigSetAttribute(algo, attr, &v, @sizeOf(@TypeOf(v))), "cublasLtMatmulAlgoConfigSetAttribute");
    try readBack(lt, algo, attr, v);
}

fn readBack(lt: *const Lt, algo: *const Algo, attr: c_int, want: anytype) Error!void {
    var got: @TypeOf(want) = undefined;
    var written: usize = 0;
    try lt.check(lt.api.cublasLtMatmulAlgoConfigGetAttribute(algo, attr, &got, @sizeOf(@TypeOf(want)), &written), "cublasLtMatmulAlgoConfigGetAttribute");
    if (written != @sizeOf(@TypeOf(want)) or got != want) {
        log.err("cuBLASLt algo attribute {d}: read back {d} ({d} bytes), set {d}", .{ attr, got, written, want });
        return error.LtReadBackMismatch;
    }
}

// ---- tests ------------------------------------------------------------------------------------------------------------

const testing = std.testing;
const quiesce = @import("quiesce.zig");

/// A fake cuBLASLt: config attributes are stored by id in the algo's own words.
const F = struct {
    var version: usize = 130101;
    var check_workspace: usize = 2281152;
    var skew_attr: c_int = -1;
    var layout_destroy_rc: S = 0;
    var layout_set_rc: S = 0;
    var handle_destroy_rc: S = 0;
    var handle_destroys: usize = 0;
    var created: usize = 0;
    var destroyed: usize = 0;

    fn ok0() callconv(.c) usize {
        return version;
    }
    fn create(h: *Handle) callconv(.c) S {
        h.* = @ptrFromInt(0x1000);
        return 0;
    }
    fn destroyH(_: Handle) callconv(.c) S {
        handle_destroys += 1;
        return handle_destroy_rc;
    }
    fn statusName(_: S) callconv(.c) ?[*:0]const u8 {
        return "CUBLAS_STATUS_TEST";
    }
    fn descCreate(d: *Desc, _: c_int, _: tf.DataType) callconv(.c) S {
        created += 1;
        d.* = @ptrFromInt(0x2000);
        return 0;
    }
    fn descDestroy(_: Desc) callconv(.c) S {
        destroyed += 1;
        return 0;
    }
    fn descSet(_: Desc, _: c_int, _: CPtr, _: usize) callconv(.c) S {
        return 0;
    }
    fn layoutCreate(l: *Layout, _: tf.DataType, _: u64, _: u64, _: i64) callconv(.c) S {
        created += 1;
        l.* = @ptrFromInt(0x3000);
        return 0;
    }
    fn layoutDestroy(_: Layout) callconv(.c) S {
        if (layout_destroy_rc != 0) return layout_destroy_rc;
        destroyed += 1;
        return 0;
    }
    fn layoutSet(_: Layout, _: c_int, _: CPtr, _: usize) callconv(.c) S {
        return layout_set_rc;
    }
    fn algoInit(_: Handle, _: c_int, _: tf.DataType, _: tf.DataType, _: tf.DataType, _: tf.DataType, _: tf.DataType, id: c_int, algo: *Algo) callconv(.c) S {
        algo.* = .{ .data = @splat(0) };
        algo.data[0] = @as(u32, @bitCast(id));
        return 0;
    }
    fn algoSet(algo: *Algo, attr: c_int, buf: CPtr, size: usize) callconv(.c) S {
        var w: u64 = 0;
        @memcpy(std.mem.asBytes(&w)[0..size], @as([*]const u8, @ptrCast(buf.?))[0..size]);
        algo.data[@intCast(attr)] = w;
        return 0;
    }
    fn algoGet(algo: *const Algo, attr: c_int, buf: ?*anyopaque, size: usize, written: *usize) callconv(.c) S {
        var w = algo.data[@intCast(attr)];
        if (attr == skew_attr) w += 1;
        @memcpy(@as([*]u8, @ptrCast(buf.?))[0..size], std.mem.asBytes(&w)[0..size]);
        written.* = size;
        return 0;
    }
    fn algoCheck(_: Handle, _: Desc, _: Layout, _: Layout, _: Layout, _: Layout, _: *const Algo, r: *CheckResult) callconv(.c) S {
        r.workspace_size = check_workspace;
        return 0;
    }
    fn matmul(_: Handle, _: Desc, _: CPtr, _: cuda.abi.DevicePtr, _: Layout, _: cuda.abi.DevicePtr, _: Layout, _: CPtr, _: cuda.abi.DevicePtr, _: Layout, _: cuda.abi.DevicePtr, _: Layout, _: ?*const Algo, _: cuda.abi.DevicePtr, _: usize, _: cuda.abi.Stream) callconv(.c) S {
        return 0;
    }

    const api: Api = .{
        .cublasLtCreate = create,
        .cublasLtDestroy = destroyH,
        .cublasLtGetVersion = ok0,
        .cublasLtGetStatusName = statusName,
        .cublasLtMatmulDescCreate = descCreate,
        .cublasLtMatmulDescDestroy = descDestroy,
        .cublasLtMatmulDescSetAttribute = descSet,
        .cublasLtMatrixLayoutCreate = layoutCreate,
        .cublasLtMatrixLayoutDestroy = layoutDestroy,
        .cublasLtMatrixLayoutSetAttribute = layoutSet,
        .cublasLtMatmulAlgoInit = algoInit,
        .cublasLtMatmulAlgoConfigSetAttribute = algoSet,
        .cublasLtMatmulAlgoConfigGetAttribute = algoGet,
        .cublasLtMatmulAlgoCheck = algoCheck,
        .cublasLtMatmul = matmul,
    };
};

/// Test ids only (slots in the fake's eight algo words), not cuBLASLt's values. Inner and cluster shape share a slot;
/// each is read back right after its own set.
const test_ids: Constants = .{
    .desc_sm_count_target = 100,
    .algo_id = 0,
    .algo_tile_id = 1,
    .algo_stages_id = 2,
    .algo_splitk_num = 3,
    .algo_reduction_scheme = 4,
    .algo_cta_swizzling = 5,
    .algo_custom_option = 6,
    .algo_inner_shape_id = 7,
    .algo_cluster_shape_id = 7,
};

fn scores() GemmSpec {
    return .{
        .m = 218,
        .n = 218,
        .k = 64,
        .transa = .t,
        .transb = .n,
        .lda = 64,
        .ldb = 64,
        .ldc = 218,
        .sm_count_target = 48,
        .algo = .{ .algo_id = 20, .tile = 15, .stages = 28, .splitk = 12, .reduction = 2, .cta_swizzling = 0, .custom_option = 0, .inner_shape = 0, .cluster_shape = 0, .workspace = 2281152 },
    };
}

test "a fully recorded plan builds, runs only with its workspace, and is released only with quiescence" {
    var deps: u32 = 0;
    const lt = try Lt.fromApi(F.api, 130101, &deps);
    var h = try lt.create();
    F.created = 0;
    F.destroyed = 0;
    var r: ?InitRetained = null;
    var p = try Plan.init(&lt, &h, scores(), test_ids, &r);
    try testing.expectEqual(@as(u32, 2), deps); // the handle and the plan
    try p.run(1, 2, 3, 4, 2281152, null);
    try testing.expectError(error.LtWorkspaceMismatch, p.run(1, 2, 3, 4, 2281151, null));

    var timed_out = quiesce.testProof();
    timed_out.obs.outcome = .timed_out;
    try testing.expectError(error.LtNotQuiesced, p.deinit(timed_out));
    try testing.expectError(error.LtNotQuiesced, lt.destroy(&h, timed_out));
    try testing.expectEqual(@as(usize, 0), F.destroyed);

    try p.deinit(quiesce.testProof());
    try testing.expectEqual(F.created, F.destroyed);
    try testing.expectError(error.LtReleased, p.deinit(quiesce.testProof())); // released once
    try testing.expectError(error.LtReleased, p.run(1, 2, 3, 4, 2281152, null));
    try lt.destroy(&h, quiesce.testProof());
    try testing.expectEqual(@as(u32, 0), deps);
}

test "a failed destroy is sticky, keeps its raw status, and keeps the plan held" {
    var deps: u32 = 0;
    const lt = try Lt.fromApi(F.api, 130101, &deps);
    var h = try lt.create();
    var r: ?InitRetained = null;
    var p = try Plan.init(&lt, &h, scores(), test_ids, &r);
    F.layout_destroy_rc = 14; // CUBLAS_STATUS_INTERNAL_ERROR
    try testing.expectError(error.LtReleaseFailed, p.deinit(quiesce.testProof()));
    try testing.expectEqual(@as(S, 14), p.release_status);
    try testing.expect(p.a != null and p.b != null and p.c != null); // still owned
    F.layout_destroy_rc = 0;
    try testing.expectError(error.LtReleaseFailed, p.deinit(quiesce.testProof())); // no second attempt
    try testing.expectEqual(@as(u32, 2), deps); // the set cannot be unloaded under it
    try lt.destroy(&h, quiesce.testProof());
}

test "the version, unpinned constants, unrecorded fields, read-back and workspace differences refuse" {
    var deps: u32 = 0;
    try testing.expectError(error.LtVersionMismatch, Lt.fromApi(F.api, 130100, &deps));
    const lt = try Lt.fromApi(F.api, 130101, &deps);
    var h = try lt.create();
    var r: ?InitRetained = null;
    // the shipped constants are not cited yet (G1)
    try testing.expectError(error.LtConstantUnpinned, Plan.init(&lt, &h, scores(), constants, &r));
    var s = scores();
    s.algo.cta_swizzling = null; // G2
    try testing.expectError(error.LtAlgoUnrecorded, Plan.init(&lt, &h, s, test_ids, &r));
    s = scores();
    s.sm_count_target = null;
    try testing.expectError(error.LtAlgoUnrecorded, Plan.init(&lt, &h, s, test_ids, &r));

    F.created = 0;
    F.destroyed = 0;
    F.skew_attr = 1; // the tile reads back different
    try testing.expectError(error.LtReadBackMismatch, Plan.init(&lt, &h, scores(), test_ids, &r));
    F.skew_attr = -1;
    try testing.expectEqual(F.created, F.destroyed);

    F.check_workspace = 0;
    defer F.check_workspace = 2281152;
    try testing.expectError(error.LtWorkspaceMismatch, Plan.init(&lt, &h, scores(), test_ids, &r));
    try testing.expectEqual(F.created, F.destroyed);
    try testing.expect(r == null); // every cleanup succeeded: nothing retained
    try testing.expectEqual(@as(u32, 1), deps); // only the handle
    try lt.destroy(&h, quiesce.testProof());
}

test "a refused plan whose cleanup fails retains the exact objects, both statuses and its hold" {
    var deps: u32 = 0;
    const lt = try Lt.fromApi(F.api, 130101, &deps);
    var h = try lt.create();
    var r: ?InitRetained = null;
    F.layout_set_rc = 7; // the first layout's attribute set refuses after its create
    F.layout_destroy_rc = 14; // and the plan's cleanup cannot destroy the layouts
    defer {
        F.layout_set_rc = 0;
        F.layout_destroy_rc = 0;
    }
    try testing.expectError(error.LtFailed, Plan.init(&lt, &h, scores(), test_ids, &r));
    const kept = &r.?;
    try testing.expectEqual(Error.LtFailed, kept.primary);
    try testing.expectEqual(@as(c_int, 14), kept.cleanup.rc);
    try testing.expectEqual(PlanState.release_failed, kept.held.state);
    try testing.expectEqual(@as(S, 14), kept.held.release_status);
    try testing.expect(kept.held.a != null); // the exact layout created before the refusal, still owned
    try testing.expect(kept.held.desc == null); // the descriptor destroyed once
    try testing.expectEqual(@as(u32, 2), deps); // the handle and the retained plan
    // not a release path: no retry
    F.layout_destroy_rc = 0;
    const before = F.destroyed;
    try testing.expectError(error.LtReleaseFailed, kept.held.deinit(quiesce.testProof()));
    try testing.expectEqual(before, F.destroyed);
    try testing.expectError(error.LtReleaseFailed, kept.held.run(1, 2, 3, 4, 2281152, null));
    try testing.expectEqual(@as(u32, 2), deps);
    try lt.destroy(&h, quiesce.testProof());
}

test "a handle is destroyed once: repeats make no vendor call and drop nothing; plans refuse it" {
    var deps: u32 = 0;
    const lt = try Lt.fromApi(F.api, 130101, &deps);
    var r: ?InitRetained = null;

    // released: a second destroy refuses, and a plan on it refuses to build or run
    var h = try lt.create();
    var p = try Plan.init(&lt, &h, scores(), test_ids, &r);
    try p.deinit(quiesce.testProof());
    F.handle_destroys = 0;
    try lt.destroy(&h, quiesce.testProof());
    try testing.expectEqual(@as(u32, 0), deps);
    try testing.expectError(error.LtReleased, lt.destroy(&h, quiesce.testProof()));
    try testing.expectEqual(@as(usize, 1), F.handle_destroys);
    try testing.expectEqual(@as(u32, 0), deps);
    try testing.expectError(error.LtReleased, Plan.init(&lt, &h, scores(), test_ids, &r));

    // failed: the raw status is kept, the handle stays held, a repeat makes no call
    var g = try lt.create();
    var live = try Plan.init(&lt, &g, scores(), test_ids, &r);
    F.handle_destroy_rc = 14;
    defer F.handle_destroy_rc = 0;
    F.handle_destroys = 0;
    try testing.expectError(error.LtReleaseFailed, lt.destroy(&g, quiesce.testProof()));
    try testing.expectEqual(State.release_failed, g.state);
    try testing.expectEqual(@as(S, 14), g.release_status);
    F.handle_destroy_rc = 0;
    try testing.expectError(error.LtReleaseFailed, lt.destroy(&g, quiesce.testProof()));
    try testing.expectEqual(@as(usize, 1), F.handle_destroys);
    try testing.expectEqual(@as(u32, 2), deps); // the failed handle and the live plan
    try testing.expectError(error.LtReleaseFailed, live.run(1, 2, 3, 4, 2281152, null)); // not against a failed handle
    try testing.expectError(error.LtReleaseFailed, Plan.init(&lt, &g, scores(), test_ids, &r));
    try live.deinit(quiesce.testProof());
    try testing.expectEqual(@as(u32, 1), deps); // the failed handle is never dropped
}
