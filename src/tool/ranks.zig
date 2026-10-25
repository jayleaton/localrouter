//! Rank 0 of a tensor-parallel group (`topology.Group`): one model split across several nodes, each running one rank of
//! its server. On rank 0's node the tool is a `RankTool` around the ordinary worker (`ProcessTool`, whose server gets
//! its rank through `options.rank_env`). Loading it starts every other rank first, through its node's daemon
//! (POST /v1/tools/{id}/load there, where that node admits its rank against its own memory and GPU lock), then rank 0,
//! which is ready when its server answers: every rank has joined. A rank that dies stops the others (the engines'
//! contract: TensorFold's ranks keep a line to rank 0 and stop together), so any failure ends with every rank unloaded,
//! and the next load starts the whole group again. Requests go to rank 0 only.

const std = @import("std");
const tool = @import("tool.zig");
const process = @import("process.zig");
const node = @import("node.zig");
const Request = @import("request.zig").Request;
const ToolConfig = @import("../config.zig").ToolConfig;

pub const RankTool = struct {
    cluster: *node.Cluster,
    rank0: *process.ProcessTool, // rank 0, here
    followers: []const usize, // the other ranks' nodes, in rank order
    fail_msg: [2048]u8 = undefined,
    fail: ?tool.Failure = null, // a failure of the group rather than of rank 0's own worker

    pub fn asTool(r: *RankTool) tool.Tool {
        return .{ .ptr = r, .vtable = &.{
            .config = config_,
            .needs = needs,
            .load = load,
            .generate = generate,
            .unload = unload,
            .abort = abort,
            .cancel = cancel,
            .state = state,
            .failure = failure,
        } };
    }

    fn self(ptr: *anyopaque) *RankTool {
        return @ptrCast(@alignCast(ptr));
    }

    fn inner(r: *RankTool) tool.Tool {
        return r.rank0.asTool();
    }

    fn config_(ptr: *anyopaque) *const ToolConfig {
        return self(ptr).rank0.cfg;
    }

    fn needs(ptr: *anyopaque, req: *const Request) tool.Needs {
        return self(ptr).inner().needs(req); // this node's rank; every other node admits its own
    }

    fn state(ptr: *anyopaque) tool.State {
        return self(ptr).inner().state();
    }

    fn failure(ptr: *anyopaque) tool.Failure {
        const r = self(ptr);
        return r.fail orelse r.inner().failure();
    }

    fn abort(ptr: *anyopaque) void {
        self(ptr).inner().abort();
    }

    fn cancel(ptr: *anyopaque) bool {
        return self(ptr).inner().cancel();
    }

    /// The other ranks first (each restarted: a rank left from a group that broke would wait for a rank 0 that is gone),
    /// then rank 0.
    fn load(ptr: *anyopaque) tool.Error!u64 {
        const r = self(ptr);
        r.fail = null;
        const id = r.rank0.cfg.id;
        var arena: std.heap.ArenaAllocator = .init(r.cluster.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        for (r.followers, 0..) |n, k| {
            r.control(a, n, "unload");
            const path = std.fmt.allocPrint(a, "/v1/tools/{s}/load", .{id}) catch return tool.Error.OutOfMemory;
            const reply = r.cluster.request(a, n, "POST", path, "{}", r.rank0.cfg.load_timeout_s) catch |err| {
                r.stopFollowers(a, k);
                return r.failed("rank {d} of '{s}' is on node {s}, which did not answer: {s}", .{ k + 1, id, r.cluster.peers[n].node.id, @errorName(err) });
            };
            if (reply.status != 200) {
                r.stopFollowers(a, k);
                return r.failed("node {s} could not start rank {d} of '{s}': {s}", .{ r.cluster.peers[n].node.id, k + 1, id, @import("upstream.zig").errorMessage(a, reply.body) });
            }
        }
        const resident = r.inner().load() catch |err| {
            r.stopFollowers(a, r.followers.len);
            return err;
        };
        std.log.info("{s}: {d} ranks up", .{ id, r.followers.len + 1 });
        return resident;
    }

    fn generate(ptr: *anyopaque, job: *const tool.Job, sink: tool.Sink, arena: std.mem.Allocator) tool.Error!tool.Output {
        const r = self(ptr);
        const out = r.inner().generate(job, sink, arena) catch |err| {
            if (r.inner().state() == .unloaded) r.stopFollowers(arena, r.followers.len); // rank 0 is gone, so the group is
            return err;
        };
        return out;
    }

    fn unload(ptr: *anyopaque) void {
        const r = self(ptr);
        r.inner().unload();
        var arena: std.heap.ArenaAllocator = .init(r.cluster.gpa);
        defer arena.deinit();
        r.stopFollowers(arena.allocator(), r.followers.len);
    }

    /// Unloads the first `n` other ranks (best effort: a node that is down has nothing loaded to stop).
    fn stopFollowers(r: *RankTool, a: std.mem.Allocator, n: usize) void {
        for (r.followers[0..n]) |f| r.control(a, f, "unload");
    }

    fn control(r: *RankTool, a: std.mem.Allocator, n: usize, verb: []const u8) void {
        const path = std.fmt.allocPrint(a, "/v1/tools/{s}/{s}", .{ r.rank0.cfg.id, verb }) catch return;
        _ = r.cluster.request(a, n, "POST", path, "{}", 60) catch {};
    }

    fn failed(r: *RankTool, comptime fmt: []const u8, args: anytype) tool.Error {
        r.fail = .{ .message = std.fmt.bufPrint(&r.fail_msg, fmt, args) catch r.fail_msg[0..], .type = "server_error" };
        return tool.Error.LoadFailed;
    }
};
