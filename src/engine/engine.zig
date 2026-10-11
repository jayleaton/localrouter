//! The worker-side contract every compiled-in engine implements, and the table of engines in this binary.
//! `needs` is a pure function of the config and request, so the daemon prices a request without starting a worker.

const std = @import("std");
const tool = @import("../tool/tool.zig");
const request = @import("../tool/request.zig");
const Request = request.Request;
pub const Capability = request.Capability;
const config = @import("../config.zig");
const ToolConfig = config.ToolConfig;

pub const Needs = tool.Needs;
pub const Job = tool.Job;
pub const Sink = tool.Sink;
pub const Output = tool.Output;

/// Raised by engines for a request they refuse (wrong size for the model, missing reference): the API's 400.
pub const Refused = error{Refused};

pub const Engine = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        load: *const fn (ptr: *anyopaque) anyerror!u64,
        generate: *const fn (ptr: *anyopaque, job: *const Job, sink: Sink, arena: std.mem.Allocator) anyerror!Output,
        unload: *const fn (ptr: *anyopaque) void,
    };

    /// Loads weights and kernels; returns the resident bytes it now holds.
    pub fn load(e: Engine) anyerror!u64 {
        return e.vtable.load(e.ptr);
    }
    /// Runs one request into `job.dir`; result strings live in `arena`.
    pub fn generate(e: Engine, job: *const Job, sink: Sink, arena: std.mem.Allocator) anyerror!Output {
        return e.vtable.generate(e.ptr, job, sink, arena);
    }
    /// Frees everything `load` took and the engine itself.
    pub fn unload(e: Engine) void {
        e.vtable.unload(e.ptr);
    }
};

/// What an engine needs at run time besides its config.
pub const Env = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
};

pub const Entry = struct {
    name: []const u8,
    capabilities: []const Capability, // what the engine can do; a tool's `kind` picks its share
    needs: *const fn (cfg: *const ToolConfig, req: *const Request) Needs,
    create: *const fn (env: Env, cfg: *const ToolConfig) anyerror!Engine,
    /// Refuses a request this model cannot serve (a step count its schedule lacks, a size past its buffers) before it
    /// is queued or loaded, writing why to `why`. Pure and cheap, like `needs`. Null: whatever `request.validate` passes.
    check: ?*const fn (cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool = null,
};

pub const entries = [_]Entry{
    @import("../engines/testpattern.zig").entry,
    @import("../engines/qwen_image_tool.zig").entry,
    @import("../engines/qwen_image_tool.zig").turbo_entry,
    @import("../engines/minimax_h3_tool.zig").entry,
};

/// Whether the tool can serve `cap`: its configured capabilities, else its engine's, else plain text to image / video;
/// always within the tool's kind.
pub fn supports(cfg: *const ToolConfig, cap: Capability) bool {
    if (cap.kind() != cfg.kind) return false;
    const base: []const Capability = if (cfg.capabilities.len > 0) cfg.capabilities else if (find(cfg.engine)) |e| e.capabilities else &.{ .text_to_image, .text_to_video };
    return std.mem.indexOfScalar(Capability, base, cap) != null;
}

/// The tool's capabilities (see `supports`), allocated in `a`.
pub fn capabilityList(a: std.mem.Allocator, cfg: *const ToolConfig) ![]const Capability {
    var list: std.ArrayList(Capability) = .empty;
    for (std.enums.values(Capability)) |c| if (supports(cfg, c)) try list.append(a, c);
    return list.items;
}

/// Whether the tool's engine takes `req` (see `Entry.check`); a `cmd` tool takes anything valid.
pub fn check(cfg: *const ToolConfig, req: *const Request, why: *std.Io.Writer) bool {
    const e = find(cfg.engine) orelse return true;
    const f = e.check orelse return true;
    return f(cfg, req, why);
}

/// Whether each configured default (`Config.defaults`) names a tool that has its capability; the first that does not
/// is written to `why`.
pub fn checkDefaults(c: *const config.Config, why: *std.Io.Writer) bool {
    for (std.enums.values(Capability)) |cap| {
        const id = c.defaults.get(cap) orelse continue;
        const t = c.tool(id) orelse {
            why.print("defaults.{s} names '{s}', which is not a configured tool", .{ @tagName(cap), id }) catch {};
            return false;
        };
        if (!supports(t, cap)) {
            why.print("defaults.{s} names '{s}', which cannot do {s}", .{ @tagName(cap), id, @tagName(cap) }) catch {};
            return false;
        }
    }
    return true;
}

test "defaults must name a tool that has the capability" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const tools =
        \\"tools": [{"id": "img", "kind": "image", "engine": "testpattern", "capabilities": ["text_to_image"]},
        \\          {"id": "vid", "kind": "video", "engine": "testpattern"}]
    ;
    var buf: [256]u8 = undefined;
    var why: std.Io.Writer = .fixed(&buf);
    try std.testing.expect(checkDefaults(&try config.parse(arena.allocator(), "{\"defaults\": {\"text_to_image\": \"img\", \"image_to_video\": \"vid\"}," ++ tools ++ "}"), &why));
    try std.testing.expect(!checkDefaults(&try config.parse(arena.allocator(), "{\"defaults\": {\"image_edit\": \"img\"}," ++ tools ++ "}"), &why));
    try std.testing.expectEqualStrings("defaults.image_edit names 'img', which cannot do image_edit", why.buffered());
    why = .fixed(&buf);
    try std.testing.expect(!checkDefaults(&try config.parse(arena.allocator(), "{\"defaults\": {\"text_to_image\": \"vid\"}," ++ tools ++ "}"), &why));
    why = .fixed(&buf);
    try std.testing.expect(!checkDefaults(&try config.parse(arena.allocator(), "{\"defaults\": {\"text_to_image\": \"nope\"}," ++ tools ++ "}"), &why));
    try std.testing.expectEqualStrings("defaults.text_to_image names 'nope', which is not a configured tool", why.buffered());
}

pub fn find(name: []const u8) ?*const Entry {
    for (&entries) |*e| if (std.mem.eql(u8, e.name, name)) return e;
    return null;
}

/// A u64 option from the tool's `options` object, else `default`.
pub fn optionInt(cfg: *const ToolConfig, key: []const u8, default: u64) u64 {
    const opts = cfg.options orelse return default;
    if (opts != .object) return default;
    const v = opts.object.get(key) orelse return default;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else default,
        else => default,
    };
}

/// A string option from the tool's `options` object, else `default`.
pub fn optionString(cfg: *const ToolConfig, key: []const u8, default: []const u8) []const u8 {
    const opts = cfg.options orelse return default;
    if (opts != .object) return default;
    const v = opts.object.get(key) orelse return default;
    return if (v == .string) v.string else default;
}
