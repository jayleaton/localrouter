//! The worker-side contract every compiled-in engine implements, and the table of engines in this binary.
//! `needs` is a pure function of the config and request, so the daemon prices a request without starting a worker.

const std = @import("std");
const tool = @import("../tool/tool.zig");
const request = @import("../tool/request.zig");
const Request = request.Request;
pub const Capability = request.Capability;
const ToolConfig = @import("../config.zig").ToolConfig;

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
};

pub const entries = [_]Entry{
    @import("../engines/testpattern.zig").entry,
    @import("../engines/qwen_image_tool.zig").entry,
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
