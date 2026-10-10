//! The daemon's configuration: where it listens and stores outputs, the memory budget, and the tools it serves.

const std = @import("std");
const Kind = @import("tool/request.zig").Kind;
const Capability = @import("tool/request.zig").Capability;
const bind = @import("bind.zig");

pub const gib: u64 = 1 << 30;
pub const mib: u64 = 1 << 20;

/// One tool. `engine` names a compiled-in engine; `cmd` runs any program that speaks the worker protocol instead.
pub const ToolConfig = struct {
    id: []const u8, // the OpenAI model id
    kind: Kind,
    name: []const u8 = "",
    engine: []const u8 = "",
    cmd: []const []const u8 = &.{},
    weights: []const u8 = "",
    resident_bytes: u64 = 0, // estimate for `cmd` tools; engines compute their own
    working_bytes: u64 = 0, // per request, for `cmd` tools
    idle_ttl_s: u32 = 120, // 0: unload right after each job
    priority: i32 = 0, // higher is kept longer: memory pressure unloads lower priorities first (then least recently used)
    keep_loaded: bool = false, // load at start if it fits, skip the idle TTL, reload when memory frees; still evictable
    capabilities: []const Capability = &.{}, // empty: the engine's (a `cmd` tool: plain text to image / video)
    load_timeout_s: u32 = 900,
    generate_timeout_s: u32 = 3600,
    options: ?std.json.Value = null, // engine-specific, passed through to the worker
};

pub const Config = struct {
    host: []const u8 = bind.default_host, // an IP, localhost, all (0.0.0.0) or tailscale (this machine's tailnet IPv4)
    port: u16 = 8190,
    data_dir: []const u8 = "/data",
    machine: []const u8 = "",
    budget_bytes: u64 = 0, // cap on loaded + working bytes; 0: only MemAvailable - reserve
    reserve_bytes: u64 = 8 * gib, // kept free for the OS and co-resident services
    keep_outputs_s: u32 = 24 * 3600,
    max_queue: u32 = 64,
    allow_private_urls: bool = false, // URL inputs (edits, image to video) may name loopback / private / link-local hosts
    defaults: Defaults = .{}, // the model a request that names none goes to, per capability
    tools: []const ToolConfig = &default_tools,

    pub fn tool(c: *const Config, id: []const u8) ?*const ToolConfig {
        for (c.tools) |*t| if (std.mem.eql(u8, t.id, id)) return t;
        return null;
    }
};

/// A tool id per capability, `{"text_to_image": "qwen-image-2.1-turbo"}`: requests that name no model use it. Unset:
/// the first tool (in config order) that has the capability. `serve` checks each names a tool that can do it.
pub const Defaults = struct {
    text_to_image: ?[]const u8 = null,
    image_edit: ?[]const u8 = null,
    text_to_video: ?[]const u8 = null,
    image_to_video: ?[]const u8 = null,

    pub fn get(d: Defaults, cap: Capability) ?[]const u8 {
        return switch (cap) {
            inline else => |c| @field(d, @tagName(c)),
        };
    }
};

/// With no config file LocalRouter serves its self-test tool only.
pub const default_tools = [_]ToolConfig{
    .{ .id = "testpattern", .kind = .image, .name = "Test pattern (self-test)", .engine = "testpattern", .idle_ttl_s = 30 },
    .{ .id = "testpattern-video", .kind = .video, .name = "Test pattern video (self-test)", .engine = "testpattern", .idle_ttl_s = 30 },
};

/// Parses a JSON config; the result lives in `arena`.
pub fn parse(arena: std.mem.Allocator, text: []const u8) !Config {
    return std.json.parseFromSliceLeaky(Config, arena, text, .{ .ignore_unknown_fields = false, .allocate = .alloc_always });
}

test "parse a config with an external tool" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try parse(arena.allocator(),
        \\{"port": 9000, "data_dir": "/tmp/x", "tools": [
        \\  {"id": "qwen-image-2.1", "kind": "image", "engine": "qwen_image", "weights": "/models/qwen", "options": {"precision": "nvfp4"}},
        \\  {"id": "py-tts", "kind": "video", "cmd": ["python3", "-m", "tts"], "resident_bytes": 1000}
        \\]}
    );
    try std.testing.expectEqual(@as(u16, 9000), c.port);
    try std.testing.expectEqualStrings("nvfp4", c.tool("qwen-image-2.1").?.options.?.object.get("precision").?.string);
    try std.testing.expectEqual(@as(usize, 3), c.tool("py-tts").?.cmd.len);
    try std.testing.expect(c.tool("nope") == null);
    try std.testing.expect(c.defaults.get(.text_to_image) == null);
}

test "defaults name a model per capability; unknown capabilities are rejected" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const c = try parse(arena.allocator(),
        \\{"defaults": {"text_to_image": "turbo", "image_to_video": "v"}}
    );
    try std.testing.expectEqualStrings("turbo", c.defaults.get(.text_to_image).?);
    try std.testing.expectEqualStrings("v", c.defaults.get(.image_to_video).?);
    try std.testing.expect(c.defaults.get(.image_edit) == null);
    try std.testing.expectError(error.UnknownField, parse(arena.allocator(), "{\"defaults\": {\"text_to_speech\": \"x\"}}"));
}

test "the default host is loopback; host accepts the keywords" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const d = try parse(arena.allocator(), "{}");
    try std.testing.expectEqualStrings("127.0.0.1", d.host);
    const t = try parse(arena.allocator(), "{\"host\": \"tailscale\"}");
    try std.testing.expectEqualStrings("tailscale", t.host);
}
