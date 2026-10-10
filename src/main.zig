//! localrouter: the one binary.
//!   localrouter serve [--config FILE] [--host H] [--port N] [--data DIR]   the daemon (API + scheduler); H: an IP, localhost (default 127.0.0.1), all, tailscale
//!   localrouter worker                                                     one tool's worker (LOCALROUTER_TOOL holds its config)
//!   localrouter gen image|video ...                                        the client (see cli/gen.zig)
//!   localrouter models [--url URL]                                         the server's models: kind, capabilities, defaults
//!   localrouter mcp-stdio [--url URL]                                      /mcp over stdio, for clients without HTTP MCP
//!   localrouter check gpu                                                  driver and memory probe

const std = @import("std");
const Io = std.Io;
const config = @import("config.zig");
const worker = @import("engine/worker.zig");
const serve = @import("serve.zig");
const gen = @import("cli/gen.zig");
const check = @import("cli/check.zig");
const mcp_stdio = @import("cli/mcp_stdio.zig");

pub const std_options: std.Options = .{ .log_level = .info };

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) return usage();
    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "worker")) {
        const json = init.environ_map.get("LOCALROUTER_TOOL") orelse {
            std.log.err("localrouter worker needs LOCALROUTER_TOOL (the tool's config as JSON)", .{});
            return 2;
        };
        const parent = std.fmt.parseInt(i32, init.environ_map.get("LOCALROUTER_PARENT") orelse "0", 10) catch 0;
        return worker.run(gpa, io, json, parent);
    }
    if (std.mem.eql(u8, cmd, "serve")) return serve.run(gpa, io, arena, args[2..], init.environ_map);
    if (std.mem.eql(u8, cmd, "gen")) return gen.run(gpa, io, arena, args[2..], init.environ_map);
    if (std.mem.eql(u8, cmd, "models")) return gen.models(gpa, io, arena, args[2..], init.environ_map);
    if (std.mem.eql(u8, cmd, "mcp-stdio")) return mcp_stdio.run(gpa, io, args[2..], init.environ_map);
    if (std.mem.eql(u8, cmd, "check")) return check.run(io, gpa, args[2..]);
    return usage();
}

fn usage() u8 {
    std.debug.print(
        \\usage:
        \\  localrouter serve [--config FILE] [--host H] [--port N] [--data DIR]
        \\  localrouter gen image PROMPT [-o FILE] [--size WxH] [--seed N] [--steps N] [--model ID] [--url URL]
        \\  localrouter gen video PROMPT [-o FILE] [--size WxH] [--seconds N] [--seed N] [--model ID] [--url URL]
        \\  localrouter models [--url URL]
        \\  localrouter mcp-stdio [--url URL]
        \\  localrouter check gpu | selftest | health [URL]
        \\
    , .{});
    return 2;
}

test {
    _ = @import("config.zig");
    _ = @import("bind.zig");
    _ = @import("tool/request.zig");
    _ = @import("tool/wire.zig");
    _ = @import("tool/process.zig");
    _ = @import("sched/memory.zig");
    _ = @import("sched/jobs.zig");
    _ = @import("sched/scheduler.zig");
    _ = @import("api/routes.zig");
    _ = @import("media/png.zig");
    _ = @import("media/mp4.zig");
    _ = @import("engine/engine.zig");
    _ = @import("engines/testpattern.zig");
    _ = @import("engines/qwen_image_tool.zig");
    _ = @import("serve.zig");
    _ = @import("cli/gen.zig");
    _ = @import("cli/check.zig");
    _ = @import("cli/mcp_stdio.zig");
    _ = @import("api/inputs.zig");
}
