//! localrouter: the one binary.
//!   localrouter serve [--config FILE] [--host H] [--port N] [--data DIR]   the daemon (API + scheduler); H: an IP, localhost (default 127.0.0.1), all, tailscale
//!   localrouter worker                                                     one tool's worker (LOCALROUTER_TOOL holds its config)
//!   localrouter gen image|video ...                                        the client (see cli/gen.zig)
//!   localrouter models [--url URL]                                         the server's models: kind, capabilities, defaults
//!   localrouter mcp-stdio [--url URL] [--token T]                          /mcp over stdio, for clients without HTTP MCP
//!   localrouter pair add|token|list|revoke ...                             paired devices (see cli/pair.zig)
//!   localrouter check gpu                                                  driver and memory probe

const std = @import("std");
const Io = std.Io;
const config = @import("config.zig");
const worker = @import("engine/worker.zig");
const serve = @import("serve.zig");
const gen = @import("cli/gen.zig");
const check = @import("cli/check.zig");
const mcp_stdio = @import("cli/mcp_stdio.zig");
const pair = @import("cli/pair.zig");

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
        return worker.run(gpa, io, json, parent, init.environ_map);
    }
    if (std.mem.eql(u8, cmd, "serve")) return serve.run(gpa, io, arena, args[2..], init.environ_map);
    if (std.mem.eql(u8, cmd, "gen")) return gen.run(gpa, io, arena, args[2..], init.environ_map);
    if (std.mem.eql(u8, cmd, "models")) return gen.models(gpa, io, arena, args[2..], init.environ_map);
    if (std.mem.eql(u8, cmd, "mcp-stdio")) return mcp_stdio.run(gpa, io, args[2..], init.environ_map);
    if (std.mem.eql(u8, cmd, "pair")) return pair.run(io, arena, args[2..]);
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
        \\  localrouter mcp-stdio [--url URL] [--token T]
        \\  localrouter pair add NAME [--config FILE] [--data DIR] [--url URL]
        \\  localrouter pair token NAME [--config FILE] [--data DIR]
        \\  localrouter pair list [--config FILE] [--data DIR]
        \\  localrouter pair revoke ID|NAME [--config FILE] [--data DIR]
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
    _ = @import("api/http.zig");
    _ = @import("api/routes.zig");
    _ = @import("api/capabilities.zig");
    _ = @import("media/png.zig");
    _ = @import("media/png_decode.zig");
    _ = @import("media/image_in.zig");
    _ = @import("media/mp4.zig");
    _ = @import("media/marker.zig");
    _ = @import("engine/engine.zig");
    _ = @import("engine/memcap_coverage.zig");
    _ = @import("engines/testpattern.zig");
    _ = @import("engines/qwen_image_tool.zig");
    _ = @import("engines/minimax_h3_tool.zig");
    _ = @import("sched/speech_gate.zig");
    _ = @import("serve.zig");
    _ = @import("cli/gen.zig");
    _ = @import("cli/check.zig");
    _ = @import("cli/mcp_stdio.zig");
    _ = @import("cli/pair.zig");
    _ = @import("pair.zig");
    _ = @import("api/pair.zig");
    _ = @import("api/inputs.zig");
    _ = @import("api/speech.zig");
    _ = @import("api/chat.zig");
    _ = @import("api/budget.zig");
    _ = @import("api/responses.zig");
    _ = @import("api/decide.zig");
    _ = @import("tool/systemone.zig");
    _ = @import("tool/upstream.zig");
    _ = @import("engines/openai_server.zig");
    _ = @import("media/audio.zig");
    _ = @import("media/voiceprep.zig");
    _ = @import("voices.zig");
    _ = @import("api/voices.zig");
    _ = @import("api/realtime.zig");
    _ = @import("api/voice_local.zig");
    _ = @import("voice/ws.zig");
    _ = @import("voice/vad.zig");
    _ = @import("voice/resample.zig");
    _ = @import("voice/chunker.zig");
    _ = @import("voice/chat.zig");
    _ = @import("voice/stop.zig");
    _ = @import("voice/events.zig");
    _ = @import("voice/session.zig");
    _ = @import("voice/respond.zig");
    _ = @import("voice/tools.zig");
    _ = @import("voice/timing.zig");
    _ = @import("voice/session_test.zig");
    _ = @import("computer/policy.zig");
    _ = @import("computer/approvals.zig");
    _ = @import("computer/decide.zig");
    _ = @import("computer/audit.zig");
    _ = @import("computer/gate.zig");
    _ = @import("computer/hub.zig");
    _ = @import("api/desktop.zig");
}
