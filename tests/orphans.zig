//! A daemon killed outright (SIGKILL: no handler runs) leaves nothing behind: its compiled-in workers, its `cmd` workers
//! and the model servers they started all exit. Linux has PR_SET_PDEATHSIG for this; macOS has the kqueue watch and
//! `localrouter guard` (src/engine/orphan.zig).

const std = @import("std");
const opts = @import("build_options");
const harness = @import("harness.zig");
const testing = std.testing;
const proc = harness.proc;

test "orphans: killing the daemon takes its workers and their servers with it" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = try std.fmt.allocPrint(a, "{{" ++ harness.reserve ++ ", \"tools\": [" ++
        "{{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\", \"keep_loaded\": true}}," ++
        "{{\"id\": \"chat\", \"kind\": \"chat\", \"engine\": \"openai_server\", \"keep_loaded\": true, \"resident_bytes\": 41943040, \"options\": {{\"server\": [\"python3\", \"{s}\", \"--port\", \"{{port}}\", \"--name\", \"{{id}}\"]}}}}," ++
        "{{\"id\": \"py\", \"kind\": \"speech\", \"cmd\": [\"python3\", \"{s}\"], \"keep_loaded\": true, \"resident_bytes\": 8388608}}]}}", .{ opts.chat_server, opts.speech_worker });
    var d = try harness.Daemon.start(testing.allocator, io, "orphans", cfg);
    defer d.stop();
    for ([_][]const u8{ "tp", "chat", "py" }) |id| try waitReady(&d, a, id);

    // every process under the daemon: three workers, the chat server under its worker
    var tree: std.ArrayList(i32) = .empty;
    try tree.append(a, d.child.id.?);
    var i: usize = 0;
    while (i < tree.items.len) : (i += 1) try tree.appendSlice(a, try proc.children(a, io, tree.items[i]));
    const below = tree.items[1..];
    std.debug.print("\norphans: {d} processes under the daemon before it is killed: {any}\n", .{ below.len, below });
    try testing.expect(below.len >= 4);

    _ = std.posix.system.kill(d.child.id.?, std.posix.SIG.KILL);
    var left: usize = below.len;
    for (0..100) |_| { // 10 s
        left = 0;
        for (below) |pid| {
            if (std.posix.system.kill(pid, @fromBackingInt(@intCast(0))) == 0) left += 1;
        }
        if (left == 0) break;
        harness.sleepMs(io, 100);
    }
    if (left > 0) for (below) |pid| if (std.posix.system.kill(pid, @fromBackingInt(@intCast(0))) == 0) {
        var buf: [256]u8 = undefined;
        std.debug.print("orphan left: {d} {s}\n", .{ pid, proc.name(a, pid, &buf) });
        _ = std.posix.system.kill(pid, std.posix.SIG.KILL);
    };
    try testing.expectEqual(@as(usize, 0), left);
}

fn waitReady(d: *harness.Daemon, a: std.mem.Allocator, id: []const u8) !void {
    for (0..300) |_| {
        if (std.mem.eql(u8, try d.toolState(a, id), "ready")) return;
        harness.sleepMs(d.io, 100);
    }
    return error.NotReady;
}

test "orphans: SIGTERM stops the daemon (exit 0 within 5 s) and its workers with it" {
    const io = testing.io;
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg = try std.fmt.allocPrint(a, "{{" ++ harness.reserve ++ ", \"tools\": [" ++
        "{{\"id\": \"tp\", \"kind\": \"image\", \"engine\": \"testpattern\", \"keep_loaded\": true}}," ++
        "{{\"id\": \"py\", \"kind\": \"speech\", \"cmd\": [\"python3\", \"{s}\"], \"keep_loaded\": true, \"resident_bytes\": 8388608}}]}}", .{opts.speech_worker});
    var d = try harness.Daemon.start(testing.allocator, io, "sigterm", cfg);
    defer { // not `stop`: the daemon is reaped here
        d.client.deinit();
        std.Io.Dir.cwd().deleteTree(io, d.dir) catch {};
        testing.allocator.free(d.dir);
    }
    for ([_][]const u8{ "tp", "py" }) |id| try waitReady(&d, a, id);
    const pid = d.child.id.?;
    const below = try proc.children(a, io, pid);
    try testing.expect(below.len >= 2);

    _ = std.posix.system.kill(pid, std.posix.SIG.TERM);
    var status: c_int = 0;
    const exited = for (0..50) |_| { // 5 s
        if (std.c.waitpid(pid, &status, std.c.W.NOHANG) == pid) break true;
        harness.sleepMs(io, 100);
    } else false;
    if (!exited) {
        _ = std.posix.system.kill(pid, std.posix.SIG.KILL);
        _ = std.c.waitpid(pid, &status, 0);
    }
    try testing.expect(exited);
    try testing.expect(std.c.W.IFEXITED(@bitCast(status)) and std.c.W.EXITSTATUS(@bitCast(status)) == 0);
    for (0..50) |_| { // the workers follow within 5 s
        if (for (below) |w| {
            if (std.posix.system.kill(w, @fromBackingInt(@intCast(0))) == 0) break false;
        } else true) break;
        harness.sleepMs(io, 100);
    }
    for (below) |w| try testing.expect(std.posix.system.kill(w, @fromBackingInt(@intCast(0))) != 0);
}
