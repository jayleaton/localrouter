//! Where each tool runs. A `single` topology (the default) serves every configured tool here, as before. A `multi`
//! topology names its nodes, their addresses and the tools each one hosts (`config.Topology`); every node's daemon gets
//! the same topology and tools, and `self` (or `serve --node`) says which node it is. A node serves the tools it hosts;
//! the front also serves every other node's, as remote tools it sends to their node (`tool/node.zig`), and runs a
//! remote tool itself while its node is down when it lists the tool under its `fallback`. A remote tool too big for the
//! front can name a `stand_in` instead: one of the front's own models of the same kind, which answers its requests while
//! its node is down (`api/routes.zig` `resolve`) and otherwise loads only when asked for by name.
//!
//! A tool hosted by several nodes is one model split across them: a tensor-parallel group of ranks, one on each. Rank 0
//! is the front when it hosts the model, else the first of its nodes in topology order; the others follow in that
//! order. Rank 0 serves the model and starts the other ranks first (`tool/ranks.zig`); each node admits its own rank.
//! The tool says how its engine is told its rank (`options.rank_env`, `engines/openai_server.zig`), so a model swap is a
//! config change. A group has no fallback: a single node cannot run its share alone.

const std = @import("std");
const config = @import("config.zig");
const Config = config.Config;
const ToolConfig = config.ToolConfig;

/// Where one served tool runs, from this daemon's point of view.
pub const Place = union(enum) {
    local, // here, as in a single topology
    remote: usize, // on that node (an index into `Plan.nodes`); never here
    fallback: usize, // on that node while it is up, here while it is down
    lead: Group, // rank 0 of a tensor-parallel group: it runs here and serves, the other ranks run on their nodes
    follower: Group, // another rank of a group: its process runs here when rank 0 asks; it serves no request
};

/// A tensor-parallel group as one of its nodes sees it.
pub const Group = struct {
    rank: usize, // this node's
    nodes: []const usize, // the group's nodes in rank order (indices into `Plan.nodes`): `nodes[0]` is rank 0

    pub fn world(g: Group) usize {
        return g.nodes.len;
    }
};

pub const Plan = struct {
    multi: bool = false,
    self: usize = 0, // this daemon's node (multi only)
    front: bool = true, // this daemon serves the API for every node (a single topology's daemon does)
    nodes: []const config.Node = &.{},
    tools: []const ToolConfig, // the tools this daemon serves, in config order
    places: []const Place, // where each of them runs, beside `tools`
    stand_ins: []const ?usize = &.{}, // the front only, beside `tools`: the tool (an index into `tools`) that answers for a remote one while its node is down

    /// Whether tool `i` (an index into `tools`) stands in for another node's tool: it is not kept loaded for itself.
    pub fn standsIn(p: *const Plan, i: usize) bool {
        for (p.stand_ins) |s| if (s == i) return true;
        return false;
    }

    /// The node that hosts tool `id`, or null for a tool of this node (or of a single topology).
    pub fn hostOf(p: *const Plan, id: []const u8) ?*const config.Node {
        for (p.tools, p.places) |t, pl| if (std.mem.eql(u8, t.id, id)) return switch (pl) {
            .local, .lead => null,
            .remote, .fallback => |n| &p.nodes[n],
            .follower => |g| &p.nodes[g.nodes[0]],
        };
        return null;
    }
};

/// The plan for this daemon, or null with `why` written when the topology is invalid. `self_flag` (from
/// `serve --node`) overrides `topology.self`.
pub fn plan(a: std.mem.Allocator, cfg: *const Config, self_flag: ?[]const u8, why: *std.Io.Writer) !?Plan {
    const t = cfg.topology;
    if (t.mode == .single) {
        if (t.nodes.len > 0) return refuse(why, "topology: mode single takes no nodes (use mode multi to name them)", .{});
        if (self_flag) |s| return refuse(why, "--node {s}: the topology is single (set topology.mode to multi and name its nodes)", .{s});
        const places = try a.alloc(Place, cfg.tools.len);
        @memset(places, .local);
        return .{ .tools = cfg.tools, .places = places };
    }
    if (t.nodes.len < 2) return refuse(why, "topology: mode multi needs at least two nodes, it names {d}", .{t.nodes.len});
    if (t.health_interval_ms < 100) return refuse(why, "topology: health_interval_ms must be at least 100", .{});
    if (t.down_after == 0) return refuse(why, "topology: down_after must be at least 1", .{});
    var front: ?usize = null;
    for (t.nodes, 0..) |n, i| {
        if (n.id.len == 0) return refuse(why, "topology: node {d} has no id", .{i});
        for (t.nodes[0..i]) |m| if (std.mem.eql(u8, m.id, n.id)) return refuse(why, "topology: two nodes are called '{s}'", .{n.id});
        if (n.addresses.len == 0) return refuse(why, "topology: node '{s}' has no addresses", .{n.id});
        for (n.addresses) |addr| if (ip4(addr) == null) return refuse(why, "topology: node '{s}' address '{s}' is not an IPv4 address", .{ n.id, addr });
        if (n.port == 0) return refuse(why, "topology: node '{s}' needs a port", .{n.id});
        if (n.role == .front) {
            if (front) |f| return refuse(why, "topology: nodes '{s}' and '{s}' are both the front; exactly one serves the API", .{ t.nodes[f].id, n.id });
            front = i;
        } else if (n.fallback.len > 0) {
            return refuse(why, "topology: node '{s}' lists fallback models, which only the front runs", .{n.id});
        } else if (n.stand_in.map.count() > 0) return refuse(why, "topology: node '{s}' lists stand_in models, which only the front runs", .{n.id});
    }
    const f = front orelse return refuse(why, "topology: no node has role front; exactly one serves the API", .{});
    const self_id = self_flag orelse t.self;
    if (self_id.len == 0) return refuse(why, "topology: this daemon's node is not named: set topology.self, or start it with `serve --node ID`", .{});
    const self = for (t.nodes, 0..) |n, i| {
        if (std.mem.eql(u8, n.id, self_id)) break i;
    } else return refuse(why, "topology: this daemon is node '{s}', which is not one of its nodes ({s})", .{ self_id, try ids(a, t.nodes) });

    // every model a node lists is a tool, and every tool is on at least one node: on several, its ranks in order
    const hosts = try a.alloc(std.ArrayList(usize), cfg.tools.len);
    @memset(hosts, .empty);
    for (t.nodes, 0..) |n, ni| for (n.models) |m| {
        const ti = toolIndex(cfg, m) orelse return refuse(why, "topology: node '{s}' hosts '{s}', which is not a configured tool", .{ n.id, m });
        for (hosts[ti].items) |other| if (other == ni) return refuse(why, "topology: node '{s}' lists '{s}' twice", .{ n.id, m });
        if (ni == f) try hosts[ti].insert(a, 0, ni) else try hosts[ti].append(a, ni); // the front is rank 0 when it hosts it
    };
    for (cfg.tools, hosts) |*tl, hs| {
        if (hs.items.len == 0) return refuse(why, "topology: tool '{s}' is on no node: list it under one node's models", .{tl.id});
        if (hs.items.len == 1) continue;
        if (!std.mem.eql(u8, tl.engine, "openai_server")) return refuse(why, "topology: '{s}' is on {d} nodes (tensor parallel), which only a model's own server can run (engine openai_server); its engine is '{s}'", .{ tl.id, hs.items.len, if (tl.engine.len > 0) tl.engine else "a cmd worker" });
        const re = if (tl.options) |o| (if (o == .object) o.object.get("rank_env") else null) else null;
        if (re == null or re.? != .object or re.?.object.count() == 0)
            return refuse(why, "topology: '{s}' is on {d} nodes (tensor parallel), but its tool has no options.rank_env telling its server which rank it is (TensorFold's GLM: {{\"TF_GLM_RANK\": \"{{rank}}\", \"TF_GLM_WORLD\": \"{{world}}\", \"NCCL_COMM_ID\": \"{{master}}:29583\"}})", .{ tl.id, hs.items.len });
        if (listed(t.nodes[f].fallback, tl.id)) return refuse(why, "topology: '{s}' is tensor parallel over {d} nodes and cannot fall back to the front alone", .{ tl.id, hs.items.len });
    }
    for (t.nodes[f].fallback) |m| {
        const ti = toolIndex(cfg, m) orelse return refuse(why, "topology: the front's fallback '{s}' is not a configured tool", .{m});
        if (hosts[ti].items[0] == f) return refuse(why, "topology: the front's fallback '{s}' is one of its own models", .{m});
    }
    // a stand-in: another node's tool (one node, not a group, without a fallback) -> one of the front's own, same kind
    var sit = t.nodes[f].stand_in.map.iterator();
    while (sit.next()) |kv| {
        const m = kv.key_ptr.*;
        const by = kv.value_ptr.*;
        const ti = toolIndex(cfg, m) orelse return refuse(why, "topology: the front's stand_in names '{s}', which is not a configured tool", .{m});
        const si = toolIndex(cfg, by) orelse return refuse(why, "topology: the front's stand-in for '{s}' is '{s}', which is not a configured tool", .{ m, by });
        if (hosts[ti].items[0] == f) return refuse(why, "topology: the front's stand_in names '{s}', one of its own models: a stand-in answers for another node's", .{m});
        if (hosts[ti].items.len > 1) return refuse(why, "topology: '{s}' is tensor parallel over {d} nodes and has no stand-in: losing a rank's node stops the group, and its requests are refused naming that node", .{ m, hosts[ti].items.len });
        if (listed(t.nodes[f].fallback, m)) return refuse(why, "topology: '{s}' is under the front's fallback and its stand_in: the front runs it itself, or another model for it, not both", .{m});
        if (hosts[si].items.len != 1 or hosts[si].items[0] != f) return refuse(why, "topology: the stand-in for '{s}' is '{s}', which is not one of the front's own models (list it under the front's models)", .{ m, by });
        if (cfg.tools[si].kind != cfg.tools[ti].kind) return refuse(why, "topology: the stand-in for '{s}' ({s}) is '{s}', a {s} model: it must be of the same kind", .{ m, @tagName(cfg.tools[ti].kind), by, @tagName(cfg.tools[si].kind) });
    }

    var tools: std.ArrayList(ToolConfig) = .empty;
    var places: std.ArrayList(Place) = .empty;
    for (cfg.tools, hosts) |tl, hs| {
        const h = hs.items[0];
        const rank = std.mem.indexOfScalar(usize, hs.items, self);
        const pl: Place = if (hs.items.len > 1 and rank != null)
            (if (rank.? == 0) .{ .lead = .{ .rank = 0, .nodes = hs.items } } else .{ .follower = .{ .rank = rank.?, .nodes = hs.items } })
        else if (h == self) .local else if (self != f) continue else if (listed(t.nodes[f].fallback, tl.id)) .{ .fallback = h } else .{ .remote = h };
        try tools.append(a, tl);
        try places.append(a, pl);
    }
    const stand_ins: []const ?usize = if (self == f) try standIns(a, t.nodes[f], tools.items) else &.{};
    return .{ .multi = true, .self = self, .front = self == f, .nodes = t.nodes, .tools = tools.items, .places = places.items, .stand_ins = stand_ins };
}

/// Beside `served` (the front's tools): the index of the tool that stands in for each one, or null.
fn standIns(a: std.mem.Allocator, front: config.Node, served: []const ToolConfig) ![]const ?usize {
    const out = try a.alloc(?usize, served.len);
    for (out, served) |*o, tl| {
        const by = front.stand_in.map.get(tl.id) orelse {
            o.* = null;
            continue;
        };
        o.* = for (served, 0..) |other, k| {
            if (std.mem.eql(u8, other.id, by)) break k;
        } else unreachable; // `plan` checked it: one of the front's own models, so served here
    }
    return out;
}

/// A node daemon that is not the front serves only its own tools: its defaults keep only those (a default for a tool
/// it does not serve is dropped, as if unset).
pub fn trimDefaults(p: *const Plan, d: *config.Defaults) void {
    inline for (@typeInfo(config.Defaults).@"struct".field_names) |name| {
        if (@field(d, name)) |id| {
            const served = for (p.tools) |tl| {
                if (std.mem.eql(u8, tl.id, id)) break true;
            } else false;
            if (!served) @field(d, name) = null;
        }
    }
}

/// The four bytes of an IPv4 address in dotted form, or null.
pub fn ip4(text: []const u8) ?[4]u8 {
    var out: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, text, '.');
    var n: usize = 0;
    while (it.next()) |part| : (n += 1) {
        if (n == 4 or part.len == 0 or part.len > 3) return null;
        out[n] = std.fmt.parseInt(u8, part, 10) catch return null;
    }
    return if (n == 4) out else null;
}

fn toolIndex(cfg: *const Config, id: []const u8) ?usize {
    for (cfg.tools, 0..) |tl, i| if (std.mem.eql(u8, tl.id, id)) return i;
    return null;
}

fn listed(list: []const []const u8, id: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, id)) return true;
    return false;
}

fn ids(a: std.mem.Allocator, nodes: []const config.Node) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (nodes, 0..) |n, i| {
        if (i > 0) try out.appendSlice(a, ", ");
        try out.appendSlice(a, n.id);
    }
    return out.items;
}

fn refuse(why: *std.Io.Writer, comptime fmt: []const u8, args: anytype) ?Plan {
    why.print(fmt, args) catch {};
    return null;
}

// ---------------------------------------------------------------- tests

const two_nodes =
    \\"tools": [{"id": "chat", "kind": "chat", "engine": "openai_server"}, {"id": "img", "kind": "image", "engine": "testpattern"},
    \\          {"id": "tts", "kind": "speech", "engine": "testpattern"}],
    \\"topology": {"mode": "multi", "self": "s2", "nodes": [
    \\  {"id": "s1", "addresses": ["192.168.100.11", "192.168.101.11"], "models": ["chat"]},
    \\  {"id": "s2", "addresses": ["192.168.100.10"], "role": "front", "models": ["img", "tts"], "fallback": ["chat"]}]}
;

fn planOf(a: std.mem.Allocator, text: []const u8, self_flag: ?[]const u8, why: *std.Io.Writer) !?Plan {
    const c = try a.create(Config);
    c.* = try config.parse(a, text);
    return plan(a, c, self_flag, why);
}

test "single: every tool here, nothing else changes; nodes or --node are refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [512]u8 = undefined;
    var why: std.Io.Writer = .fixed(&buf);
    const p = (try planOf(a, "{}", null, &why)).?;
    try std.testing.expect(!p.multi and p.front);
    try std.testing.expectEqual(config.default_tools.len, p.tools.len);
    for (p.places) |pl| try std.testing.expect(pl == .local);
    try std.testing.expect((try planOf(a, "{}", "s1", &why)) == null);
    try std.testing.expect(std.mem.indexOf(u8, why.buffered(), "topology is single") != null);
}

test "multi: the front serves every tool, remote or fallback; a node only its own" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [512]u8 = undefined;
    var why: std.Io.Writer = .fixed(&buf);
    const front = (try planOf(a, "{" ++ two_nodes ++ "}", null, &why)).?;
    try std.testing.expect(front.multi and front.front);
    try std.testing.expectEqual(@as(usize, 3), front.tools.len);
    try std.testing.expectEqual(Place{ .fallback = 0 }, front.places[0]);
    try std.testing.expect(front.places[1] == .local and front.places[2] == .local);
    try std.testing.expectEqualStrings("s1", front.hostOf("chat").?.id);
    try std.testing.expect(front.hostOf("img") == null);

    const node = (try planOf(a, "{" ++ two_nodes ++ "}", "s1", &why)).?; // --node overrides topology.self
    try std.testing.expect(!node.front);
    try std.testing.expectEqual(@as(usize, 1), node.tools.len);
    try std.testing.expectEqualStrings("chat", node.tools[0].id);
    var d: config.Defaults = .{ .chat_completion = "chat", .text_to_image = "img" };
    trimDefaults(&node, &d);
    try std.testing.expectEqualStrings("chat", d.chat_completion.?);
    try std.testing.expect(d.text_to_image == null);

    // without a fallback the front's remote tool never runs there
    const nofb = (try planOf(a, try std.mem.replaceOwned(u8, a, "{" ++ two_nodes ++ "}", ", \"fallback\": [\"chat\"]", ""), null, &why)).?;
    try std.testing.expectEqual(Place{ .remote = 0 }, nofb.places[0]);
}

test "multi: invalid topologies are refused with the reason" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const base = "{" ++ two_nodes ++ "}";
    const cases = [_]struct { from: []const u8, to: []const u8, reason: []const u8 }{
        .{ .from = "\"self\": \"s2\"", .to = "\"self\": \"\"", .reason = "serve --node ID" },
        .{ .from = "\"self\": \"s2\"", .to = "\"self\": \"s9\"", .reason = "not one of its nodes (s1, s2)" },
        .{ .from = "\"role\": \"front\", \"models\": [\"img\", \"tts\"], \"fallback\": [\"chat\"]", .to = "\"models\": [\"img\", \"tts\"]", .reason = "no node has role front" },
        .{ .from = "\"models\": [\"chat\"]", .to = "\"models\": [\"chat\"], \"role\": \"front\"", .reason = "both the front" },
        .{ .from = "\"models\": [\"img\", \"tts\"]", .to = "\"models\": [\"img\", \"tts\", \"chat\"]", .reason = "tensor parallel" },
        .{ .from = "\"models\": [\"img\", \"tts\"]", .to = "\"models\": [\"img\"]", .reason = "tool 'tts' is on no node" },
        .{ .from = "\"models\": [\"img\", \"tts\"]", .to = "\"models\": [\"img\", \"tts\", \"nope\"]", .reason = "'nope', which is not a configured tool" },
        .{ .from = "\"192.168.100.10\"", .to = "\"spark-2.local\"", .reason = "is not an IPv4 address" },
        .{ .from = "\"fallback\": [\"chat\"]", .to = "\"fallback\": [\"img\"]", .reason = "is one of its own models" },
        .{ .from = "\"192.168.101.11\"], ", .to = "\"192.168.101.11\"], \"fallback\": [\"img\"], ", .reason = "which only the front runs" },
        .{ .from = "\"id\": \"s1\"", .to = "\"id\": \"s2\"", .reason = "two nodes are called 's2'" },
    };
    for (cases) |c| {
        var buf: [512]u8 = undefined;
        var why: std.Io.Writer = .fixed(&buf);
        const text = try std.mem.replaceOwned(u8, a, base, c.from, c.to);
        try std.testing.expect(!std.mem.eql(u8, text, base)); // the case changed something
        try std.testing.expect((try planOf(a, text, null, &why)) == null);
        std.testing.expect(std.mem.indexOf(u8, why.buffered(), c.reason) != null) catch |err| {
            std.debug.print("case {s} -> {s}: {s}\n", .{ c.from, c.to, why.buffered() });
            return err;
        };
    }
    var buf: [512]u8 = undefined;
    var why: std.Io.Writer = .fixed(&buf);
    try std.testing.expect((try planOf(a, "{\"topology\": {\"mode\": \"multi\", \"nodes\": [{\"id\": \"x\", \"addresses\": [\"10.0.0.1\"]}]}}", null, &why)) == null);
    try std.testing.expect(std.mem.indexOf(u8, why.buffered(), "at least two nodes") != null);
    try std.testing.expect((try planOf(a, "{\"topology\": {\"nodes\": [{\"id\": \"x\", \"addresses\": [\"10.0.0.1\"]}]}}", null, &why)) == null);
}

const stand_in_nodes =
    \\{"tools": [{"id": "big", "kind": "chat", "engine": "openai_server"}, {"id": "small", "kind": "chat", "engine": "openai_server"},
    \\           {"id": "tts", "kind": "speech", "engine": "testpattern"}],
    \\ "topology": {"mode": "multi", "self": "s2", "nodes": [
    \\  {"id": "s1", "addresses": ["192.168.100.11"], "models": ["big"]},
    \\  {"id": "s2", "addresses": ["192.168.100.10"], "role": "front", "models": ["small", "tts"], "stand_in": {"big": "small"}}]}}
;

test "stand_in: the front names one of its own models to answer for another node's while that node is down" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [512]u8 = undefined;
    var why: std.Io.Writer = .fixed(&buf);
    const front = (try planOf(a, stand_in_nodes, null, &why)).?;
    try std.testing.expectEqual(Place{ .remote = 0 }, front.places[0]); // big: on s1, no fallback of its own
    try std.testing.expect(front.places[1] == .local);
    try std.testing.expectEqual(@as(?usize, 1), front.stand_ins[0]); // small answers for big
    try std.testing.expectEqual(@as(?usize, null), front.stand_ins[1]);
    try std.testing.expect(front.standsIn(1) and !front.standsIn(0) and !front.standsIn(2));
    const node = (try planOf(a, stand_in_nodes, "s1", &why)).?; // the node knows nothing of it
    try std.testing.expectEqual(@as(usize, 1), node.tools.len);
    try std.testing.expectEqual(@as(usize, 0), node.stand_ins.len);
    try std.testing.expect(!node.standsIn(0));
    // a single topology has none
    const single = (try planOf(a, "{}", null, &why)).?;
    try std.testing.expectEqual(@as(usize, 0), single.stand_ins.len);
}

test "the two-Spark placement of deploy/jarvis/topologies/flashnext-2spark.json plans on both nodes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [512]u8 = undefined;
    var why: std.Io.Writer = .fixed(&buf);
    // the tools and the block as install.sh writes them (the file's `defaults` go to the config's defaults)
    const text =
        \\{"defaults": {"chat_completion": "qwen3.8-flash-next"},
        \\ "tools": [{"id": "qwen3.8-27b", "kind": "chat", "engine": "openai_server", "keep_loaded": false},
        \\           {"id": "qwen3.8-flash-next", "kind": "chat", "engine": "openai_server", "keep_loaded": true},
        \\           {"id": "fishaudio/drama-3", "kind": "speech", "cmd": ["x"]}, {"id": "whisper-large-v3-turbo", "kind": "transcription", "cmd": ["x"]},
        \\           {"id": "qwen-image-2.1-turbo", "kind": "image", "engine": "testpattern"}, {"id": "minimax-h3", "kind": "video", "engine": "testpattern"},
        \\           {"id": "knowline-4b", "kind": "decision", "engine": "openai_server"}, {"id": "idol", "kind": "asset3d", "cmd": ["x"]}],
        \\ "topology": {"mode": "multi", "token_file": "/home/dgx01/jarvis/lr/node-token", "health_interval_ms": 2000, "down_after": 3, "self": "spark-2",
        \\  "nodes": [{"id": "spark-1", "addresses": ["192.168.100.11", "192.168.101.11"], "port": 8190, "role": "node", "models": ["qwen3.8-flash-next"]},
        \\            {"id": "spark-2", "addresses": ["192.168.100.10", "192.168.101.10"], "port": 8190, "role": "front",
        \\             "models": ["qwen3.8-27b", "fishaudio/drama-3", "whisper-large-v3-turbo", "qwen-image-2.1-turbo", "minimax-h3", "knowline-4b", "idol"],
        \\             "stand_in": {"qwen3.8-flash-next": "qwen3.8-27b"}}]}}
    ;
    const front = (try planOf(a, text, null, &why)) orelse {
        std.debug.print("{s}\n", .{why.buffered()});
        return error.TestUnexpectedResult;
    };
    try std.testing.expectEqual(@as(usize, 8), front.tools.len);
    try std.testing.expectEqual(Place{ .remote = 0 }, front.places[1]); // Flash-Next on spark-1
    try std.testing.expectEqual(@as(?usize, 0), front.stand_ins[1]); // the 27B answers for it while spark-1 is down
    try std.testing.expect(front.standsIn(0));
    for (front.places, 0..) |pl, i| if (i != 1) try std.testing.expect(pl == .local);
    const chat_node = (try planOf(a, text, "spark-1", &why)).?;
    try std.testing.expectEqual(@as(usize, 1), chat_node.tools.len);
    try std.testing.expectEqualStrings("qwen3.8-flash-next", chat_node.tools[0].id);
    var d: config.Defaults = .{ .chat_completion = "qwen3.8-flash-next", .text_to_speech = "fishaudio/drama-3" };
    trimDefaults(&chat_node, &d);
    try std.testing.expectEqualStrings("qwen3.8-flash-next", d.chat_completion.?);
    try std.testing.expect(d.text_to_speech == null);
}

test "stand_in: refused unless it is the front's own model of the same kind, for one remote tool without a fallback" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { from: []const u8, to: []const u8, reason: []const u8 }{
        .{ .from = "{\"big\": \"small\"}", .to = "{\"nope\": \"small\"}", .reason = "stand_in names 'nope', which is not a configured tool" },
        .{ .from = "{\"big\": \"small\"}", .to = "{\"big\": \"nope\"}", .reason = "stand-in for 'big' is 'nope', which is not a configured tool" },
        .{ .from = "{\"big\": \"small\"}", .to = "{\"small\": \"tts\"}", .reason = "one of its own models" },
        .{ .from = "{\"big\": \"small\"}", .to = "{\"big\": \"tts\"}", .reason = "a speech model: it must be of the same kind" },
        .{ .from = "\"models\": [\"small\", \"tts\"]", .to = "\"models\": [\"tts\"]", .reason = "tool 'small' is on no node" },
        .{ .from = "\"models\": [\"big\"]}", .to = "\"models\": [\"big\", \"small\"]}", .reason = "tensor parallel" },
        .{ .from = "\"stand_in\"", .to = "\"fallback\": [\"big\"], \"stand_in\"", .reason = "under the front's fallback and its stand_in" },
        .{ .from = "\"models\": [\"big\"]}", .to = "\"models\": [\"big\"], \"stand_in\": {\"small\": \"big\"}}", .reason = "lists stand_in models, which only the front runs" },
    };
    for (cases) |c| {
        var buf: [512]u8 = undefined;
        var why: std.Io.Writer = .fixed(&buf);
        const text = try std.mem.replaceOwned(u8, a, stand_in_nodes, c.from, c.to);
        try std.testing.expect(!std.mem.eql(u8, text, stand_in_nodes)); // the case changed something
        try std.testing.expect((try planOf(a, text, null, &why)) == null);
        std.testing.expect(std.mem.indexOf(u8, why.buffered(), c.reason) != null) catch |err| {
            std.debug.print("case {s} -> {s}: {s}\n", .{ c.from, c.to, why.buffered() });
            return err;
        };
    }
    // a group has none: a stand-in is refused for a tool on several nodes
    var buf: [512]u8 = undefined;
    var why: std.Io.Writer = .fixed(&buf);
    const group =
        \\{"tools": [{"id": "big", "kind": "chat", "engine": "openai_server", "options": {"rank_env": {"R": "{rank}"}}},
        \\           {"id": "small", "kind": "chat", "engine": "openai_server"}],
        \\ "topology": {"mode": "multi", "nodes": [
        \\  {"id": "n1", "addresses": ["10.0.0.2"], "models": ["big"]}, {"id": "n2", "addresses": ["10.0.0.3"], "models": ["big"]},
        \\  {"id": "f", "addresses": ["10.0.0.1"], "role": "front", "models": ["small"], "stand_in": {"big": "small"}}]}}
    ;
    try std.testing.expect((try planOf(a, group, "f", &why)) == null);
    try std.testing.expect(std.mem.indexOf(u8, why.buffered(), "tensor parallel over 2 nodes and has no stand-in") != null);
}

test "tensor parallel: a model on several nodes is a group; the front is rank 0 when it hosts it, else the first node" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [512]u8 = undefined;
    var why: std.Io.Writer = .fixed(&buf);
    const text =
        \\{"tools": [{"id": "big", "kind": "chat", "engine": "openai_server", "options": {"rank_env": {"R": "{rank}"}}},
        \\           {"id": "wide", "kind": "chat", "engine": "openai_server", "options": {"rank_env": {"R": "{rank}"}}}],
        \\ "topology": {"mode": "multi", "nodes": [
        \\  {"id": "n2", "addresses": ["10.0.0.2"], "models": ["big", "wide"]},
        \\  {"id": "n3", "addresses": ["10.0.0.3"], "models": ["big", "wide"]},
        \\  {"id": "f", "addresses": ["10.0.0.1"], "role": "front", "models": ["wide"]}]}}
    ;
    const front = (try planOf(a, text, "f", &why)).?;
    try std.testing.expectEqual(Place{ .remote = 0 }, front.places[0]); // big: rank 0 is n2, the first of its nodes
    try std.testing.expect(front.places[1] == .lead); // wide: the front hosts it, so it is rank 0
    try std.testing.expectEqualSlices(usize, &.{ 2, 0, 1 }, front.places[1].lead.nodes);
    try std.testing.expectEqual(@as(usize, 3), front.places[1].lead.world());
    const n2 = (try planOf(a, text, "n2", &why)).?;
    try std.testing.expect(n2.places[0] == .lead and n2.places[0].lead.world() == 2);
    try std.testing.expectEqual(@as(usize, 1), n2.places[1].follower.rank);
    const n3 = (try planOf(a, text, "n3", &why)).?;
    try std.testing.expectEqual(@as(usize, 1), n3.places[0].follower.rank);
    try std.testing.expectEqual(@as(usize, 2), n3.places[1].follower.rank);
    try std.testing.expectEqualStrings("n2", n3.hostOf("big").?.id); // requests for it go to rank 0
    try std.testing.expectEqualStrings("f", n3.hostOf("wide").?.id);
    // a group needs a server engine and its rank_env, and has no fallback
    const no_env = try std.mem.replaceOwned(u8, a, text, "\"rank_env\": {\"R\": \"{rank}\"}", "\"x\": 1");
    try std.testing.expect((try planOf(a, no_env, "f", &why)) == null);
    try std.testing.expect(std.mem.indexOf(u8, why.buffered(), "no options.rank_env") != null);
}

test "ip4 parses dotted IPv4 only" {
    try std.testing.expectEqual([4]u8{ 192, 168, 100, 11 }, ip4("192.168.100.11").?);
    for ([_][]const u8{ "", "1.2.3", "1.2.3.4.5", "256.1.1.1", "a.b.c.d", "1..2.3", "localhost" }) |s| try std.testing.expect(ip4(s) == null);
}
