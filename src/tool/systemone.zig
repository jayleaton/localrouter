//! Typed decisions in the System One shape (TypeSafe's public `POST /v1/systemone`, which SideBud's Jev client speaks):
//! a request is `{model, state, questions}` and each question is a `choice` among named options, a `noul` (the
//! probability that a statement is true) or a `score` against 2 to 10 levels. A decision model answers each with
//! probabilities; code, not the model, acts on them.
//!
//! This file is the contract both ways: `parseQuestions` checks a request before it is queued (the API's 400s), and
//! `parseAnswers` checks a model's reply against the questions that were asked before anything reads it: every question
//! answered once with its own type, a choice that is one of the offered options, every probability finite and in
//! [0, 1], a distribution that sums to 1 (within the rounding a server may apply), a score inside its levels. A reply
//! that fails is the model's error (a 500), never a decision.

const std = @import("std");

pub const max_questions = 64;
pub const max_options = 255; // a choice's options, as TypeSafe's API
pub const max_levels = 10; // a score's levels, as TypeSafe's API
pub const max_name = 64;

/// One option of a choice: its id (what the answer names) and what it means ("" : the id alone).
pub const Option = struct { id: []const u8, description: []const u8 = "" };

pub const Question = struct {
    name: []const u8,
    instructions: []const u8 = "",
    ask: Ask,

    pub const Ask = union(enum) {
        choice: []const Option,
        noul: struct { yes: []const u8 = "", no: []const u8 = "" },
        score: []const []const u8, // level i scores i
    };
};

/// A validated answer. `probabilities` follow the question's options (choice) or levels (score), in order.
pub const Answer = union(enum) {
    choice: struct { choice: usize, confidence: f64, probabilities: []const f64 },
    noul: f64,
    score: struct { score: f64, confidence: f64, probabilities: []const f64 },

    /// The probability of option `id` (0 for an id the question did not offer).
    pub fn probabilityOf(an: Answer, q: Question, id: []const u8) f64 {
        if (an != .choice or q.ask != .choice) return 0;
        for (q.ask.choice, 0..) |o, i| if (std.mem.eql(u8, o.id, id)) return an.choice.probabilities[i];
        return 0;
    }
};

pub const Invalid = error{Invalid};

/// How far a distribution may stray from summing to 1: servers round each probability (TypeSafe to two decimals).
fn sumSlack(n: usize) f64 {
    return @max(0.02, 0.005 * @as(f64, @floatFromInt(n)));
}

// ---------------------------------------------------------------- requests

/// The questions of a request body (`questions`, an object of name -> question), checked; `why` names the problem.
pub fn parseQuestions(a: std.mem.Allocator, body: std.json.Value, why: *[]const u8) (Invalid || std.mem.Allocator.Error)![]Question {
    if (body != .object) return fail(why, "the body must be a JSON object with state and questions");
    const state = body.object.get("state") orelse return fail(why, "state is required: what every question refers to");
    switch (state) {
        .string, .object, .array => {},
        else => return fail(why, "state must be a string, an object or an array"),
    }
    const qs = body.object.get("questions") orelse return fail(why, "questions is required: an object of name -> question");
    if (qs != .object or qs.object.count() == 0) return fail(why, "questions must be a non-empty object of name -> question");
    if (qs.object.count() > max_questions) return fail(why, "at most 64 questions a request");
    const out = try a.alloc(Question, qs.object.count());
    var it = qs.object.iterator();
    var i: usize = 0;
    while (it.next()) |kv| : (i += 1) {
        const name = kv.key_ptr.*;
        if (!isName(name)) return fail(why, "a question name must be 1 to 64 letters, digits, '-', '_' or '.'");
        const q = kv.value_ptr.*;
        if (q != .object) return fail(why, "each question must be an object with a type");
        const typ = str(q.object.get("type")) orelse return fail(why, "each question needs a type: choice, noul or score");
        const instructions = text(a, q.object.get("instructions")) catch return fail(why, "instructions must be text");
        const criteria = q.object.get("criteria");
        const ask: Question.Ask = if (eql(typ, "choice")) blk: {
            const c = criteria orelse return fail(why, "a choice needs criteria: an object of 2 to 255 option -> description");
            if (c != .object or c.object.count() < 2) return fail(why, "a choice needs criteria: an object of 2 to 255 option -> description");
            if (c.object.count() > max_options) return fail(why, "a choice offers at most 255 options");
            const opts = try a.alloc(Option, c.object.count());
            var ci = c.object.iterator();
            var k: usize = 0;
            while (ci.next()) |o| : (k += 1) {
                if (o.key_ptr.len == 0 or o.key_ptr.len > 256) return fail(why, "an option id must be 1 to 256 bytes");
                opts[k] = .{ .id = o.key_ptr.*, .description = text(a, o.value_ptr.*) catch return fail(why, "an option's description must be text") };
            }
            break :blk .{ .choice = opts };
        } else if (eql(typ, "noul")) blk: {
            var yes: []const u8 = "";
            var no: []const u8 = "";
            if (criteria) |c| switch (c) {
                .null => {},
                .object => |o| {
                    var ci = o.iterator();
                    while (ci.next()) |kv2| {
                        const t = text(a, kv2.value_ptr.*) catch return fail(why, "a noul's criteria must be text");
                        if (eql(kv2.key_ptr.*, "true")) yes = t else if (eql(kv2.key_ptr.*, "false")) no = t else return fail(why, "a noul's criteria may only name true and false");
                    }
                },
                else => return fail(why, "a noul's criteria must be an object with true and/or false"),
            };
            break :blk .{ .noul = .{ .yes = yes, .no = no } };
        } else if (eql(typ, "score")) blk: {
            const c = criteria orelse return fail(why, "a score needs criteria: a list of 2 to 10 levels, lowest first");
            if (c != .array or c.array.items.len < 2 or c.array.items.len > max_levels) return fail(why, "a score needs criteria: a list of 2 to 10 levels, lowest first");
            const levels = try a.alloc([]const u8, c.array.items.len);
            for (levels, c.array.items) |*l, v| l.* = text(a, v) catch return fail(why, "a score level must be text");
            break :blk .{ .score = levels };
        } else return fail(why, "a question's type must be choice, noul or score");
        out[i] = .{ .name = name, .instructions = instructions, .ask = ask };
    }
    return out;
}

/// The request body for `questions` about `state` (JSON text), in the System One shape.
pub fn encodeRequest(a: std.mem.Allocator, model: []const u8, state_json: []const u8, questions: []const Question) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    var s: std.json.Stringify = .{ .writer = &w.writer };
    try s.beginObject();
    try s.objectField("model");
    try s.write(model);
    try s.objectField("state");
    try s.beginWriteRaw();
    try w.writer.writeAll(state_json);
    s.endWriteRaw();
    try s.objectField("questions");
    try s.beginObject();
    for (questions) |q| {
        try s.objectField(q.name);
        try s.beginObject();
        try s.objectField("type");
        try s.write(@tagName(q.ask));
        try s.objectField("instructions");
        try s.write(q.instructions);
        try s.objectField("criteria");
        switch (q.ask) {
            .choice => |opts| {
                try s.beginObject();
                for (opts) |o| {
                    try s.objectField(o.id);
                    try s.write(o.description);
                }
                try s.endObject();
            },
            .noul => |n| try s.write(.{ .true = n.yes, .false = n.no }),
            .score => |levels| try s.write(levels),
        }
        try s.endObject();
    }
    try s.endObject();
    try s.endObject();
    return w.written();
}

// ---------------------------------------------------------------- answers

/// The answers in a model's reply `body`, one per question in `questions`' order, each checked against its question.
pub fn parseAnswers(a: std.mem.Allocator, questions: []const Question, body: []const u8, why: *[]const u8) (Invalid || std.mem.Allocator.Error)![]Answer {
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}) catch return fail(why, "the decision model's reply is not JSON");
    if (v != .object) return fail(why, "the decision model's reply is not a JSON object");
    const answers = v.object.get("answers") orelse return fail(why, "the decision model's reply has no answers");
    if (answers != .object) return fail(why, "the decision model's answers are not an object");
    const out = try a.alloc(Answer, questions.len);
    for (questions, out) |q, *an| {
        const got = answers.object.get(q.name) orelse return failFmt(a, why, "the decision model did not answer '{s}'", .{q.name});
        var bad: []const u8 = "";
        an.* = answer(a, q, got, &bad) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Invalid => return failFmt(a, why, "the decision model's answer to '{s}' is invalid: {s}", .{ q.name, bad }),
        };
    }
    var extra = answers.object.iterator();
    while (extra.next()) |kv| {
        for (questions) |q| {
            if (eql(q.name, kv.key_ptr.*)) break;
        } else return failFmt(a, why, "the decision model answered '{s}', which was not asked", .{kv.key_ptr.*});
    }
    return out;
}

fn answer(a: std.mem.Allocator, q: Question, v: std.json.Value, why: *[]const u8) (Invalid || std.mem.Allocator.Error)!Answer {
    if (v != .object) return fail(why, "not an object");
    const typ = str(v.object.get("type")) orelse return fail(why, "no type");
    if (!eql(typ, @tagName(q.ask))) return fail(why, "its type is not the question's");
    switch (q.ask) {
        .noul => {
            const p = probability(v.object.get("noul")) orelse return fail(why, "noul is not a probability in [0, 1]");
            return .{ .noul = p };
        },
        .choice => |opts| {
            const chosen = str(v.object.get("choice")) orelse return fail(why, "no choice");
            const at = indexOf(opts, chosen) orelse return fail(why, "the choice is not one of the offered options");
            const confidence = probability(v.object.get("confidence")) orelse return fail(why, "confidence is not a probability in [0, 1]");
            const p = (try distribution(a, opts.len, v.object.get("probabilities"), opts, null, why)) orelse blk: {
                // SideBud's client takes a reply without probabilities: only the chosen option's is known
                const only = try a.alloc(f64, opts.len);
                @memset(only, 0);
                only[at] = confidence;
                break :blk only;
            };
            if (std.mem.max(f64, p) - p[at] > 0.011) return fail(why, "the choice is not the most probable option");
            return .{ .choice = .{ .choice = at, .confidence = confidence, .probabilities = p } };
        },
        .score => |levels| {
            const score = number(v.object.get("score")) orelse return fail(why, "no score");
            const top: f64 = @floatFromInt(levels.len - 1);
            if (!(score >= -0.005 and score <= top + 0.005)) return fail(why, "the score is outside its levels");
            const confidence = probability(v.object.get("confidence")) orelse return fail(why, "confidence is not a probability in [0, 1]");
            const p = (try distribution(a, levels.len, v.object.get("probabilities"), null, levels.len, why)) orelse try a.alloc(f64, 0);
            return .{ .score = .{ .score = std.math.clamp(score, 0, top), .confidence = confidence, .probabilities = p } };
        },
    }
}

/// `probabilities` as a list in option (or level) order; null when absent. Keys must be the offered options (or
/// "0".."n-1"); every value finite in [0, 1]; the sum 1 within rounding. A missing key is 0 only when the rest sum to 1.
fn distribution(a: std.mem.Allocator, n: usize, v: ?std.json.Value, opts: ?[]const Option, levels: ?usize, why: *[]const u8) (Invalid || std.mem.Allocator.Error)!?[]const f64 {
    const given = v orelse return null;
    if (given == .null) return null;
    if (given != .object) return fail(why, "probabilities is not an object");
    const out = try a.alloc(f64, n);
    @memset(out, 0);
    var it = given.object.iterator();
    var sum: f64 = 0;
    while (it.next()) |kv| {
        const at = if (opts) |o| indexOf(o, kv.key_ptr.*) else levelIndex(kv.key_ptr.*, levels.?);
        const i = at orelse return fail(why, "probabilities names an option that was not offered");
        out[i] = probability(kv.value_ptr.*) orelse return fail(why, "a probability is not a number in [0, 1]");
        sum += out[i];
    }
    if (@abs(sum - 1) > sumSlack(n)) return fail(why, "the probabilities do not sum to 1");
    return out;
}

fn levelIndex(key: []const u8, n: usize) ?usize {
    const i = std.fmt.parseInt(usize, key, 10) catch return null;
    return if (i < n) i else null;
}

fn indexOf(opts: []const Option, id: []const u8) ?usize {
    for (opts, 0..) |o, i| if (eql(o.id, id)) return i;
    return null;
}

fn number(v: ?std.json.Value) ?f64 {
    const x: f64 = switch (v orelse return null) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f64, s) catch return null,
        else => return null,
    };
    return if (std.math.isFinite(x)) x else null;
}

/// A finite number in [0, 1].
fn probability(v: ?std.json.Value) ?f64 {
    const x = number(v) orelse return null;
    return if (x >= 0 and x <= 1) x else null;
}

// ---------------------------------------------------------------- helpers

/// A string as is; an object or array as its JSON text (TypeSafe accepts structured instructions); null as "".
fn text(a: std.mem.Allocator, v: ?std.json.Value) error{ Invalid, OutOfMemory }![]const u8 {
    return switch (v orelse return "") {
        .null => "",
        .string => |s| s,
        .object, .array => std.json.Stringify.valueAlloc(a, v.?, .{}) catch return error.OutOfMemory,
        else => error.Invalid,
    };
}

fn isName(s: []const u8) bool {
    if (s.len == 0 or s.len > max_name) return false;
    for (s) |c| if (!(std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.')) return false;
    return true;
}

fn str(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn eql(x: []const u8, y: []const u8) bool {
    return std.mem.eql(u8, x, y);
}

fn fail(why: *[]const u8, msg: []const u8) Invalid {
    why.* = msg;
    return error.Invalid;
}

fn failFmt(a: std.mem.Allocator, why: *[]const u8, comptime fmt: []const u8, args: anytype) (Invalid || std.mem.Allocator.Error) {
    why.* = try std.fmt.allocPrint(a, fmt, args);
    return error.Invalid;
}

test "a request's questions are checked before they are queued" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: []const u8 = "";
    const ok =
        \\{"model":"jev","state":{"userWords":"make a picture of a fox"},"questions":{
        \\ "route":{"type":"choice","instructions":"What does the user want?","criteria":{"image":"a picture","chat":null}},
        \\ "compound":{"type":"noul","instructions":"Is it two requests?","criteria":{"true":"two","false":"one"}},
        \\ "urgency":{"type":"score","criteria":["none","some","high"]}}}
    ;
    const qs = try parseQuestions(a, try std.json.parseFromSliceLeaky(std.json.Value, a, ok, .{}), &why);
    try std.testing.expectEqual(@as(usize, 3), qs.len);
    try std.testing.expectEqualStrings("chat", qs[0].ask.choice[1].id);
    try std.testing.expectEqualStrings("", qs[0].ask.choice[1].description);
    try std.testing.expectEqualStrings("two", qs[1].ask.noul.yes);
    try std.testing.expectEqual(@as(usize, 3), qs[2].ask.score.len);

    const bad = [_]struct { body: []const u8, msg: []const u8 }{
        .{ .body = "[]", .msg = "the body must be a JSON object with state and questions" },
        .{ .body = "{\"questions\":{}}", .msg = "state is required: what every question refers to" },
        .{ .body = "{\"state\":1,\"questions\":{}}", .msg = "state must be a string, an object or an array" },
        .{ .body = "{\"state\":\"s\",\"questions\":{}}", .msg = "questions must be a non-empty object of name -> question" },
        .{ .body = "{\"state\":\"s\",\"questions\":{\"a b\":{\"type\":\"noul\"}}}", .msg = "a question name must be 1 to 64 letters, digits, '-', '_' or '.'" },
        .{ .body = "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"box\"}}}", .msg = "a question's type must be choice, noul or score" },
        .{ .body = "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"choice\",\"criteria\":{\"only\":null}}}}", .msg = "a choice needs criteria: an object of 2 to 255 option -> description" },
        .{ .body = "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"score\",\"criteria\":[\"0\",\"1\",\"2\",\"3\",\"4\",\"5\",\"6\",\"7\",\"8\",\"9\",\"10\"]}}}", .msg = "a score needs criteria: a list of 2 to 10 levels, lowest first" },
        .{ .body = "{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"noul\",\"criteria\":{\"maybe\":\"x\"}}}}", .msg = "a noul's criteria may only name true and false" },
    };
    for (bad) |b| {
        try std.testing.expectError(error.Invalid, parseQuestions(a, try std.json.parseFromSliceLeaky(std.json.Value, a, b.body, .{}), &why));
        try std.testing.expectEqualStrings(b.msg, why);
    }
}

test "the encoded request round-trips through the parser" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const qs = [_]Question{
        .{ .name = "route", .instructions = "Which?", .ask = .{ .choice = &.{ .{ .id = "image", .description = "a \"picture\"" }, .{ .id = "chat" } } } },
        .{ .name = "yes", .ask = .{ .noul = .{ .yes = "y" } } },
        .{ .name = "level", .ask = .{ .score = &.{ "low", "high" } } },
    };
    const body = try encodeRequest(a, "jev", "{\"userWords\":\"hi\"}", &qs);
    var why: []const u8 = "";
    const back = try parseQuestions(a, try std.json.parseFromSliceLeaky(std.json.Value, a, body, .{}), &why);
    try std.testing.expectEqual(@as(usize, 3), back.len);
    try std.testing.expectEqualStrings("a \"picture\"", back[0].ask.choice[0].description);
    try std.testing.expectEqualStrings("y", back[1].ask.noul.yes);
    try std.testing.expectEqualStrings("high", back[2].ask.score[1]);
}

test "a model's answers are checked against the questions asked" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const qs = [_]Question{
        .{ .name = "route", .ask = .{ .choice = &.{ .{ .id = "image" }, .{ .id = "chat" }, .{ .id = "none" } } } },
        .{ .name = "compound", .ask = .{ .noul = .{} } },
        .{ .name = "urgency", .ask = .{ .score = &.{ "none", "some", "high" } } },
    };
    var why: []const u8 = "";
    const ok =
        \\{"model":"jev-1","answers":{
        \\ "route":{"type":"choice","choice":"image","confidence":0.91,"probabilities":{"image":0.91,"chat":0.06,"none":0.02}},
        \\ "compound":{"type":"noul","noul":0.03},
        \\ "urgency":{"type":"score","score":1.2,"confidence":0.7,"probabilities":{"0":0.1,"1":0.6,"2":0.3}}},
        \\ "usage":{"input_tokens":120,"output_tokens":0}}
    ;
    const got = try parseAnswers(a, &qs, ok, &why);
    try std.testing.expectEqual(@as(usize, 0), got[0].choice.choice);
    try std.testing.expectApproxEqAbs(@as(f64, 0.06), got[0].probabilityOf(qs[0], "chat"), 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.03), got[1].noul, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 1.2), got[2].score.score, 1e-9);

    const without = "{\"answers\":{\"route\":{\"type\":\"choice\",\"choice\":\"chat\",\"confidence\":0.8},\"compound\":{\"type\":\"noul\",\"noul\":1},\"urgency\":{\"type\":\"score\",\"score\":0,\"confidence\":1}}}";
    const sparse = try parseAnswers(a, &qs, without, &why);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), sparse[0].probabilityOf(qs[0], "chat"), 1e-9);

    const bad = [_]struct { body: []const u8, msg: []const u8 }{
        .{ .body = "not json", .msg = "the decision model's reply is not JSON" },
        .{ .body = "{\"answers\":{}}", .msg = "the decision model did not answer 'route'" },
        .{ .body = "{\"answers\":{\"route\":{\"type\":\"choice\",\"choice\":\"video\",\"confidence\":0.9},\"compound\":{\"type\":\"noul\",\"noul\":0},\"urgency\":{\"type\":\"score\",\"score\":0,\"confidence\":1}}}", .msg = "the decision model's answer to 'route' is invalid: " },
        .{ .body = "{\"answers\":{\"route\":{\"type\":\"choice\",\"choice\":\"chat\",\"confidence\":0.9},\"compound\":{\"type\":\"noul\",\"noul\":1.5},\"urgency\":{\"type\":\"score\",\"score\":0,\"confidence\":1}}}", .msg = "the decision model's answer to 'compound' is invalid: " },
        .{ .body = "{\"answers\":{\"route\":{\"type\":\"choice\",\"choice\":\"chat\",\"confidence\":0.9,\"probabilities\":{\"chat\":0.9,\"image\":0.5}},\"compound\":{\"type\":\"noul\",\"noul\":0},\"urgency\":{\"type\":\"score\",\"score\":0,\"confidence\":1}}}", .msg = "the decision model's answer to 'route' is invalid: " },
        .{ .body = "{\"answers\":{\"route\":{\"type\":\"noul\",\"noul\":0.5},\"compound\":{\"type\":\"noul\",\"noul\":0},\"urgency\":{\"type\":\"score\",\"score\":0,\"confidence\":1}}}", .msg = "the decision model's answer to 'route' is invalid: " },
        .{ .body = "{\"answers\":{\"route\":{\"type\":\"choice\",\"choice\":\"chat\",\"confidence\":0.9},\"compound\":{\"type\":\"noul\",\"noul\":0},\"urgency\":{\"type\":\"score\",\"score\":3,\"confidence\":1}}}", .msg = "the decision model's answer to 'urgency' is invalid: " },
        .{ .body = "{\"answers\":{\"route\":{\"type\":\"choice\",\"choice\":\"chat\",\"confidence\":0.9},\"compound\":{\"type\":\"noul\",\"noul\":0},\"urgency\":{\"type\":\"score\",\"score\":0,\"confidence\":1},\"x\":{\"type\":\"noul\",\"noul\":0}}}", .msg = "the decision model answered 'x', which was not asked" },
    };
    for (bad) |b| {
        try std.testing.expectError(error.Invalid, parseAnswers(a, &qs, b.body, &why));
        try std.testing.expect(std.mem.startsWith(u8, why, b.msg));
    }
}
