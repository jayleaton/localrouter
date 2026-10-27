//! The assistant route: what the user calls their assistant (`assistant.zig`), one name per box for every client.
//!
//! - GET /v1/assistant: `{"object": "assistant", "name", "default_name", "custom", "voice": {"model", "voice"} or
//!   null}`; `custom` is false while the config's name applies, `voice` the voice a call speaks in when it names none
//!   (a hint to show beside the name). Any paired device (`routes.need`).
//! - PATCH /v1/assistant `{"name": "..."}`: renames it (`""`: back to the config's name); the new name counts from a
//!   voice session's next turn. Needs `manage`.

const std = @import("std");
const h = @import("http.zig");
const routes = @import("routes.zig");
const store = @import("../assistant.zig");

const App = routes.App;

pub fn route(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    switch (req.head.method) {
        .GET => {},
        .PATCH => {
            const text = h.body(req, a) catch return h.fail(req, a, 413, "invalid_request_error", "body too large");
            const b = std.json.parseFromSliceLeaky(struct { name: ?[]const u8 = null }, a, text, .{ .ignore_unknown_fields = true }) catch
                return h.fail(req, a, 400, "invalid_request_error", "send {\"name\": \"...\"}");
            const name = b.name orelse return h.fail(req, a, 400, "invalid_request_error", "send {\"name\": \"...\"}");
            app.assistant.set(app.io, name) catch |err| return switch (err) {
                error.InvalidName => h.fail(req, a, 400, "invalid_request_error", std.fmt.comptimePrint("name must be 1 to {d} characters: letters, digits, spaces, apostrophes, hyphens and full stops, with at least one letter or digit (\"\" goes back to the default)", .{store.max_chars})),
                error.WriteFailed => h.fail(req, a, 500, "server_error", "could not save the name"),
            };
            std.log.info("assistant: renamed to \"{s}\"", .{try app.assistant.name(app.io, a)});
        },
        else => return h.fail(req, a, 405, "invalid_request_error", "method not allowed"),
    }
    return h.jsonValue(req, a, .ok, try object(app, a));
}

const Voice = struct { model: []const u8, voice: []const u8 };

/// The route's answer, for GET and after a rename.
pub fn object(app: *App, a: std.mem.Allocator) !struct { object: []const u8 = "assistant", name: []const u8, default_name: []const u8, custom: bool, voice: ?Voice } {
    const v = @import("voice_local.zig").defaultVoice(app, a) catch null;
    return .{
        .name = try app.assistant.name(app.io, a),
        .default_name = app.cfg.assistant.name,
        .custom = app.assistant.custom(app.io),
        .voice = if (v) |x| Voice{ .model = x.model, .voice = x.voice } else null,
    };
}
