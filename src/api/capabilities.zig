//! GET /v1/capabilities: what this daemon offers a harness, by contract version (docs/HARNESS.md). A harness reads it
//! once to learn which surfaces exist here (chat, realtime, MCP, decisions, model management), their profiles, how
//! they are authorized and what is not offered, then fails closed on anything it needs that is missing. It reports
//! this daemon's configuration and models, never a secret, and is open like `/v1/models`: knowing that a route needs a
//! token gives nobody the token.

const std = @import("std");
const h = @import("http.zig");
const routes = @import("routes.zig");
const mcp = @import("mcp.zig");
const responses = @import("responses.zig");
const engine = @import("../engine/engine.zig");
const request = @import("../tool/request.zig");
const pair = @import("../pair.zig");

const App = routes.App;

/// The small-limit rule of the chat path (`budget.zig`), for both chat APIs.
const thinking_budget = .{
    .rule = "a limit below the model's min_thinking_tokens, with reasoning not chosen, answers with thinking off",
    .warnings = "warnings: [{code, message}] (streamed chat: a last chunk with choices: [])",
    .codes = [_][]const u8{ "thinking_off_for_small_budget", "reasoning_used_budget" },
};

/// The contract this daemon implements: docs/HARNESS.md under the same version. A date, changed whenever a documented
/// surface changes shape; additions a client may ignore keep the date.
pub const contract = "2026-10-23";
/// The daemon's version (build.zig.zon), also MCP's `serverInfo.version`.
pub const server_version = "0.1.0";

/// Client events `/v1/realtime` accepts (`voice/session.zig` `dispatch`); any other type is an `unknown_event_type` error.
pub const realtime_client_events = [_][]const u8{
    "session.update",           "input_audio_buffer.append",  "input_audio_buffer.commit", "input_audio_buffer.clear",
    "conversation.item.create", "conversation.item.truncate", "conversation.item.delete",  "response.create",
    "response.cancel",
};

/// Server events `/v1/realtime` sends: the OpenAI Realtime beta vocabulary, then this daemon's extensions.
pub const realtime_server_events = [_][]const u8{
    "session.created",                                    "session.updated",
    "input_audio_buffer.speech_started",                  "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",                       "input_audio_buffer.cleared",
    "conversation.item.created",                          "conversation.item.deleted",
    "conversation.item.truncated",                        "conversation.item.input_audio_transcription.completed",
    "conversation.item.input_audio_transcription.failed", "response.created",
    "response.output_item.added",                         "response.output_item.done",
    "response.content_part.added",                        "response.content_part.done",
    "response.text.delta",                                "response.text.done",
    "response.audio_transcript.delta",                    "response.audio_transcript.done",
    "response.audio.delta",                               "response.audio.done",
    "response.function_call_arguments.delta",             "response.function_call_arguments.done",
    "response.done",                                      "error",
};

pub const realtime_extension_events = [_][]const u8{
    "localrouter.transcription.partial", "localrouter.response.timing", "localrouter.response.text_done",
    "localrouter.response.speech_done",  "localrouter.job.created",     "localrouter.job.done",
    "localrouter.voice.paused",          "localrouter.voice.resumed",   "localrouter.function_call.cancelled",
};

/// A route that needs a paired device's token with `scope`: `always`, or only `when` pairing is required.
const Guard = struct { route: []const u8, scope: []const u8, when: []const u8 };

/// The routes a harness uses, each with the scope `routes.need` asks of it under `pairing.required` (so this list
/// cannot drift from what is enforced), then computer control's, which need their scopes always.
const Route = struct { method: std.http.Method, path: []const u8, route: []const u8 };
const routed = [_]Route{
    .{ .method = .POST, .path = "/v1/chat/completions", .route = "/v1/chat/completions" },
    .{ .method = .POST, .path = "/v1/responses", .route = "/v1/responses" },
    .{ .method = .GET, .path = "/v1/realtime", .route = "/v1/realtime" },
    .{ .method = .POST, .path = "/mcp", .route = "/mcp" },
    .{ .method = .POST, .path = "/v1/systemone", .route = "/v1/systemone" },
    .{ .method = .POST, .path = "/v1/images/generations", .route = "/v1/images/*" },
    .{ .method = .POST, .path = "/v1/videos", .route = "/v1/videos" },
    .{ .method = .POST, .path = "/v1/audio/speech", .route = "/v1/audio/*" },
    .{ .method = .GET, .path = "/v1/models", .route = "GET /v1/models, /v1/tools, /v1/skills/*" },
    .{ .method = .POST, .path = "/v1/tools/release", .route = "POST /v1/tools/*" },
    .{ .method = .GET, .path = "/v1/jobs/x", .route = "/v1/jobs/*, /v1/files/*, /v1/videos/*" },
};
const always = [_]Guard{
    .{ .route = "/v1/desktop", .scope = "desktop_helper", .when = "always" },
    .{ .route = "/v1/computer/*", .scope = "approve", .when = "always" },
    .{ .route = "mcp:computer_control,computer_use,computer_status,computer_stop", .scope = "computer", .when = "always" },
};

fn scopeOf(n: pair.Need) []const u8 {
    return switch (n) {
        .inference, .media, .decisions, .read, .manage => @tagName(n),
        .realtime => "inference (or legacy voice)",
        .mcp => "mcp or any service scope; each tool then needs its own",
        .owned => "any paired device; the job must be its own",
    };
}

fn guards(a: std.mem.Allocator) ![]const Guard {
    var out: std.ArrayList(Guard) = .empty;
    for (routed) |r| if (routes.need(r.method, r.path)) |n| try out.append(a, .{ .route = r.route, .scope = scopeOf(n), .when = "pairing_required" });
    try out.appendSlice(a, &always);
    return out.items;
}

/// The MCP tools for media and its jobs, as `media.mcp_tools`.
const media_tools = [_][]const u8{ "generate_image", "edit_image", "generate_video", "transcribe_audio", "synthesize_speech", "get_job", "cancel_job" };

/// One thing a harness might expect that this daemon does not offer (or not yet), and what to do instead.
const Gap = struct { feature: []const u8, status: []const u8, note: []const u8 };

const gaps = [_]Gap{
    .{ .feature = "responses_stored_state", .status = "unsupported", .note = "nothing is stored: previous_response_id, conversation, store true, item_reference and /v1/responses/{id} are refused; replay the whole conversation in input with store false" },
    .{ .feature = "responses_hosted_tools", .status = "unsupported", .note = "web_search, file_search, code_interpreter, computer_use, image_generation, mcp and other hosted tools are refused: function tools (and namespaces of them) only, run by the caller (Codex: web_search = \"disabled\")" },
    .{ .feature = "responses_custom_tools", .status = "unsupported", .note = "custom (freeform) tools are refused: function tools only" },
    .{ .feature = "responses_background", .status = "unsupported", .note = "background true is refused: stream the response and close the connection to cancel it" },
    .{ .feature = "responses_reasoning_summaries", .status = "unsupported", .note = "reasoning.summary concise or detailed is refused (auto and none make no summary); reasoning comes back as reasoning_text, never encrypted" },
    .{ .feature = "responses_multimodal_input", .status = "unsupported", .note = "input_image, input_file and input_audio are refused: the chat model takes text only" },
    .{ .feature = "responses_logprobs", .status = "unsupported", .note = "top_logprobs and include message.output_text.logprobs are refused" },
    .{ .feature = "realtime_ga_schema", .status = "unsupported", .note = "the beta event names only (response.audio.delta, flat session audio fields)" },
    .{ .feature = "realtime_webrtc", .status = "unsupported", .note = "WebSocket only" },
    .{ .feature = "realtime_ephemeral_keys", .status = "unsupported", .note = "no /v1/realtime/sessions; use a paired device token" },
    .{ .feature = "realtime_audio_formats", .status = "unsupported", .note = "pcm16 at 24 kHz mono only; no g711" },
    .{ .feature = "semantic_vad", .status = "unsupported", .note = "server_vad or push to talk (turn_detection null)" },
    .{ .feature = "chat_multimodal_input", .status = "unsupported", .note = "the chat model takes text only" },
    .{ .feature = "chat_n", .status = "unsupported", .note = "n must be 1" },
    .{ .feature = "chat_thinking_field", .status = "unsupported", .note = "DeepSeek's thinking: {type} reaches the model's server, which ignores it: switch reasoning off with reasoning_effort: none" },
    .{ .feature = "own_urls_as_inputs_via_proxy", .status = "unsupported", .note = "behind a TLS proxy (tailscale serve) this daemon's https output URLs are refused as inputs: send the file as base64" },
    .{ .feature = "audio_translations", .status = "unsupported", .note = "/v1/audio/translations is a 404" },
    .{ .feature = "transcription_streaming", .status = "unsupported", .note = "stream=true is refused; transcription answers once per file" },
    .{ .feature = "speech_sse", .status = "unsupported", .note = "stream_format sse is refused; instructions are refused by a speech model whose /v1/models entry lacks instructions: true" },
    .{ .feature = "embeddings", .status = "unsupported", .note = "/v1/embeddings is not served" },
    .{ .feature = "completions_legacy", .status = "unsupported", .note = "/v1/completions is not served" },
    .{ .feature = "files_upload", .status = "unsupported", .note = "/v1/files serves outputs only (GET, kept keep_outputs_s)" },
    .{ .feature = "mcp_oauth", .status = "unsupported", .note = "MCP takes a paired device's bearer token, not OAuth" },
    .{ .feature = "mcp_sessions", .status = "unsupported", .note = "stateless Streamable HTTP: no Mcp-Session-Id, no server-sent notifications, no resources or prompts" },
};

/// A model's licence as `/v1/models` lists it (`license` omitted when unknown): stated, not enforced.
const ModelLicense = struct { id: []const u8, license: ?@import("../config.zig").License };

pub fn serve(app: *App, req: *h.Request, a: std.mem.Allocator) !void {
    var kinds: std.ArrayList([]const u8) = .empty;
    var caps: std.ArrayList(engine.Capability) = .empty;
    var chat_models: std.ArrayList([]const u8) = .empty;
    var licenses: std.ArrayList(ModelLicense) = .empty;
    for (app.scheduler.slots) |slot| {
        const c = slot.tool.config();
        const k = @tagName(c.kind);
        if (!contains(kinds.items, k)) try kinds.append(a, k);
        for (try engine.capabilityList(a, c)) |cap| if (std.mem.indexOfScalar(engine.Capability, caps.items, cap) == null) try caps.append(a, cap);
        if (c.kind == .chat) try chat_models.append(a, c.id);
        try licenses.append(a, .{ .id = c.id, .license = engine.license(c) });
    }
    const has = struct {
        fn f(list: []const engine.Capability, cap: request.Capability) bool {
            return std.mem.indexOfScalar(engine.Capability, list, cap) != null;
        }
    }.f;
    const cfg = app.cfg;
    const computer = app.computer != null;
    try h.jsonValue(req, a, .ok, .{
        .object = "localrouter.capabilities",
        .contract = contract,
        .docs = "docs/HARNESS.md",
        .server = .{ .name = "localrouter", .version = server_version, .machine = cfg.machine },
        .auth = .{
            .scheme = "bearer",
            .pairing_required = cfg.pairing.required,
            .pair = "/v1/pair",
            .scopes = pair.scope_names,
            .service_scopes = pair.service_scopes,
            .legacy_scopes = [_][]const u8{ "voice", "mcp" }, // either (or none listed) grants every service scope
            .harness_scopes = [_][]const u8{ "inference", "media", "read" }, // what a harness is usually issued
            .guarded = try guards(a),
            .open = [_][]const u8{ "/health", "/v1/pair", "/v1/capabilities", "/voice" }, // with pairing off, every route is open
            .jobs_owned = true, // a job and its files answer only the device that asked for them (when paired)
        },
        .models = .{ .list = "/v1/models", .kinds = kinds.items, .capabilities = caps.items, .licenses = licenses.items },
        .chat = .{
            .available = has(caps.items, .chat_completion),
            .api = "chat_completions",
            .path = "/v1/chat/completions",
            .models = chat_models.items,
            .stream = true,
            .stream_usage = true,
            .tools = true,
            .tool_choice = [_][]const u8{ "none", "auto", "required", "function" },
            .response_format = [_][]const u8{ "json_object", "json_schema" },
            .reasoning = .{ .field = "reasoning_content", .off = "reasoning_effort: none, or chat_template_kwargs.enable_thinking: false" },
            .thinking_budget = thinking_budget,
            .input = [_][]const u8{"text"},
            .n = 1,
            .cancel = "close the connection",
        },
        .responses = .{
            .available = has(caps.items, .chat_completion),
            .api = "responses",
            .path = "/v1/responses",
            .models = chat_models.items,
            .stream = true,
            .store = false,
            .previous_response_id = false,
            .input = [_][]const u8{"text"},
            .items = [_][]const u8{ "message", "function_call", "function_call_output", "reasoning" },
            .roles = [_][]const u8{ "user", "assistant", "system", "developer" },
            .tools = [_][]const u8{ "function", "namespace" },
            .tool_choice = [_][]const u8{ "none", "auto", "required", "function" },
            .text_format = [_][]const u8{ "text", "json_object", "json_schema" },
            .reasoning = .{ .effort = "reasoning_effort on the model's server: none switches thinking off", .summary = [_][]const u8{ "auto", "none" }, .item = "reasoning_text", .encrypted_content = false },
            .thinking_budget = thinking_budget,
            .events = responses.stream_events,
            .cancel = "close the connection",
        },
        .realtime = .{
            .path = "/v1/realtime",
            .transport = "websocket",
            .profile = "openai-realtime-beta",
            .remote = cfg.voice.remote,
            .audio = .{ .format = "pcm16", .sample_rate = 24000, .channels = 1 },
            .turn_detection = [_][]const u8{ "server_vad", "none" },
            .modalities = [_][]const u8{ "audio", "text" },
            .tool_execution = std.meta.fieldNames(@import("../voice/events.zig").ToolExecution),
            .tool_execution_default = "server",
            .tool_execution_set = "session.update {session: {localrouter: {tool_execution}}} before the first response",
            .server_tools = .{ .media = cfg.voice.tools, .image_edit = cfg.voice.tools and has(caps.items, .image_edit), .image_to_video = cfg.voice.tools and has(caps.items, .image_to_video), .model_management = cfg.voice.tools and cfg.voice.manage_models, .computer = computer and cfg.computer.voice },
            .client_events = realtime_client_events,
            .server_events = realtime_server_events,
            .extension_events = realtime_extension_events,
            .token_transports = [_][]const u8{ "authorization", "sec-websocket-protocol", "api_key_query" },
        },
        .mcp = .{
            .path = "/mcp",
            .transport = "streamable_http",
            .stateless = true,
            .protocol_versions = mcp.versions,
            .stdio_bridge = "localrouter mcp-stdio --url URL [--token TOKEN]",
            .tools = mcp.tool_name_list,
        },
        .decisions = .{ .available = has(caps.items, .typed_decision), .path = "/v1/systemone", .shape = "system_one", .mcp_tool = "decide" },
        .media = .{
            .images = "/v1/images/generations",
            .image_edits = "/v1/images/edits",
            // from the resolved models (engine.resolve): an image model edits only with its edit pack installed
            .image_edit = has(caps.items, .image_edit),
            .image_to_video = has(caps.items, .image_to_video),
            .mcp_tools = media_tools,
            .videos = "/v1/videos",
            .speech = "/v1/audio/speech",
            .transcriptions = "/v1/audio/transcriptions",
            .jobs = "/v1/jobs/{id}",
            .files = "/v1/files/{job}/{name}",
        },
        .model_management = .{ .status = "/v1/tools", .load = "/v1/tools/{id}/load", .unload = "/v1/tools/{id}/unload", .release = "/v1/tools/release", .mcp_tools = [_][]const u8{ "list_models", "reserve_for", "load_model", "unload_model", "release_models" } },
        .computer = .{ .enabled = computer, .helper = "/v1/desktop", .approvals = "/v1/computer/approvals", .placement = "box_policy" },
        .unsupported = gaps,
    });
}

fn contains(list: []const []const u8, s: []const u8) bool {
    for (list) |x| if (std.mem.eql(u8, x, s)) return true;
    return false;
}

test "every gap names a status a client can branch on" {
    for (gaps) |g| try std.testing.expect(std.mem.eql(u8, g.status, "coming") or std.mem.eql(u8, g.status, "unsupported"));
}

test "every media tool discovery names is an MCP tool, and no gap claims edits or image to video are missing" {
    for (media_tools) |t| try std.testing.expect(contains(&mcp.tool_name_list, t));
    for (gaps) |g| {
        try std.testing.expect(std.mem.indexOf(u8, g.feature, "edit") == null);
        try std.testing.expect(std.mem.indexOf(u8, g.feature, "image_to_video") == null);
    }
}
