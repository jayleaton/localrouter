//! What a voice session needs from the rest of LocalRouter, behind one interface: speech to text, text to speech, the
//! chat stream and image or video jobs. The daemon's implementation is `api/voice_local.zig` (scheduler jobs and the
//! chat client); the unit tests use in-process fakes.

const std = @import("std");
const chat = @import("chat.zig");
const Stop = @import("stop.zig").Stop;

pub const Error = chat.Error;
pub const Why = chat.Why;

/// Receives speech as it is made: mono 16-bit samples at `rate`, in order, any number of calls.
pub const PcmSink = struct {
    ctx: *anyopaque,
    write: *const fn (ctx: *anyopaque, samples: []const i16, rate: u32) void,
};

/// worker_ms: the time the model itself took; detected: the language it heard (in the call's allocator), when given;
/// greedy: one greedy decoding pass instead of the model's default recipe.
pub const Hear = struct { model: []const u8 = "", language: []const u8 = "", prompt: []const u8 = "", worker_ms: ?*u64 = null, detected: ?*[]const u8 = null, greedy: bool = false };
pub const Say = struct { model: []const u8 = "", voice: []const u8 = "" };

/// What a warm-up speaks (the audio is dropped): one short word, since the engine speaks one request at a time and a
/// reply waits behind it.
pub const warm_sentence = "Hello.";

pub const JobKind = enum { image, video, asset3d };

/// A tool call's job: what the model asked for. An input image (an image's: an edit; a video's: its first frame) is
/// `input_file`, a file an earlier job of this session made, or `input_url`, a URL the model gave.
/// A 3D asset's `input_mode` ("": synthetic) is what its picture is.
pub const JobSpec = struct { kind: JobKind, prompt: []const u8, size: []const u8 = "", seconds: u32 = 0, input_file: []const u8 = "", input_url: []const u8 = "", input_mode: []const u8 = "" };

/// A started job: its id, and the implementation's handle.
pub const Job = struct { id: []const u8, kind: JobKind, handle: *anyopaque };

pub const Outcome = struct {
    status: []const u8, // completed, failed, cancelled or timeout
    urls: []const []const u8 = &.{},
    files: []const []const u8 = &.{}, // the same outputs as paths on this machine (for the next job's input image)
    @"error": []const u8 = "",
};

/// A model management tool's result: its JSON (or the reason it failed) for the model to read.
pub const Managed = struct { ok: bool, text: []const u8 };

/// A finished turn for a shadow decision: what the user said, the tools the chat model was offered and what it did
/// (the tools it called, in order; none: it replied in speech). Borrowed: the hook copies what it keeps.
pub const Shadow = struct {
    session_id: []const u8,
    response_id: []const u8,
    words: []const u8,
    previous: []const u8 = "", // the assistant's reply before this turn, for context
    tools: []const chat.Tool,
    called: []const []const u8,
    turn_ms: u64 = 0, // from the end of the user's turn to the end of this reply
};

/// A model the scheduler loads for an open call (residency `call`) and how far it is: `queued` (not started yet),
/// `loading`, `warming` (a speech model, loaded: its voice's warm-up runs), `ready`, `failed` (its load failed: `error`
/// says why), `no_room` (it does not fit beside what is kept) or `disabled` (turned off in settings since the call
/// asked for it: it is not coming, by choice). The last four are settled. `elapsed_ms`: since the
/// call asked for it.
pub const CallModel = struct { id: []const u8, state: []const u8, elapsed_ms: u64 = 0, @"error": []const u8 = "" };

pub fn settled(m: CallModel) bool {
    for ([_][]const u8{ "queued", "loading", "warming" }) |x| if (std.mem.eql(u8, m.state, x)) return false;
    return true;
}

/// Which model a session setting names.
pub const Role = enum { chat, speech, transcription };

/// What the built-in tools can do here: a tool is offered only when a model can serve it.
pub const Offer = struct { image: bool = false, image_edit: bool = false, video: bool = false, image_to_video: bool = false, asset3d: bool = false, computer: bool = false };

/// Lines the session speaks while a computer task runs ("I need your OK on screen to send that"), from any thread.
pub const Speaker = struct {
    ctx: *anyopaque,
    say: *const fn (ctx: *anyopaque, text: []const u8) void,
};

pub const Backend = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        transcribe: *const fn (ptr: *anyopaque, a: std.mem.Allocator, pcm16k: []const i16, o: Hear, stop: *Stop, why: *Why) Error![]const u8,
        speak: *const fn (ptr: *anyopaque, a: std.mem.Allocator, text: []const u8, o: Say, sink: PcmSink, stop: *Stop, why: *Why) Error!void,
        stream: *const fn (ptr: *anyopaque, a: std.mem.Allocator, r: *const chat.Request, sink: chat.Sink, stop: *Stop, why: *Why) Error!chat.Result,
        start_job: *const fn (ptr: *anyopaque, a: std.mem.Allocator, spec: JobSpec, why: *Why) Error!Job,
        wait_job: *const fn (ptr: *anyopaque, a: std.mem.Allocator, job: Job, timeout_s: u32, stop: *Stop) Outcome,
        offer: *const fn (ptr: *anyopaque) Offer,
        check: *const fn (ptr: *anyopaque, role: Role, name: []const u8, why: *Why) bool,
        paused: ?*const fn (ptr: *anyopaque) bool = null, // a voice model is unloaded to make room (it comes back)
        manage: ?*const fn (ptr: *anyopaque, a: std.mem.Allocator, name: []const u8, arguments: []const u8) Managed = null,
        computer: ?*const fn (ptr: *anyopaque, a: std.mem.Allocator, name: []const u8, arguments: []const u8, utterance: []const u8, speaker: Speaker, stop: *Stop) Managed = null,
        shadow: ?*const fn (ptr: *anyopaque, turn: Shadow) void = null, // must not block: called under the session's lock
        client_tools: ?*const fn (ptr: *anyopaque, why: *Why) bool = null, // may the session own its tools (null: yes)
        yield_chat: ?*const fn (ptr: *anyopaque, model: []const u8, op: Yield, holder: usize) bool = null, // speech first
        assistant_name: ?*const fn (ptr: *anyopaque, a: std.mem.Allocator) error{OutOfMemory}![]const u8 = null, // null: the default
        model_skill: ?*const fn (ptr: *anyopaque, a: std.mem.Allocator) error{OutOfMemory}![]const u8 = null, // null: no models line
        speech_warm: ?*const fn (ptr: *anyopaque, a: std.mem.Allocator, o: Say) bool = null, // null: never known warm
        call_models: ?*const fn (ptr: *anyopaque, a: std.mem.Allocator) error{OutOfMemory}![]const CallModel = null, // null: none
    };

    /// The models this machine loads for a call (residency `call`) and their state now, in `a`; none when it has none.
    /// Takes the scheduler's lock: never call it under the session's.
    pub fn callModels(b: Backend, a: std.mem.Allocator) error{OutOfMemory}![]const CallModel {
        const f = b.vtable.call_models orelse return &.{};
        return f(b.ptr, a);
    }

    /// Whether `o`'s voice is warm in its speech model's loaded instance: its first sentence starts at the warm speed.
    /// False when the backend cannot tell (the session then warms it).
    pub fn speechWarm(b: Backend, a: std.mem.Allocator, o: Say) bool {
        const f = b.vtable.speech_warm orelse return false;
        return f(b.ptr, a, o);
    }

    /// The model management procedure (`skill.zig`) with this machine's models in it, in `a`.
    pub fn modelSkill(b: Backend, a: std.mem.Allocator) error{OutOfMemory}![]const u8 {
        const f = b.vtable.model_skill orelse return @import("skill.zig").render(a, 0, &.{});
        return f(b.ptr, a);
    }

    /// What the assistant is called now (`assistant.zig`), in `a`: read each turn, so a rename counts from the next one.
    pub fn assistantName(b: Backend, a: std.mem.Allocator) error{OutOfMemory}![]const u8 {
        const f = b.vtable.assistant_name orelse return @import("../assistant.zig").default_name;
        return f(b.ptr, a);
    }

    /// Speech first (sched/yielder.zig): `check` whether chat model `model` ("": the default) can be paused for speech,
    /// `hold` it paused for one more lease (`holder`: the response), `release` it.
    pub const Yield = enum { check, hold, release };

    /// False when the chat model cannot be paused (another server, another node, not loaded): speech shares the GPU.
    pub fn yieldChat(b: Backend, model: []const u8, op: Yield, holder: usize) bool {
        const f = b.vtable.yield_chat orelse return false;
        return f(b.ptr, model, op, holder);
    }

    /// Whether the session's caller may take client tool execution (the `inference` scope); false with `why` set.
    pub fn clientTools(b: Backend, why: *Why) bool {
        return if (b.vtable.client_tools) |f| f(b.ptr, why) else true;
    }

    /// Runs a computer tool (computer_control, computer_use) for the user's words `utterance`, until the task ends
    /// (or `stop`: the task is stopped first); its JSON outcome. Progress worth hearing goes to `speaker`.
    pub fn computer(b: Backend, a: std.mem.Allocator, name: []const u8, arguments: []const u8, utterance: []const u8, speaker: Speaker, stop: *Stop) Managed {
        const f = b.vtable.computer orelse return .{ .ok = false, .text = "computer control is not available here" };
        return f(b.ptr, a, name, arguments, utterance, speaker, stop);
    }

    /// Asks the decision model how it would have routed a finished turn, and logs it beside what the chat model did.
    /// Shadow only: returns at once, and nothing acts on the decision.
    pub fn shadow(b: Backend, turn: Shadow) void {
        if (b.vtable.shadow) |f| f(b.ptr, turn);
    }

    /// True while a voice model is unloaded to make room for another job; it comes back by itself.
    pub fn paused(b: Backend) bool {
        return if (b.vtable.paused) |f| f(b.ptr) else false;
    }

    /// Runs a model management tool (list_models, reserve_for, unload_model, load_model); its JSON text.
    pub fn manage(b: Backend, a: std.mem.Allocator, name: []const u8, arguments: []const u8) Managed {
        const f = b.vtable.manage orelse return .{ .ok = false, .text = "model management is not available here" };
        return f(b.ptr, a, name, arguments);
    }

    /// The text of 16 kHz mono speech, in `a`.
    pub fn transcribe(b: Backend, a: std.mem.Allocator, pcm16k: []const i16, o: Hear, stop: *Stop, why: *Why) Error![]const u8 {
        return b.vtable.transcribe(b.ptr, a, pcm16k, o, stop, why);
    }
    /// Speaks `text` into `sink` as the audio is made (streamed when the speech model streams).
    pub fn speak(b: Backend, a: std.mem.Allocator, text: []const u8, o: Say, sink: PcmSink, stop: *Stop, why: *Why) Error!void {
        return b.vtable.speak(b.ptr, a, text, o, sink, stop, why);
    }
    /// Streams a completion; text deltas go to `sink` as they arrive.
    pub fn stream(b: Backend, a: std.mem.Allocator, r: *const chat.Request, sink: chat.Sink, stop: *Stop, why: *Why) Error!chat.Result {
        return b.vtable.stream(b.ptr, a, r, sink, stop, why);
    }
    /// Queues an image or video job.
    pub fn startJob(b: Backend, a: std.mem.Allocator, spec: JobSpec, why: *Why) Error!Job {
        return b.vtable.start_job(b.ptr, a, spec, why);
    }
    /// Waits for `job` (at most `timeout_s`, or until `stop`); its URLs in `a`.
    pub fn waitJob(b: Backend, a: std.mem.Allocator, job: Job, timeout_s: u32, stop: *Stop) Outcome {
        return b.vtable.wait_job(b.ptr, a, job, timeout_s, stop);
    }
    pub fn offer(b: Backend) Offer {
        return b.vtable.offer(b.ptr);
    }
    /// True when `name` is a model that can play `role`; else `why` says which can.
    pub fn check(b: Backend, role: Role, name: []const u8, why: *Why) bool {
        return b.vtable.check(b.ptr, role, name, why);
    }
};

/// Where a session's server events go (the WebSocket); one complete JSON text a call, from any thread.
pub const Out = struct {
    ctx: *anyopaque,
    send: *const fn (ctx: *anyopaque, text: []const u8) void,
};
