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

pub const Hear = struct { model: []const u8 = "", language: []const u8 = "", prompt: []const u8 = "", worker_ms: ?*u64 = null }; // worker_ms: the time the model itself took
pub const Say = struct { model: []const u8 = "", voice: []const u8 = "" };

pub const JobKind = enum { image, video };

/// A tool call's job: what the model asked for.
pub const JobSpec = struct { kind: JobKind, prompt: []const u8, size: []const u8 = "", seconds: u32 = 0 };

/// A started job: its id, and the implementation's handle.
pub const Job = struct { id: []const u8, kind: JobKind, handle: *anyopaque };

pub const Outcome = struct {
    status: []const u8, // completed, failed, cancelled or timeout
    urls: []const []const u8 = &.{},
    @"error": []const u8 = "",
};

/// Which model a session setting names.
pub const Role = enum { chat, speech, transcription };

/// What the built-in tools can do here: a tool is offered only when a model can serve it.
pub const Offer = struct { image: bool = false, video: bool = false };

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
    };

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
