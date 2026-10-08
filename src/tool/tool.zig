//! The Tool contract the scheduler drives: needs, load, generate, unload, abort, state. One implementation today
//! (`process.ProcessTool`); the interface keeps the scheduler independent of how a tool runs.

const std = @import("std");
const Request = @import("request.zig").Request;
const ToolConfig = @import("../config.zig").ToolConfig;

/// Memory a tool holds while loaded, and the peak a request adds on top while it runs.
pub const Needs = struct {
    resident: u64,
    working: u64,

    pub fn total(n: Needs) u64 {
        return n.resident + n.working;
    }
};

pub const State = enum(u8) { unloaded, loading, ready, busy, failed };

/// Reported while generating; `of` 0 means unknown.
pub const Progress = struct {
    phase: []const u8 = "",
    step: u32 = 0,
    of: u32 = 0,
};

/// What one generate call produced: file names inside the job directory.
pub const Output = struct {
    files: []const []const u8,
    seed: u64 = 0,
    ms: u64 = 0,
};

/// One request for a tool: where its inputs and outputs live and what to make.
pub const Job = struct {
    id: []const u8,
    dir: []const u8,
    request: Request,
};

/// Receives progress from a running generate; called on the scheduler's thread.
pub const Sink = struct {
    ctx: *anyopaque,
    progress: *const fn (ctx: *anyopaque, p: Progress) void,

    pub fn report(s: Sink, p: Progress) void {
        s.progress(s.ctx, p);
    }
};

pub const Error = error{
    LoadFailed, // the tool could not start; `failure` has the reason
    GenerateFailed, // the request failed; the tool may still be loaded
    WorkerDied, // the tool's process ended; it is unloaded
    Cancelled, // `abort` ended the call; the tool is unloaded
    Timeout, // a deadline passed; the tool is unloaded
    OutOfMemory,
};

/// The reason and log tail of the last failure, owned by the tool until the next call.
pub const Failure = struct {
    message: []const u8 = "",
    type: []const u8 = "server_error",
};

pub const Tool = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        config: *const fn (ptr: *anyopaque) *const ToolConfig,
        needs: *const fn (ptr: *anyopaque, req: *const Request) Needs,
        load: *const fn (ptr: *anyopaque) Error!u64,
        generate: *const fn (ptr: *anyopaque, job: *const Job, sink: Sink, arena: std.mem.Allocator) Error!Output,
        unload: *const fn (ptr: *anyopaque) void,
        abort: *const fn (ptr: *anyopaque) void,
        state: *const fn (ptr: *anyopaque) State,
        failure: *const fn (ptr: *anyopaque) Failure,
    };

    /// The tool's configuration (id, kind, timeouts).
    pub fn config(t: Tool) *const ToolConfig {
        return t.vtable.config(t.ptr);
    }
    /// Pure and cheap: what loading it and running `req` would take.
    pub fn needs(t: Tool, req: *const Request) Needs {
        return t.vtable.needs(t.ptr, req);
    }
    /// Makes the tool ready; returns the measured resident bytes.
    pub fn load(t: Tool) Error!u64 {
        return t.vtable.load(t.ptr);
    }
    /// Runs one request; output files land in `job.dir`; strings in the result live in `arena`.
    pub fn generate(t: Tool, job: *const Job, sink: Sink, arena: std.mem.Allocator) Error!Output {
        return t.vtable.generate(t.ptr, job, sink, arena);
    }
    /// Releases everything; never fails.
    pub fn unload(t: Tool) void {
        t.vtable.unload(t.ptr);
    }
    /// Ends a running load or generate from another thread; that call returns `error.Cancelled`.
    pub fn abort(t: Tool) void {
        t.vtable.abort(t.ptr);
    }
    /// Safe to read from any thread.
    pub fn state(t: Tool) State {
        return t.vtable.state(t.ptr);
    }
    pub fn failure(t: Tool) Failure {
        return t.vtable.failure(t.ptr);
    }
};
