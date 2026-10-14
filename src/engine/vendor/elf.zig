//! The ELF facts the pinned loader needs, read from a library's bytes before it is loaded: the private memory its
//! writable segments take once mapped (from the program headers alone, so it is known before anything is copied), its
//! DT_SONAME and its DT_NEEDED list. ELF64 little-endian only (aarch64 and x86_64). Every offset is bounds-checked;
//! a malformed file refuses, nothing is guessed.

const std = @import("std");

pub const Error = error{ NotElf64, Truncated, NoDynamic, BadOffset, TooManyNeeded };

pub const ehdr_size = 64;
const phdr_size = 56;
const pt_load = 1;
const pt_dynamic = 2;
const pf_w = 2;
const dt_null = 0;
const dt_needed = 1;
const dt_strtab = 5;
const dt_strsz = 10;
const dt_soname = 14;

pub const max_needed = 32;

/// Where the program headers are, from the 64-byte ELF header.
pub const Header = struct {
    phoff: u64,
    phnum: u16,

    pub fn phdrBytes(self: Header) u64 {
        return @as(u64, self.phnum) * phdr_size;
    }
};

pub fn header(ehdr: []const u8) Error!Header {
    if (ehdr.len < ehdr_size) return error.Truncated;
    if (!std.mem.eql(u8, ehdr[0..4], "\x7fELF")) return error.NotElf64;
    if (ehdr[4] != 2 or ehdr[5] != 1) return error.NotElf64; // ELFCLASS64, ELFDATA2LSB
    if (rd(u16, ehdr, 54) != phdr_size) return error.NotElf64; // e_phentsize
    return .{ .phoff = rd(u64, ehdr, 32), .phnum = rd(u16, ehdr, 56) };
}

/// The private bytes the loader's mapping adds: every writable PT_LOAD's memsz (data, relro, bss), each rounded out to
/// whole pages of `page` bytes from its page-aligned start. Read-only segments share the sealed file's pages.
pub fn writableBytes(phdrs: []const u8, phnum: u16, page: u64) Error!u64 {
    if (phdrs.len < @as(usize, phnum) * phdr_size) return error.Truncated;
    var total: u64 = 0;
    for (0..phnum) |i| {
        const p = phdrs[i * phdr_size ..][0..phdr_size];
        if (rd(u32, p, 0) != pt_load or rd(u32, p, 4) & pf_w == 0) continue;
        const vaddr = rd(u64, p, 16);
        const memsz = rd(u64, p, 40);
        const start = vaddr - vaddr % page;
        const end = std.math.add(u64, vaddr, memsz) catch return error.BadOffset;
        total += std.mem.alignForward(u64, end, page) - start;
    }
    return total;
}

pub const Dynamic = struct {
    soname: ?[]const u8 = null,
    needed_buf: [max_needed][]const u8 = undefined,
    needed_len: usize = 0,

    pub fn needed(self: *const Dynamic) []const []const u8 {
        return self.needed_buf[0..self.needed_len];
    }
};

/// DT_SONAME and DT_NEEDED from the whole file's bytes (the strings point into `bytes`).
pub fn dynamic(bytes: []const u8) Error!Dynamic {
    const h = try header(bytes);
    const phdrs = try slice(bytes, h.phoff, h.phdrBytes());
    var dyn: ?[]const u8 = null;
    for (0..h.phnum) |i| {
        const p = phdrs[i * phdr_size ..][0..phdr_size];
        if (rd(u32, p, 0) == pt_dynamic) dyn = try slice(bytes, rd(u64, p, 8), rd(u64, p, 32));
    }
    const d = dyn orelse return error.NoDynamic;

    var strtab_vaddr: ?u64 = null;
    var strsz: u64 = 0;
    var i: usize = 0;
    while (i + 16 <= d.len) : (i += 16) {
        const tag = rd(i64, d, i);
        if (tag == dt_null) break;
        if (tag == dt_strtab) strtab_vaddr = rd(u64, d, i + 8);
        if (tag == dt_strsz) strsz = rd(u64, d, i + 8);
    }
    const strtab = try slice(bytes, try fileOffset(phdrs, h.phnum, strtab_vaddr orelse return error.NoDynamic), strsz);

    var out: Dynamic = .{};
    i = 0;
    while (i + 16 <= d.len) : (i += 16) {
        const tag = rd(i64, d, i);
        if (tag == dt_null) break;
        if (tag != dt_needed and tag != dt_soname) continue;
        const s = try string(strtab, rd(u64, d, i + 8));
        if (tag == dt_soname) {
            out.soname = s;
        } else {
            if (out.needed_len == max_needed) return error.TooManyNeeded;
            out.needed_buf[out.needed_len] = s;
            out.needed_len += 1;
        }
    }
    return out;
}

/// The file offset of `vaddr`, through the PT_LOAD that maps it from the file.
fn fileOffset(phdrs: []const u8, phnum: u16, vaddr: u64) Error!u64 {
    for (0..phnum) |i| {
        const p = phdrs[i * phdr_size ..][0..phdr_size];
        if (rd(u32, p, 0) != pt_load) continue;
        const v = rd(u64, p, 16);
        if (vaddr >= v and vaddr - v < rd(u64, p, 32)) return rd(u64, p, 8) + (vaddr - v);
    }
    return error.BadOffset;
}

fn slice(bytes: []const u8, off: u64, len: u64) Error![]const u8 {
    if (off > bytes.len or len > bytes.len - off) return error.Truncated;
    return bytes[@intCast(off)..][0..@intCast(len)];
}

fn string(strtab: []const u8, off: u64) Error![]const u8 {
    if (off >= strtab.len) return error.BadOffset;
    const rest = strtab[@intCast(off)..];
    const n = std.mem.indexOfScalar(u8, rest, 0) orelse return error.BadOffset;
    return rest[0..n];
}

fn rd(comptime T: type, b: []const u8, off: usize) T {
    return std.mem.readInt(T, b[off..][0..@sizeOf(T)], .little);
}

/// A minimal ELF64 shared object for the tests: one read-only and one writable PT_LOAD, a PT_DYNAMIC naming `soname`
/// and `needed`. Laid out flat (vaddr == file offset).
pub fn testObject(buf: []u8, soname: []const u8, needed: []const []const u8, data_memsz: u64) []u8 {
    @memset(buf, 0);
    const phoff = ehdr_size;
    const dyn_off = phoff + 3 * phdr_size;
    const n_dyn = needed.len + 4; // NEEDED..., SONAME, STRTAB, STRSZ, NULL
    const str_off = dyn_off + n_dyn * 16;
    var str_len: usize = 1; // a leading NUL
    const so_at = str_len;
    @memcpy(buf[str_off + str_len ..][0..soname.len], soname);
    str_len += soname.len + 1;
    var need_at: [8]usize = undefined;
    for (needed, 0..) |n, k| {
        need_at[k] = str_len;
        @memcpy(buf[str_off + str_len ..][0..n.len], n);
        str_len += n.len + 1;
    }
    const total = str_off + str_len;

    @memcpy(buf[0..4], "\x7fELF");
    buf[4] = 2;
    buf[5] = 1;
    wr(u64, buf, 32, phoff);
    wr(u16, buf, 54, phdr_size);
    wr(u16, buf, 56, 3);
    const ph = buf[phoff..];
    // PT_LOAD r-x covering the whole file
    wr(u32, ph, 0, pt_load);
    wr(u32, ph, 4, 5);
    wr(u64, ph, 32, total);
    wr(u64, ph, 40, total);
    // PT_LOAD rw- (data + bss), past the file
    wr(u32, ph, 56 + 0, pt_load);
    wr(u32, ph, 56 + 4, 6);
    wr(u64, ph, 56 + 16, 0x10000);
    wr(u64, ph, 56 + 40, data_memsz);
    // PT_DYNAMIC
    wr(u32, ph, 112 + 0, pt_dynamic);
    wr(u64, ph, 112 + 8, dyn_off);
    wr(u64, ph, 112 + 16, dyn_off);
    wr(u64, ph, 112 + 32, n_dyn * 16);
    var e: usize = dyn_off;
    for (needed, 0..) |_, k| {
        wr(i64, buf, e, dt_needed);
        wr(u64, buf, e + 8, need_at[k]);
        e += 16;
    }
    wr(i64, buf, e, dt_soname);
    wr(u64, buf, e + 8, so_at);
    wr(i64, buf, e + 16, dt_strtab);
    wr(u64, buf, e + 24, str_off);
    wr(i64, buf, e + 32, dt_strsz);
    wr(u64, buf, e + 40, str_len);
    return buf[0..total];
}

fn wr(comptime T: type, b: []u8, off: usize, v: T) void {
    std.mem.writeInt(T, b[off..][0..@sizeOf(T)], v, .little);
}

test "soname, needed and writable bytes from a built object" {
    var buf: [1024]u8 = undefined;
    const obj = testObject(&buf, "libcudnn.so.9", &.{ "libc.so.6", "libcublasLt.so.13" }, 0x2345);
    const d = try dynamic(obj);
    try std.testing.expectEqualStrings("libcudnn.so.9", d.soname.?);
    try std.testing.expectEqual(@as(usize, 2), d.needed().len);
    try std.testing.expectEqualStrings("libc.so.6", d.needed()[0]);
    try std.testing.expectEqualStrings("libcublasLt.so.13", d.needed()[1]);
    const h = try header(obj);
    // 0x10000 .. 0x12345 rounded out to 4 KiB pages is 0x3000; to 64 KiB pages, 0x10000
    try std.testing.expectEqual(@as(u64, 0x3000), try writableBytes(obj[@intCast(h.phoff)..], h.phnum, 4096));
    try std.testing.expectEqual(@as(u64, 0x10000), try writableBytes(obj[@intCast(h.phoff)..], h.phnum, 65536));
}

test "malformed objects refuse" {
    var buf: [1024]u8 = undefined;
    const obj = testObject(&buf, "liba.so.1", &.{}, 16);
    try std.testing.expectError(error.Truncated, dynamic(obj[0..100]));
    var bad = buf;
    bad[4] = 1; // ELFCLASS32
    try std.testing.expectError(error.NotElf64, dynamic(bad[0..obj.len]));
    try std.testing.expectError(error.Truncated, header(obj[0..10]));
}
