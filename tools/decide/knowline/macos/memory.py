"""A macOS process's physical footprint, now and its lifetime peak (bytes): what the system charges it, Metal (unified
memory) buffers included, from proc_pid_rusage (RUSAGE_INFO_V4), the figures `footprint` and `/usr/bin/time -l` print."""

import ctypes

_RUSAGE_INFO_V4 = 4
_PHYS_FOOTPRINT = 7  # ri_phys_footprint, counting the uint64 fields after ri_uuid
_LIFETIME_MAX_PHYS_FOOTPRINT = 28  # ri_lifetime_max_phys_footprint


def _info(pid):
    buf = (ctypes.c_uint64 * 64)()
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    if libc.proc_pid_rusage(int(pid), _RUSAGE_INFO_V4, ctypes.byref(buf)) != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage({pid})")
    return buf[2:]  # past the 16-byte uuid


def footprint(pid):
    return _info(pid)[_PHYS_FOOTPRINT]


def peak(pid):
    return _info(pid)[_LIFETIME_MAX_PHYS_FOOTPRINT]


if __name__ == "__main__":
    import os
    import sys
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else os.getpid()
    print(f"footprint {footprint(pid)} B, peak {peak(pid)} B")
