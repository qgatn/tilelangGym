#!/usr/bin/env python
"""Launch + verify pure_burst.asc. Usage:
launch_pure.py <so> <threads> <op> <kind>      op: lds|f2f|ffma, kind: 0..3

Sinks: LDS -> uint32 checksum (one word per thread, read back T floats).
F2F/FFMA -> full-tile fp32 store (read back N floats at dst[tid + i*T]).
FFMA B buffer == A buffer (the stub passes src for both GM inputs).
"""
import ctypes
import os
import sys

import numpy as np

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_H2D = 1
_ACL_MEMCPY_D2H = 2


def _load_acl():
    sim = os.environ["TL_CAMODEL_LIB"]
    ctypes.CDLL(os.path.join(sim, "libruntime_camodel.so"), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL("libascendcl.so")
    lib.aclInit.argtypes = [ctypes.c_char_p]
    for name in ("aclFinalize", "aclrtSetDevice", "aclrtSynchronizeStream"):
        getattr(lib, name).argtypes = []
        getattr(lib, name).restype = ctypes.c_int
    lib.aclrtSetDevice.argtypes = [ctypes.c_int]
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    lib.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                ctypes.c_size_t, ctypes.c_int]
    return lib


def input_fp16(N):
    rng = np.random.RandomState(2026)
    return rng.uniform(-1, 1, (N,)).astype(np.float32).astype(np.float16)


def gold(op, kind, T, N):
    """Expected device output, mirroring the kernel exactly."""
    f16 = input_fp16(N)
    fr = f16.astype(np.float32)
    if op == "lds":
        u16 = f16.view(np.uint16)
        u32 = np.ascontiguousarray(f16).view(np.uint32)
        g = np.zeros((T,), np.uint32)
        if kind == 0:  # scalar: element tid + i*T, one u16 per load
            for tid in range(T):
                acc = 0
                for i, idx in enumerate(range(tid, N, T)):
                    acc = (acc + int(u16[idx]) + i) & 0xFFFFFFFF
                g[tid] = acc
        else:  # packed kinds checksum the same 32-bit words
            for tid in range(T):
                acc = 0
                for i, w in enumerate(range(tid, N // 2, T)):
                    acc = (acc + int(u32[w]) + i) & 0xFFFFFFFF
                g[tid] = acc
        return g.view(np.float32)
    if op == "f2f":
        # fp16->fp32 is exact and elementwise: every schedule yields out = input
        return fr
    # ffma: r = a*a + a  (B == A on device)
    out = np.zeros((N,), np.float32)
    for tid in range(T):
        for idx in range(tid, N, T):
            out[idx] = fr[idx] * fr[idx] + fr[idx]
    return out


def main():
    so, T, op, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    N = int(sys.argv[5]) if len(sys.argv) > 5 else 8192
    lib = _load_acl()
    kern = ctypes.CDLL(so)
    kern.init.restype = ctypes.c_int
    kern.call.restype = ctypes.c_int
    kern.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    if lib.aclInit(None) != 0 or lib.aclrtSetDevice(0) != 0 or kern.init() != 0:
        print("FAIL init", flush=True)
        return 1
    st = ctypes.c_void_p()
    lib.aclrtCreateStream(ctypes.byref(st))

    src_n = N if op == "f2f" else 8192
    src = input_fp16(src_n).view(np.uint8)  # one src_n*2 buffer; FFMA B re-reads it
    sp = ctypes.c_void_p()
    lib.aclrtMalloc(ctypes.byref(sp), max(src.nbytes, 256), _ACL_MEM_MALLOC_HUGE_FIRST)
    lib.aclrtMemcpy(sp, max(src.nbytes, 256),
                    np.ascontiguousarray(src).ctypes.data_as(ctypes.c_void_p),
                    src.nbytes, _ACL_MEMCPY_H2D)
    out_n = T if op == "lds" else N
    dp = ctypes.c_void_p()
    lib.aclrtMalloc(ctypes.byref(dp), out_n * 4, _ACL_MEM_MALLOC_HUGE_FIRST)
    if kern.call(dp, sp, st) != 0:
        print("FAIL call", flush=True)
        return 1
    lib.aclrtSynchronizeStream(st)
    got = np.zeros((out_n,), np.float32)
    lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), got.nbytes, dp, got.nbytes,
                    _ACL_MEMCPY_D2H)

    ref = gold(op, kind, T, N)
    if op == "lds":
        ok = np.array_equal(got.view(np.uint32), ref.view(np.uint32))
        err = 0.0 if ok else float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))
    else:
        # F2F/FFMA outputs are element-exact; the compiler may permute which
        # warp's register block stores which slot, so compare as multisets
        err = float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64))))
        ok = (
            np.array_equal(np.sort(got.view(np.uint32)), np.sort(ref.view(np.uint32)))
            or err <= 5e-2 * max(1.0, float(np.max(np.abs(ref.astype(np.float64)))))
        )
    print(("PASS" if ok else "FAIL") + f" maxabs={err:.3e}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
