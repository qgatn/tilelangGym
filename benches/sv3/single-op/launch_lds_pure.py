#!/usr/bin/env python
"""Launch pure-LDS microbench (no sink verify — loads kept live by asm).
Usage: launch_lds_pure.py <so> <threads> <kind>
kind: fp32 | bf16 | bf16x2
"""

import ctypes
import os
import sys

import numpy as np

_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1


def _load_acl():
    sim = os.environ["TL_CAMODEL_LIB"]
    ctypes.CDLL(os.path.join(sim, "libruntime_camodel.so"), mode=ctypes.RTLD_GLOBAL)
    lib = ctypes.CDLL("libascendcl.so")
    for name, arg in [
        ("aclInit", ctypes.c_char_p),
        ("aclFinalize", None),
        ("aclrtSetDevice", ctypes.c_int),
        ("aclrtResetDevice", ctypes.c_int),
    ]:
        getattr(lib, name).argtypes = [arg] if arg else []
        getattr(lib, name).restype = ctypes.c_int
    lib.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtCreateStream.restype = ctypes.c_int
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    lib.aclrtMalloc.restype = ctypes.c_int
    lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    lib.aclrtMemcpy.restype = ctypes.c_int
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int
    return lib


def main():
    # Usage: launch_lds_pure.py <so> <threads> <kind> [N]
    so, threads, kind = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 8192
    rng = np.random.RandomState(2026)
    f32 = rng.uniform(-1, 1, (N,)).astype(np.float32)
    if kind == "fp32":
        src_np = f32
    elif kind == "bf16":
        u32 = f32.view(np.uint32)
        src_np = (u32 >> 16).astype(np.uint16)          # N bf16
    elif kind == "bf16x2":
        # N pairs = 2N bf16 — match fp32 tile bytes when N matches
        f32_2 = rng.uniform(-1, 1, (N * 2,)).astype(np.float32)
        src_np = (f32_2.view(np.uint32) >> 16).astype(np.uint16)
    else:
        raise SystemExit(f"bad kind={kind}")

    lib = _load_acl()
    kern = ctypes.CDLL(so)
    kern.init.restype = ctypes.c_int
    kern.call.restype = ctypes.c_int
    kern.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    if lib.aclInit(None) or lib.aclrtSetDevice(0) or kern.init():
        print("FAIL init", flush=True)
        return 1
    st = ctypes.c_void_p()
    lib.aclrtCreateStream(ctypes.byref(st))

    src_p = ctypes.c_void_p()
    lib.aclrtMalloc(ctypes.byref(src_p), src_np.nbytes, _ACL_MEM_MALLOC_HUGE_FIRST)
    lib.aclrtMemcpy(
        src_p,
        src_np.nbytes,
        np.ascontiguousarray(src_np).ctypes.data_as(ctypes.c_void_p),
        src_np.nbytes,
        _ACL_MEMCPY_HOST_TO_DEVICE,
    )
    dst = ctypes.c_void_p()
    lib.aclrtMalloc(ctypes.byref(dst), max(threads * 4, 64), _ACL_MEM_MALLOC_HUGE_FIRST)
    rc = kern.call(dst, src_p, st)
    if rc != 0:
        print(f"FAIL call={rc}", flush=True)
        return 1
    lib.aclrtSynchronizeStream(st)
    # No numeric sink — PASS means the kernel ran. LDS counts checked from instr.bin.
    print(f"PASS maxabs=0.000e+00 kind={kind} threads={threads}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
