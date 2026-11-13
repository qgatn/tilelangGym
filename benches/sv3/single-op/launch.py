#!/usr/bin/env python
"""Launch one single-op case. Usage: launch.py <so> <op> <threads> <dt> [N]
op: lds|sts|ffma|f2f ; dt: fp32|bf16|fp16 ; default N = 8192"""

import ctypes
import os
import sys

import numpy as np
_ACL_MEM_MALLOC_HUGE_FIRST = 0
_ACL_MEMCPY_HOST_TO_DEVICE = 1
_ACL_MEMCPY_DEVICE_TO_HOST = 2


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
    so, op, threads, dt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    N = int(sys.argv[5]) if len(sys.argv) > 5 else 8192

    rng = np.random.RandomState(2026)
    f32 = rng.uniform(-1, 1, (N,)).astype(np.float32)
    if dt == "bf16":
        u32 = f32.view(np.uint32)
        src_np = (u32 >> 16).astype(np.uint16)  # bf16 bits
        # reference values after truncation to bf16
        f16r = (u32 & np.uint32(0xFFFF0000)).view(np.float32)
    elif dt == "fp16":
        src_np = f32.astype(np.float16)  # fp16 storage
        f16r = src_np.astype(np.float32)
    else:
        src_np = f32
        f16r = f32

    if op == "sts":
        gold = np.ones((N,), np.float32)
    elif op == "ffma":
        # b and c copies come from the same source buffer
        gold = f16r * f16r + f16r
    elif op == "lds" and dt == "fp16":
        # LDS burst kernel sinks a per-thread checksum into the first T floats
        vals = f16r.reshape(N // threads, threads)
        gold = np.zeros((N,), np.float32)
        gold[:threads] = vals.sum(axis=0)
    else:  # lds (bf16/fp32 full copy), f2f
        gold = f16r

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

    def alloc(arr):
        p = ctypes.c_void_p()
        lib.aclrtMalloc(ctypes.byref(p), arr.nbytes, _ACL_MEM_MALLOC_HUGE_FIRST)
        lib.aclrtMemcpy(p, arr.nbytes, np.ascontiguousarray(arr).ctypes.data_as(ctypes.c_void_p), arr.nbytes, _ACL_MEMCPY_HOST_TO_DEVICE)
        return p

    src_p = alloc(src_np)
    dst = ctypes.c_void_p()
    lib.aclrtMalloc(ctypes.byref(dst), gold.nbytes, _ACL_MEM_MALLOC_HUGE_FIRST)
    rc = kern.call(dst, src_p, st)
    if rc != 0:
        print(f"FAIL call={rc}", flush=True)
        return 1
    lib.aclrtSynchronizeStream(st)
    got = np.zeros_like(gold)
    lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), gold.nbytes, dst, gold.nbytes, _ACL_MEMCPY_DEVICE_TO_HOST)
    err = float(np.max(np.abs(got - gold)))
    print(f"PASS maxabs={err:.3e}" if err <= 1e-2 else f"FAIL maxabs={err:.3e} got0={got[:4]} gold0={gold[:4]}", flush=True)
    return 0 if err <= 1e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
