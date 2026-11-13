#!/usr/bin/env python
"""Check packed F2F: dst holds fp32 of N fp16 values laid out as
word w -> elements 2w (lo) and 2w+1 (hi). Usage: launch_f2f_packed.py <so> [N]"""

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
    so = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    rng = np.random.RandomState(2026)
    f16 = rng.uniform(-1, 1, (N,)).astype(np.float16)
    gold = f16.astype(np.float32)

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
        lib.aclrtMemcpy(
            p, arr.nbytes,
            np.ascontiguousarray(arr).ctypes.data_as(ctypes.c_void_p),
            arr.nbytes, _ACL_MEMCPY_HOST_TO_DEVICE,
        )
        return p

    src_p = alloc(f16)
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
