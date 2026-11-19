#!/usr/bin/env python
"""Launch a_row8_pack benches. Usage: launch_row8.py <so> <threads> <pack>

Payload: 1024 rows x 8 fp16 (= 4096 u32 = 16 KB) of random bits.
PACK=2 wants a transposed UB; since this bench measures the VF-side LDS rate
only (post-MTE2), we simply fill UB linearly — the .asc's PACK=2 index shape
(tid + i*T over 4096 u32) is then exactly the pure-strided bench at slab size.
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
    lib.aclrtMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.aclrtSynchronizeStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
    lib.aclrtFree.restype = ctypes.c_int
    lib.aclrtResetDevice.argtypes = [ctypes.c_int]
    lib.aclrtResetDevice.restype = ctypes.c_int
    lib.aclrtDestroyStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.aclrtDestroyStream.restype = ctypes.c_int
    return lib


def main():
    so, threads, pack = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    rng = np.random.RandomState(2026)
    src_np = rng.uniform(-1, 1, (1024 * 8,)).astype(np.float16)
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
        src_p, src_np.nbytes,
        np.ascontiguousarray(src_np).ctypes.data_as(ctypes.c_void_p), src_np.nbytes,
        _ACL_MEMCPY_HOST_TO_DEVICE,
    )
    dst = ctypes.c_void_p()
    lib.aclrtMalloc(ctypes.byref(dst), max(threads * 4, 64), _ACL_MEM_MALLOC_HUGE_FIRST)
    rc = kern.call(dst, src_p, st)
    lib.aclrtSynchronizeStream(st)
    ok = rc == 0
    print(f"{'PASS' if ok else 'FAIL'} maxabs=0.000e+00 pack={pack} threads={threads}", flush=True)
    # no aclrtFree: free before teardown corrupts the sim's buffer registry
    # and kills instr.bin record flush (single-op launchers never free either)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
