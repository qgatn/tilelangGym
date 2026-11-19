"""Launch an A[k, n] kernel. The host packs A before the call.

Usage: launch_flip.py kernel.so N K <packT|pair8>

- packT: Aᵀ packed to (K/2, N) uint32  — word (c,n) = fp16 pair (2c,2c+1)
- pair8: Aᵀ packed to (K/4, N) uint64  — word (c,n) = fp16 quad (4c..4c+3)
Packing models what an MTE2/CCE pre-pass would produce.
"""

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
    lib.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
    lib.aclrtMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_int]
    lib.aclrtMalloc.restype = ctypes.c_int
    lib.aclrtFree.argtypes = [ctypes.c_void_p]
    lib.aclrtFree.restype = ctypes.c_int
    lib.aclrtMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int
    ]
    lib.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
    lib.aclrtSynchronizeStream.restype = ctypes.c_int
    return lib


def _chk(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed {rc}")


def pack_at(A: np.ndarray, mode: str):
    """A (N,K) fp16 → flipped packed bytes + dtype."""
    a16 = A.view(np.uint16)  # (N, K)
    if mode == "packT":
        w = a16[:, 0::2].astype(np.uint32) | (
            a16[:, 1::2].astype(np.uint32) << np.uint32(16)
        )  # (N, K/2) u32
        return np.ascontiguousarray(w.T), np.uint32  # (K/2, N)
    if mode == "pair8":
        # u64 quads then split into u32 lo/hi rows: GM (K/2, N) u32 where
        # rows 2c/2c+1 hold lo/hi halves of quad c (kernel views u64 in UB).
        wq = (a16[:, 0::4].astype(np.uint64)
              | (a16[:, 1::4].astype(np.uint64) << np.uint64(16))
              | (a16[:, 2::4].astype(np.uint64) << np.uint64(32))
              | (a16[:, 3::4].astype(np.uint64) << np.uint64(48)))  # (N, K/4)
        wq = np.ascontiguousarray(wq.T)                              # (K/4, N)
        lo = (wq & np.uint64(0xFFFFFFFF)).astype(np.uint32)          # (K/4, N)
        hi = (wq >> np.uint64(32)).astype(np.uint32)                 # (K/4, N)
        out = np.empty((wq.shape[0] * 2, wq.shape[1]), dtype=np.uint32)
        out[0::2] = lo
        out[1::2] = hi
        return out, np.uint32  # (K/2, N) u32
    raise ValueError(f"unknown mode {mode}")


def main():
    so, n, k, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rng = np.random.RandomState(2026)
    A = rng.uniform(-1, 1, (n, k)).astype(np.float16)
    X = rng.uniform(-1, 1, (k,)).astype(np.float16)
    gold = (A.astype(np.float32) * X.astype(np.float32)[None, :]).sum(axis=1)
    Agm, dt = pack_at(A, mode)
    a_bytes, x_bytes, o_bytes = Agm.nbytes, X.nbytes, n * 4
    lib = _load_acl()
    kern = ctypes.CDLL(so)
    kern.init.restype = ctypes.c_int
    kern.call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    kern.call.restype = ctypes.c_int
    kern.get_last_error.restype = ctypes.c_char_p
    _chk(lib.aclInit(None), "init")
    _chk(lib.aclrtSetDevice(0), "dev")
    _chk(kern.init(), "kern.init")
    st = ctypes.c_void_p()
    _chk(lib.aclrtCreateStream(ctypes.byref(st)), "stream")
    ad, xd, od = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
    _chk(lib.aclrtMalloc(ctypes.byref(ad), a_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "ma")
    _chk(lib.aclrtMalloc(ctypes.byref(xd), x_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mx")
    _chk(lib.aclrtMalloc(ctypes.byref(od), o_bytes, _ACL_MEM_MALLOC_HUGE_FIRST), "mo")
    _chk(lib.aclrtMemcpy(ad, a_bytes, np.ascontiguousarray(Agm, dtype=dt).ctypes.data_as(ctypes.c_void_p), a_bytes, _ACL_MEMCPY_HOST_TO_DEVICE), "h2d-a")
    _chk(lib.aclrtMemcpy(xd, x_bytes, np.ascontiguousarray(X).ctypes.data_as(ctypes.c_void_p), x_bytes, _ACL_MEMCPY_HOST_TO_DEVICE), "h2d-x")
    rc = kern.call(ad, xd, od, st)
    if rc != 0:
        print(f"FAIL call={rc} {kern.get_last_error()}", flush=True)
        return 1
    _chk(lib.aclrtSynchronizeStream(st), "sync")
    got = np.zeros((n,), dtype=np.float32)
    _chk(lib.aclrtMemcpy(got.ctypes.data_as(ctypes.c_void_p), o_bytes, od, o_bytes, _ACL_MEMCPY_DEVICE_TO_HOST), "d2h")
    err = float(np.max(np.abs(got - gold)))
    print(f"PASS maxabs={err:.3e}" if err <= 1e-2 else f"FAIL maxabs={err:.3e} got0={got[:4]} gold0={gold[:4]}", flush=True)
    return 0 if err <= 1e-2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
