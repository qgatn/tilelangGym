"""Scheduling probes: keep the pun, change when x is live.

  reorder  One SimtVF. Unpack every row into float UB first, barrier, then
           load x and MAC from those floats. x is not live across the pun.
  split    Two SimtVFs. The first only pun-unpacks into float UB. The second
           loads x and MACs, and contains no reinterpret.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang
import tilelang.ascend.language as T
import tilelang.transform
import tvm

HERE = Path(__file__).resolve().parent
SRC = "/home/onasismelchiorjeyakumar/workspace/tilelang/src"
THREADS = 256
N = 2048
K = 8
KH = K // 2
SLAB = N // 2


def build(arm: str):
    @T.prim_func
    def main(
        A_t: T.Tensor((KH, N), "uint32"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_t_ub = T.alloc_shared((KH, SLAB), "uint32")
            a_lo = T.alloc_shared((KH, SLAB), "float32")
            a_hi = T.alloc_shared((KH, SLAB), "float32")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // SLAB):
                T.copy(A_t[:, s * SLAB : (s + 1) * SLAB], a_t_ub)
                if arm == "split":
                    with T.SimtVF(THREADS):
                        for q in T.serial(SLAB // THREADS):
                            for c in T.serial(KH):
                                for n in T.Parallel(THREADS):
                                    w = a_t_ub[c, q * THREADS + n]
                                    a_lo[c, q * THREADS + n] = T.Cast(
                                        "float32",
                                        T.reinterpret(T.Cast("uint16", w & T.uint32(0xFFFF)), "float16"),
                                    )
                                    a_hi[c, q * THREADS + n] = T.Cast(
                                        "float32",
                                        T.reinterpret(T.Cast("uint16", w >> T.uint32(16)), "float16"),
                                    )
                    with T.SimtVF(THREADS):
                        xfa = T.alloc_fragment((KH,), "float32")
                        xfb = T.alloc_fragment((KH,), "float32")
                        for c in T.Parallel(KH):
                            xfa[c] = T.Cast("float32", x_ub[2 * c])
                            xfb[c] = T.Cast("float32", x_ub[2 * c + 1])
                        for q in T.serial(SLAB // THREADS):
                            accA = T.alloc_fragment((THREADS,), "float32")
                            accB = T.alloc_fragment((THREADS,), "float32")
                            for n in T.Parallel(THREADS):
                                accA[n] = T.float32(0)
                                accB[n] = T.float32(0)
                            for c in T.serial(KH):
                                for n in T.Parallel(THREADS):
                                    accA[n] = accA[n] + a_lo[c, q * THREADS + n] * xfa[c]
                                    accB[n] = accB[n] + a_hi[c, q * THREADS + n] * xfb[c]
                            for n in T.Parallel(THREADS):
                                out_ub[s * SLAB + q * THREADS + n] = accA[n] + accB[n]
                else:
                    with T.SimtVF(THREADS):
                        for q in T.serial(SLAB // THREADS):
                            for c in T.serial(KH):
                                for n in T.Parallel(THREADS):
                                    w = a_t_ub[c, q * THREADS + n]
                                    a_lo[c, q * THREADS + n] = T.Cast(
                                        "float32",
                                        T.reinterpret(T.Cast("uint16", w & T.uint32(0xFFFF)), "float16"),
                                    )
                                    a_hi[c, q * THREADS + n] = T.Cast(
                                        "float32",
                                        T.reinterpret(T.Cast("uint16", w >> T.uint32(16)), "float16"),
                                    )
                        T.sync_threads()
                        xfa = T.alloc_fragment((KH,), "float32")
                        xfb = T.alloc_fragment((KH,), "float32")
                        for c in T.Parallel(KH):
                            xfa[c] = T.Cast("float32", x_ub[2 * c])
                            xfb[c] = T.Cast("float32", x_ub[2 * c + 1])
                        for q in T.serial(SLAB // THREADS):
                            accA = T.alloc_fragment((THREADS,), "float32")
                            accB = T.alloc_fragment((THREADS,), "float32")
                            for n in T.Parallel(THREADS):
                                accA[n] = T.float32(0)
                                accB[n] = T.float32(0)
                            for c in T.serial(KH):
                                for n in T.Parallel(THREADS):
                                    accA[n] = accA[n] + a_lo[c, q * THREADS + n] * xfa[c]
                                    accB[n] = accB[n] + a_hi[c, q * THREADS + n] * xfb[c]
                            for n in T.Parallel(THREADS):
                                out_ub[s * SLAB + q * THREADS + n] = accA[n] + accB[n]
            T.copy(out_ub, Out)

    return main


def lower_source(arm: str) -> str:
    with tvm.target.Target("ascend"), tilelang.transform.PassContext(
        config={"tl.disable_data_race_check": True}
    ):
        return tilelang.lower(build(arm), target="ascend").kernel_source


def host_wrapper(smem: int) -> str:
    return f"""

#ifdef _WIN32
#define TL_EXPORT __declspec(dllexport)
#else
#define TL_EXPORT
#endif
#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];
extern "C" TL_EXPORT const char* get_last_error() {{ return error_buf; }}
extern "C" TL_EXPORT int init() {{ error_buf[0] = '\\0'; return 0; }}
extern "C" TL_EXPORT int call(void* a, void* x, void* o, void* stream) {{
  main_kernel<<<(1 * 1 * 1), {smem}, stream>>>((__gm__ uint32_t*)a, (__gm__ float*)o, (__gm__ half*)x);
  return 0;
}}
"""


def compile_arm(arm: str) -> Path:
    out = HERE / "runs" / f"probe-{arm}-n2048-k8-t256"
    out.mkdir(parents=True, exist_ok=True)
    src = lower_source(arm)
    (out / "kernel.cpp").write_text(src)
    # A u32 slab + two f32 slabs + x + full-N out.
    smem = (SLAB * KH * 4) + (SLAB * KH * 4) * 2 + K * 2 + N * 4
    asc = out / "kernel.asc"
    asc.write_text(src + host_wrapper(smem))
    subprocess.run(
        ["bisheng", "-O2", "-fPIC", "-std=c++20",
         "-mllvm", "-cce-aicore-dcpreload-args=false",
         "--npu-arch=dav-3510",
         "-Wl,--no-mmap-output-file", "--shared",
         str(asc), "-I", SRC, "-o", str(out / "kernel.so")],
        check=True,
    )
    print(f"OK {arm}", flush=True)
    return out


if __name__ == "__main__":
    arm = sys.argv[1]
    src = lower_source(arm)
    puns = src.count("*(half *)(&") + src.count("*(half*)(&")
    print(f"{arm}: pun={puns} vf={src.count('__simt_vf__')}")
    if len(sys.argv) > 2 and sys.argv[2] == "build":
        compile_arm(arm)
