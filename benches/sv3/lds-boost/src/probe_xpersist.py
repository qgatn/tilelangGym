"""Spill-elimination probes for V3, run end to end. Final forms.

  xv_scalars  x as 8 T.alloc_var scalars; the unpack writes 8 SEPARATE
              one-column fragments (pl0..pl3, ph0..ph3), each with a single
              index pattern (n); the MAC reads them fully unrolled. No
              fragment column index >0 anywhere.
  xh_vec      x as four half2 ramps → float32x2 (__half22float2 path),
              vector MAC folded with tirx.Shuffle.
  v3_base     today's V3 lowering (control).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang
import tilelang.ascend.language as T
import tvm
import tilelang.transform
from tvm import tirx

ROOT = Path(__file__).resolve().parents[1]
SRC = "/home/onasismelchiorjeyakumar/workspace/tilelang/src"
THREADS = 256
KH = 4
SLAB = 1024


def build(arm: str):
    @T.prim_func
    def main(
        A_t: T.Tensor((KH, 2048), "uint32"),
        X: T.Tensor((8,), "float16"),
        Out: T.Tensor((2048,), "float32"),
    ):
        with T.Kernel(1):
            a_t_ub = T.alloc_shared((KH, SLAB), "uint32")
            x_ub = T.alloc_shared((8,), "float16")
            out_ub = T.alloc_shared((2048,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(2):
                T.copy(A_t[:, s * SLAB : (s + 1) * SLAB], a_t_ub)
                with T.SimtVF(THREADS):
                    wa = T.alloc_fragment((THREADS, KH), "uint32")
                    accA = T.alloc_fragment((THREADS,), "float32")
                    accB = T.alloc_fragment((THREADS,), "float32")
                    if arm == "xv_scalars":
                        x0 = T.alloc_var("float32", init=0.0)
                        x1 = T.alloc_var("float32", init=0.0)
                        x2 = T.alloc_var("float32", init=0.0)
                        x3 = T.alloc_var("float32", init=0.0)
                        x4 = T.alloc_var("float32", init=0.0)
                        x5 = T.alloc_var("float32", init=0.0)
                        x6 = T.alloc_var("float32", init=0.0)
                        x7 = T.alloc_var("float32", init=0.0)
                        x0 = T.Cast("float32", x_ub[0])
                        x1 = T.Cast("float32", x_ub[1])
                        x2 = T.Cast("float32", x_ub[2])
                        x3 = T.Cast("float32", x_ub[3])
                        x4 = T.Cast("float32", x_ub[4])
                        x5 = T.Cast("float32", x_ub[5])
                        x6 = T.Cast("float32", x_ub[6])
                        x7 = T.Cast("float32", x_ub[7])
                        w0 = T.alloc_fragment((THREADS,), "uint32")
                        w1 = T.alloc_fragment((THREADS,), "uint32")
                        w2 = T.alloc_fragment((THREADS,), "uint32")
                        w3 = T.alloc_fragment((THREADS,), "uint32")
                    elif arm == "xh_vec":
                        xa2 = T.alloc_fragment((4,), "float32x2")
                        xb2 = T.alloc_fragment((4,), "float32x2")
                        for i in T.Parallel(4):
                            xa2[i] = T.Cast("float32x2", x_ub[T.Ramp(2 * i, 1, 2)])
                            xb2[i] = T.Cast("float32x2", x_ub[T.Ramp(2 * i + 1, 1, 2)])
                    else:
                        xfa = T.alloc_fragment((KH,), "float32")
                        xfb = T.alloc_fragment((KH,), "float32")
                        for c in T.Parallel(KH):
                            xfa[c] = T.Cast("float32", x_ub[2 * c])
                            xfb[c] = T.Cast("float32", x_ub[2 * c + 1])
                    for q in T.serial(SLAB // THREADS):
                        for n in T.Parallel(THREADS):
                            accA[n] = T.float32(0)
                            accB[n] = T.float32(0)
                        if arm == "xv_scalars":
                            for n in T.Parallel(THREADS):
                                w0[n] = a_t_ub[0, q * THREADS + n]
                            for n in T.Parallel(THREADS):
                                w1[n] = a_t_ub[1, q * THREADS + n]
                            for n in T.Parallel(THREADS):
                                w2[n] = a_t_ub[2, q * THREADS + n]
                            for n in T.Parallel(THREADS):
                                w3[n] = a_t_ub[3, q * THREADS + n]
                            l0 = T.alloc_fragment((THREADS,), "float16")
                            h0 = T.alloc_fragment((THREADS,), "float16")
                            for n in T.Parallel(THREADS):
                                l0[n] = T.reinterpret(T.Cast("uint16", w0[n] & T.uint32(0xFFFF)), "float16")
                                h0[n] = T.reinterpret(T.Cast("uint16", w0[n] >> T.uint32(16)), "float16")
                            for n in T.Parallel(THREADS):
                                accA[n] = accA[n] + T.Cast("float32", l0[n]) * x0
                                accB[n] = accB[n] + T.Cast("float32", h0[n]) * x1
                            l1 = T.alloc_fragment((THREADS,), "float16")
                            h1 = T.alloc_fragment((THREADS,), "float16")
                            for n in T.Parallel(THREADS):
                                l1[n] = T.reinterpret(T.Cast("uint16", w1[n] & T.uint32(0xFFFF)), "float16")
                                h1[n] = T.reinterpret(T.Cast("uint16", w1[n] >> T.uint32(16)), "float16")
                            for n in T.Parallel(THREADS):
                                accA[n] = accA[n] + T.Cast("float32", l1[n]) * x2
                                accB[n] = accB[n] + T.Cast("float32", h1[n]) * x3
                            l2 = T.alloc_fragment((THREADS,), "float16")
                            h2 = T.alloc_fragment((THREADS,), "float16")
                            for n in T.Parallel(THREADS):
                                l2[n] = T.reinterpret(T.Cast("uint16", w2[n] & T.uint32(0xFFFF)), "float16")
                                h2[n] = T.reinterpret(T.Cast("uint16", w2[n] >> T.uint32(16)), "float16")
                            for n in T.Parallel(THREADS):
                                accA[n] = accA[n] + T.Cast("float32", l2[n]) * x4
                                accB[n] = accB[n] + T.Cast("float32", h2[n]) * x5
                            l3 = T.alloc_fragment((THREADS,), "float16")
                            h3 = T.alloc_fragment((THREADS,), "float16")
                            for n in T.Parallel(THREADS):
                                l3[n] = T.reinterpret(T.Cast("uint16", w3[n] & T.uint32(0xFFFF)), "float16")
                                h3[n] = T.reinterpret(T.Cast("uint16", w3[n] >> T.uint32(16)), "float16")
                            for n in T.Parallel(THREADS):
                                accA[n] = accA[n] + T.Cast("float32", l3[n]) * x6
                                accB[n] = accB[n] + T.Cast("float32", h3[n]) * x7
                        else:
                            plo = T.alloc_fragment((THREADS, KH), "float16")
                            phi = T.alloc_fragment((THREADS, KH), "float16")
                            for n, c in T.Parallel(THREADS, KH):
                                wa[n, c] = a_t_ub[c, q * THREADS + n]
                            for n, c in T.Parallel(THREADS, KH):
                                plo[n, c] = T.reinterpret(
                                    T.Cast("uint16", wa[n, c] & T.uint32(0xFFFF)), "float16")
                                phi[n, c] = T.reinterpret(
                                    T.Cast("uint16", wa[n, c] >> T.uint32(16)), "float16")
                            if arm == "xh_vec":
                                for c in T.serial(KH):
                                    pa = T.alloc_fragment((THREADS,), "float32x2")
                                    pb = T.alloc_fragment((THREADS,), "float32x2")
                                    accA2 = T.alloc_fragment((THREADS,), "float32x2")
                                    accB2 = T.alloc_fragment((THREADS,), "float32x2")
                                    for n in T.Parallel(THREADS):
                                        pa[n] = T.float32x2(T.Cast("float32", plo[n, c]), T.Cast("float32", plo[n, c]))
                                        pb[n] = T.float32x2(T.Cast("float32", phi[n, c]), T.Cast("float32", phi[n, c]))
                                    for n in T.Parallel(THREADS):
                                        accA2[n] = pa[n] * xa2[c]
                                        accB2[n] = pb[n] * xb2[c]
                                    for n in T.Parallel(THREADS):
                                        accA[n] = accA[n] + tirx.Shuffle([accA2[n]], [0]) + tirx.Shuffle([accA2[n]], [1])
                                        accB[n] = accB[n] + tirx.Shuffle([accB2[n]], [0]) + tirx.Shuffle([accB2[n]], [1])
                            else:
                                xfa = T.alloc_fragment((KH,), "float32")
                                xfb = T.alloc_fragment((KH,), "float32")
                                for c in T.serial(KH):
                                    for n in T.Parallel(THREADS):
                                        accA[n] = accA[n] + T.Cast("float32", plo[n, c]) * xfa[c]
                                        accB[n] = accB[n] + T.Cast("float32", phi[n, c]) * xfb[c]
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
    out = ROOT / "runs" / f"probe-{arm}-n2048-k8-t256"
    out.mkdir(parents=True, exist_ok=True)
    src = lower_source(arm)
    (out / "kernel.cpp").write_text(src)
    asc = out / "kernel.asc"
    asc.write_text(src + host_wrapper((SLAB * KH * 2) + 16 + 2048 * 4))
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
    h22 = src.count("__half22float2")
    print(f"{arm}: pun={puns} half22float2={h22}")
    if len(sys.argv) > 2 and sys.argv[2] == "build":
        compile_arm(arm)
