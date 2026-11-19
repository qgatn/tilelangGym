"""Spill-free x handling probes. Same math as V3 (A[k,n] u32-packed), two arms:

  xa_scalar  x via 8 scalar fp16 LDS + 8 scalar casts (today's lowering, control)
  xc_pair    the whole x as one half2-pair: xfa2 is TWO fragments written with
             DIFFERENT index patterns in SEPARATE Parallel loops — xa[c]=x[2c],
             xb[c]=x[2c+1] — then acc pairs read xfa[c]/xb[c] alternately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang
import tilelang.ascend.language as T
import tvm
import tilelang.transform

HERE = Path(__file__).resolve().parent
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
                    plo = T.alloc_fragment((THREADS, KH), "float16")
                    phi = T.alloc_fragment((THREADS, KH), "float16")
                    accA = T.alloc_fragment((THREADS,), "float32")
                    accB = T.alloc_fragment((THREADS,), "float32")
                    xfa = T.alloc_fragment((KH,), "float32")
                    xfb = T.alloc_fragment((KH,), "float32")
                    if arm == "xa_scalar":
                        for c in T.Parallel(KH):
                            xfa[c] = T.Cast("float32", x_ub[2 * c])
                            xfb[c] = T.Cast("float32", x_ub[2 * c + 1])
                    else:
                        # one index pattern per Parallel body
                        for c in T.Parallel(KH):
                            xfa[c] = T.Cast("float32", x_ub[2 * c])
                        for c in T.Parallel(KH):
                            xfb[c] = T.Cast("float32", x_ub[2 * c + 1])
                    for q in T.serial(SLAB // THREADS):
                        for n in T.Parallel(THREADS):
                            accA[n] = T.float32(0)
                            accB[n] = T.float32(0)
                        for n, c in T.Parallel(THREADS, KH):
                            wa[n, c] = a_t_ub[c, q * THREADS + n]
                        for n, c in T.Parallel(THREADS, KH):
                            plo[n, c] = T.reinterpret(
                                T.Cast("uint16", wa[n, c] & T.uint32(0xFFFF)), "float16")
                            phi[n, c] = T.reinterpret(
                                T.Cast("uint16", wa[n, c] >> T.uint32(16)), "float16")
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


if __name__ == "__main__":
    arm = sys.argv[1]
    src = lower_source(arm)
    (HERE / f"probe_{arm}.cpp").write_text(src)
    puns = src.count("*(half *)(&") + src.count("*(half*)(&")
    h22 = src.count("__half22float2")
    print(f"{arm}: pun={puns} half22float2={h22}")
