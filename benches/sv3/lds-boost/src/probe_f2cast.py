"""Probe: can TileLang spell the packed unpack so codegen emits
__half22float2 (the register-safe pair convert)?

Arm v_ramp: wa is the u32 fragment; per word, T.reinterpret to float16x2,
T.Cast to float32x2 (vectorized-cast path), then Shuffle lanes into accA/accB.
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
from tvm import tirx

ROOT = Path(__file__).resolve().parents[1]
SRC = "/home/onasismelchiorjeyakumar/workspace/tilelang/src"
THREADS = 256
N = 2048
K = 8
KH = K // 2
SLAB = N // 2


@T.macro
def pair_mac(acc, pa, x2, n, c):
    # pa.x * x_even + pa.y * x_odd. Shuffle lane 0 is .x, lane 1 is .y.
    acc[n] = acc[n] + tirx.Shuffle([pa[n, c]], [0]) * tirx.Shuffle([x2[c]], [0]) \
                    + tirx.Shuffle([pa[n, c]], [1]) * tirx.Shuffle([x2[c]], [1])


def lower(arm: str, nosync: bool = False) -> str:
    if arm == "serial":
        return _lower_serial(nosync)
    if arm == "qset":
        return _lower_qset(True)
    if arm != "v_ramp":
        raise ValueError(f"unknown arm {arm}")
    return _lower_vramp(nosync)


@T.macro
def lane_mac(ax, ay, pa, x2, n, c):
    ax[n] = ax[n] + tirx.Shuffle([pa[n, c]], [0]) * tirx.Shuffle([x2[c]], [0])
    ay[n] = ay[n] + tirx.Shuffle([pa[n, c]], [1]) * tirx.Shuffle([x2[c]], [1])


def _finish(main, nosync: bool) -> str:
    config = {"tl.disable_data_race_check": True}
    if nosync:
        config["tl.disable_thread_storage_sync"] = True
    with tvm.target.Target("ascend"), tilelang.transform.PassContext(config=config):
        return tilelang.lower(main, target="ascend").kernel_source


def _lower_vramp(nosync: bool) -> str:
    @T.prim_func
    def main(
        A_t: T.Tensor((KH, N), "uint32"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_t_ub = T.alloc_shared((KH, SLAB), "uint32")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // SLAB):
                T.copy(A_t[:, s * SLAB : (s + 1) * SLAB], a_t_ub)
                with T.SimtVF(THREADS):
                    x2 = T.alloc_fragment((KH,), "float32x2")
                    for c in T.Parallel(KH):
                        x2[c] = T.Cast("float32x2", T.reinterpret(x_ub[T.Ramp(2 * c, 1, 2)], "float16x2"))
                    wa = T.alloc_fragment((THREADS, KH), "uint32")
                    for q in T.serial(SLAB // THREADS):
                        acc = T.alloc_fragment((THREADS,), "float32")
                        for n in T.Parallel(THREADS):
                            acc[n] = T.float32(0)
                        for n, c in T.Parallel(THREADS, KH):
                            wa[n, c] = a_t_ub[c, q * THREADS + n]
                        pa = T.alloc_fragment((THREADS, KH), "float32x2")
                        for n, c in T.Parallel(THREADS, KH):
                            pa[n, c] = T.Cast("float32x2", T.reinterpret(wa[n, c], "float16x2"))
                        for c in T.serial(KH):
                            for n in T.Parallel(THREADS):
                                pair_mac(acc, pa, x2, n, c)
                        for n in T.Parallel(THREADS):
                            out_ub[s * SLAB + q * THREADS + n] = acc[n]
            T.copy(out_ub, Out)

    return _finish(main, nosync)


def _lower_serial(nosync: bool) -> str:
    """§3.3 shape with the packed cast: convert the whole slab, then multiply.

    af holds every packed word of the slab as float32x2 before any FFMA.
    One accumulator per output, walked across the 4 packed k-steps.
    """

    @T.prim_func
    def main(
        A_t: T.Tensor((KH, N), "uint32"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_t_ub = T.alloc_shared((KH, SLAB), "uint32")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // SLAB):
                T.copy(A_t[:, s * SLAB : (s + 1) * SLAB], a_t_ub)
                with T.SimtVF(THREADS):
                    x2 = T.alloc_fragment((KH,), "float32x2")
                    for c in T.Parallel(KH):
                        x2[c] = T.Cast(
                            "float32x2",
                            T.reinterpret(x_ub[T.Ramp(2 * c, 1, 2)], "float16x2"),
                        )
                    wa = T.alloc_fragment((SLAB, KH), "uint32")
                    pa = T.alloc_fragment((SLAB, KH), "float32x2")
                    acc = T.alloc_fragment((SLAB,), "float32")
                    for n, c in T.Parallel(SLAB, KH):
                        wa[n, c] = a_t_ub[c, n]
                    for n, c in T.Parallel(SLAB, KH):
                        pa[n, c] = T.Cast(
                            "float32x2", T.reinterpret(wa[n, c], "float16x2")
                        )
                    for n in T.Parallel(SLAB):
                        acc[n] = T.float32(0)
                    for c in T.serial(KH):
                        for n in T.Parallel(SLAB):
                            pair_mac(acc, pa, x2, n, c)
                    for n in T.Parallel(SLAB):
                        out_ub[s * SLAB + n] = acc[n]
            T.copy(out_ub, Out)

    return _finish(main, nosync)


def _lower_qset(nosync: bool) -> str:
    """Packed cast in the quad-acc shape: one load phase, then one multiply phase.

    Four strips, two accumulators per strip (the two lanes of each packed word).
    The k loop updates all eight accumulators together. Storage sync is off,
    same as quad-acc, so the block barrier does not split the loads.
    """

    @T.prim_func
    def main(
        A_t: T.Tensor((KH, N), "uint32"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_t_ub = T.alloc_shared((KH, SLAB), "uint32")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // SLAB):
                T.copy(A_t[:, s * SLAB : (s + 1) * SLAB], a_t_ub)
                with T.SimtVF(THREADS):
                    x2 = T.alloc_fragment((KH,), "float32x2")
                    for c in T.Parallel(KH):
                        x2[c] = T.Cast(
                            "float32x2",
                            T.reinterpret(x_ub[T.Ramp(2 * c, 1, 2)], "float16x2"),
                        )
                    wa0 = T.alloc_fragment((THREADS, KH), "uint32")
                    wa1 = T.alloc_fragment((THREADS, KH), "uint32")
                    wa2 = T.alloc_fragment((THREADS, KH), "uint32")
                    wa3 = T.alloc_fragment((THREADS, KH), "uint32")
                    pa0 = T.alloc_fragment((THREADS, KH), "float32x2")
                    pa1 = T.alloc_fragment((THREADS, KH), "float32x2")
                    pa2 = T.alloc_fragment((THREADS, KH), "float32x2")
                    pa3 = T.alloc_fragment((THREADS, KH), "float32x2")
                    a0x = T.alloc_fragment((THREADS,), "float32")
                    a0y = T.alloc_fragment((THREADS,), "float32")
                    a1x = T.alloc_fragment((THREADS,), "float32")
                    a1y = T.alloc_fragment((THREADS,), "float32")
                    a2x = T.alloc_fragment((THREADS,), "float32")
                    a2y = T.alloc_fragment((THREADS,), "float32")
                    a3x = T.alloc_fragment((THREADS,), "float32")
                    a3y = T.alloc_fragment((THREADS,), "float32")
                    for n, c in T.Parallel(THREADS, KH):
                        wa0[n, c] = a_t_ub[c, n]
                    for n, c in T.Parallel(THREADS, KH):
                        wa1[n, c] = a_t_ub[c, THREADS + n]
                    for n, c in T.Parallel(THREADS, KH):
                        wa2[n, c] = a_t_ub[c, (2 * THREADS) + n]
                    for n, c in T.Parallel(THREADS, KH):
                        wa3[n, c] = a_t_ub[c, (3 * THREADS) + n]
                    for n, c in T.Parallel(THREADS, KH):
                        pa0[n, c] = T.Cast("float32x2", T.reinterpret(wa0[n, c], "float16x2"))
                    for n, c in T.Parallel(THREADS, KH):
                        pa1[n, c] = T.Cast("float32x2", T.reinterpret(wa1[n, c], "float16x2"))
                    for n, c in T.Parallel(THREADS, KH):
                        pa2[n, c] = T.Cast("float32x2", T.reinterpret(wa2[n, c], "float16x2"))
                    for n, c in T.Parallel(THREADS, KH):
                        pa3[n, c] = T.Cast("float32x2", T.reinterpret(wa3[n, c], "float16x2"))
                    for n in T.Parallel(THREADS):
                        a0x[n] = T.float32(0)
                        a0y[n] = T.float32(0)
                        a1x[n] = T.float32(0)
                        a1y[n] = T.float32(0)
                        a2x[n] = T.float32(0)
                        a2y[n] = T.float32(0)
                        a3x[n] = T.float32(0)
                        a3y[n] = T.float32(0)
                    for c in T.serial(KH):
                        for n in T.Parallel(THREADS):
                            lane_mac(a0x, a0y, pa0, x2, n, c)
                            lane_mac(a1x, a1y, pa1, x2, n, c)
                            lane_mac(a2x, a2y, pa2, x2, n, c)
                            lane_mac(a3x, a3y, pa3, x2, n, c)
                    base = s * SLAB
                    for n in T.Parallel(THREADS):
                        out_ub[base + n] = a0x[n] + a0y[n]
                        out_ub[base + THREADS + n] = a1x[n] + a1y[n]
                        out_ub[base + (2 * THREADS) + n] = a2x[n] + a2y[n]
                        out_ub[base + (3 * THREADS) + n] = a3x[n] + a3y[n]
            T.copy(out_ub, Out)

    return _finish(main, nosync)


if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "v_ramp"
    nosync = "nosync" in sys.argv or arm == "qset"
    try:
        src = lower(arm, nosync=nosync)
    except Exception as e:
        print(f"{arm}: LOWER_FAIL {type(e).__name__}: {str(e)[:400]}")
        raise SystemExit(1)
    puns = src.count("*(half *)(&") + src.count("*(half*)(&")
    h22 = src.count("__half22float2")
    print(f"{arm}: pun={puns} half22float2={h22}")
    tag = f"probe-f2-{arm}" + ("-nosync" if nosync else "") + "-n2048-k8-t256"
    out = ROOT / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.cpp").write_text(src)
    if "build" in sys.argv:
        smem = (SLAB * KH * 4) + N * 4 + K * 2
        wrapper = f"""
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
        (out / "kernel.asc").write_text(src + wrapper)
        subprocess.run(
            ["bisheng", "-O2", "-fPIC", "-std=c++20",
             "-mllvm", "-cce-aicore-dcpreload-args=false",
             "--npu-arch=dav-3510",
             "-Wl,--no-mmap-output-file", "--shared",
             str(out / "kernel.asc"), "-I", SRC, "-o", str(out / "kernel.so")],
            check=True)
        print(f"OK {arm}")
