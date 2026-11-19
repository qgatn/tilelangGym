"""A[k, n] GEMV kernels. n is contiguous; each lane loads a uint32 pair.

Write-up: lds_performance.md
Arms (N=2048, K=8, T=256), launched via launch_flip.py:

  v0_base   unpack inside the load batch (control)
  v1_burst  load all 4 rows, then unpack + FFMA
  v2_pair   one row, two adjacent uint32 loads (8 B between lanes)
  v3_row    one row: load, unpack, FFMA, store, then the next row
"""

from __future__ import annotations

import tilelang.ascend.language as T

THREADS = 256
GM_DTYPE = {
    "v0_base": "uint32",
    "v1_burst": "uint32",
    "v2_pair": "uint32",
    "v3_row": "uint32",
}


def _check(N: int, K: int, threads: int) -> tuple[int, int, int]:
    if N < 32 or N & (N - 1):
        raise ValueError(f"N={N} must be a power of two >= 32")
    if K < 8 or K & (K - 1):
        raise ValueError(f"K={K} must be a power of two >= 8")
    if threads % 32:
        raise ValueError("threads must be a warp multiple")
    slab = N // 2
    if slab % threads:
        raise ValueError("slab must be divisible by threads")
    return slab, threads, K // 2


# ------------------------------------------------------------------ v0 control
def build_v0_base(N: int, K: int, threads: int | None = None):
    """Control: unpack sits in the same loop as the load."""
    threads = threads or THREADS
    slab, threads, kh = _check(N, K, threads)

    @T.prim_func
    def main(
        A_t: T.Tensor((kh, N), "uint32"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_t_ub = T.alloc_shared((kh, slab), "uint32")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A_t[:, s * slab : (s + 1) * slab], a_t_ub)
                with T.SimtVF(threads):
                    xfa = T.alloc_fragment((kh,), "float32")
                    xfb = T.alloc_fragment((kh,), "float32")
                    wa = T.alloc_fragment((threads, kh), "uint32")
                    plo = T.alloc_fragment((threads, kh), "float16")
                    phi = T.alloc_fragment((threads, kh), "float16")
                    for c in T.Parallel(kh):
                        xfa[c] = T.Cast("float32", x_ub[2 * c])
                        xfb[c] = T.Cast("float32", x_ub[2 * c + 1])
                    for q in T.serial(slab // threads):
                        accA = T.alloc_fragment((threads,), "float32")
                        accB = T.alloc_fragment((threads,), "float32")
                        for n in T.Parallel(threads):
                            accA[n] = T.float32(0)
                            accB[n] = T.float32(0)
                        for n, c in T.Parallel(threads, kh):
                            w = a_t_ub[c, q * threads + n]
                            wa[n, c] = w
                            plo[n, c] = T.reinterpret(
                                T.Cast("uint16", w & T.uint32(0xFFFF)), "float16")
                            phi[n, c] = T.reinterpret(
                                T.Cast("uint16", w >> T.uint32(16)), "float16")
                        for c in T.serial(kh):
                            for n in T.Parallel(threads):
                                accA[n] = accA[n] + T.Cast("float32", plo[n, c]) * xfa[c]
                                accB[n] = accB[n] + T.Cast("float32", phi[n, c]) * xfb[c]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + q * threads + n] = accA[n] + accB[n]
            T.copy(out_ub, Out)

    return main


# --------------------------------------------------------------- v1 burst order
def build_v1_burst(N: int, K: int, threads: int | None = None):
    """Burst-ordered: ALL kh word batches load (q-blocks back-to-back) BEFORE
    any unpack/MAC; fragments live across q so per-thread words stay hoisted.

    q indexes the 4 output blocks (rows q*T + t); kh words per row.
    Load phase: 2 loads per thread → the pure-BURST span. Then unpack+mac.
    """
    threads = threads or THREADS
    slab, threads, kh = _check(N, K, threads)

    @T.prim_func
    def main(
        A_t: T.Tensor((kh, N), "uint32"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_t_ub = T.alloc_shared((kh, slab), "uint32")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A_t[:, s * slab : (s + 1) * slab], a_t_ub)
                with T.SimtVF(threads):
                    xfa = T.alloc_fragment((kh,), "float32")
                    xfb = T.alloc_fragment((kh,), "float32")
                    wa = T.alloc_fragment((4, threads, kh), "uint32")
                    plo = T.alloc_fragment((4, threads, kh), "float16")
                    phi = T.alloc_fragment((4, threads, kh), "float16")
                    for c in T.Parallel(kh):
                        xfa[c] = T.Cast("float32", x_ub[2 * c])
                        xfb[c] = T.Cast("float32", x_ub[2 * c + 1])
                    # ---- LOAD PHASE: 4 q-batches back-to-back, no other work
                    for q in T.serial(4):
                        for n, c in T.Parallel(threads, kh):
                            wa[q, n, c] = a_t_ub[c, q * threads + n]
                    # ---- UNPACK + MAC phase
                    for q in T.serial(4):
                        accA = T.alloc_fragment((threads,), "float32")
                        accB = T.alloc_fragment((threads,), "float32")
                        for n in T.Parallel(threads):
                            accA[n] = T.float32(0)
                            accB[n] = T.float32(0)
                        for n, c in T.Parallel(threads, kh):
                            plo[q, n, c] = T.reinterpret(
                                T.Cast("uint16", wa[q, n, c] & T.uint32(0xFFFF)), "float16")
                            phi[q, n, c] = T.reinterpret(
                                T.Cast("uint16", wa[q, n, c] >> T.uint32(16)), "float16")
                        for c in T.serial(kh):
                            for n in T.Parallel(threads):
                                accA[n] = accA[n] + T.Cast("float32", plo[q, n, c]) * xfa[c]
                                accB[n] = accB[n] + T.Cast("float32", phi[q, n, c]) * xfb[c]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + q * threads + n] = accA[n] + accB[n]
            T.copy(out_ub, Out)

    return main


# --------------------------------------------------------------- v3 one row
def build_v3_row(N: int, K: int, threads: int | None = None):
    """One output row in flight. Same consecutive-n address as v1, but the
    thread finishes load → unpack → mac → store before touching the next row.

    Live set per thread: 4 uint32 + 8 fp16 (one K=8 vector), not v1's
    16 uint32 + 32 fp16 held across all four rows.
    """
    threads = threads or THREADS
    slab, threads, kh = _check(N, K, threads)

    @T.prim_func
    def main(
        A_t: T.Tensor((kh, N), "uint32"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_t_ub = T.alloc_shared((kh, slab), "uint32")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A_t[:, s * slab : (s + 1) * slab], a_t_ub)
                with T.SimtVF(threads):
                    xfa = T.alloc_fragment((kh,), "float32")
                    xfb = T.alloc_fragment((kh,), "float32")
                    wa = T.alloc_fragment((threads, kh), "uint32")
                    plo = T.alloc_fragment((threads, kh), "float16")
                    phi = T.alloc_fragment((threads, kh), "float16")
                    for c in T.Parallel(kh):
                        xfa[c] = T.Cast("float32", x_ub[2 * c])
                        xfb[c] = T.Cast("float32", x_ub[2 * c + 1])
                    for q in T.serial(slab // threads):
                        accA = T.alloc_fragment((threads,), "float32")
                        accB = T.alloc_fragment((threads,), "float32")
                        for n in T.Parallel(threads):
                            accA[n] = T.float32(0)
                            accB[n] = T.float32(0)
                        # four packed loads of THIS row, then unpack, then mac
                        for n, c in T.Parallel(threads, kh):
                            wa[n, c] = a_t_ub[c, q * threads + n]
                        for n, c in T.Parallel(threads, kh):
                            plo[n, c] = T.reinterpret(
                                T.Cast("uint16", wa[n, c] & T.uint32(0xFFFF)), "float16")
                            phi[n, c] = T.reinterpret(
                                T.Cast("uint16", wa[n, c] >> T.uint32(16)), "float16")
                        for c in T.serial(kh):
                            for n in T.Parallel(threads):
                                accA[n] = accA[n] + T.Cast("float32", plo[n, c]) * xfa[c]
                                accB[n] = accB[n] + T.Cast("float32", phi[n, c]) * xfb[c]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + q * threads + n] = accA[n] + accB[n]
            T.copy(out_ub, Out)

    return main


# ------------------------------------------------------------- v2 8-B pair probe
def build_v2_pair(N: int, K: int, threads: int | None = None):
    """8-B-per-lane probe WITHOUT a u64 view: each lane issues two adjacent
    u32 loads (pairs lo=row 2c, hi=row 2c+1 of the (K/2, N) u32 GM slab).

    Lane stride between THREADS stays 2 words (8 B) → the 2-way replay
    prediction is unchanged; we simply avoid the T.view/DMA layout trap
    (the planner picked up the u64 view's 8192-B row stride and overlapped
    stores). Needs K % 4 == 0.
    """
    threads = threads or THREADS
    slab, threads, kh = _check(N, K, threads)
    if K % 4:
        raise ValueError("v2_pair needs K % 4 == 0")
    kw = K // 4  # u64 quads per row → 2 u32 rows per quad

    @T.prim_func
    def main(
        A_t: T.Tensor((2 * kw, N), "uint32"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2 * kw, slab), "uint32")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A_t[:, s * slab : (s + 1) * slab], a_ub)
                with T.SimtVF(threads):
                    # quad x fragments (structural-equal rule: one index
                    # pattern per buffer per Parallel body):
                    #   xa0[c]=x[4c]  xa1[c]=x[4c+1]  xb0[c]=x[4c+2]  xb1[c]=x[4c+3]
                    xa0 = T.alloc_fragment((kh,), "float32")
                    xa1 = T.alloc_fragment((kh,), "float32")
                    xb0 = T.alloc_fragment((kh,), "float32")
                    xb1 = T.alloc_fragment((kh,), "float32")
                    for c in T.Parallel(kh):
                        xa0[c] = T.Cast("float32", x_ub[4 * c])
                    for c in T.Parallel(kh):
                        xa1[c] = T.Cast("float32", x_ub[4 * c + 1])
                    for c in T.Parallel(kh):
                        xb0[c] = T.Cast("float32", x_ub[4 * c + 2])
                    for c in T.Parallel(kh):
                        xb1[c] = T.Cast("float32", x_ub[4 * c + 3])
                    for q in T.serial(slab // threads):
                        accA = T.alloc_fragment((threads,), "float32")
                        accB = T.alloc_fragment((threads,), "float32")
                        for n in T.Parallel(threads):
                            accA[n] = T.float32(0)
                            accB[n] = T.float32(0)
                        for c in T.serial(kw):
                            plo = T.alloc_fragment((threads,), "float16")
                            phi = T.alloc_fragment((threads,), "float16")
                            qlo = T.alloc_fragment((threads,), "float16")
                            qhi = T.alloc_fragment((threads,), "float16")
                            for n in T.Parallel(threads):
                                wlo = a_ub[2 * c, q * threads + n]
                                whi = a_ub[2 * c + 1, q * threads + n]
                                plo[n] = T.reinterpret(
                                    T.Cast("uint16", wlo & T.uint32(0xFFFF)), "float16")
                                phi[n] = T.reinterpret(
                                    T.Cast("uint16", wlo >> T.uint32(16)), "float16")
                                qlo[n] = T.reinterpret(
                                    T.Cast("uint16", whi & T.uint32(0xFFFF)), "float16")
                                qhi[n] = T.reinterpret(
                                    T.Cast("uint16", whi >> T.uint32(16)), "float16")
                            for n in T.Parallel(threads):
                                accA[n] = accA[n] + T.Cast("float32", plo[n]) * xa0[c]
                                accB[n] = accB[n] + T.Cast("float32", phi[n]) * xa1[c]
                                accA[n] = accA[n] + T.Cast("float32", qlo[n]) * xb0[c]
                                accB[n] = accB[n] + T.Cast("float32", qhi[n]) * xb1[c]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + q * threads + n] = accA[n] + accB[n]
            T.copy(out_ub, Out)

    return main


BUILDERS = {
    "v0_base": build_v0_base,
    "v1_burst": build_v1_burst,
    "v2_pair": build_v2_pair,
    "v3_row": build_v3_row,
}
