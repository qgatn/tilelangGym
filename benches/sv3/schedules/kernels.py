"""SV3 GEMV schedule TileLang kernels (the schedules/ tree).

Out[n] = sum_k A[n, k] * x[k], fp16 in, fp32 out.

Read with: benches/sv3/simt-uarch/SV3-theoretical.md
Compile:   python compile.py <name> N K [threads] [--nosync]

Published ladder (§3) — file order matches the doc:
  build_thread_ub       §3.1 Base (default TileLang buffer GEMV)
  build_ub_methodo      §3.2 Op-matched
  build_ub_serial       §3.3 Serial staging
  build_ub_warp_pipe    §3.4 Strip / warp-pipeline
  build_ub_dbl_buf      §3.5 Double-buffer (prefetch strip t+1)
  build_ub_dbl_buf_quad §3.6 Quad-buffer (4 strips in flight)
  build_ub_quad_acc     §3.7 Quad-acc (ILP chains + straight-line; needs n_strips==4)

Exploration / dead ends (§5) — kept for provenance:
  build_ub_dbl_buf_twice  I-cache probe (same body twice per slab)
  build_ub_vfsplit        UB relayout attempt (does NOT compile)
  build_ub_quad_reord     compute-first reorder (worse than stock)
  build_ub_quad_ilv       k-chunk F2F/FFMA interleave (overlap yes, span unchanged)

Legacy ANALYSIS.md builders (thread/warp/simd) live in kernels_legacy.py.
Numbers: results/second_vf.txt, results/lds_improve.txt.
"""

import tilelang.ascend.language as T

from kernels_legacy import build_thread_ub  # §3.1 Base — re-exported for compile.py

THREADS = 256
WARP = 32


def _check(N: int, K: int) -> None:
    if N < 32 or N & (N - 1):
        raise ValueError(f"N={N} must be a power of two and at least 32")
    if K < 8 or K & (K - 1):
        raise ValueError(f"K={K} must be a power of two and at least 8")


# ---------------------------------------------------------------------------
# Published ladder (§3)
# ---------------------------------------------------------------------------

def build_ub_methodo(N: int, K: int, threads: int | None = None):
    """§3.2: case shape, single-op methodology inside the K walk."""
    _check(N, K)
    threads = threads or THREADS
    if N % threads:
        raise ValueError("N must divide threads")
    slab = N // 2  # two launches like the case at N=4096

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((slab, K), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A[s * slab : (s + 1) * slab, :], a_ub)
                with T.SimtVF(threads):
                    acc = T.alloc_fragment((slab,), "float32")
                    for n in T.Parallel(slab):
                        acc[n] = T.float32(0)
                    # stage 1+2 per thread: load the thread's x once, convert once
                    xf = T.alloc_fragment((K,), "float32")
                    for k in T.Parallel(K):
                        xf[k] = T.Cast("float32", x_ub[k])
                    # stage 3: batched multiply over the thread's slab columns
                    for k in T.serial(K):
                        for n in T.Parallel(slab):
                            acc[n] = acc[n] + T.Cast("float32", a_ub[n, k]) * xf[k]
                    for n in T.Parallel(slab):
                        out_ub[s * slab + n] = acc[n]
            T.copy(out_ub, Out)

    return main

def build_ub_serial(N: int, K: int, threads: int | None = None):
    """§3.3: serial LDS -> F2F -> FFMA -> STS over the whole slab."""
    _check(N, K)
    threads = threads or THREADS
    if N % threads:
        raise ValueError("N must divide threads")
    slab = N // 2

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((slab, K), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A[s * slab : (s + 1) * slab, :], a_ub)
                with T.SimtVF(threads):
                    acc = T.alloc_fragment((slab,), "float32")
                    xf = T.alloc_fragment((K,), "float32")
                    af = T.alloc_fragment((slab, K), "float32")
                    # stage 1+2: load + convert ALL values first
                    for k in T.Parallel(K):
                        xf[k] = T.Cast("float32", x_ub[k])
                    for n, k in T.Parallel(slab, K):
                        af[n, k] = T.Cast("float32", a_ub[n, k])
                    # stage 3: pure multiply
                    for n in T.Parallel(slab):
                        acc[n] = T.float32(0)
                    for k in T.serial(K):
                        for n in T.Parallel(slab):
                            acc[n] = acc[n] + af[n, k] * xf[k]
                    # stage 4: store
                    for n in T.Parallel(slab):
                        out_ub[s * slab + n] = acc[n]
            T.copy(out_ub, Out)

    return main

def build_ub_warp_pipe(N: int, K: int, threads: int | None = None):
    """§3.4: strip / warp-pipeline schedule.

    Same live state as serial (xf, af, acc), but the slab is processed in
    strips of `threads` rows (one row per thread). Each strip runs
    LDS+F2F → FFMA → STS before the next strip starts — so later work is not
    gated on a whole-slab LDS drain before any convert/multiply begins.
    """
    _check(N, K)
    threads = threads or THREADS
    if N % threads:
        raise ValueError("N must divide threads")
    slab = N // 2
    if slab % threads:
        raise ValueError("slab must be divisible by threads")
    n_strips = slab // threads  # e.g. 1024/256 = 4

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((slab, K), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A[s * slab : (s + 1) * slab, :], a_ub)
                with T.SimtVF(threads):
                    # strip-local scratch (smaller than serial's whole-slab af)
                    acc = T.alloc_fragment((threads,), "float32")
                    xf = T.alloc_fragment((K,), "float32")
                    af = T.alloc_fragment((threads, K), "float32")
                    for k in T.Parallel(K):
                        xf[k] = T.Cast("float32", x_ub[k])
                    for t in T.serial(n_strips):
                        for n, k in T.Parallel(threads, K):
                            af[n, k] = T.Cast(
                                "float32", a_ub[t * threads + n, k]
                            )
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + t * threads + n] = acc[n]
            T.copy(out_ub, Out)

    return main

def build_ub_dbl_buf(N: int, K: int, threads: int | None = None):
    """§3.5: double-buffered strip pipeline (overlap LDS with FFMA).

    Prolog loads strip 0 into af0. Each steady step (Python-unrolled) prefetches
    strip t+1 into the idle half, then multiplies+stores strip t from the ready
    half so in-flight LDS of t+1 can overlap FFMA of t. Epilog mac+stores last.
    """
    _check(N, K)
    threads = threads or THREADS
    if N % threads:
        raise ValueError("N must divide threads")
    slab = N // 2
    if slab % threads:
        raise ValueError("slab must be divisible by threads")
    n_strips = slab // threads
    if n_strips < 2:
        raise ValueError("dbl_buf needs at least 2 strips")

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((slab, K), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A[s * slab : (s + 1) * slab, :], a_ub)
                with T.SimtVF(threads):
                    acc = T.alloc_fragment((threads,), "float32")
                    xf = T.alloc_fragment((K,), "float32")
                    af0 = T.alloc_fragment((threads, K), "float32")
                    af1 = T.alloc_fragment((threads, K), "float32")
                    for k in T.Parallel(K):
                        xf[k] = T.Cast("float32", x_ub[k])

                    # Prolog: load strip 0 → af0
                    for n, k in T.Parallel(threads, K):
                        af0[n, k] = T.Cast("float32", a_ub[n, k])

                    # Steady (unrolled): prefetch t+1, then mac+store t
                    for t in range(n_strips - 1):
                        if t % 2 == 0:
                            for n, k in T.Parallel(threads, K):
                                af1[n, k] = T.Cast(
                                    "float32",
                                    a_ub[(t + 1) * threads + n, k],
                                )
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc[n] = acc[n] + af0[n, k] * xf[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                        else:
                            for n, k in T.Parallel(threads, K):
                                af0[n, k] = T.Cast(
                                    "float32",
                                    a_ub[(t + 1) * threads + n, k],
                                )
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc[n] = acc[n] + af1[n, k] * xf[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]

                    # Epilog: last strip already in af0 (even last) or af1 (odd last)
                    t_last = n_strips - 1
                    if t_last % 2 == 0:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af0[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + t_last * threads + n] = acc[n]
                    else:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af1[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + t_last * threads + n] = acc[n]
            T.copy(out_ub, Out)

    return main

def build_ub_dbl_buf_quad(N: int, K: int, threads: int | None = None):
    """§3.6: quad-buffered strip pipeline (4 strips in flight).

    dbl_buf keeps one strip (2 uint2 LDS issues) in flight per thread, so the
    ~28-cycle LDS return latency is not covered at 8 warps: the VF[1] trace
    shows three short LDS bursts separated by ~270-tick drains. Prefetching
    strips t+1..t+3 before the FFMA/STS of strip t puts 8 uint2 issues in
    flight per thread — the whole slab's loads issue as one burst.

    Needs n_strips >= 4 (N=2048, t=256 -> slab=1024, strips=4: prolog loads
    strips 0..2, the single steady step prefetches strip 3 while strip 0 is
    multiplied+stored, epilog macs strips 1..3).
    """
    _check(N, K)
    threads = threads or THREADS
    if N % threads:
        raise ValueError("N must divide threads")
    slab = N // 2
    if slab % threads:
        raise ValueError("slab must be divisible by threads")
    n_strips = slab // threads
    if n_strips < 4:
        raise ValueError("dbl_buf_quad needs at least 4 strips")

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((slab, K), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A[s * slab : (s + 1) * slab, :], a_ub)
                with T.SimtVF(threads):
                    acc = T.alloc_fragment((threads,), "float32")
                    xf = T.alloc_fragment((K,), "float32")
                    af0 = T.alloc_fragment((threads, K), "float32")
                    af1 = T.alloc_fragment((threads, K), "float32")
                    af2 = T.alloc_fragment((threads, K), "float32")
                    af3 = T.alloc_fragment((threads, K), "float32")
                    for k in T.Parallel(K):
                        xf[k] = T.Cast("float32", x_ub[k])

                    # Prolog: strips 0..2 in flight before any compute
                    for n, k in T.Parallel(threads, K):
                        af0[n, k] = T.Cast("float32", a_ub[n, k])
                    for n, k in T.Parallel(threads, K):
                        af1[n, k] = T.Cast(
                            "float32", a_ub[threads + n, k]
                        )
                    for n, k in T.Parallel(threads, K):
                        af2[n, k] = T.Cast(
                            "float32", a_ub[(2 * threads) + n, k]
                        )

                    # Steady: prefetch strip t+3, then mac+store strip t
                    for t in range(n_strips - 3):
                        if (t % 4) == 0:
                            for n, k in T.Parallel(threads, K):
                                af3[n, k] = T.Cast(
                                    "float32",
                                    a_ub[((t + 3) * threads) + n, k],
                                )
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc[n] = acc[n] + af0[n, k] * xf[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                        elif (t % 4) == 1:
                            for n, k in T.Parallel(threads, K):
                                af0[n, k] = T.Cast(
                                    "float32",
                                    a_ub[((t + 3) * threads) + n, k],
                                )
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc[n] = acc[n] + af1[n, k] * xf[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                        elif (t % 4) == 2:
                            for n, k in T.Parallel(threads, K):
                                af1[n, k] = T.Cast(
                                    "float32",
                                    a_ub[((t + 3) * threads) + n, k],
                                )
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc[n] = acc[n] + af2[n, k] * xf[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                        else:
                            for n, k in T.Parallel(threads, K):
                                af2[n, k] = T.Cast(
                                    "float32",
                                    a_ub[((t + 3) * threads) + n, k],
                                )
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc[n] = acc[n] + af3[n, k] * xf[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]

                    # Epilog: mac+store strips n_strips-3 .. n_strips-1.
                    # Strip j lives in af[j%4], so branch k=(n_strips-3)%4
                    # reads af[k], af[(k+1)%4], af[(k+2)%4] in order.
                    if ((n_strips - 3) % 4) == 0:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af0[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 3) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af1[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 2) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af2[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 1) * threads + n] = acc[n]
                    elif ((n_strips - 3) % 4) == 1:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af1[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 3) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af2[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 2) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af3[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 1) * threads + n] = acc[n]
                    elif ((n_strips - 3) % 4) == 2:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af2[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 3) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af3[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 2) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af0[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 1) * threads + n] = acc[n]
                    else:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af3[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 3) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af0[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 2) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af1[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 1) * threads + n] = acc[n]
            T.copy(out_ub, Out)

    return main

def build_ub_quad_acc(N: int, K: int, threads: int | None = None):
    """§3.7: quad pipeline with interleaved FFMA chains (needs n_strips==4).

    After barrier removal the VF tail is the epilog: 3 strips x 8-deep serial
    FFMA dependency chain (acc += af[n,k]*xf[k], k=0..7) — ~11.5 cycles per
    link, ~285 ticks for 24 FFMAs. The three strips are mutually independent
    (different af buffers, different outputs), so their chains can interleave:
    one accumulator per strip, all three advanced in the same k loop -> 3-way
    ILP. The steady mac gets an even/odd 2-chain split (depth 8 -> 4).
    Prefetch/prolog layout identical to dbl_buf_quad.
    """
    _check(N, K)
    threads = threads or THREADS
    if N % threads:
        raise ValueError("N must divide threads")
    slab = N // 2
    if slab % threads:
        raise ValueError("slab must be divisible by threads")
    n_strips = slab // threads
    if n_strips != 4:
        raise ValueError("quad_acc requires exactly 4 strips (n_strips == 4)")

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((slab, K), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A[s * slab : (s + 1) * slab, :], a_ub)
                with T.SimtVF(threads):
                    # One accumulator per chain; fragments are SPLIT so each
                    # buffer sees exactly one index expression per parallel
                    # body (structural-equal rule): *a/*b hold cols 0..3 / 4..7.
                    accA = T.alloc_fragment((threads,), "float32")
                    accB = T.alloc_fragment((threads,), "float32")
                    e0 = T.alloc_fragment((threads,), "float32")
                    e0b = T.alloc_fragment((threads,), "float32")
                    e1 = T.alloc_fragment((threads,), "float32")
                    e1b = T.alloc_fragment((threads,), "float32")
                    e2 = T.alloc_fragment((threads,), "float32")
                    e2b = T.alloc_fragment((threads,), "float32")
                    xfa = T.alloc_fragment((4,), "float32")
                    xfb = T.alloc_fragment((4,), "float32")
                    af0a = T.alloc_fragment((threads, 4), "float32")
                    af0b = T.alloc_fragment((threads, 4), "float32")
                    af1a = T.alloc_fragment((threads, 4), "float32")
                    af1b = T.alloc_fragment((threads, 4), "float32")
                    af2a = T.alloc_fragment((threads, 4), "float32")
                    af2b = T.alloc_fragment((threads, 4), "float32")
                    af3a = T.alloc_fragment((threads, 4), "float32")
                    af3b = T.alloc_fragment((threads, 4), "float32")
                    for k in T.Parallel(4):
                        xfa[k] = T.Cast("float32", x_ub[k])
                        xfb[k] = T.Cast("float32", x_ub[4 + k])

                    # Prolog: strips 0..2 in flight before any compute
                    for n, k in T.Parallel(threads, 4):
                        af0a[n, k] = T.Cast("float32", a_ub[n, k])
                        af0b[n, k] = T.Cast("float32", a_ub[n, 4 + k])
                    for n, k in T.Parallel(threads, 4):
                        af1a[n, k] = T.Cast("float32", a_ub[threads + n, k])
                        af1b[n, k] = T.Cast("float32", a_ub[threads + n, 4 + k])
                    for n, k in T.Parallel(threads, 4):
                        af2a[n, k] = T.Cast("float32", a_ub[(2 * threads) + n, k])
                        af2b[n, k] = T.Cast("float32", a_ub[(2 * threads) + n, 4 + k])

                    for n in T.Parallel(threads):
                        accA[n] = T.float32(0)
                        accB[n] = T.float32(0)
                        e0[n] = T.float32(0)
                        e0b[n] = T.float32(0)
                        e1[n] = T.float32(0)
                        e1b[n] = T.float32(0)
                        e2[n] = T.float32(0)
                        e2b[n] = T.float32(0)

                    # Steady: prefetch strip 3, then mac strip 0 over the two
                    # half-fragments (two chains, dependency depth 4 each)
                    for n, k in T.Parallel(threads, 4):
                        af3a[n, k] = T.Cast("float32", a_ub[(3 * threads) + n, k])
                        af3b[n, k] = T.Cast("float32", a_ub[(3 * threads) + n, 4 + k])
                    for k in T.serial(4):
                        for n in T.Parallel(threads):
                            accA[n] = accA[n] + af0a[n, k] * xfa[k]
                            accB[n] = accB[n] + af0b[n, k] * xfb[k]
                    for n in T.Parallel(threads):
                        out_ub[s * slab + n] = accA[n] + accB[n]

                    # Epilog: strips 1..3, one accumulator PER HALF (6 fully
                    # independent chains), all advanced in ONE k loop ->
                    # 6-way ILP at dependency depth 4 (stock: 3 chains x 8).
                    for k in T.serial(4):
                        for n in T.Parallel(threads):
                            e0[n] = e0[n] + af1a[n, k] * xfa[k]
                            e0b[n] = e0b[n] + af1b[n, k] * xfb[k]
                            e1[n] = e1[n] + af2a[n, k] * xfa[k]
                            e1b[n] = e1b[n] + af2b[n, k] * xfb[k]
                            e2[n] = e2[n] + af3a[n, k] * xfa[k]
                            e2b[n] = e2b[n] + af3b[n, k] * xfb[k]
                    for n in T.Parallel(threads):
                        out_ub[s * slab + threads + n] = e0[n] + e0b[n]
                        out_ub[s * slab + (2 * threads) + n] = e1[n] + e1b[n]
                        out_ub[s * slab + (3 * threads) + n] = e2[n] + e2b[n]
            T.copy(out_ub, Out)

    return main

# ---------------------------------------------------------------------------
# Exploration / dead ends (§5)
# ---------------------------------------------------------------------------

def build_ub_dbl_buf_twice(N: int, K: int, threads: int | None = None):
    """§5: identical dbl_buf VF body run twice per slab (I-cache probe).

    Same work twice on purpose — diagnosis for Figure 5's long 0-IPC blank.
    Hypothesis: first VF pays the cold / post-END gap; second launch of the
    same body should be cleaner. Not a rate-table winner (double work).
    Body inlined twice (TileLang cannot call a Python helper from prim_func).
    """
    _check(N, K)
    threads = threads or THREADS
    if N % threads:
        raise ValueError("N must divide threads")
    slab = N // 2
    if slab % threads:
        raise ValueError("slab must be divisible by threads")
    n_strips = slab // threads
    if n_strips < 2:
        raise ValueError("dbl_buf_twice needs at least 2 strips")

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((slab, K), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A[s * slab : (s + 1) * slab, :], a_ub)
                # --- identical VF #1 (warm / first) ---
                with T.SimtVF(threads):
                    acc = T.alloc_fragment((threads,), "float32")
                    xf = T.alloc_fragment((K,), "float32")
                    af0 = T.alloc_fragment((threads, K), "float32")
                    af1 = T.alloc_fragment((threads, K), "float32")
                    for k in T.Parallel(K):
                        xf[k] = T.Cast("float32", x_ub[k])
                    for n, k in T.Parallel(threads, K):
                        af0[n, k] = T.Cast("float32", a_ub[n, k])
                    for t in range(n_strips - 1):
                        if t % 2 == 0:
                            for n, k in T.Parallel(threads, K):
                                af1[n, k] = T.Cast(
                                    "float32", a_ub[(t + 1) * threads + n, k]
                                )
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc[n] = acc[n] + af0[n, k] * xf[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                        else:
                            for n, k in T.Parallel(threads, K):
                                af0[n, k] = T.Cast(
                                    "float32", a_ub[(t + 1) * threads + n, k]
                                )
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc[n] = acc[n] + af1[n, k] * xf[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                    t_last = n_strips - 1
                    if t_last % 2 == 0:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af0[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + t_last * threads + n] = acc[n]
                    else:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af1[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + t_last * threads + n] = acc[n]
                # --- identical VF #2 (redundant replay) ---
                with T.SimtVF(threads):
                    acc2 = T.alloc_fragment((threads,), "float32")
                    xf2 = T.alloc_fragment((K,), "float32")
                    af0b = T.alloc_fragment((threads, K), "float32")
                    af1b = T.alloc_fragment((threads, K), "float32")
                    for k in T.Parallel(K):
                        xf2[k] = T.Cast("float32", x_ub[k])
                    for n, k in T.Parallel(threads, K):
                        af0b[n, k] = T.Cast("float32", a_ub[n, k])
                    for t in range(n_strips - 1):
                        if t % 2 == 0:
                            for n, k in T.Parallel(threads, K):
                                af1b[n, k] = T.Cast(
                                    "float32", a_ub[(t + 1) * threads + n, k]
                                )
                            for n in T.Parallel(threads):
                                acc2[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc2[n] = acc2[n] + af0b[n, k] * xf2[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc2[n]
                        else:
                            for n, k in T.Parallel(threads, K):
                                af0b[n, k] = T.Cast(
                                    "float32", a_ub[(t + 1) * threads + n, k]
                                )
                            for n in T.Parallel(threads):
                                acc2[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc2[n] = acc2[n] + af1b[n, k] * xf2[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc2[n]
                    t_last = n_strips - 1
                    if t_last % 2 == 0:
                        for n in T.Parallel(threads):
                            acc2[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc2[n] = acc2[n] + af0b[n, k] * xf2[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + t_last * threads + n] = acc2[n]
                    else:
                        for n in T.Parallel(threads):
                            acc2[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc2[n] = acc2[n] + af1b[n, k] * xf2[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + t_last * threads + n] = acc2[n]
            T.copy(out_ub, Out)

    return main

def build_ub_vfsplit(N: int, K: int, threads: int | None = None):
    """§5 (does NOT compile): strided UB relayout via exchange buffer.

    Two VFs per slab: VF A loads A with the pure-bench strided word pattern
    (lane L reads word L + threads*i, 4-B lane stride) and stores per-word
    partials to a flat exchange buffer p_ub[w]; VF B reloads lane-per-row and
    finishes each dot. The store half alone is the clean relayout.

    Blocked by Ascend layout inference (ProveFragmentContains ->
    InverseAffineIterMap, "CanProveEqual(abs(scale),1)"):
      - store-side form  p_ub[w // 4, w % 4]  (2-D row-major) -> nonlinear
        index, inverter cannot invert;
      - reload-side form p_ub[base + n*4 + q] (flat, lane stride 4) -> scale-4
        iters, inverter requires scale 1.
    Both halves of the transpose need layout support the inverter lacks today.
    See results/lds_improve.txt experiment 3.
    """
    _check(N, K)
    if K != 8:
        raise ValueError("vfsplit requires K=8")
    threads = threads or THREADS
    if N % threads:
        raise ValueError("N must divide threads")
    slab = N // 2
    if slab % threads:
        raise ValueError("slab must be divisible by threads")
    n_strips = slab // threads
    words_per_strip = threads * 4          # 256 rows * 4 words (K=8 fp16)
    n_groups = (slab * K) // (2 * threads)  # u32 loads per thread (16)

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((slab, K), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            p_ub = T.alloc_shared((slab * 4,), "float32")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A[s * slab : (s + 1) * slab, :], a_ub)
                # ---- VF A: strided pure-load band + per-word partial sums ----
                with T.SimtVF(threads):
                    xf = T.alloc_fragment((K,), "float32")
                    # word view of a_ub: word w = halfs {2w, 2w+1}
                    a_w = T.view(a_ub, (slab * K // 2,), "uint32")
                    aw = T.alloc_fragment((n_groups, threads), "uint32")
                    for k in T.Parallel(K):
                        xf[k] = T.Cast("float32", x_ub[k])
                    # pure-bench strided pattern: word w = gi*threads + L
                    for gi, L in T.Parallel(n_groups, threads):
                        aw[gi, L] = a_w[gi * threads + L]
                    hf0 = T.alloc_fragment((n_groups, threads), "float32")
                    hf1 = T.alloc_fragment((n_groups, threads), "float32")
                    for gi, L in T.Parallel(n_groups, threads):
                        hf0[gi, L] = T.Cast(
                            "float32",
                            T.Cast("float16", aw[gi, L] & T.uint32(0xFFFF)),
                        )
                    for gi, L in T.Parallel(n_groups, threads):
                        hf1[gi, L] = T.Cast(
                            "float32",
                            T.Cast(
                                "float16",
                                (aw[gi, L] >> T.uint32(16)) & T.uint32(0xFFFF),
                            ),
                        )
                    even = T.alloc_fragment((n_groups, threads), "float32")
                    odd = T.alloc_fragment((n_groups, threads), "float32")
                    for gi, L in T.Parallel(n_groups, threads):
                        w = gi * threads + L
                        even[gi, L] = hf0[gi, L] * xf[(w % 4) * 2]
                    for gi, L in T.Parallel(n_groups, threads):
                        w = gi * threads + L
                        odd[gi, L] = hf1[gi, L] * xf[(w % 4) * 2 + 1]
                    for gi, L in T.Parallel(n_groups, threads):
                        w = gi * threads + L
                        p_ub[w] = even[gi, L] + odd[gi, L]
                    T.sync_threads()
                # ---- VF B: reload partials lane-per-row, sum, store ----
                with T.SimtVF(threads):
                    acc = T.alloc_fragment((threads,), "float32")
                    pf0 = T.alloc_fragment((threads,), "float32")
                    pf1 = T.alloc_fragment((threads,), "float32")
                    pf2 = T.alloc_fragment((threads,), "float32")
                    pf3 = T.alloc_fragment((threads,), "float32")
                    for t in T.serial(n_strips):
                        base = t * threads * 4
                        for n in T.Parallel(threads):
                            pf0[n] = p_ub[base + n * 4]
                        for n in T.Parallel(threads):
                            pf1[n] = p_ub[base + n * 4 + 1]
                        for n in T.Parallel(threads):
                            pf2[n] = p_ub[base + n * 4 + 2]
                        for n in T.Parallel(threads):
                            pf3[n] = p_ub[base + n * 4 + 3]
                        for n in T.Parallel(threads):
                            acc[n] = pf0[n] + pf1[n] + pf2[n] + pf3[n]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + t * threads + n] = acc[n]
            T.copy(out_ub, Out)

    return main

def build_ub_quad_reord(N: int, K: int, threads: int | None = None):
    """§5: quad pipeline with reordered steady step (compute first).

    Stock quad steady step is: prefetch strip t+3 (LDS+F2F) THEN mac+store
    strip t. Trace shows F2F bulk sits ahead of FFMA in program order, so
    FFMA starts only after most of the prefetch band. Here the steady step is
    reversed: mac+store strip t first, prefetch strip t+3 last, so the LDS/F2F
    band overlaps the STS drain of the previous half-step. Everything else
    (prolog, epilog, buffers) is identical to build_ub_dbl_buf_quad.
    """
    _check(N, K)
    threads = threads or THREADS
    if N % threads:
        raise ValueError("N must divide threads")
    slab = N // 2
    if slab % threads:
        raise ValueError("slab must be divisible by threads")
    n_strips = slab // threads
    if n_strips < 4:
        raise ValueError("quad_reord needs at least 4 strips")

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((slab, K), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A[s * slab : (s + 1) * slab, :], a_ub)
                with T.SimtVF(threads):
                    acc = T.alloc_fragment((threads,), "float32")
                    xf = T.alloc_fragment((K,), "float32")
                    af0 = T.alloc_fragment((threads, K), "float32")
                    af1 = T.alloc_fragment((threads, K), "float32")
                    af2 = T.alloc_fragment((threads, K), "float32")
                    af3 = T.alloc_fragment((threads, K), "float32")
                    for k in T.Parallel(K):
                        xf[k] = T.Cast("float32", x_ub[k])

                    # Prolog: strips 0..2 in flight before any compute
                    for n, k in T.Parallel(threads, K):
                        af0[n, k] = T.Cast("float32", a_ub[n, k])
                    for n, k in T.Parallel(threads, K):
                        af1[n, k] = T.Cast(
                            "float32", a_ub[threads + n, k]
                        )
                    for n, k in T.Parallel(threads, K):
                        af2[n, k] = T.Cast(
                            "float32", a_ub[(2 * threads) + n, k]
                        )

                    # Steady: mac+store strip t FIRST, prefetch strip t+3 LAST
                    for t in range(n_strips - 3):
                        if (t % 4) == 0:
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc[n] = acc[n] + af0[n, k] * xf[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                            for n, k in T.Parallel(threads, K):
                                af3[n, k] = T.Cast(
                                    "float32",
                                    a_ub[((t + 3) * threads) + n, k],
                                )
                        elif (t % 4) == 1:
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc[n] = acc[n] + af1[n, k] * xf[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                            for n, k in T.Parallel(threads, K):
                                af0[n, k] = T.Cast(
                                    "float32",
                                    a_ub[((t + 3) * threads) + n, k],
                                )
                        elif (t % 4) == 2:
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc[n] = acc[n] + af2[n, k] * xf[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                            for n, k in T.Parallel(threads, K):
                                af1[n, k] = T.Cast(
                                    "float32",
                                    a_ub[((t + 3) * threads) + n, k],
                                )
                        else:
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for k in T.serial(K):
                                for n in T.Parallel(threads):
                                    acc[n] = acc[n] + af3[n, k] * xf[k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                            for n, k in T.Parallel(threads, K):
                                af2[n, k] = T.Cast(
                                    "float32",
                                    a_ub[((t + 3) * threads) + n, k],
                                )

                    # Epilog: mac+store strips n_strips-3 .. n_strips-1.
                    # Strip j lives in af[j%4], so branch k=(n_strips-3)%4
                    # reads af[k], af[(k+1)%4], af[(k+2)%4] in order.
                    if ((n_strips - 3) % 4) == 0:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af0[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 3) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af1[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 2) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af2[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 1) * threads + n] = acc[n]
                    elif ((n_strips - 3) % 4) == 1:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af1[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 3) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af2[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 2) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af3[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 1) * threads + n] = acc[n]
                    elif ((n_strips - 3) % 4) == 2:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af2[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 3) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af3[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 2) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af0[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 1) * threads + n] = acc[n]
                    else:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af3[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 3) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af0[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 2) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af1[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 1) * threads + n] = acc[n]
            T.copy(out_ub, Out)

    return main

def build_ub_quad_ilv(N: int, K: int, threads: int | None = None, kc: int = 2):
    """§5: quad pipeline with k-chunk interleaved steady step.

    Instead of a monolithic prefetch block followed by a monolithic mac, the
    steady step walks k in chunks of kc: per chunk, convert that chunk of
    strip t+3 (LDS+F2F) then FFMA that chunk of strip t. The prefetch band is
    spread evenly across the whole steady body, overlapping F2F/LDS with FFMA
    at chunk granularity. Same buffers/prolog/epilog as dbl_buf_quad.
    """
    _check(N, K)
    threads = threads or THREADS
    if N % threads:
        raise ValueError("N must divide threads")
    if K % kc:
        raise ValueError("K must divide kc")
    slab = N // 2
    if slab % threads:
        raise ValueError("slab must be divisible by threads")
    n_strips = slab // threads
    if n_strips < 4:
        raise ValueError("quad_ilv needs at least 4 strips")

    @T.prim_func
    def main(
        A: T.Tensor((N, K), "float16"),
        X: T.Tensor((K,), "float16"),
        Out: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((slab, K), "float16")
            x_ub = T.alloc_shared((K,), "float16")
            out_ub = T.alloc_shared((N,), "float32")
            T.copy(X, x_ub)
            for s in T.serial(N // slab):
                T.copy(A[s * slab : (s + 1) * slab, :], a_ub)
                with T.SimtVF(threads):
                    acc = T.alloc_fragment((threads,), "float32")
                    xf = T.alloc_fragment((K,), "float32")
                    af0 = T.alloc_fragment((threads, K), "float32")
                    af1 = T.alloc_fragment((threads, K), "float32")
                    af2 = T.alloc_fragment((threads, K), "float32")
                    af3 = T.alloc_fragment((threads, K), "float32")
                    for k in T.Parallel(K):
                        xf[k] = T.Cast("float32", x_ub[k])

                    # Prolog: strips 0..2 in flight before any compute
                    for n, k in T.Parallel(threads, K):
                        af0[n, k] = T.Cast("float32", a_ub[n, k])
                    for n, k in T.Parallel(threads, K):
                        af1[n, k] = T.Cast(
                            "float32", a_ub[threads + n, k]
                        )
                    for n, k in T.Parallel(threads, K):
                        af2[n, k] = T.Cast(
                            "float32", a_ub[(2 * threads) + n, k]
                        )

                    # Steady: per k-chunk, convert chunk of strip t+3, then
                    # mac chunk of strip t.
                    for t in range(n_strips - 3):
                        if (t % 4) == 0:
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for kb in T.serial(K // kc):
                                k0 = kb * kc
                                for n, k in T.Parallel(threads, kc):
                                    af3[n, k0 + k] = T.Cast(
                                        "float32",
                                        a_ub[((t + 3) * threads) + n, k0 + k],
                                    )
                                for k in T.serial(kc):
                                    for n in T.Parallel(threads):
                                        acc[n] = acc[n] + af0[n, k0 + k] * xf[k0 + k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                        elif (t % 4) == 1:
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for kb in T.serial(K // kc):
                                k0 = kb * kc
                                for n, k in T.Parallel(threads, kc):
                                    af0[n, k0 + k] = T.Cast(
                                        "float32",
                                        a_ub[((t + 3) * threads) + n, k0 + k],
                                    )
                                for k in T.serial(kc):
                                    for n in T.Parallel(threads):
                                        acc[n] = acc[n] + af1[n, k0 + k] * xf[k0 + k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                        elif (t % 4) == 2:
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for kb in T.serial(K // kc):
                                k0 = kb * kc
                                for n, k in T.Parallel(threads, kc):
                                    af1[n, k0 + k] = T.Cast(
                                        "float32",
                                        a_ub[((t + 3) * threads) + n, k0 + k],
                                    )
                                for k in T.serial(kc):
                                    for n in T.Parallel(threads):
                                        acc[n] = acc[n] + af2[n, k0 + k] * xf[k0 + k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]
                        else:
                            for n in T.Parallel(threads):
                                acc[n] = T.float32(0)
                            for kb in T.serial(K // kc):
                                k0 = kb * kc
                                for n, k in T.Parallel(threads, kc):
                                    af2[n, k0 + k] = T.Cast(
                                        "float32",
                                        a_ub[((t + 3) * threads) + n, k0 + k],
                                    )
                                for k in T.serial(kc):
                                    for n in T.Parallel(threads):
                                        acc[n] = acc[n] + af3[n, k0 + k] * xf[k0 + k]
                            for n in T.Parallel(threads):
                                out_ub[s * slab + t * threads + n] = acc[n]

                    # Epilog: mac+store strips n_strips-3 .. n_strips-1.
                    # Strip j lives in af[j%4], so branch k=(n_strips-3)%4
                    # reads af[k], af[(k+1)%4], af[(k+2)%4] in order.
                    if ((n_strips - 3) % 4) == 0:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af0[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 3) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af1[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 2) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af2[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 1) * threads + n] = acc[n]
                    elif ((n_strips - 3) % 4) == 1:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af1[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 3) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af2[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 2) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af3[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 1) * threads + n] = acc[n]
                    elif ((n_strips - 3) % 4) == 2:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af2[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 3) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af3[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 2) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af0[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 1) * threads + n] = acc[n]
                    else:
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af3[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 3) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af0[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 2) * threads + n] = acc[n]
                        for n in T.Parallel(threads):
                            acc[n] = T.float32(0)
                        for k in T.serial(K):
                            for n in T.Parallel(threads):
                                acc[n] = acc[n] + af1[n, k] * xf[k]
                        for n in T.Parallel(threads):
                            out_ub[s * slab + (n_strips - 1) * threads + n] = acc[n]
            T.copy(out_ub, Out)

    return main

