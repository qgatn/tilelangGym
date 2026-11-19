# lds-boost results — A / B / C

These three runs are the **old `A[n, k]`** quad-acc (28.51 B/cyc). They are why the layout changed. The current kernels are V1–V3 in this folder; the write-up is `lds_performance.md`.

Reference lines: pure pack 81.11 · pure size-scaled @256t 76.38 · scalar S1 34.79.

## A — packing works; the lane stride is the killer

`cce/a_row8_pack.asc`: same 16 KB slab, same **128 × LDS.b32 (RVECLD)** issues,
only the lane→address pattern differs (`logs/results_a.txt`):

| Layout | Index | Lane stride | LDS busy | B/cyc |
|---|---|---:|---:|---:|
| **row** (old quad-acc, `A[n, k]`) | `row=tid+r*T; src[row*4+j]` | **16 B** | 558 | **29.36** |
| **transposed** (pure-bench shape) | `src[tid + i*T]` | 4 B | 205 | **79.92** |
| flat per-thread (conflict control) | `src[tid*kL+i]` | – | 2096 | 7.82 |

Answers:

- **A: packed fp16x2 already happens.** The generated quad-acc loads are
  `uint1` = u32 (`half2`) — 128 B/warp issue (see `runs/quad_acc…/kernel.cpp`,
  `*(__ubuf__ uint1*)`). TileLang lowering is NOT the blocker.
- The 29.4 vs 79.9 gap is **pure 16-B-lane-stride bank replay**: same dtype,
  same issue width, same bytes — only the address pattern changed.
- 512t does not fix the row shape (26.4) and actually hurts transposed at this
  small slab (61.8; the pure 96-line needs WPT-scaled N).

That transposed address is what V1–V3 in this folder run. V1's load band is **63.75 B/cyc**. The write-up is `lds_performance.md`.

## B — 512 threads: slight LDS gain, net VF loss

Real kernels, same N=2048 K=8 shape, 2 slabs (`logs/results_b.txt`, rates from
VF[1]; t256 refs from SV3-theoretical Tables 6d/4):

| Kernel | t | VF[1] span | LDS B/cyc | el/cyc (full case) |
|---|---:|---:|---:|---:|
| quad_acc2 (n_strips=2) | 512 | 994 | **26.77** | 2.06 |
| quad_acc | 256 | 850 | **28.51** | 2.41 |
| dbl_buf | 512 | 1015 | **26.86** | 2.00 |
| dbl_buf | 256 | 1070 | **25.71** | 1.92 |
| serial | 512 | 1505 | **14.60** | 1.21 |
| serial | 256 | 1472 | **15.12** | 1.39 |

- LDS rate is **warp-count-insensitive here** (+1.1% dbl-buf, −1.2% serial,
  −2.3% quad-acc): the 16-B stride replay dominates, extra warps just queue
  behind the same conflicts.
- End-to-end VF **loses** ~17% at 512t (850→994; 1070→1015; 1472→1505) —
  wider warps dilute per-warp FFMA chains (quad-acc's ILP depth halves).
- **B verdict: 512t is not the fix.** Consistent with `lds_fp32_vs_bf16.md`
  §4.4 — thread-count ceilings only pay when latency (not conflicts) binds.

## C — IADD is NOT the kernel's problem

`logs/results_c.txt` (addr ops inside the LDS busy window):

| Trace | IADD_I / LEA in LDS busy | other-pipe ticks in LDS busy |
|---|---|---:|
| pure fp32 burst @256t (76.38) | **9 / 10** | 4218 |
| pure pack @256t (81.11) | **0 / 10** | 1566 |
| quad_acc VF[1] (28.51) | **0 / 0** | 9260 |
| dbl_buf VF[1] | 0 / 0 | 9997 |
| serial VF[1] | 0 / 0 | 11264 |
| base VF | 0 / 9 | 23986 |

- The single-op band carried a small `IADD_I`/`LEA` tax; the **kernel VF has
  none** (addressing is loop-hoisted base + immediate). So the single-op
  ~81/97 lines were measured *with* address ops — the kernel's 28.5 cannot be
  blamed on them.
- The kernel LDS busy window is drowned by **F2F/FFMA ticks** (9260 vs pure's
  4218/1566): the band stretches because the strip loop interleaves converts
  and MACs into the load trickle — a scheduling/shape artifact, not IADD.

## Bottom line

| Question | Answer |
|---|---|
| A: pack or stride? | Pack is already on. 16 B stride is 29.36; 4 B stride is 79.92. |
| B: 512t? | No. LDS flat, VF ~17% slower. |
| C: IADD? | No. None in the kernel's LDS band. |
| Where is the TileLang `A[k, n]`? | This folder: V1 63.75 B/cyc band, V3 0.47 el/cyc. Write-up: `lds_performance.md`. |

Provenance: `logs/results_a.txt`, `logs/results_b.txt`, `logs/results_c.txt`.
