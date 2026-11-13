# uarch

- Single-op SIMT microbenches on one block (`dav-3510`). Source: `../single-op/pure_burst.asc` (+ `lds_pure_burst.asc`, `hide32.asc`). Charts: opcode IPC + per-warp, from the trace's VF window.
- Index: `tid + i * threads` (warp coalesces 32 consecutive els / issue).
- **Busy** = union of `[issue, completion)` for the target opcode across warps.
- **Purity**: target opcode + forced helpers only (`IADD_I`/`LEA` addr, `LDS` preload for F2F/FFMA, checksum/`STS` sink). No spills. No cross-op contamination in LDS.
- **I-cache / first-launch**: every chart reads the trace's **VF[0]** window. Microbenches are single-launch; the known first-launch post-`END` blank (`I-cache_miss.md`) is **outside** the target-op busy union, so rates are unaffected. Multi-slab case kernels are the ones needing VF[1] accounting (`SV3-theoretical.md`).

## 1. Case config

- GEMV case: `N=4096`, `K=8`, 256t, `A`/`x` = fp16. Theory ref: `N=2048`, `K=8`, `ub-n2048-k8-t256`.

| Op | Dtype | B or els / warp-issue | Active cyc |
|---|---|---|---|
| LDS | fp16 | 64 B (scalar) / 128 B (pack) | 27–29 |
| F2F | fp16→fp32 | 32 (scalar) / 64 (pack) | 8 |
| FFMA | fp32 | 32 products | 11 |
| STS | fp32 | 128 B | 20 |

## 2. LDS

- Index: `tid + i * threads` (strided; warp coalesces). Contiguous `tid*WPT+i` bank-conflicts → ~4 B/cyc (rejected). Per-lane float4 also conflicts (31.21, `lds_variants_results.txt`).
- Sink = 1 checksum word/thread (`STS` outside load band). Source: `../single-op/lds_pure_burst.asc`.

### 2.1 Size-scaled peaks (fp32 LDS.b32)

| Threads | N | WPT | Issues | Busy | B/cyc | iss/cyc |
|---:|---:|---:|---:|---:|---:|---:|
| **256** | 24576 | 96 | 768 | 1287 | **76.38** | 0.597 |
| **512** | 49152 | 96 | 1536 | 2049 | **95.95** | 0.750 |
| **1024** | 49152 | 48 | 1536 | 2012 | **97.72** | 0.763 |

- @256t: denser WPT past 96 **drops** (WPT=192 → ~58). 8 warps cannot hide LDS latency — **76.38 is the 256t ceiling**.
- @512t: best at WPT=96; larger N softens (N=61440 → 91.98).
- @1024t: **97.72** — best measured; saturates with more warps. @2048t collapses (30.5).
- bf16x2 ties fp32 at 512t within ~1.5 B/cyc and **exactly at 1024t** (both busy 2012).
- SoftLib ~114 still open: `IADD_I` 1:1 with `LDS`; pointer-walk / pre-addr / unroll all CSE back (`lds_variants_results.txt`).

![LDS fp32 peak 256t](plots/lds_fp32_peak_t256.png)

![LDS fp32 peak 256t warps](plots/lds_fp32_peak_t256_warps.png)

![LDS fp32 peak 512t](plots/lds_fp32_peak_t512.png)

![LDS fp32 peak 512t warps](plots/lds_fp32_peak_t512_warps.png)

### 2.2 Schedules @256t (N = 8192, matched small tile)

| Schedule | Load shape | Issues | Busy | B/cyc | iss/cyc |
|---|---|---:|---:|---:|---:|
| S1 scalar fp16 | 64 B/issue | 256 | 471 | 34.79 | 0.54 |
| S2 `uint32` pack | 128 B/issue | 128 | 202 | **81.11** | 0.63 |
| S3 `half2` pack | 128 B/issue | 128 | 202 | **81.11** | 0.63 |
| S4 `bfloat16x2` pack | 128 B/issue | 128 | 202 | **81.11** | 0.63 |

- Small-tile matched compare: fp32 scalar (256 issues, 471, 69.57) ≡ bf16x2 doubled (256, 471, 69.57) — width packing never *beats* fp32; best case equality (`lds_fp32_vs_bf16.md`).

![LDS S1 IPC](plots/lds_s1_scalar_t256.png)

![LDS S1 warps](plots/lds_s1_scalar_t256_warps.png)

![LDS S2 IPC](plots/lds_s2_u32_t256.png)

![LDS S2 warps](plots/lds_s2_u32_t256_warps.png)

![LDS S4 IPC](plots/lds_s4_bf16x2_t256.png)

![LDS S4 warps](plots/lds_s4_bf16x2_t256_warps.png)

### 2.3 Comparison

| Case | Width | Busy | B/cyc | Note |
|---|---|---:|---:|---|
| S1 scalar @256 | 64 B | 471 | 34.79 | 2× issues |
| S2/S3/S4 pack @256 | 128 B | 202 | **81.11** | GEMV half-tile |
| fp32 size peak @256 | 128 B | 1287 | **76.38** | warp-limited ceiling |
| fp32 size peak @512 | 128 B | 2049 | **95.95** | |
| fp32/bf16x2 @1024 | 128 B | 2012 | **97.72** | best measured |

## 3. F2F

- Fixed work/thread: 32 el → `N = 32 × threads`. Preload `LDS` then convert band. el/cyc = `N / busy`.

### 3.1 Schedules

| Schedule | Band | 256t Issues | 256t Busy | 256t el/cyc | 512t el/cyc | 1024t el/cyc |
|---|---|---:|---:|---:|---:|---:|
| S1 scalar `__cvt_float` | stagger | 256 | 275 | 29.79 | 26.64 | 12.02 |
| S2 packed + `PRMT` | stagger + PRMT | 128 | 281 | 29.15 | 23.74 | – |
| S3 word-preload | stagger | 128 | 208 | 39.38 | 17.58 | 28.59 |
| **S4 barrier-aligned** | 4 co-issue/tick | **128** | **76** | **107.79** | **39.57** | **30.65** |

- **S4 wins at every T, but only 256t holds the ceiling**: busy stays 76 @256t, then 414 @512t and 1069 @1024t — 32 warps over-fill the issue window (iss/cyc 0.48).
- Multi-thread rows: `f2f_wpt32_results.txt` (128/256/512), `t1024_results.txt` (1024).

![F2F S4 IPC](plots/f2f_s4_barrier_t256.png)

![F2F S4 warps](plots/f2f_s4_barrier_t256_warps.png)

![F2F S4 barrier 512t](plots/pure_f2f_s4_barrier_t512.png)

![F2F S4 barrier 512t warps](plots/pure_f2f_s4_barrier_t512_warps.png)

![F2F S4 barrier 1024t](plots/pure_f2f_s4_barrier_t1024.png)

![F2F S4 barrier 1024t warps](plots/pure_f2f_s4_barrier_t1024_warps.png)

![F2F S1 IPC](plots/f2f_s1_scalar_t256.png)

![F2F S3 IPC](plots/f2f_s3_word_t256.png)

### 3.2 Takeaway

- Winner: **S4 — 107.79 el/cyc @256t**. Packing halves issues; the barrier is what makes the rate scale.
- The 107.79 line is a **256t-only property** (see §6 tradeoff).

## 4. FFMA

- Fixed `N=8192` (issues stay 256 at every T). Preload+convert both operands (`LDS`+`F2F`), then multiply band. No packed FFMA ISA.
- el/cyc = `8192 / busy`.

### 4.1 Schedules

| Schedule | Band | 256t Busy | 256t el/cyc | 512t el/cyc | 1024t el/cyc |
|---|---|---:|---:|---:|---:|
| S1 scalar | stagger | 479 | 17.10 | 9.91 | 8.54 |
| S2 float2-src | tighter schedule | 422 | 19.41 | 14.47 | – |
| **S3 barrier-aligned** | pure band | **392** | **20.90** | **24.67** | **26.01** |

- S3 improves with warps (20.90 → 24.67 → 26.01): fewer preload leaks per warp inside the barrier window.
- All S3 numbers are **lower bounds** — other warps' preload `F2F`/`LDS` can still sit in the busy union. Serial fused case reaches **32.38** (`SV3-theoretical.md` §3.3), which is the practical staged ceiling.

![FFMA S3 IPC](plots/ffma_s3_barrier_t256.png)

![FFMA S3 warps](plots/ffma_s3_barrier_t256_warps.png)

![FFMA S3 barrier 512t](plots/pure_ffma_s3_barrier_t512.png)

![FFMA S3 barrier 1024t](plots/pure_ffma_s3_barrier_t1024.png)

![FFMA S1 IPC](plots/ffma_s1_scalar_t256.png)

### 4.2 Takeaway

- Winner: **S3**. No packed FFMA on this ISA; measured microbench line stays a lower bound below serial's 32.38.

## 5. STS

| Threads | Issues | Busy | B/cyc |
|---:|---:|---:|---:|
| 256 | 256 | 355 | 92.30 |
| 512 | 512 | 687 | **95.34** |
| 1024 | 1024 | 1316 | **99.60** |

- Single coalesced schedule; every issue active 20 cyc. Rate scales mildly with warps — **99.60 @1024t is the best measured store rate**.

![STS S1 IPC](plots/sts_s1_t256.png)

![STS S1 warps](plots/sts_s1_t256_warps.png)

![STS coalesced 512t](plots/hide32_sts_t512.png)

![STS coalesced 1024t](plots/hide32_sts_t1024.png)

## 6. Conclusion

### 6.1 Winners per thread count

| T | LDS | F2F | FFMA | STS |
|---:|---:|---:|---:|---:|
| **256** | 76.38 | **107.79** | 20.90 | 92.30 |
| 512 | 95.95 | 39.57 | 24.67 | 95.34 |
| 1024 | **97.72** | 30.65 | **26.01** | **99.60** |

- **LDS / STS / FFMA** improve with warps. **F2F collapses** past 256t — the packed barrier band needs tight warp interleave.

### 6.2 256t vs 512t vs 1024t — SV3 GEMV serial model

Work: LDS 32784 B, F2F 16392 el, FFMA 16384 el, STS 8192 B (one kernel, `N=2048`, `K=8`).

| T | LDS cyc | F2F cyc | FFMA cyc | STS cyc | **Total** | el/cyc |
|---:|---:|---:|---:|---:|---:|---:|
| **256** | 429 | **152** | 784 | 89 | **1454** | **1.41** |
| 512 | 342 | 414 | 664 | 86 | **1506** | 1.36 |
| 1024 | 336 | 535 | 630 | 82 | **1583** | 1.29 |

- Warps buy ~190 cycles on LDS+FFMA+STS (256→1024t) but **cost ~383 on F2F** — net loss for a convert-heavy kernel.
- Break-even: 1024t needs F2F ≥ **40.3 el/cyc** to tie 256t (has 30.65). With FFMA at its true staged 32.38 everywhere, 1024t needs F2F ≥ **30.9** — a near-tie, still not a win.
- FFMA dominates the mix (~50% of cycles at every T); its rates here are lower bounds, so the 256t total (with staged FFMA 32.38 → **1176 cyc, 1.74 el/cyc**) is the strongest model line.
- **Verdict: 256t is the right launch for the F2F-heavy GEMV; 1024t only pays for pure streaming (LDS/STS-dominated, no converts).**

### 6.3 Ceiling summary

| Op | Best rate | Where | Note |
|---|---:|---|---|
| LDS | **97.72 B/cyc** | 1024t size-scaled | 256t ceiling 76.38; SoftLib ~114 open |
| F2F | **107.79 el/cyc** | 256t S4 barrier | 256t-only; fused ~47 |
| FFMA | **32.38 el/cyc** | serial fused 256t | microbench S3 is a lower bound |
| STS | **99.60 B/cyc** | 1024t coalesced | ~flat 92–100 |

## Appendix — provenance

- Env: Ascend950PR / npusim, `dav-3510`, bisheng `-O2 --npu-arch=dav-3510`, CANN `9.2.0-beta.2`.
- Sources: `../single-op/pure_burst.asc` (`-DPURE_OP` / `-DPURE_KIND`), `lds_pure_burst.asc` (+ `launch_lds_pure.py`), `hide32.asc` (STS).
- Results: `pure_results.txt`, `lds_size_peak_results.txt`, `lds_variants_results.txt`, `f2f_wpt32_results.txt`, `sv3_burst_results.txt`, `f2f_ffma_packed_results.txt`, `t512_ref` runs, `t1024_results.txt` (`t1024_probe.sh`).
- Charts: `tools/instr_vs_ipc.py` + `tools/warpchart.py` → `plots/` (`*_s*_t256.png`, `lds_fp32_peak_t{256,512}*.png`, `pure_f2f_s4_barrier_t{512,1024}*.png`, `pure_ffma_s3_barrier_t{512,1024}*.png`, `hide32_sts_t{512,1024}*.png`).
- Rebuild: `bash pure_sweep.sh` / `bash t1024_probe.sh` under `../single-op/` (after `set_env.sh` + `TL_CAMODEL_LIB`).
- Note: `npusim record` teardown segfaults on the 512/1024t runs (stub double-free, exit 139) — traces and rates are valid; F2F outputs verified bit-exact in a direct run.
