# Spill elimination — what we tried today

Probes live in this folder. Every arm is the V3 math (`A[k, n]` u32-packed, N=2048, K=8, 256 threads), compiled with the current library (`562cbc7c`), run under npusim. `pun` counts `*(half*)(&v_)` occurrences in the generated device C; STK/LDK are the trace spill counts.

## Result table

| Arm | Idea | pun | `__half22float2` | PASS | VF | el/cyc | band | STK/LDK |
|---|---|---:|---:|---|---:|---:|---:|---:|
| `v3_row` (control) | today's lowering | 2 | 0 | yes | 4424 | 0.46 | 28.15 | 512/512 |
| `xv_scalars` | x as 8 `T.alloc_var` scalars, unpack/MAC fully unrolled per k | 8 | 0 | yes | 5266 | 0.39 | 24.27 | **512/512** |
| `xh_vec` | x as 4 half2 ramps → `float32x2` (the `__half22float2` path), vector MAC + `Shuffle` fold | 2 | 2 | bisheng FAIL | — | — | — | — |

## Findings

1. **There is no "make x persistent" knob.** TileLang has no per-value register pinning: `alloc_var` / `alloc_fragment` / `alloc_local` are storage *scopes* (`local.var`, `local.fragment`, `local`), and the CUDA-only `T.set_max_nreg` / L2-persistence annotations do not reach the Ascend backend. Register allocation is entirely bisheng's job.

2. **`alloc_var` does NOT keep x in registers.** `xv_scalars` spells every x as its own scalar `T.alloc_var` and never indexes an array — the generated C is exactly `float x0..x7` (see `runs/probe-xv_scalars-n2048-k8-t256/kernel.cpp`), the shape you'd think cannot spill. bisheng still emits **512 STK + 512 LDK**. The trace PC schedule shows the same pattern as V3: after the four A-word loads (+199…+224), STK/LDK pairs start at +237 and interleave with FFMA to +407. The spill source is therefore **not** the x array indexing and **not** the A fragments — it is the `*(half*)(&v_)` pun forcing a memory round-trip per converted half, which the compiler generalizes to the whole frame (`l0[1]`, `h0[1]`, … all become stack slots once one value in the region is stack-addressable).

3. **The pun count tracks the spill.** `xv_scalars` printed 8 puns (two per k-column after unrolling) and still 512 spills. The only generated C in this folder with `__half22float2` is `xh_vec`, whose x path is pun-free — and bisheng rejects its store section (`float2 pa[1]` written lane-wise through `.x`/`.y`), so it has no measurement yet.

4. **Half-frames do not help.** `probe_xv_scalars` had one-column fragments (`l0[1]`…`h3[1]`) — the smallest possible arrays — and still spilled. Frame shape is irrelevant; any `*(half*)(&…)` in the region drags the frame to the stack.

## What this means for the fix

The remaining lever is inside codegen, not the schedule: when TileLang lowers the packed-pair unpack (`reinterpret(Cast uint16, w & 0xFFFF)`), emit the register-safe form

```c
float2 _; uint1 v_ = *(uint1*)ptr; ((float2*)&_)[0] = __half22float2(((half2*)&v_)[0]);
```

(the exact shape `codegen_ascend.cc` already prints for a vectorized `float16x2 → float32x2` cast, `PrintVectorizedCast("__half22float2", ...)` at the `from_ty.is_float16() && target_ty.is_float()` branch) instead of the pointer pun. The `xh_vec` arm shows that lowering exists and reaches the device C; what fails is the *store* side of the vector MAC (`PrintVecElemStore` on a `float2` temp writes `.x`/`.y` separately, which bisheng rejects for `float2` locals — same issue `test_local_var_vector_initializer_is_emitted_before_declaration` guards on the load side).

Provenance: `results_probe.txt`, `probe-xv_scalars-n2048-k8-t256/` (trace + source), `probe_xpersist.py` (arms), `probe_pc.py` (PC schedule).

## Scheduling and bisheng flags (same day)

Two TileLang schedules, both PASS at maxabs 1.19e-07, and six rebuilds of the existing V3 device source. `results_sched.txt`.

| Arm | What changed | VF | el/cyc | STK/LDK |
|---|---|---:|---:|---:|
| `reorder` | one VF: pun-unpack every row into float UB, `sync_threads`, then load x and MAC | 10358 | 0.20 | 512/512 |
| `split` | two VFs: pun-only, then a MAC VF with no reinterpret | 10509 | 0.19 | 512/512 |
| V3 + `--cce-simt-maxrregcount=128` | register cap at the top of the legal range | 4274 | 0.48 | 512/512 |
| V3 + `--cce-simt-maxrregcount=32` | register cap forced down | 4337 | 0.47 | 512/512 |
| V3 + frame elimination / frame-stack opt | `-mllvm -cce-aicore-frame-elimination-opt` and `-cce-aicore-frame-stack-opt` | 4340 | 0.47 | 512/512 |
| V3 + remat | `-cce-aicore-remat-opt`, `-cce-aicore-always-remat-opt`, `-cce-aicore-remat-reg-operand-instr` | 4315 | 0.47 | 512/512 |
| V3 + SIMT pattern opts | `-cce-simt-enable-pattern-opt` and `-cce-simt-enable-memop-combine-opt` | 4246 | 0.48 | 512/512 |
| V3 + no reserved VF stack | `--cce-disable-vf-stack-reserved-ubuf` | 4314 | 0.47 | 512/512 |

`--cce-simt-maxrregcount=255` is rejected: the flag only accepts `(0, 128]`.

Per-VF counts:

- `split` pun VF: 256 STK + 256 LDK. MAC VF: **0 / 0** (span 856, 320 LDS).
- `reorder` keeps a single VF, and the PC schedule shows the same split inside it. Every STK sits in the pun loop, one STK+LDK+F2F+STS per half (`+87 STK`, `+91 LDK`, then the next half). After `asc_syncthreads` the x loads and the FFMA block have no STK.

So a TileLang reorder does stop the compiler from spilling `x`: once the pun is finished, the MAC uses registers. The 512 that remain are the pun itself. `*(half *)(&(v_))` is a stack store of the `uint16` and a stack reload as `half`, and none of the register-cap, frame, rematerialize, pattern, or "stack size 0" flags turn that into a register bitcast. Capping the file at 32 registers did not add spills either, which matches the register math: this is not a budget decision.
