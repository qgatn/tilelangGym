# lds-boost/ — `A[k, n]` GEMV

`x[k]` is the vector. `A[k, n]` has `n` contiguous. Write-up: `lds_performance.md`.

## TileLang — V1, V2, V3

| Arm | Order | Log |
|---|---|---|
| `v1_burst` | Load all 4 rows, then FFMA | `results_flip.txt` (old lib), `results_rebuild.txt` (2026-11-18) |
| `v2_pair` | One row, 8 B between lanes | `results_flip.txt` (old lib), `results_rebuild.txt` (2026-11-18) |
| `v3_row` | Load one row, FFMA, store, next row | `results_onerow.txt` (old lib), `results_rebuild.txt` (2026-11-18) |
| `v0_base` | Unpack inside the load (control) | `results_flip.txt` (old lib), `results_rebuild.txt` (2026-11-18) |

Instruction charts: `plots/v{1,3}_instr_lds.png` and `plots/v{1,3}_warps.png` (steady VF, `charts/web` method).

| Path | Role |
|---|---|
| `kernels_flip.py` | The four arms |
| `compile_flip.py` | TileLang compile + bisheng |
| `launch_flip.py` | Host packs `A` to `A[k, n]`, then launches |
| `run_flip.sh` | Builds and measures v0–v3 |
| `rate_kernel.py` | LDS busy-union B/cyc from `instr.bin` |
| `runs/v1_burst-n2048-k8-t256/` | Generated V1 kernel |
| `plots/` | V1/V3 instr + warp charts |

```bash
cd benches/sv3/lds-boost
bash run_flip.sh
```

## CCE check that set the layout

`a_row8_pack.asc` is the same packed load at two addresses: 16 B stride **29.36 B/cyc**, 4 B stride **79.92 B/cyc**. `RESULTS.md` also records that 512 threads and `IADD` do not move the old layout.

```bash
bash run_a_pack.sh
```
