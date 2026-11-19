# SV3

Ascend SIMT GEMV study (`dav-3510` / npusim).

| Folder | What |
|---|---|
| `single-op/` | Single-opcode microbenchmarks (rates for the theoretical model) |
| `schedules/` | Full GEMV schedule ladder (TileLang kernels + kept traces) |
| `simt-uarch/` | Write-ups, charts, chart tools (`SV3-theoretical.md`, …) |
| `cce/` | Hand-written CCE comparison kernel |
| `ANALYSIS.md` | Earlier end-to-end case write-up (uses `schedules/`) |
| `plots/` | Older ANALYSIS comparison plots |
| `lds-boost/` | `A[k, n]` layout study (`lds_performance.md`) |

Start here for the schedule story: `simt-uarch/SV3-theoretical.md`.
