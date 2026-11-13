# single-op/ — SIMT microbenchmarks

Was previously `simt-uarch/simt_ops/`. One-opcode (or one-pipe) bursts used to
set the theoretical rates in `simt-uarch/simt-uarch.md` and
`simt-uarch/lds_fp32_vs_bf16.md`.

| Path | Role |
|---|---|
| `*.asc` / `launch_*.py` | Microbench sources + host launchers |
| `analyze.py` | Per-opcode busy-union analysis |
| `*_sweep.sh` / `*_probe.sh` | Rebuild / remeasure scripts |
| `results/` | Provenance `.txt` logs cited by the docs |
| `build/` | Compiled `.so` + npusim artifacts |

Narratives and charts live next door in `../simt-uarch/`.
