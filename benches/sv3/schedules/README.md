# schedules/ — GEMV schedule experiments

Was previously named `nk/` (N×K case directory). Renamed to match
`simt-uarch/SV3-theoretical.md`, which talks about **schedules**.

| Path | Role |
|---|---|
| `kernels.py` | All TileLang builders for §3 + §5 (one file) |
| `kernels_legacy.py` | Older ANALYSIS.md builders (thread/warp/simd) |
| `compile.py` | Codegen + bisheng → `runs/<name>-…/kernel.so` |
| `compile_legacy.py` | Compiler for legacy ANALYSIS kernels |
| `launch.py` / `decode.py` | Run + summarize an `instr.bin` |
| `results/` | Provenance logs cited by the docs |
| `runs/` | Kept traces only (cited in `*.md`) |
| `tools/` | Trace helpers used while writing §5 |

```bash
cd benches/sv3/schedules
python compile.py quad_acc 2048 8 256 --nosync
npusim record -s Ascend950 -o runs/quad_acc-n2048-k8-t256-nosync/npusim -g -n 0 \
  "python launch.py runs/quad_acc-n2048-k8-t256-nosync/kernel.so 2048 8"
python decode.py $(find runs/quad_acc-n2048-k8-t256-nosync/npusim -name instr.bin | head -1) 2048 8 simt
```
