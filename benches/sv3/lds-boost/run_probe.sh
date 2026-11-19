#!/bin/bash
# Run the spill probes under npusim and decode.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
ASC=/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2
SIM="$ASC/tools/simulator/Ascend950PR_9599/lib"
PY=/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python
export TL_CAMODEL_LIB="$SIM"
unset CAMODEL_CONFIG_PATH
LD=$(printf '%s\n' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | rg -v '/runtime/lib64$|/Ascend/driver/' | paste -sd: -)
export LD_LIBRARY_PATH="$SIM:$ASC/lib64:${LD}"
export CANN_PYTHON="$ASC/python/site-packages" PYTHONPATH="$ASC/python/site-packages"
OUTF="$ROOT/results_probe.txt"
: > "$OUTF"

for arm in xv_scalars; do
  tag="probe-${arm}-n2048-k8-t256"
  out="$ROOT/runs/$tag"
  rm -rf "$out/npusim"
  echo "== $tag" | tee -a "$OUTF"
  npusim record -s Ascend950 -o "$out/npusim" -g -n 0 \
    "$PY $ROOT/launch_flip.py $out/kernel.so 2048 8 packT" > "$out/sim.log" 2>&1 || true
  rg -o "PASS.*|FAIL.*" "$out/sim.log" | tail -1 | tee -a "$OUTF"
  bin=$(find "$out/npusim" -name instr.bin | head -1)
  [ -z "$bin" ] && { echo "no instr.bin" | tee -a "$OUTF"; continue; }
  "$PY" ../schedules/decode.py "$bin" 2048 8 simt 2>&1 | tee -a "$OUTF"
  "$PY" rate_kernel.py "$bin" 2048 8 2>&1 | tee -a "$OUTF"
  echo "$bin" | tee -a "$OUTF"
done
echo DONE | tee -a "$OUTF"
