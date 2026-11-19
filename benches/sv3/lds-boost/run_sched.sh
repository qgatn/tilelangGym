#!/bin/bash
# Schedule probes + bisheng flag rebuilds of V3. One npusim at a time.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
ASC="${ASCEND_HOME_PATH:-/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2}"
SIM="$ASC/tools/simulator/Ascend950PR_9599/lib"
PY=/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python
export PATH="$ASC/aarch64-linux/bin:$PATH"
export TL_CAMODEL_LIB="$SIM"
unset CAMODEL_CONFIG_PATH
export LD_LIBRARY_PATH="$SIM:$ASC/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TILELANG_DISABLE_CACHE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 TILELANG_DISABLE_DATA_RACE_CHECK=1
export PYTHONPATH="/home/onasismelchiorjeyakumar/workspace/tilelang:${PYTHONPATH:-}"
export CANN_PYTHON="$ASC/python/site-packages"
SRC=/home/onasismelchiorjeyakumar/workspace/tilelang/src
OUTF="$ROOT/results_sched.txt"
: > "$OUTF"

run_sim() {  # tag so-path
  local tag="$1" so="$2"
  local out
  out=$(dirname "$so")
  rm -rf "$out/npusim"
  echo "== $tag" | tee -a "$OUTF"
  npusim record -s Ascend950 -o "$out/npusim" -g -n 0 \
    "$PY $ROOT/launch_flip.py $so 2048 8 packT" > "$out/sim.log" 2>&1 || true
  rg -o "PASS.*|FAIL.*" "$out/sim.log" | tail -1 | tee -a "$OUTF"
  local bin
  bin=$(find "$out/npusim" -name instr.bin | head -1)
  if [ -z "$bin" ]; then echo "no instr.bin" | tee -a "$OUTF"; return; fi
  PYTHONPATH="$ASC/python/site-packages${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" "$ROOT/../schedules/decode.py" "$bin" 2048 8 simt 2>&1 | tee -a "$OUTF"
  echo "$bin" | tee -a "$OUTF"
}

for arm in reorder split; do
  "$PY" "$ROOT/probe_sched.py" "$arm" build > "$ROOT/build/probe-${arm}.compile.log" 2>&1 \
    || { echo "COMPILE_FAIL $arm" | tee -a "$OUTF"; tail -20 "$ROOT/build/probe-${arm}.compile.log" | tee -a "$OUTF"; continue; }
  rg -n "pun=|__simt_vf__|asc_syncthreads|\*\(half" "$ROOT/runs/probe-${arm}-n2048-k8-t256/kernel.cpp" | head -20 | tee -a "$OUTF"
  run_sim "probe-$arm" "$ROOT/runs/probe-${arm}-n2048-k8-t256/kernel.so"
done

BASE="$ROOT/runs/v3_row-n2048-k8-t256/kernel.asc"
bisheng_flags() {  # name extra-args...
  local name="$1"; shift
  local out="$ROOT/runs/flag-${name}-v3"
  mkdir -p "$out"
  echo "== compile flag-$name" | tee -a "$OUTF"
  if ! bisheng -O2 -fPIC -std=c++20 \
      -mllvm -cce-aicore-dcpreload-args=false \
      --npu-arch=dav-3510 \
      "$@" \
      -Wl,--no-mmap-output-file --shared \
      "$BASE" -I "$SRC" -o "$out/kernel.so" > "$out/compile.log" 2>&1; then
    echo "COMPILE_FAIL flag-$name" | tee -a "$OUTF"
    tail -15 "$out/compile.log" | tee -a "$OUTF"
    return
  fi
  run_sim "flag-$name" "$out/kernel.so"
}

bisheng_flags maxrreg255 --cce-simt-maxrregcount=255
bisheng_flags maxrreg32 --cce-simt-maxrregcount=32
bisheng_flags frame -mllvm -cce-aicore-frame-elimination-opt -mllvm -cce-aicore-frame-stack-opt
bisheng_flags remat -mllvm -cce-aicore-remat-opt -mllvm -cce-aicore-always-remat-opt -mllvm -cce-aicore-remat-reg-operand-instr
bisheng_flags pattern -mllvm -cce-simt-enable-pattern-opt -mllvm -cce-simt-enable-memop-combine-opt
bisheng_flags nostack --cce-disable-vf-stack-reserved-ubuf
echo DONE | tee -a "$OUTF"
