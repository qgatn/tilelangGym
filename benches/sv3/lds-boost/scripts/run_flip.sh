#!/usr/bin/env bash
# lds-boost: A[k,n] arms (v0 / v1 / v2 / v3) → results_flip.txt
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
ASC="${ASCEND_HOME_PATH:-/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2}"
SIM="$ASC/tools/simulator/Ascend950PR_9599/lib"
PY=/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python
PYV=/home/onasismelchiorjeyakumar/venv/bin/python
export PATH="$ASC/aarch64-linux/bin:$PATH"
export TL_CAMODEL_LIB="$SIM"
export LD_LIBRARY_PATH="$SIM:$ASC/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TILELANG_DISABLE_CACHE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export TILELANG_DISABLE_DATA_RACE_CHECK=1
export PYTHONPATH="/home/onasismelchiorjeyakumar/workspace/tilelang:${PYTHONPATH:-}"
export BISHENG_LIBSTDCXX_FIX="${ROOT}/../../bisheng_libstdcxx_clang_fix.h"
OUT="$ROOT/logs/results_flip.txt"
mkdir -p runs build
: > "$OUT"

run_one() {  # name N K T gm-mode
  local name="$1" N="$2" K="$3" T="$4" MODE="$5"
  local tag="${name}-n${N}-k${K}-t${T}"
  echo "== $tag" | tee -a "$OUT"
  ( cd "$ROOT" && "$PY" "$ROOT/scripts/compile_flip.py" "$name" "$N" "$K" "$T" "$ROOT/runs" 0 ) \
    > "$ROOT/build/${tag}.compile.log" 2>&1 \
    || { echo COMPILE_FAIL | tee -a "$OUT"; tail -8 "$ROOT/build/${tag}.compile.log"; return 1; }
  export CAMODEL_LOG_PATH="$ROOT/build/${tag}_ca"
  export CORE_LOG_PATH="$ROOT/build/${tag}_ca"
  mkdir -p "$ROOT/build/${tag}_ca"
  rm -rf "$ROOT/runs/${tag}/npusim"
  npusim record -s Ascend950 -o "$ROOT/runs/${tag}/npusim" -g -n 0 \
    "$PY $ROOT/scripts/launch_flip.py $ROOT/runs/${tag}/kernel.so $N $K $MODE" \
    > "$ROOT/build/${tag}.log" 2>&1 || true
  rg -o "PASS.*|FAIL.*" "$ROOT/build/${tag}.log" | tail -1 | tee -a "$OUT"
  local bin; bin=$(find "$ROOT/runs/${tag}/npusim" -name instr.bin | head -1)
  [ -z "$bin" ] && { echo "no instr.bin" | tee -a "$OUT"; return 1; }
  "$PYV" "$ROOT/../schedules/decode.py" "$bin" "$N" "$K" simt 2>&1 | tee -a "$OUT"
  "$PYV" "$ROOT/scripts/rate_kernel.py" "$bin" "$N" "$K" 2>&1 | tee -a "$OUT"
}

run_one v0_base 2048 8 256 packT
run_one v1_burst 2048 8 256 packT
run_one v2_pair 2048 8 256 pair8
run_one v3_row 2048 8 256 packT

echo "done -> $OUT"
cat "$OUT"
