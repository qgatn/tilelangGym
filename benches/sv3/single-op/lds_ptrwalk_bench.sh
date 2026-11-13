#!/bin/bash
# LDS pointer-walk experiments: try to close the IADD_I 1:1 gap and push B/cyc.
# Variants (LDS_VARIANT):
#   0 = baseline strided  src[tid + i*T]        (current peak, IADD_I 1:1)
#   1 = pointer walk      p += 32; load p       (carried dependency — may or may not clear IADD_I)
#   2 = unroll x4 stride  src[tid + (i+u*4)*T]  (same IADD count, batched — control)
# Kinds: fp32 / bf16x2. Threads: 256 / 512 / 1024. Size-scaled N chosen per T.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export TL_CAMODEL_LIB="$SIM_LIB"
export LD_LIBRARY_PATH="$SIM_LIB:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PY=/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python
OUT="$ROOT/lds_ptrwalk_results.txt"
mkdir -p build
: > "$OUT"

run() {  # T kind(0 fp32|2 bf16x2) variant N bytes_per_unit units_is_pairs(0|1)
  local T="$1" KN="$2" VAR="$3" N="$4"
  local KINDSTR=$([ "$KN" = "0" ] && echo fp32 || echo bf16x2)
  local BYTES=$([ "$KN" = "0" ] && echo $((N*4)) || echo $((N*4)))
  local tag="ldspw_${KINDSTR}_t${T}_v${VAR}_n${N}"
  echo "== $tag"
  bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
    -I /home/onasismelchiorjeyakumar/workspace/tilelang/src \
    -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
    -DSIMT_THREADS="$T" -DLDS_KIND="$KN" -DLDS_N="$N" -DLDS_VARIANT="$VAR" -Wl,--no-mmap-output-file \
    lds_ptrwalk_bench.asc -o "build/${tag}.so" > "build/${tag}.build.log" 2>&1 \
    || { echo COMPILE_FAIL; tail -3 "build/${tag}.build.log"; return 1; }
  rm -rf "build/npusim_${tag}"
  npusim record -s Ascend950 -o "build/npusim_${tag}" -g -n 0 \
    "$PY launch_lds_pure.py $PWD/build/${tag}.so $T $KINDSTR $N" > "build/${tag}.log" 2>&1 || true
  rg -o "PASS.*|FAIL.*" "build/${tag}.log" | tail -1
  local bin; bin=$(find "build/npusim_${tag}" -name instr.bin | head -1)
  [ -z "$bin" ] && { echo "no instr.bin"; return 1; }
  local anal; anal=$($PY analyze.py "$bin" SIMT_LDS 2>&1 | tail -1)
  echo "  $anal"
  echo "$tag $anal" >> "$OUT"
}

# 512t fp32: baseline v0 (repro), pointer-walk v1, unroll-x4 v2
run 512 0 0 49152
run 512 0 1 49152
run 512 0 2 49152
# 512t bf16x2 same variants
run 512 2 0 49152
run 512 2 1 49152
# 1024t fp32 (SV8-adjacent config)
run 1024 0 0 49152
run 1024 0 1 49152
echo SWEEP_DONE >> "$OUT"
echo SWEEP_DONE
cat "$OUT"
