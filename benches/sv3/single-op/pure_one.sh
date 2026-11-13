#!/bin/bash
# Targeted rerun of one pure config (no chart regen): ./pure_one.sh <T> <opnum> <kind> <opstr> <bytes> <opcode>
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export TL_CAMODEL_LIB="$SIM_LIB"
export LD_LIBRARY_PATH="$SIM_LIB:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PY=/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python
T=$1; OPN=$2; K=$3; OPS=$4; BYTES=$5; OPCODE=$6
tag="pure_${OPS}k${K}_t${T}"
bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
  -I /home/onasismelchiorjeyakumar/workspace/tilelang/src \
  -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
  -DSIMT_THREADS="$T" -DPURE_OP="$OPN" -DPURE_KIND="$K" -Wl,--no-mmap-output-file \
  pure_burst.asc -o "build/${tag}.so" > "build/${tag}.build.log" 2>&1 || { echo COMPILE_FAIL; exit 1; }
rm -rf "build/npusim_${tag}"
npusim record -s Ascend950 -o "build/npusim_${tag}" -g -n 0 \
  "$PY launch_pure.py $PWD/build/${tag}.so $T $OPS $K" > "build/${tag}.log" 2>&1 || true
rg -o "PASS.*|FAIL.*" "build/${tag}.log" | tail -1
BIN=$(find "build/npusim_${tag}" -name instr.bin | head -1)
$PY analyze.py "$BIN" "$OPCODE" 2>&1 | tail -1
