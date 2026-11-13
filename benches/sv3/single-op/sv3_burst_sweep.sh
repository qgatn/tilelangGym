#!/bin/bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export TL_CAMODEL_LIB="${SIM_LIB}"
export LD_LIBRARY_PATH="${SIM_LIB}:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PY="/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python"
OUT="${ROOT}/sv3_burst_results.txt"
mkdir -p build
: > "${OUT}"

run_one() {
  local kind="$1" opid="$2" opmap="$3" threads="$4"
  local tag="sv3_${kind}_t${threads}"
  local so="build/${tag}.so"
  echo "BUILD ${tag}"
  bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
    -I /home/onasismelchiorjeyakumar/workspace/tilelang/src \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -DSIMT_THREADS="${threads}" -DBURST_OP="${opid}" \
    -Wl,--no-mmap-output-file \
    sv3_burst.asc -o "${so}" 2> "build/${tag}.build.log" || { echo "COMPILE_FAIL ${tag}" | tee -a "${OUT}"; return; }
  rm -rf "build/npusim_${tag}"
  npusim record -s Ascend950 -o "build/npusim_${tag}" -g -n 0 \
    "${PY} launch.py ${ROOT}/${so} ${kind} ${threads} fp16" > "build/${tag}.log" 2>&1 || true
  local line
  line=$(rg -o "PASS maxabs=[0-9.e+-]+|FAIL maxabs=[0-9.e+-]+" "build/${tag}.log" | tail -1 || true)
  echo "${tag} ${line:-NO_VERDICT}" >> "${OUT}"
  local bin
  bin=$(find "build/npusim_${tag}" -name instr.bin 2>/dev/null | head -1)
  [ -z "$bin" ] && { echo "  no instr.bin" >> "${OUT}"; return; }
  "${PY}" analyze.py "${bin}" "${opmap}" >> "${OUT}" 2>&1
}

for T in 128 256 512; do
  run_one lds 0 SIMT_LDS "$T"
done
for T in 128 256 512; do
  run_one f2f 1 SIMT_F2F "$T"
done
for T in 128 256 512; do
  run_one ffma 2 SIMT_FFMA "$T"
done
echo SWEEP_DONE >> "${OUT}"
echo SWEEP_DONE
