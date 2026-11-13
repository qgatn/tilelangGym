#!/bin/bash
# F2F thread sweep at fixed work-per-thread (32 el/thread).
# N = 32 * threads → 128t:4096, 256t:8192, 512t:16384.
# Compares Attempts A (scalar), B (packed), C (v3), D (v4).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export TL_CAMODEL_LIB="${SIM_LIB}"
export LD_LIBRARY_PATH="${SIM_LIB}:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PY="/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python"
OUT="${ROOT}/f2f_wpt32_results.txt"
WPT=32
mkdir -p build
: > "${OUT}"

run_one() {
  local name="$1" asc="$2" threads="$3" n="$4" launch="$5"
  local tag="${name}_wpt${WPT}_t${threads}"
  local so="build/${tag}.so"
  echo "BUILD ${tag} N=${n}"
  bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
    -I /home/onasismelchiorjeyakumar/workspace/tilelang/src \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -DSIMT_THREADS="${threads}" -DSIMT_N="${n}" \
    ${EXTRA_FLAGS:-} \
    -Wl,--no-mmap-output-file \
    "${asc}" -o "${so}" 2> "build/${tag}.build.log" || {
      echo "COMPILE_FAIL ${tag}" | tee -a "${OUT}"
      return
    }
  rm -rf "build/npusim_${tag}"
  npusim record -s Ascend950 -o "build/npusim_${tag}" -g -n 0 \
    "${launch}" > "build/${tag}.log" 2>&1 || true
  local line
  line=$(rg -o "PASS maxabs=[0-9.e+-]+|FAIL maxabs=[0-9.e+-]+" "build/${tag}.log" | tail -1 || true)
  echo "${tag} N=${n} ${line:-NO_VERDICT}" | tee -a "${OUT}"
  local bin
  bin=$(find "build/npusim_${tag}" -name instr.bin 2>/dev/null | head -1)
  [ -z "$bin" ] && { echo "  no instr.bin" >> "${OUT}"; return; }
  "${PY}" analyze.py "${bin}" SIMT_F2F >> "${OUT}" 2>&1
}

for T in 128 256 512; do
  N=$((WPT * T))
  EXTRA_FLAGS="-DBURST_OP=1"
  run_one f2f_A sv3_burst.asc "$T" "$N" \
    "${PY} launch.py ${ROOT}/build/f2f_A_wpt${WPT}_t${T}.so f2f ${T} fp16 ${N}"
  EXTRA_FLAGS=""
  run_one f2f_B f2f_packed_burst.asc "$T" "$N" \
    "${PY} launch.py ${ROOT}/build/f2f_B_wpt${WPT}_t${T}.so f2f ${T} fp16 ${N}"
  run_one f2f_C f2f_packed_v3.asc "$T" "$N" \
    "${PY} launch_f2f_packed.py ${ROOT}/build/f2f_C_wpt${WPT}_t${T}.so ${N}"
  run_one f2f_D f2f_packed_v4.asc "$T" "$N" \
    "${PY} launch_f2f_packed.py ${ROOT}/build/f2f_D_wpt${WPT}_t${T}.so ${N}"
done
echo WPT32_DONE | tee -a "${OUT}"
