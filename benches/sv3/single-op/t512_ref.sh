set -u
cd "$(dirname "$0")"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export TL_CAMODEL_LIB="$SIM_LIB"
export LD_LIBRARY_PATH="$SIM_LIB:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PY=/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python
T=512
run() { local OP=$1 KIND=$2 NAME=$3 TARGET=$4 EL=$5 NARG=$6
  local tag="pure_${NAME}_t${T}"
  bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
    -I /home/onasismelchiorjeyakumar/workspace/tilelang/src \
    -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
    -DSIMT_THREADS=$T -DPURE_OP=$OP -DPURE_KIND=$KIND -Wl,--no-mmap-output-file \
    pure_burst.asc -o build/${tag}.so > build/${tag}.build.log 2>&1 || { echo COMPILE_FAIL $tag; return; }
  rm -rf build/npusim_${tag}
  npusim record -s Ascend950 -o build/npusim_${tag} -g -n 0 \
    "$PY launch_pure.py $PWD/build/${tag}.so $T $OP $KIND $NARG" > /tmp/${tag}.log 2>&1 || true
  rg -o "PASS.*|FAIL.*" /tmp/${tag}.log | tail -1
  local bin; bin=$(find build/npusim_${tag} -name instr.bin | head -1)
  [ -z "$bin" ] && { echo "no instr.bin"; return; }
  $PY analyze.py "$bin" $TARGET 2>&1 | tail -1
}
echo "== f2f_s4_barrier_t512"; run 1 3 f2f_s4_barrier SIMT_F2F 16384 16384
echo "== ffma_s3_barrier_t512"; run 2 2 ffma_s3_barrier SIMT_FFMA 8192 8192
tag=hide32_sts_t512
bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
  -I /home/onasismelchiorjeyakumar/workspace/tilelang/src \
  -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
  -DSIMT_THREADS=$T -DSIMT_OP=2 -Wl,--no-mmap-output-file \
  hide32.asc -o build/${tag}.so > build/${tag}.build.log 2>&1 || echo COMPILE_FAIL sts
rm -rf build/npusim_${tag}
npusim record -s Ascend950 -o build/npusim_${tag} -g -n 0 \
  "$PY launch.py $PWD/build/${tag}.so sts $T fp32" > /tmp/${tag}.log 2>&1 || true
echo "== hide32_sts_t512  $(rg -o 'PASS.*|FAIL.*' /tmp/${tag}.log | tail -1)"
bin=$(find build/npusim_${tag} -name instr.bin | head -1)
[ -n "$bin" ] && $PY analyze.py "$bin" SIMT_STS 2>&1 | tail -1
echo DONE_512REF
