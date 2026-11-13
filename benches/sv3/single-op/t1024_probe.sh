set -u
cd "$(dirname "$0")"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export TL_CAMODEL_LIB="$SIM_LIB"
export LD_LIBRARY_PATH="$SIM_LIB:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PY=/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python
T=1024

# --- pure_burst: F2F kinds 0,2,3 ; FFMA kinds 0,2 ---
for spec in "1 0 f2f_s1_scalar" "1 2 f2f_s3_word" "1 3 f2f_s4_barrier" "2 0 ffma_s1_scalar" "2 2 ffma_s3_barrier"; do
  set -- $spec; OP=$1; KIND=$2; NAME=$3
  tag="pure_${NAME}_t${T}"
  echo "== $tag"
  bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
    -I /home/onasismelchiorjeyakumar/workspace/tilelang/src \
    -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
    -DSIMT_THREADS=$T -DPURE_OP=$OP -DPURE_KIND=$KIND -Wl,--no-mmap-output-file \
    pure_burst.asc -o build/${tag}.so > build/${tag}.build.log 2>&1 \
    || { echo COMPILE_FAIL; tail -3 build/${tag}.build.log; continue; }
  rm -rf build/npusim_${tag}
  NARG=8192
  [ "$OP" = "1" ] && NARG=$((32*T))
  "$PY" launch_pure.py "$PWD/build/${tag}.so" $T $OP $KIND $NARG > build/${tag}.log 2>&1 || true
  rg -o "PASS.*|FAIL.*" build/${tag}.log | tail -1
  bin=$(find build/npusim_${tag} -name instr.bin | head -1)
  [ -z "$bin" ] && { echo "no instr.bin"; continue; }
  case $OP in
    1) TARGET=SIMT_F2F; EL=$((32*T)) ;;
    2) TARGET=SIMT_FFMA; EL=$((8192)) ;;
  esac
  $PY analyze.py "$bin" $TARGET 2>&1 | tail -1
done

# --- STS via hide32.asc SIMT_OP=2 ---
tag="hide32_sts_t${T}"
echo "== $tag"
bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
  -I /home/onasismelchiorjeyakumar/workspace/tilelang/src \
  -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
  -DSIMT_THREADS=$T -DSIMT_OP=2 -Wl,--no-mmap-output-file \
  hide32.asc -o build/${tag}.so > build/${tag}.build.log 2>&1 \
  || { echo COMPILE_FAIL; tail -3 build/${tag}.build.log; }
rm -rf build/npusim_${tag}
"$PY" launch.py "$PWD/build/${tag}.so" sts $T > build/${tag}.log 2>&1 || true
rg -o "PASS.*|FAIL.*" build/${tag}.log | tail -1
bin=$(find build/npusim_${tag} -name instr.bin | head -1)
[ -n "$bin" ] && $PY analyze.py "$bin" SIMT_STS 2>&1 | tail -1
echo DONE_1024
