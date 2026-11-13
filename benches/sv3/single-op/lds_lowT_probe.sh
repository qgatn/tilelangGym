set -u
cd "$(dirname "$0")"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export TL_CAMODEL_LIB="$SIM_LIB"
export LD_LIBRARY_PATH="$SIM_LIB:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PY=/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python
# 256t sparser WPT + intermediate warp counts
for spec in "256 fp32 12288" "256 fp32 16384" "384 fp32 49152" "768 fp32 49152" "384 bf16x2 49152" "768 bf16x2 49152"; do
  set -- $spec; T=$1; KS=$2; N=$3
  KN=$([ "$KS" = "fp32" ] && echo 0 || echo 2)
  tag="ldslo_${KS}_t${T}_n${N}"
  echo "== $tag"
  bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
    -I /home/onasismelchiorjeyakumar/workspace/tilelang/src \
    -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
    -DSIMT_THREADS=$T -DLDS_KIND=$KN -DLDS_N=$N -Wl,--no-mmap-output-file \
    lds_pure_burst.asc -o build/${tag}.so > build/${tag}.build.log 2>&1 \
    || { echo COMPILE_FAIL; tail -3 build/${tag}.build.log; continue; }
  rm -rf build/npusim_${tag}
  npusim record -s Ascend950 -o build/npusim_${tag} -g -n 0 \
    "$PY launch_lds_pure.py $PWD/build/${tag}.so $T $KS $N" > build/${tag}.log 2>&1 || true
  rg -o "PASS.*|FAIL.*" build/${tag}.log | tail -1
  bin=$(find build/npusim_${tag} -name instr.bin | head -1)
  $PY analyze.py "$bin" SIMT_LDS 2>&1 | tail -1
done
echo DONE_LOW
