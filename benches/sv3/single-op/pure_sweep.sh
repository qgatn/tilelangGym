#!/bin/bash
# Pure single-op sweep: one source per op family, checksum sink, one chart each.
# run <T> <opnum> <kind> <chart> <label> <bytes> <opcode> [opstr]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export TL_CAMODEL_LIB="$SIM_LIB"
export LD_LIBRARY_PATH="$SIM_LIB:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PY=/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python
PYV=/home/onasismelchiorjeyakumar/venv/bin/python
TOOLS=${ROOT}/../simt-uarch/tools
PLOTS=${ROOT}/../simt-uarch/plots
mkdir -p "$PLOTS" build
OUT="$ROOT/pure_results.txt"
: > "$OUT"

audit() {
  $PY - "$1" <<'PY'
import re, sys
from collections import Counter
sys.path.insert(0, str(__import__("pathlib").Path("..", "schedules").resolve()))
from decode import decode_instr_logs_from_file
c = Counter()
for r in decode_instr_logs_from_file(sys.argv[1]):
    d = vars(r)
    if d.get("is_popped") != 1: continue
    m = re.search(r"(SIMT_[A-Z0-9_]+)", d.get("dec", "") or "")
    if m: c[m.group(1)] += 1
print("  ops:", dict(c.most_common(10)))
PY
}

measure() {  # bin opcode bytes
  local bin="$1" opcode="$2" bytes="$3"
  local anal; anal=$($PY analyze.py "$bin" "$opcode" 2>&1 | tail -1)
  echo "  $anal"
  local busy n
  busy=$(echo "$anal" | sed -n 's/.*busy=\([0-9]*\).*/\1/p')
  n=$(echo "$anal" | sed -n 's/.*n=\([0-9]*\).*/\1/p')
  [ -n "$busy" ] && $PY - "$bytes" "$busy" "$n" <<'PY'
import sys
b, busy, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
print(f"  -> B/cyc={b/busy:.2f} iss/cyc={n/busy:.3f}")
PY
}

run() {  # T opnum kind chart label bytes opcode opstr
  local T="$1" OPN="$2" K="$3" CHART="$4" LABEL="$5" BYTES="$6" OPCODE="$7" OPS="$8"
  local tag="pure_${OPS}k${K}_t${T}"
  echo "== $tag (PURE_OP=$OPN PURE_KIND=$K)"
  bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
    -I /home/onasismelchiorjeyakumar/workspace/tilelang/src \
    -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
    -DSIMT_THREADS="$T" -DPURE_OP="$OPN" -DPURE_KIND="$K" -Wl,--no-mmap-output-file \
    pure_burst.asc -o "build/${tag}.so" > "build/${tag}.build.log" 2>&1 \
    || { echo "COMPILE_FAIL $tag"; tail -5 "build/${tag}.build.log"; return 1; }
  rm -rf "build/npusim_${tag}"
  npusim record -s Ascend950 -o "build/npusim_${tag}" -g -n 0 \
    "$PY launch_pure.py $PWD/build/${tag}.so $T $OPS $K" > "build/${tag}.log" 2>&1 || true
  rg -o "PASS.*|FAIL.*" "build/${tag}.log" | tail -1
  local bin; bin=$(find "build/npusim_${tag}" -name instr.bin | head -1)
  [ -z "$bin" ] && { echo "  no instr.bin"; return 1; }
  audit "$bin"
  measure "$bin" "$OPCODE" "$BYTES"
  echo "${tag} OK" >> "$OUT"
  "$PYV" "$TOOLS/instr_vs_ipc.py" "$bin" simt "$PLOTS/$CHART" --label "$LABEL"
  "$PYV" "$TOOLS/warpchart.py" "$bin" "$PLOTS/${CHART}_warps" --label "$LABEL, warp activity"
}

# ---- LDS @256t: 4 schedules ----
run 256 0 0 lds_s1_scalar_t256 "LDS S1 scalar fp16 (pure), 256 threads"   16384 SIMT_LDS lds
run 256 0 1 lds_s2_u32_t256    "LDS S2 uint32 pack (pure), 256 threads"   16384 SIMT_LDS lds
run 256 0 2 lds_s3_half2_t256  "LDS S3 half2 typed (pure), 256 threads"   16384 SIMT_LDS lds
run 256 0 3 lds_s4_bf16x2_t256 "LDS S4 bfloat16x2 typed (pure), 256t"     16384 SIMT_LDS lds

# ---- F2F @256t: 4 schedules (LDS preload unavoidable) ----
run 256 1 0 f2f_s1_scalar_t256  "F2F S1 scalar convert (pure), 256t"      8192 SIMT_F2F f2f
run 256 1 1 f2f_s2_packed_t256  "F2F S2 packed+PRMT (pure), 256 threads" 8192 SIMT_F2F f2f
run 256 1 2 f2f_s3_word_t256    "F2F S3 word-preload (pure), 256t"       8192 SIMT_F2F f2f
run 256 1 3 f2f_s4_barrier_t256 "F2F S4 barrier-aligned (pure), 256t"    8192 SIMT_F2F f2f

# ---- FFMA @256t: 3 schedules (LDS+F2F preload unavoidable) ----
run 256 2 0 ffma_s1_scalar_t256  "FFMA S1 scalar (pure), 256 threads"    16384 SIMT_FFMA ffma
run 256 2 1 ffma_s2_f2src_t256   "FFMA S2 float2-src (pure), 256t"       16384 SIMT_FFMA ffma
run 256 2 2 ffma_s3_barrier_t256 "FFMA S3 barrier-aligned (pure), 256t"  16384 SIMT_FFMA ffma

# ---- STS: reuse the pure hide32 trace (already store-only) ----
STS=$(find build/npusim_hide32_sts_t256 -name instr.bin 2>/dev/null | head -1)
if [ -n "$STS" ]; then
  echo "== STS hide32_t256"
  audit "$STS"; measure "$STS" SIMT_STS 32768
  "$PYV" "$TOOLS/instr_vs_ipc.py" "$STS" simt "$PLOTS/sts_s1_t256" --label "STS S1 coalesced (pure), 256 threads"
  "$PYV" "$TOOLS/warpchart.py" "$STS" "$PLOTS/sts_s1_t256_warps" --label "STS S1 coalesced, 256t, warp activity"
fi

echo SWEEP_DONE >> "$OUT"
echo SWEEP_DONE
