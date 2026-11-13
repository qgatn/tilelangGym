#!/bin/bash
# Pure LDS fp32 vs bf16 vs bf16x2 — rates + charts for lds_fp32_vs_bf16.md
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
OUT="$ROOT/lds_pure_results.txt"
mkdir -p build "$PLOTS"
: > "$OUT"

run() {  # T kindnum kindstr chart bytes
  local T="$1" KN="$2" KS="$3" CHART="$4" BYTES="$5"
  local tag="lds_pure_${KS}_t${T}"
  echo "== $tag"
  bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
    -I /home/onasismelchiorjeyakumar/workspace/tilelang/src \
    -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
    -DSIMT_THREADS="$T" -DLDS_KIND="$KN" -Wl,--no-mmap-output-file \
    lds_pure_burst.asc -o "build/${tag}.so" > "build/${tag}.build.log" 2>&1 \
    || { echo COMPILE_FAIL; tail -3 "build/${tag}.build.log"; return 1; }
  rm -rf "build/npusim_${tag}"
  npusim record -s Ascend950 -o "build/npusim_${tag}" -g -n 0 \
    "$PY launch_lds_pure.py $PWD/build/${tag}.so $T $KS" > "build/${tag}.log" 2>&1 || true
  rg -o "PASS.*|FAIL.*" "build/${tag}.log" | tail -1
  local bin; bin=$(find "build/npusim_${tag}" -name instr.bin | head -1)
  [ -z "$bin" ] && { echo "no instr.bin"; return 1; }
  # audit SIMT ops
  $PY - "$bin" <<'PY'
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
print("  ops:", dict(c.most_common(8)))
bad = [k for k in c if k in ("SIMT_F2F", "SIMT_FFMA", "SIMT_LOP3", "SIMT_STK", "SIMT_LDK")]
print("  impure:", bad if bad else "none")
PY
  local anal; anal=$($PY analyze.py "$bin" SIMT_LDS 2>&1 | tail -1)
  echo "  $anal"
  local busy n
  busy=$(echo "$anal" | sed -n 's/.*busy=\([0-9]*\).*/\1/p')
  n=$(echo "$anal" | sed -n 's/.*n=\([0-9]*\).*/\1/p')
  $PY - "$BYTES" "$busy" "$n" "$tag" <<'PY'
import sys
b, busy, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
tag = sys.argv[4]
line = f"{tag} busy={busy} issues={n} B/cyc={b/busy:.2f} iss/cyc={n/busy:.3f}"
print(" ", line)
open("lds_pure_results.txt", "a").write(line + "\n")
PY
  "$PYV" "$TOOLS/instr_vs_ipc.py" "$bin" simt "$PLOTS/$CHART" --label "LDS ${KS} (pure), ${T} threads"
  "$PYV" "$TOOLS/warpchart.py" "$bin" "$PLOTS/${CHART}_warps" --label "LDS ${KS} (pure), ${T}t, warp activity"
}

# @256 — all three kinds
run 256 0 fp32   lds_fp32_t256        32768
run 256 1 bf16   lds_bf16_scalar_t256 16384
run 256 2 bf16x2 lds_bf16x2_t256      16384

# @1024 — fp32 vs bf16x2 (width-matched LDS.32)
run 1024 0 fp32   lds_fp32_t1024   32768
run 1024 2 bf16x2 lds_bf16x2_t1024 16384

echo SWEEP_DONE >> "$OUT"
echo SWEEP_DONE
cat "$OUT"
