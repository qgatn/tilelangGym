#!/usr/bin/env bash
# A: packed-load ceilings at GEMV row shape (and 512t variants).
# Usage: bash run_a_pack.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source /mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/set_env.sh
SIM_LIB="${ASCEND_HOME_PATH}/tools/simulator/Ascend950PR_9599/lib"
export TL_CAMODEL_LIB="$SIM_LIB"
export LD_LIBRARY_PATH="$SIM_LIB:${ASCEND_HOME_PATH}/runtime/lib64/stub:${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PY=/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python
PYV=/home/onasismelchiorjeyakumar/venv/bin/python
TOOLS="${ROOT}/../simt-uarch/tools"
SRC=/home/onasismelchiorjeyakumar/workspace/tilelang/src
OUT="$ROOT/results_a.txt"
mkdir -p build
: > "$OUT"

# audit: total SIMT_LDS bytes/issue must be 128 for u32 loads
audit() {
  local bin="$1" threads="$2"
  $PYV - "$bin" "$threads" <<'PY'
import re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/python/site-packages")
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file
MN = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")
PIPE = re.compile(r"\)\s+([A-Z0-9]+)\s+:")
rows = [r for r in decode_instr_logs_from_file(Path(sys.argv[1])) if r.core == 0 and r.sub == 1]
issues = {}
for r in rows:
    m, p = MN.search(r.dec or ""), PIPE.search(r.dec or "")
    if not m:
        continue
    key = (r.pc, int(m.group(1)), m.group(2), bool(p and p.group(1).startswith("RVEC")))
    if r.is_popped == 1 and key not in issues:
        issues[key] = r.time
vf = [t for k, t in issues.items() if k[2] == "VF_SIMT"]
start = min(vf) if vf else 0
lds = [k for k, t in issues.items() if k[2] == "SIMT_LDS" and t >= start]
iadd = [k for k, t in issues.items() if k[2] in ("SIMT_IADD_I", "SIMT_IADD", "SIMT_LEA") and t >= start]
pipe_of = {}
for r in rows:
    m, p = MN.search(r.dec or ""), PIPE.search(r.dec or "")
    if m and p:
        pipe_of[(r.pc, int(m.group(1)), m.group(2))] = p.group(1)
kinds = Counter(pipe_of.get(k[:3], "?") for k in lds)
print(f"  audit: LDS={len(lds)} IADD+LEA={len(iadd)} lds_pipes={dict(kinds)}")
PY
}

run() {  # tag threads packmacro rows
  local tag="$1" T="$2" PACK="$3" ROWS="$4"
  echo "== $tag" | tee -a "$OUT"
  bisheng -O2 -fPIC -std=c++20 -shared --npu-arch=dav-3510 \
    -I "$SRC" \
    -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 \
    -DSIMT_THREADS="$T" -DPACK="$PACK" -DROWS="$ROWS" -Wl,--no-mmap-output-file \
    a_row8_pack.asc -o "build/${tag}.so" > "build/${tag}.build.log" 2>&1 \
    || { echo COMPILE_FAIL | tee -a "$OUT"; tail -3 "build/${tag}.build.log"; return 1; }
  rm -rf "build/npusim_${tag}"
  npusim record -s Ascend950 -o "build/npusim_${tag}" -g -n 0 \
    "$PY launch_row8.py $PWD/build/${tag}.so $T $PACK" \
    > "build/${tag}.log" 2>&1 || true
  grep -ao "PASS.*\|FAIL.*" "build/${tag}.log" | tail -1 | tee -a "$OUT"
  local bin; bin=$(find "build/npusim_${tag}" -name instr.bin | head -1)
  [ -z "$bin" ] && { echo "no instr.bin" | tee -a "$OUT"; return 1; }
  # busy-union rate on SIMT_LDS inside VF[0]
  $PYV "$TOOLS/instr_vs_ipc.py" "$bin" simt "build/${tag}_chart.png" --vf 0 --label "$tag" >/dev/null 2>&1 || true
  $PYV - "$bin" >> "$OUT" <<'PY'
import re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/python/site-packages")
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file
MN = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")
rows = [r for r in decode_instr_logs_from_file(Path(sys.argv[1])) if r.core == 0 and r.sub == 1]
iss, don = {}, {}
for r in rows:
    m = MN.search(r.dec or "")
    if not m:
        continue
    key = (r.pc, int(m.group(1)), m.group(2))
    if r.is_popped == 1 and key not in iss:
        iss[key] = r.time
    elif r.is_popped == 0:
        don[key] = r.time
vfs = sorted((iss[k], don[k]) for k in iss if k[2] == "VF_SIMT" and k in don)
if not vfs:
    print("  NO_VF"); raise SystemExit
lo, hi = vfs[0]
busy = set()
for k, t in iss.items():
    if k[2] != "SIMT_LDS" or k not in don:
        continue
    d = don[k]
    if d < lo or t > hi:
        continue
    a, b = max(t, lo), min(d, hi)
    busy.update(range(a, b + 1))
n_issues = sum(1 for k, t in iss.items() if k[2] == "SIMT_LDS" and lo <= t <= hi)
# GEMV-unique bytes: A rows only (ROWS*16 B) — rate vs busy union
bytes_unique = int(sys.argv[2]) * 0 + 1024 * 16  # ROWS fixed 1024 in this bench
print(f"  rate: issues={n_issues} busy={len(busy)} B/cyc={bytes_unique / max(len(busy),1):.2f}")
PY
}

# GEMV row shape (1024×16B lane-stride 16B) vs transposed (lane-stride 4B) vs flat
run a_row_t256   256 1 1024   # PACK=1 today's lowering shape
run a_row_t512   512 1 1024
run a_trans_t256 256 2 1024   # PACK=2 transposed, conflict-free
run a_trans_t512 512 2 1024
run a_flat_t256  256 0 1024   # control

echo "done -> $OUT"
cat "$OUT"
