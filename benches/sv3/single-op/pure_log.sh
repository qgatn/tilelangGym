#!/bin/bash
# Append regenerated pure rates to pure_results.txt.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=/home/onasismelchiorjeyakumar/workspace/tilelang/.venv/bin/python
{
  for t in ldsk0 ldsk1 ldsk2 ldsk3 f2fk0 f2fk1 f2fk2 f2fk3 ffmak0 ffmak1 ffmak2; do
    BIN=$(find build/npusim_pure_${t}_t256 -name instr.bin | head -1)
    case $t in lds*) OP=SIMT_LDS; B=16384; U=B/cyc;; f2f*) OP=SIMT_F2F; B=8192; U=el/cyc;; ffma*) OP=SIMT_FFMA; B=8192; U=el/cyc;; esac
    BUSY=$($PY analyze.py "$BIN" $OP 2>&1 | tail -1 | sed -n 's/.*busy=\([0-9]*\).*/\1/p')
    N=$($PY analyze.py "$BIN" $OP 2>&1 | tail -1 | sed -n 's/.*n=\([0-9]*\).*/\1/p')
    RATE=$($PY -c "print(f'{$B/$BUSY:.2f}')")
    echo "$t busy=$BUSY issues=$N $U=$RATE"
  done
  STS=$(find build/npusim_hide32_sts_t256 -name instr.bin | head -1)
  echo "STS (hide32) $($PY analyze.py "$STS" SIMT_STS 2>&1 | tail -1)"
} >> pure_results.txt
tail -14 pure_results.txt
