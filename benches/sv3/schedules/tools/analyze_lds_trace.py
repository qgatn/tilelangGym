"""Busy-union analysis for one dbl_buf-style instr.bin.

Per VF window: list opcodes, issue counts, and per-opcode busy union
(union of [issue, completion) intervals). Also the whole-VF union for LDS.

Usage: analyze_lds_trace.py <instr.bin> [--window 1]
"""

import argparse
import re
import sys
from pathlib import Path

CANN = "/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/python/site-packages"
sys.path.insert(0, CANN)
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file

OPCODE = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")


def union_len(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total, cur_lo, cur_hi = 0, intervals[0][0], intervals[0][1]
    for lo, hi in intervals[1:]:
        if lo > cur_hi:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    return total + (cur_hi - cur_lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instr")
    ap.add_argument("--window", type=int, default=1, help="VF index (default 1 = 2nd)")
    args = ap.parse_args()

    entries = [
        e for e in decode_instr_logs_from_file(Path(args.instr))
        if e.core == 0 and e.sub == 1
    ]
    events = []
    for e in entries:
        m = OPCODE.search(e.dec or "")
        if m:
            events.append((e.time, e.is_popped, e.pc, int(m.group(1)), m.group(2)))

    issues, dones = {}, {}
    for time, popped, pc, ident, op in events:
        key = (pc, ident, op)
        if popped == 1 and key not in issues:
            issues[key] = time
        elif popped == 0:
            dones[key] = time
    windows = sorted(
        (issues[key], dones[key])
        for key in issues if key[2] == "VF_SIMT" and key in dones
    )
    print(f"VF windows: {len(windows)}  spans={[b - a for a, b in windows]}")
    lo, hi = windows[args.window]
    print(f"window {args.window}: [{lo}, {hi}] span={hi - lo}")

    in_win = [ev for ev in events if lo <= ev[0] <= hi and ev[1] == 1]
    per_op = {}
    for ev in in_win:
        per_op[ev[4]] = per_op.get(ev[4], 0) + 1
    print("\nopcode counts in window:")
    for op, n in sorted(per_op.items(), key=lambda kv: -kv[1]):
        print(f"  {op:14s} {n}")

    # busy unions per opcode (intervals from issue->completion pairs)
    print("\nbusy unions in window:")
    for op in sorted(per_op, key=lambda o: -per_op[o]):
        ivals = []
        for key, start in issues.items():
            if key[2] != op or key not in dones:
                continue
            end = dones[key]
            if start >= lo and end <= hi:
                ivals.append((start, end))
        u = union_len(ivals)
        n = len(ivals)
        print(f"  {op:14s} n={n:4d} busy={u:5d}  sum={sum(b - a for a, b in ivals):5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
