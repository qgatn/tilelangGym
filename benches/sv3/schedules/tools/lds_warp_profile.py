"""Which warp issues LDS, when, relative to LDS busy-union segments.

Usage: lds_warp_profile.py <instr.bin> [--window 1]
"""

import argparse
import re
import sys
from pathlib import Path

CANN = "/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/python/site-packages"
sys.path.insert(0, CANN)
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file

OPCODE = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")
WARP = 32


def union_segments(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    segs = []
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo > cur_hi:
            segs.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    segs.append((cur_lo, cur_hi))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instr")
    ap.add_argument("--window", type=int, default=1)
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
        (issues[k], dones[k]) for k in issues
        if k[2] == "VF_SIMT" and k in dones
    )
    lo, hi = windows[args.window]

    lds_ivals = sorted(
        (t, dones[k]) for k, t in issues.items()
        if k[2] == "SIMT_LDS" and k in dones and lo <= t and dones[k] <= hi
    )
    segs = union_segments(lds_ivals)
    print("LDS busy segments (start end len gap_from_prev):")
    prev_end = None
    for s, e in segs:
        gap = "" if prev_end is None else str(s - prev_end)
        print(f"  {s:5d} {e:5d} len={e - s:4d} gap={gap}")
        prev_end = e

    # per-LDS: warp id, pc, issue time, completion, latency
    pairs = sorted(
        (
            (t, dones[k], k)
            for k, t in issues.items()
            if k[2] == "SIMT_LDS" and k in dones and lo <= t and dones[k] <= hi
        ),
        key=lambda p: p[0],
    )
    print("\nfirst 40 LDS (warp pc issue done lat):")
    for t, done, key in pairs[:40]:
        warp = key[0] % WARP
        print(f"  w{warp:2d} pc={key[0]:4d} t={t:5d} done={done:5d} lat={done - t:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
