"""Full LDS/F2F/FFMA issue timeline inside one VF window.

Prints a per-bin issue histogram for chosen opcodes plus every LDS issue
with completion and latency. Bins 20 ticks.

Usage: lds_issue_timeline.py <instr.bin> [--window 1] [--ops SIMT_LDS,SIMT_F2F]
"""

import argparse
import re
import sys
from pathlib import Path

CANN = "/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/python/site-packages"
sys.path.insert(0, CANN)
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file

OPCODE = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instr")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--ops", default="SIMT_LDS")
    ap.add_argument("--bin", type=int, default=20)
    args = ap.parse_args()
    watch = set(args.ops.split(","))

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

    # histogram of watched-op issues per bin
    nbins = (hi - lo + args.bin - 1) // args.bin
    hist = [0] * nbins
    for time, popped, pc, ident, op in events:
        if popped == 1 and op in watch and lo <= time < hi:
            hist[(time - lo) // args.bin] += 1
    print(f"window [{lo},{hi}] bin={args.bin} ops={sorted(watch)}")
    for i, c in enumerate(hist):
        bar = "#" * min(c, 60)
        print(f"  {i * args.bin:5d} {c:3d} {bar}")

    # all watched-op issues with latency (issue->completion)
    print("\nall issues (t rel, done rel, lat, op, pc_tail):")
    rows = []
    for key, t in issues.items():
        op = key[2]
        if op not in watch or key not in dones:
            continue
        done = dones[key]
        if lo <= t < hi:
            rows.append((t - lo, done - lo, done - t, op, key[0] % 10000))
    for t, done, lat, op, pc in sorted(rows):
        print(f"  {t:5d} {done:5d} {lat:4d} {op} …{pc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
