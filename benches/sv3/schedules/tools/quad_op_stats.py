"""Quad dbl_buf: FFMA union, F2F/STS busy on VF windows, post-issue tail gaps."""
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
    total, lo, hi = 0, intervals[0][0], intervals[0][1]
    for a, b in intervals[1:]:
        if a > hi:
            total += hi - lo
            lo, hi = a, b
        else:
            hi = max(hi, b)
    return total + (hi - lo)


bin_path = sys.argv[1]
entries = [e for e in decode_instr_logs_from_file(Path(bin_path)) if e.core == 0 and e.sub == 1]
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

windows = sorted((issues[k], dones[k]) for k in issues if k[2] == "VF_SIMT" and k in dones)
for widx, (lo, hi) in enumerate(windows):
    last_issue = max((t for k, t in issues.items() if lo <= t <= hi), default=lo)
    print(f"VF[{widx}]: span={hi - lo} last_issue_rel={last_issue - lo} tail={hi - last_issue}")
    for name, label in [
        (("SIMT_FFMA", "SIMT_FFMA_I"), "FFMA(+I) union"),
        (("SIMT_FFMA",), "FFMA only"),
        (("SIMT_LDS",), "LDS"),
        (("SIMT_F2F",), "F2F"),
        (("SIMT_STS",), "STS"),
    ]:
        ivals = [
            (t, dones[k])
            for k, t in issues.items()
            if k[2] in name and k in dones and lo <= t and dones[k] <= hi
        ]
        print(f"  {label:16s} n={len(ivals):4d} busy={union_len(ivals):5d}")
