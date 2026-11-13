#!/usr/bin/env python
# Analyze one npusim instr.bin with busy-cycle accounting.
# Busy cycles = union of [issue, completion) intervals of the target opcode:
# overlapping warps are counted once (merge intervals, then sum).
# Usage: analyze.py <instr.bin> <OPNAME>
import re
import sys
from collections import defaultdict

from cannsim.core.public.instr_decoder import decode_instr_logs_from_file

binp, opname = sys.argv[1], sys.argv[2]
pat = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")

issues, dones = {}, {}
for e in decode_instr_logs_from_file(binp):
    if e.core != 0 or e.sub != 1:
        continue
    m = pat.search(e.dec or "")
    if not m:
        continue
    key = (e.pc, int(m.group(1)), m.group(2))
    if e.is_popped == 1 and key not in issues:
        issues[key] = e.time
    elif e.is_popped == 0:
        dones[key] = e.time

rows = sorted((issues[k], dones[k]) for k in issues if k[2] == opname and k in dones)
if not rows:
    print("no", opname)
    sys.exit(1)

# union of intervals (busy cycles)
merged = []
start, end = rows[0]
for s, d in rows[1:]:
    if s <= end:
        end = max(end, d)
    else:
        merged.append((start, end))
        start, end = s, d
merged.append((start, end))
busy = sum(e - s for s, e in merged)

acts = sorted(d - s for s, d in rows)
span = rows[-1][1] - rows[0][0]
print(f"op={opname} n={len(rows)} span={span} busy={busy} act_min={acts[0]} act_med={acts[len(acts)//2]} act_max={acts[-1]}")
