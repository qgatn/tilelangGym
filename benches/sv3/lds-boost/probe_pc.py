"""Where do the xv_scalars spills come from? PC schedule of the probe trace."""
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/python/site-packages")
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file

OP = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")
bin_path = Path(sys.argv[1])
rows = [r for r in decode_instr_logs_from_file(bin_path) if r.core == 0 and r.sub == 1]
events = []
for r in rows:
    m = OP.search(r.dec or "")
    if m:
        events.append((r.time, r.is_popped, r.pc, int(m.group(1)), m.group(2)))
issues, dones = {}, {}
for t, pop, pc, i, op in events:
    key = (pc, i, op)
    if pop == 1 and key not in issues:
        issues[key] = t
    elif pop == 0:
        dones[key] = t
vfs = sorted((issues[k], dones[k]) for k in issues if k[2] == "VF_SIMT" and k in dones)
lo, hi = vfs[1]
first, count = {}, defaultdict(int)
for t, pop, pc, i, op in events:
    if pop == 1 and lo <= t <= hi:
        count[(pc, op)] += 1
        if (pc, op) not in first or t < first[(pc, op)]:
            first[(pc, op)] = t
print(f"VF[1] span {hi - lo}")
for (pc, op), t in sorted(first.items(), key=lambda kv: kv[1]):
    if op in ("SIMT_STK", "SIMT_LDK", "SIMT_LDS", "SIMT_F2F", "SIMT_FFMA", "SIMT_FFMA_I", "SIMT_STS", "SIMT_BAR_THREAD_BLOCK", "SIMT_END", "SIMT_BRANCH"):
        print(f"  +{t - lo:5d}  {op:22s} n={count[(pc, op)]}")
