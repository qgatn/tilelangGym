import re, sys, math
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/python/site-packages")
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file
MN = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")
bin_path, rows_n, threads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [r for r in decode_instr_logs_from_file(Path(bin_path)) if r.core == 0 and r.sub == 1]
iss, don = {}, {}
pipes = {}
PIPE = re.compile(r"\)\s+([A-Z0-9]+)\s+:")
for r in rows:
    m, p = MN.search(r.dec or ""), PIPE.search(r.dec or "")
    if not m: continue
    key = (r.pc, int(m.group(1)), m.group(2))
    if p: pipes[key] = p.group(1)
    if r.is_popped == 1 and key not in iss: iss[key] = r.time
    elif r.is_popped == 0: don[key] = r.time
vfs = sorted((iss[k], don[k]) for k in iss if k[2] == "VF_SIMT" and k in don)
if not vfs:
    print("  rate: NO_VF_SIMT (ops not inside a VF window)")
    raise SystemExit
lo, hi = vfs[0]
busy = set(); n_issues = 0; pipes_lds = Counter()
for k, t in iss.items():
    if k[2] != "SIMT_LDS" or k not in don: continue
    d = don[k]
    if d < lo or t > hi: continue
    n_issues += 1
    pipes_lds[pipes.get(k, "?")] += 1
    busy.update(range(max(t, lo), min(d, hi) + 1))
bytes_unique = rows_n * 16
print(f"  rate: issues={n_issues} busy={len(busy)} B/cyc={bytes_unique / max(len(busy),1):.2f} pipes={dict(pipes_lds)}")
