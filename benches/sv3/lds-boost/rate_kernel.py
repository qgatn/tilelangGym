import re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/python/site-packages")
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file
MN = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")
bin_path, n, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [r for r in decode_instr_logs_from_file(Path(bin_path)) if r.core == 0 and r.sub == 1]
iss, don = {}, {}
for r in rows:
    m = MN.search(r.dec or "")
    if not m: continue
    key = (r.pc, int(m.group(1)), m.group(2))
    if r.is_popped == 1 and key not in iss: iss[key] = r.time
    elif r.is_popped == 0: don[key] = r.time
vfs = sorted((iss[k], don[k]) for k in iss if k[2] == "VF_SIMT" and k in don)
for vi,(lo,hi) in enumerate(vfs):
    busy=set(); ni=0
    for k2,t in iss.items():
        if k2[2]!="SIMT_LDS" or k2 not in don: continue
        d=don[k2]
        if d<lo or t>hi: continue
        ni+=1; busy.update(range(max(t,lo),min(d,hi)+1))
    bytes_l = n//2*k*2
    print(f"VF[{vi}] span={hi-lo} LDS_issues={ni} LDS_busy={len(busy)} LDS_B/cyc={bytes_l/max(len(busy),1):.2f}")
