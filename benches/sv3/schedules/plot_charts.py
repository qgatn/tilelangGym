"""SV3 charts for the contiguous-K runs."""

import math
import re
import sys
from collections import Counter, defaultdict
from functools import reduce
from math import gcd
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CANN = "/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/python/site-packages"
sys.path.insert(0, CANN)
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "plots"
MN = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")
PC = re.compile(r"PC: (0x[0-9a-f]+)")
PIPE = re.compile(r"\)\s+([A-Z0-9]+)\s+:")
PALETTE = [
    "#1f4e79", "#c45c26", "#1a7a7a", "#e07a1f", "#1b5e20", "#5c4e78",
    "#3d5a80", "#7cb342", "#6b4c7a", "#8fb4d4", "#888888", "#b5b5b5",
]
# Address math / control on RVEC* — keep off opcode rows (still need RVEC filter).
OFF_ROWS = {
    "SIMT_IADD", "SIMT_IADD_I", "SIMT_LEA", "SIMT_S2R", "SIMT_SHFI",
    "SIMT_BRANCH", "SIMT_ISETP", "SIMT_ISETP_I", "SIMT_END", "SIMT_MOVI",
    "SIMT_MOV", "SIMT_PLOP3", "SIMT_FSETP", "SIMT_LOP3",
    "RV_SLDI", "RV_SSTI", "RV_VMOV", "RV_SMOV", "RV_SMOVI", "RV_SMOVK",
    "RV_PSET", "RV_SEND", "RV_VCI", "RV_VLOOP",
    "RV_SADD", "RV_SSHLI", "RV_SZEROEXT",
}


def is_rvec(dec):
    m = PIPE.search(dec or "")
    return bool(m and m.group(1).startswith("RVEC"))


def opcode(dec):
    match = MN.search(dec or "")
    return match.group(2) if match else None


def newest_bin(case: Path):
    bins = sorted(case.glob("**/instr.bin"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not bins:
        raise FileNotFoundError(case)
    return bins[0]


def opcode_id(dec):
    match = MN.search(dec or "")
    return (int(match.group(1)), match.group(2)) if match else (None, None)


def decode_pc(dec):
    m = PC.search(dec or "")
    return m.group(1) if m else None


def _infer_n_warps(per_pc_lens):
    counts = Counter(c for c in per_pc_lens if c != 1)
    if not counts:
        return 1
    common = [c for c, _ in counts.most_common()]
    n = reduce(gcd, common)
    if n <= 1:
        n = max(counts.items(), key=lambda kv: kv[1])[0]
    return n


def windows(path: Path, vf_name: str, warp: int | None = 0):
    """Return (span, ivals) where ivals are (issue_rel, done_rel, op) sorted by issue.

    Body rows/IPC use RVEC* pipes only (SIMT and SIMD vector path).
    For VF_SIMT, Instr IPC keeps one warp only (default w0); pass warp=None for all.
    """
    rows = [row for row in decode_instr_logs_from_file(path) if row.core == 0 and row.sub == 1]
    vf_issues, vf_dones = {}, {}
    all_issues, all_dones, rvec, pc_of = {}, {}, set(), {}
    for row in rows:
        ident, op = opcode_id(row.dec)
        if op is None:
            continue
        key = (row.pc, ident, op)
        if row.is_popped == 1 and key not in all_issues:
            all_issues[key] = row.time
            if is_rvec(row.dec):
                rvec.add(key)
                pc = decode_pc(row.dec)
                if pc:
                    pc_of[key] = pc
        elif row.is_popped == 0:
            all_dones[key] = row.time
        if op != vf_name:
            continue
        vkey = (row.pc, op)
        if row.is_popped == 1 and vkey not in vf_issues:
            vf_issues[vkey] = row.time
        elif row.is_popped == 0:
            vf_dones[vkey] = row.time
    spans = sorted(
        (vf_issues[k], vf_dones[k]) for k in vf_issues if k in vf_dones
    )
    if not spans:
        return 0, []
    # First VF only for ANALYSIS charts (first launch)
    start0, done0 = spans[0]
    raw = []
    for key, start in all_issues.items():
        op = key[2]
        if op == vf_name or key not in all_dones or key not in rvec:
            continue
        end = all_dones[key]
        if end < start0 or start > done0:
            continue
        lo = max(start, start0) - start0
        hi = min(end, done0) - start0
        if hi < lo:
            continue
        raw.append((lo, hi, op, pc_of.get(key), start))

    use_single = vf_name == "VF_SIMT" and warp is not None
    if use_single:
        per_pc = defaultdict(list)
        for lo, hi, op, pc, abs_start in raw:
            if pc is None:
                continue
            per_pc[pc].append((lo, hi, op, abs_start))
        n_warps = _infer_n_warps([len(v) for v in per_pc.values()])
        bad = {pc: len(v) for pc, v in per_pc.items()
               if len(v) != 1 and (len(v) % n_warps) != 0}
        if bad or n_warps <= 1:
            print(f"note: warp filter unsafe ({len(bad)} bad PCs, n_warps={n_warps}); "
                  f"Instr IPC keeps all warps")
            ivals = [(lo, hi, op) for lo, hi, op, _, _ in raw]
        else:
            ivals = []
            for pc, seq in per_pc.items():
                seq = sorted(seq, key=lambda x: (x[0], x[3]))
                if len(seq) == 1:
                    continue
                for k, (lo, hi, op, _) in enumerate(seq):
                    if (k % n_warps) == warp:
                        ivals.append((lo, hi, op))
            print(f"note: Instr IPC filtered to w{warp}/{n_warps}")
    else:
        ivals = [(lo, hi, op) for lo, hi, op, _, _ in raw]

    ivals.sort(key=lambda x: x[0])
    last = max((hi for _, hi, _ in ivals), default=0)
    return last + 1, ivals


def draw_instr(case: Path, title: str, out_name: str, kind: str):
    vf_name = "VF" if kind == "simd" else "VF_SIMT"
    ticks, ivals = windows(newest_bin(case), vf_name)
    counts = Counter(op for _, _, op in ivals)
    shown = [
        (op.removeprefix("SIMT_").removeprefix("RV_"), {op}, PALETTE[i % len(PALETTE)])
        for i, op in enumerate(
            op for op in sorted(counts, key=lambda name: (-counts[name], name))
            if op not in OFF_ROWS
        )
    ]
    # Bars: is_popped=1 → start, is_popped=0 → end; width = end-start+1; later overlaps on top.
    fig = plt.figure(figsize=(10.6, 0.34 * max(len(shown), 1) + 2.1), dpi=200, facecolor="white")
    fig.suptitle(title, x=0.01, ha="left", y=0.98, fontsize=13, fontweight="semibold")
    grid = GridSpec(2, 1, figure=fig, height_ratios=[2.15, 0.95], hspace=0.08, top=0.88, bottom=0.12, left=0.18, right=0.90)
    top, bot = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    n = len(shown)
    for i in range(n):
        top.axhspan(i - 0.5, i + 0.5, color="#f7f8f9" if i % 2 == 0 else "white", linewidth=0)
    op_to_y = {}
    for i, (label, names, color) in enumerate(shown):
        y = n - 1 - i
        for op in names:
            op_to_y[op] = (y, color)
        n_op = sum(1 for _, _, op in ivals if op in names)
        top.text(ticks * 1.015, y, f"×{n_op}", va="center", ha="left", fontsize=8.5, color="#444", clip_on=False)
    for z, (lo, hi, op) in enumerate(ivals):
        if op not in op_to_y:
            continue
        y, color = op_to_y[op]
        top.barh(y, hi - lo + 1, left=lo, height=0.62, color=color,
                 edgecolor="none", linewidth=0, align="center", zorder=2 + z)
    top.set_yticks(range(n), [label for label, _, _ in reversed(shown)])
    top.set_xlim(0, ticks)
    top.set_ylim(-0.55, n - 0.45)
    top.tick_params(axis="x", labelbottom=False, length=0)
    width = 20
    bins = max((ticks + width - 1) // width, 1)
    bucket = [0] * bins
    for lo, _, _ in ivals:
        if 0 <= lo < ticks:
            bucket[lo // width] += 1
    xs, ys = [], []
    for i, count in enumerate(bucket):
        a, b = i * width, min(ticks, (i + 1) * width)
        xs.append(a + (b - a) / 2)
        ys.append(count / (b - a) if b > a else 0)
    bot.fill_between(xs, ys, color="#5c4e78", alpha=0.14, linewidth=0)
    bot.plot(xs, ys, color="#5c4e78", linewidth=1.15)
    bot.set_xlim(0, ticks)
    peak = max(ys) if ys else 0.0
    ylim = 1.0 if peak <= 1.0 else math.ceil(peak * 2) / 2.0
    bot.set_ylim(0, ylim)
    bot.set_yticks([i * 0.5 for i in range(int(round(ylim * 2)) + 1)])
    bot.set_ylabel("Instr IPC (w0)" if kind == "simt" else "Instr IPC")
    bot.set_xlabel("tick (issue → completion)")
    bot.yaxis.grid(True, color="#eef0f2", linewidth=0.7)
    bot.set_axisbelow(True)
    for ax in (top, bot):
        ax.tick_params(length=0, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    out = OUT / out_name
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"VF {ticks}  ivals {len(ivals)}  IPC {len(ivals) / ticks:.2f}  -> {out.name}")


def draw_n():
    ns = [128, 256, 512, 1024, 2048]
    thread = [64493, 68114, 80067, 82842, 86750]
    warp = [20730, 38033, 93753, 184315, 449953]
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=200)
    ax.plot(ns, thread, marker="o", color="#1f4e79", label="Thread")
    ax.plot(ns, warp, marker="o", color="#c45c26", label="Warp")
    ax.set_xticks(ns)
    ax.set_xlabel("N")
    ax.set_ylabel("VF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUT / "n_vf.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"-> {out.name}")


def main():
    OUT.mkdir(exist_ok=True)
    for stale in ("n_vf.png", "simd_instr.png", "simt_first_launch.png"):
        path = OUT / stale
        if path.exists():
            path.unlink()
    draw_n()
    draw_instr(
        HERE / "runs" / "simd-n4096-k32",
        "SIMD, N = 4096, K = 32",
        "simd_instr.png",
        "simd",
    )
    draw_instr(
        HERE / "runs" / "ub-n4096-k32",
        "Buffer SIMT, N = 4096, K = 32, 256 threads",
        "simt_first_launch.png",
        "simt",
    )


if __name__ == "__main__":
    main()
