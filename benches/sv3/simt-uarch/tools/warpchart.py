#!/usr/bin/env python3
"""Warp-activity chart: one row per warp, bars from issue → completion.

Each instruction instance is paired by (pc, id, opcode):
  is_popped=1 → start, is_popped=0 → end.
Warp id = k % n_warps on each PC's issue sequence (tick order). Bars that
overlap are drawn in issue order so later starts sit on top.

IPC line = issues / tick. --vf selects the VF launch.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from math import gcd
from functools import reduce
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CANN = "/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/python/site-packages"
sys.path.insert(0, CANN)
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file

MN = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")
PC = re.compile(r"PC: (0x[0-9a-f]+)")
PALETTE = [
    "#1f4e79", "#e07a1f", "#1a7a7a", "#1b5e20", "#7cb342", "#5c4e78",
    "#c45c26", "#3d5a80", "#6b4c7a", "#b5b5b5",
]
COLOR_SKIP = {"IADD_I", "IADD", "LEA", "IMAD", "S2R", "SHFI", "MOV", "MOVI",
              "BRANCH", "ISETP_I", "STK", "LDK", "END", "END_LABEL", "CALL"}


def fields(dec):
    m = MN.search(dec or "")
    p = PC.search(dec or "")
    ident = int(m.group(1)) if m else None
    op = m.group(2) if m else None
    pc = p.group(1) if p else None
    return op, pc, ident


def vf_spans(rows):
    issues, dones = {}, {}
    for r in rows:
        op, pc, _ = fields(r.dec)
        if op != "VF_SIMT":
            continue
        key = (pc, op)
        if r.is_popped == 1 and key not in issues:
            issues[key] = r.time
        elif r.is_popped == 0:
            dones[key] = r.time
    spans = []
    for key, start in issues.items():
        if key in dones:
            spans.append((start, dones[key]))
    spans.sort()
    return spans


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("instr")
    ap.add_argument("out")
    ap.add_argument("--bin", type=int, default=20)
    ap.add_argument("--label", default="", help="Figure number and name for the chart title")
    ap.add_argument("--vf", type=int, default=0, help="0-based VF launch index (default 0)")
    args = ap.parse_args(argv[1:])

    rows = [r for r in decode_instr_logs_from_file(Path(args.instr)) if r.core == 0 and r.sub == 1]
    spans = vf_spans(rows)
    if not spans:
        print("REFUSE: no VF_SIMT windows found.")
        return 1
    if args.vf < 0 or args.vf >= len(spans):
        print(f"REFUSE: --vf {args.vf} out of range (launches={len(spans)}).")
        return 1
    iss, done = spans[args.vf]
    vf_ticks = done - iss

    # pair issue→completion
    issues, dones = {}, {}
    for r in rows:
        op, pc, ident = fields(r.dec)
        if op is None:
            continue
        key = (r.pc, ident, op)
        if r.is_popped == 1 and key not in issues:
            issues[key] = r.time
        elif r.is_popped == 0:
            dones[key] = r.time

    # per-PC list of (issue_rel, done_rel, short_op) inside VF, sorted by issue
    per_pc = defaultdict(list)
    for key, start in issues.items():
        if key not in dones:
            continue
        op = key[2]
        if op == "VF_SIMT":
            continue
        end = dones[key]
        if end < iss or start > done:
            continue
        lo = max(start, iss) - iss
        hi = min(end, done) - iss
        if hi < lo:
            continue
        short = op.removeprefix("SIMT_").removeprefix("RV_")
        # use hex PC string from first matching decode if available; fall back to int
        pc_key = hex(key[0]) if isinstance(key[0], int) else str(key[0])
        # Prefer the PC string from the decode text when present
        per_pc[pc_key].append((lo, hi, short))

    # Rebuild per_pc keyed by decode PC string for stable warp inference
    per_pc2 = defaultdict(list)
    for r in rows:
        if r.is_popped != 1:
            continue
        op, pc, ident = fields(r.dec)
        if op in (None, "VF_SIMT") or pc is None:
            continue
        key = (r.pc, ident, op)
        if key not in issues or key not in dones:
            continue
        start, end = issues[key], dones[key]
        if end < iss or start > done:
            continue
        lo = max(start, iss) - iss
        hi = min(end, done) - iss
        if hi < lo:
            continue
        short = op.removeprefix("SIMT_").removeprefix("RV_")
        per_pc2[pc].append((lo, hi, short, start))  # start abs for sort stability
    per_pc = per_pc2

    counts_pc = Counter(len(v) for v in per_pc.values() if len(v) != 1)
    if not counts_pc:
        print("REFUSE: no per-warp PCs found.")
        return 1
    common = [c for c, _ in counts_pc.most_common()]
    n_warps = reduce(gcd, common)
    if n_warps <= 1:
        n_warps = max(counts_pc.items(), key=lambda kv: kv[1])[0]
    once = {pc: len(v) for pc, v in per_pc.items() if len(v) == 1}
    bad = {pc: len(v) for pc, v in per_pc.items()
           if len(v) != 1 and (len(v) % n_warps) != 0}
    if bad:
        print(f"REFUSE: {len(bad)}/{len(per_pc)} PCs not multiple of {n_warps} "
              f"(e.g. {dict(list(bad.items())[:3])}). Warp inference is unsafe here.")
        return 1
    if once:
        print(f"note: skipping {len(once)} once-per-kernel PC(s) from warp assignment")

    events = []  # (issue_rel, done_rel, warp, op) — sorted by issue for z-order
    ipc_extra = []
    for pc, seq in per_pc.items():
        seq = sorted(seq, key=lambda x: (x[0], x[3] if len(x) > 3 else 0))
        if len(seq) == 1:
            lo, hi, op = seq[0][0], seq[0][1], seq[0][2]
            ipc_extra.append((lo, hi, op))
            continue
        for k, item in enumerate(seq):
            lo, hi, op = item[0], item[1], item[2]
            events.append((lo, hi, k % n_warps, op))
    events.sort(key=lambda e: e[0])

    counts = Counter(op for _, _, _, op in events)
    for _, _, op in ipc_extra:
        counts[op] += 1
    opcolors = {}
    ci = 0
    for op in sorted(counts, key=lambda n: -counts[n]):
        if op in COLOR_SKIP:
            opcolors[op] = "#b5b5b5"
        else:
            opcolors[op] = PALETTE[ci % len(PALETTE)]
            ci += 1

    last = max([hi for _, hi, _, _ in events] + [hi for _, hi, _ in ipc_extra] + [0])
    span = last + 1

    title = (f"{args.label}  ·  {n_warps} warps  ·  span {span} (VF {vf_ticks})" if args.label
             else f"SIMT warps  ·  {n_warps} warps  ·  span {span} (VF {vf_ticks})")
    fig = plt.figure(figsize=(10.6, 0.30 * n_warps + 2.1), dpi=200, facecolor="white")
    fig.suptitle(title, x=0.01, ha="left", y=0.98, fontsize=13, fontweight="semibold")
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.15, 0.85], hspace=0.08,
                  top=0.88, bottom=0.12, left=0.09, right=0.97)
    top, bot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for w in range(n_warps):
        top.axhspan(w - 0.5, w + 0.5, color="#f7f8f9" if w % 2 == 0 else "white", linewidth=0)
    for z, (lo, hi, w, op) in enumerate(events):
        top.barh(w, hi - lo + 1, left=lo, height=0.62, color=opcolors[op],
                 edgecolor="none", linewidth=0, align="center", zorder=2 + z)
    top.set_yticks(range(n_warps), [f"w{w}" for w in range(n_warps)])
    top.set_xlim(0, span)
    top.set_ylim(-0.55, n_warps - 0.45)
    top.tick_params(axis="x", labelbottom=False, length=0)

    n = (span + args.bin - 1) // args.bin
    cnt = [0] * max(n, 1)
    for lo, _, _, _ in events:
        if 0 <= lo < span:
            cnt[lo // args.bin] += 1
    for lo, _, _ in ipc_extra:
        if 0 <= lo < span:
            cnt[lo // args.bin] += 1
    xs, ys = [], []
    for i, c in enumerate(cnt):
        a, b = i * args.bin, min(span, (i + 1) * args.bin)
        w = b - a
        if w <= 0:
            continue
        xs.append(a + w / 2)
        ys.append(c / w)
    bot.fill_between(xs, ys, color="#5c4e78", alpha=0.14, linewidth=0)
    bot.plot(xs, ys, color="#5c4e78", linewidth=1.15)
    bot.set_xlim(0, span)
    bot.set_ylim(0, max(4, (max(ys) if ys else 0) + 0.4))
    bot.set_ylabel("IPC")
    bot.set_xlabel("tick (issue → completion)")
    bot.yaxis.grid(True, color="#eef0f2", linewidth=0.7)
    bot.set_axisbelow(True)
    for ax in (top, bot):
        ax.tick_params(length=0, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [plt.Line2D([], [], color=opcolors[o], linewidth=2) for o in opcolors]
    bot.legend(handles, list(opcolors), fontsize=6, ncol=5, loc="upper right", frameon=False)

    out = Path(args.out)
    if out.suffix != ".png":
        out = out.with_suffix(".png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    n_issues = len(events) + len(ipc_extra)
    print(f"VF[{args.vf}] span {span}  warps {n_warps}  ivals {n_issues}  "
          f"IPC {n_issues / span:.2f}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
