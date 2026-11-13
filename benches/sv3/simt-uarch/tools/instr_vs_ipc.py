#!/usr/bin/env python3
"""Instruction rows plus the IPC line, from one instr.bin.

One row per opcode found in the VF window. --bin sets the IPC bin width in ticks.
Adds --label so charts get a figure number and name: "Figure 3. LDS packed v2, 256 threads".
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

CANN = "/mnt/fluxdata/Ascend/cann_92b2/cann-9.2.0-beta.2/python/site-packages"
sys.path.insert(0, CANN)
from cannsim.core.public.instr_decoder import decode_instr_logs_from_file

MN = re.compile(r"\(ID:\s*(\d+)\)\s+([A-Z0-9_]+)")
PALETTE = [
    "#1f4e79", "#8fb4d4", "#1a7a7a", "#e07a1f", "#1b5e20", "#7cb342",
    "#5c4e78", "#c45c26", "#3d5a80", "#6b4c7a", "#b5b5b5", "#888888",
]


def opcode(dec):
    m = MN.search(dec)
    return m.group(2) if m else None


def vf_spans(rows, vf_name):
    starts, dones = [], []
    for r in rows:
        if opcode(r.dec) != vf_name:
            continue
        if r.is_popped == 1:
            starts.append(r.time)
        elif r.is_popped == 0:
            dones.append(r.time)
    n = min(len(starts), len(dones))
    return list(zip(starts[:n], dones[:n]))


def window(path, kind, vf_idx=0):
    vf_name = "VF_SIMT" if kind == "simt" else "VF"
    rows = [r for r in decode_instr_logs_from_file(Path(path)) if r.core == 0 and r.sub == 1]
    spans = vf_spans(rows, vf_name)
    if not spans:
        raise SystemExit(f"no {vf_name} windows found")
    if vf_idx < 0 or vf_idx >= len(spans):
        raise SystemExit(f"--vf {vf_idx} out of range (launches={len(spans)})")
    issue, done = spans[vf_idx]
    vf_ticks = done - issue
    ops = []
    for r in rows:
        if r.is_popped != 1:
            continue
        op = opcode(r.dec)
        if op is None or op == vf_name:
            continue
        if issue <= r.time <= done:
            ops.append((r.time - issue, op))
    # Clip the plot to the last issue — VF drain after that is empty axis.
    last = max((t for t, _ in ops), default=0)
    span = last + 1
    return vf_ticks, span, ops


def rows_from(ops):
    counts = Counter(op for _, op in ops)
    rows = []
    for i, op in enumerate(sorted(counts, key=lambda name: (-counts[name], name))):
        label = op.removeprefix("RV_").removeprefix("SIMT_")
        rows.append((label, {op}, PALETTE[i % len(PALETTE)]))
    return rows


def ipc_line(ops, ticks, width):
    n = (ticks + width - 1) // width
    counts = [0] * n
    for t, _ in ops:
        if 0 <= t < ticks:
            counts[t // width] += 1
    xs, ys = [], []
    for i, c in enumerate(counts):
        lo, hi = i * width, min(ticks, (i + 1) * width)
        w = hi - lo
        xs.append(lo + w / 2)
        ys.append(c / w)
    return xs, ys


def draw(span, ops, rows, title, out, width):
    fig = plt.figure(figsize=(10.6, 0.34 * len(rows) + 2.1), dpi=200, facecolor="white")
    fig.suptitle(title, x=0.01, ha="left", y=0.98, fontsize=13, fontweight="semibold")
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.15, 0.95], hspace=0.08,
                  top=0.88, bottom=0.12, left=0.12, right=0.90)
    top, bot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    n = len(rows)
    for i in range(n):
        top.axhspan(i - 0.5, i + 0.5, color="#f7f8f9" if i % 2 == 0 else "white", linewidth=0)
    for i, (label, names, color) in enumerate(rows):
        y = n - 1 - i
        ts = [t for t, op in ops if op in names]
        top.vlines(ts, y - 0.34, y + 0.34, color=color, linewidth=0.7)
        top.text(span * 1.015, y, f"×{len(ts)}", va="center", ha="left", fontsize=8.5, color="#444", clip_on=False)
    top.set_yticks(range(n), [r[0] for r in reversed(rows)])
    top.set_xlim(0, span)
    top.set_ylim(-0.55, n - 0.45)
    top.tick_params(axis="x", labelbottom=False, length=0)
    xs, ys = ipc_line(ops, span, width)
    bot.fill_between(xs, ys, color="#5c4e78", alpha=0.14, linewidth=0)
    bot.plot(xs, ys, color="#5c4e78", linewidth=1.15)
    bot.set_xlim(0, span)
    bot.set_ylim(0, max(4, max(ys) + 0.4) if ys else 4)
    bot.set_ylabel("IPC")
    bot.set_xlabel("tick (to last issue)")
    bot.yaxis.grid(True, color="#eef0f2", linewidth=0.7)
    bot.set_axisbelow(True)
    for ax in (top, bot):
        ax.tick_params(length=0, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.08)
    print(f"span {span}  issues {len(ops)}  IPC {len(ops) / span:.2f}  -> {out}")


def main(argv):
    p = argparse.ArgumentParser(description="Instruction rows plus the IPC line, from one instr.bin.")
    p.add_argument("instr")
    p.add_argument("kind", choices=("simd", "simt"))
    p.add_argument("out")
    p.add_argument("--bin", type=int, default=20, help="IPC bin width in ticks (default 20)")
    p.add_argument("--label", default="", help="Figure number and name for the chart title")
    p.add_argument("--vf", type=int, default=0, help="0-based VF launch index (default 0)")
    args = p.parse_args(argv[1:])
    if args.bin < 1:
        p.error("--bin must be a positive number of ticks")
    vf_ticks, span, ops = window(args.instr, args.kind, args.vf)
    rows = rows_from(ops)
    for label, names, _ in rows:
        print(f"{label} {sum(op in names for _, op in ops)}")
    title = (f"{args.label}  ·  span {span} (VF {vf_ticks})" if args.label
             else f"{args.kind.upper()} instructions  ·  span {span} (VF {vf_ticks})")
    draw(span, ops, rows, title, args.out, args.bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
