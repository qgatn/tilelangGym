#!/usr/bin/env python3
"""Opcode activity chart: one row per opcode, bars from issue → completion.

Each instruction instance is paired by (pc, id, opcode):
  is_popped=1 → start (instr_popped_log)
  is_popped=0 → end   (instr_log)
Bar spans [start, end] inclusive (drawn to end+1 so the last cycle is visible).
Overlapping bars on the same row are drawn in issue order — later starts on top.

Only RVEC* pipes (RVECEX/RVECLD/RVECST/…) — drops MTE/VEC/PUSHQ/FLOWCTRL/SCALAR.
IPC line = RVEC issues / tick. --vf selects the VF launch.
"""

from __future__ import annotations

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
PIPE = re.compile(r"\)\s+([A-Z0-9]+)\s+:")
PALETTE = [
    "#1f4e79", "#8fb4d4", "#1a7a7a", "#e07a1f", "#1b5e20", "#7cb342",
    "#5c4e78", "#c45c26", "#3d5a80", "#6b4c7a", "#b5b5b5", "#888888",
]


def opcode(dec):
    m = MN.search(dec or "")
    return m.group(2) if m else None


def opcode_id(dec):
    m = MN.search(dec or "")
    return (int(m.group(1)), m.group(2)) if m else (None, None)


def is_rvec(dec):
    m = PIPE.search(dec or "")
    return bool(m and m.group(1).startswith("RVEC"))


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


def pair_intervals(rows, vf_name):
    """Return list of (issue, done, op) for every completed RVEC* instruction."""
    issues, dones, rvec = {}, {}, set()
    for r in rows:
        ident, op = opcode_id(r.dec)
        if op is None:
            continue
        key = (r.pc, ident, op)
        if r.is_popped == 1 and key not in issues:
            issues[key] = r.time
            if is_rvec(r.dec):
                rvec.add(key)
        elif r.is_popped == 0:
            dones[key] = r.time
    out = []
    for key, start in issues.items():
        if key[2] == vf_name or key not in dones or key not in rvec:
            continue
        out.append((start, dones[key], key[2]))
    out.sort(key=lambda x: x[0])  # issue order → later overlaps draw on top
    return out


def window(path, kind, vf_idx=0):
    vf_name = "VF_SIMT" if kind == "simt" else "VF"
    rows = [r for r in decode_instr_logs_from_file(Path(path)) if r.core == 0 and r.sub == 1]
    spans = vf_spans(rows, vf_name)
    if not spans:
        raise SystemExit(f"no {vf_name} windows found")
    # vf_idx == -1 → full window from first VF issue to last VF completion
    if vf_idx == -1:
        issue0, done0 = spans[0][0], spans[-1][1]
    else:
        if vf_idx < 0 or vf_idx >= len(spans):
            raise SystemExit(f"--vf {vf_idx} out of range (launches={len(spans)}; use -1 for full)")
        issue0, done0 = spans[vf_idx]
    vf_ticks = done0 - issue0
    ivals = []
    for start, end, op in pair_intervals(rows, vf_name):
        if end < issue0 or start > done0:
            continue
        # clip to VF window, relative ticks; keep inclusive end
        lo = max(start, issue0) - issue0
        hi = min(end, done0) - issue0
        if hi < lo:
            continue
        ivals.append((lo, hi, op))
    # clip axis to last completion inside the window
    last = max((hi for _, hi, _ in ivals), default=0)
    span = last + 1
    return vf_ticks, span, ivals


def rows_from(ivals):
    counts = Counter(op for _, _, op in ivals)
    rows = []
    for i, op in enumerate(sorted(counts, key=lambda name: (-counts[name], name))):
        label = op.removeprefix("RV_").removeprefix("SIMT_")
        rows.append((label, {op}, PALETTE[i % len(PALETTE)]))
    return rows


def ipc_line(ivals, ticks, width):
    """Issue-rate IPC (one count per instruction at its issue tick)."""
    n = (ticks + width - 1) // width
    counts = [0] * max(n, 1)
    for lo, _, _ in ivals:
        if 0 <= lo < ticks:
            counts[lo // width] += 1
    xs, ys = [], []
    for i, c in enumerate(counts):
        a, b = i * width, min(ticks, (i + 1) * width)
        w = b - a
        if w <= 0:
            continue
        xs.append(a + w / 2)
        ys.append(c / w)
    return xs, ys


def draw(span, ivals, rows, title, out, width):
    fig = plt.figure(figsize=(10.6, 0.34 * len(rows) + 2.1), dpi=200, facecolor="white")
    fig.suptitle(title, x=0.01, ha="left", y=0.98, fontsize=13, fontweight="semibold")
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.15, 0.95], hspace=0.08,
                  top=0.88, bottom=0.12, left=0.12, right=0.90)
    top, bot = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    n = len(rows)
    for i in range(n):
        top.axhspan(i - 0.5, i + 0.5, color="#f7f8f9" if i % 2 == 0 else "white", linewidth=0)

    # draw in issue order so later-starting overlaps sit on top
    op_to_y = {}
    for i, (label, names, color) in enumerate(rows):
        y = n - 1 - i
        for op in names:
            op_to_y[op] = (y, color)
        n_op = sum(1 for _, _, op in ivals if op in names)
        top.text(span * 1.015, y, f"×{n_op}", va="center", ha="left",
                 fontsize=8.5, color="#444", clip_on=False)

    for z, (lo, hi, op) in enumerate(ivals):
        if op not in op_to_y:
            continue
        y, color = op_to_y[op]
        # inclusive end: bar covers cycles lo..hi
        top.barh(y, hi - lo + 1, left=lo, height=0.62, color=color,
                 edgecolor="none", linewidth=0, align="center", zorder=2 + z)

    top.set_yticks(range(n), [r[0] for r in reversed(rows)])
    top.set_xlim(0, span)
    top.set_ylim(-0.55, n - 0.45)
    top.tick_params(axis="x", labelbottom=False, length=0)

    xs, ys = ipc_line(ivals, span, width)
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
    out = Path(out)
    if out.suffix != ".png":
        out = out.with_suffix(".png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"span {span}  ivals {len(ivals)}  IPC {len(ivals) / span:.2f}  -> {out}")


def main(argv):
    p = argparse.ArgumentParser(description="Opcode rows with issue→completion bars.")
    p.add_argument("instr")
    p.add_argument("kind", choices=("simd", "simt"))
    p.add_argument("out")
    p.add_argument("--bin", type=int, default=20, help="IPC bin width in ticks (default 20)")
    p.add_argument("--label", default="", help="Figure number and name for the chart title")
    p.add_argument("--vf", type=int, default=0, help="0-based VF launch index (default 0)")
    args = p.parse_args(argv[1:])
    if args.bin < 1:
        p.error("--bin must be a positive number of ticks")
    vf_ticks, span, ivals = window(args.instr, args.kind, args.vf)
    rows = rows_from(ivals)
    for label, names, _ in rows:
        print(f"{label} {sum(1 for _, _, op in ivals if op in names)}")
    title = (f"{args.label}  ·  span {span} (VF {vf_ticks})" if args.label
             else f"{args.kind.upper()} instructions  ·  span {span} (VF {vf_ticks})")
    draw(span, ivals, rows, title, args.out, args.bin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
