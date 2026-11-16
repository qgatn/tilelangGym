"""IPC and instruction charts for the TileLang buffer GEMV and the CCE GEMV.

Bars: is_popped=1 → start, is_popped=0 → end (width end-start+1); later overlaps
on top, no borders. Only RVEC* pipes (via plot_charts.windows).
"""

import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCHED = Path(__file__).resolve().parents[1] / "schedules"
sys.path.insert(0, str(SCHED))
import plot_charts as pc

OUT = Path(__file__).resolve().parents[1] / "plots"
CCE_BIN = pc.newest_bin(Path(__file__).resolve().parent)
TL_BIN = pc.newest_bin(SCHED / "runs" / "ub-n4096-k32")


def shown_rows(ivals):
    counts = Counter(op for _, _, op in ivals)
    return [
        (op.removeprefix("SIMT_").removeprefix("RV_"), {op}, pc.PALETTE[i % len(pc.PALETTE)])
        for i, op in enumerate(
            op for op in sorted(counts, key=lambda name: (-counts[name], name))
            if op not in pc.OFF_ROWS
        )
    ]


def ipc(ticks, ivals, width=20):
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
    return xs, ys


# Figures 4/5: first 2000 ticks of the first VF (see cce-run.md).
ZOOM_TICKS = 2000


def draw(path, title, out_name, xlim=None):
    ticks, ivals = pc.windows(path, "VF_SIMT")
    rows = shown_rows(ivals)
    view = ticks if xlim is None else min(ticks, xlim)
    fig = plt.figure(figsize=(10.6, 0.34 * max(len(rows), 1) + 2.1), dpi=160, facecolor="white")
    fig.suptitle(title, x=0.01, ha="left", y=0.98, fontsize=13, fontweight="semibold")
    grid = GridSpec(2, 1, figure=fig, height_ratios=[2.15, 0.95], hspace=0.08, top=0.88, bottom=0.12, left=0.16, right=0.90)
    top, bot = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    n = len(rows)
    for i in range(n):
        top.axhspan(i - 0.5, i + 0.5, color="#f7f8f9" if i % 2 == 0 else "white", linewidth=0)
    op_to_y = {}
    for i, (label, names, color) in enumerate(rows):
        y = n - 1 - i
        for op in names:
            op_to_y[op] = (y, color)
        n_op = sum(1 for lo, _, op in ivals if op in names and lo < view)
        top.text(view * 1.01, y, f"×{n_op}", va="center", ha="left", fontsize=8.5, color="#444", clip_on=False)
    for z, (lo, hi, op) in enumerate(ivals):
        if op not in op_to_y or lo >= view:
            continue
        y, color = op_to_y[op]
        hi_c = min(hi, view - 1)
        if hi_c < lo:
            continue
        top.barh(y, hi_c - lo + 1, left=lo, height=0.62, color=color,
                 edgecolor="none", linewidth=0, align="center", zorder=2 + z)
    top.set_yticks(range(n), [label for label, _, _ in reversed(rows)])
    top.set_xlim(0, view)
    top.set_ylim(-0.55, n - 0.45)
    top.tick_params(axis="x", labelbottom=False, length=0)
    xs, ys = ipc(ticks, ivals)
    xs = [x for x in xs if x <= view]
    ys = ys[: len(xs)]
    bot.fill_between(xs, ys, color="#5c4e78", alpha=0.14, linewidth=0)
    bot.plot(xs, ys, color="#5c4e78", linewidth=1.15)
    bot.set_xlim(0, view)
    peak = max(ys) if ys else 0.0
    ylim = 1.0 if peak <= 1.0 else math.ceil(peak * 2) / 2.0
    bot.set_ylim(0, ylim)
    bot.set_yticks([i * 0.5 for i in range(int(round(ylim * 2)) + 1)])
    bot.set_ylabel("Instr IPC (w0)")
    bot.set_xlabel("tick (issue → completion)")
    bot.yaxis.grid(True, color="#eef0f2", linewidth=0.7)
    bot.set_axisbelow(True)
    for ax in (top, bot):
        ax.tick_params(length=0, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    out = OUT / out_name
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"{out}  VF {ticks}  IPC {len(ivals) / ticks:.2f}")


def main():
    OUT.mkdir(exist_ok=True)
    draw(TL_BIN, "TileLang buffer, N = 4096, K = 32, 256 threads, VF 132303", "cmp_tilelang.png")
    draw(CCE_BIN, "CCE, N = 4096, K = 32, 256 threads, VF 55100", "cmp_cce.png")
    draw(
        TL_BIN,
        f"TileLang buffer, first launch, first {ZOOM_TICKS} ticks",
        "cmp_tilelang_zoom.png",
        xlim=ZOOM_TICKS,
    )
    draw(
        CCE_BIN,
        f"CCE, first launch, first {ZOOM_TICKS} ticks",
        "cmp_cce_zoom.png",
        xlim=ZOOM_TICKS,
    )


if __name__ == "__main__":
    main()
