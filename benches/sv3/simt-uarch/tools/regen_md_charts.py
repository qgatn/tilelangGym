#!/usr/bin/env python3
"""Regenerate every PNG referenced by SV3 *.md with issue→completion bars.

Deletes obsolete PNGs in simt-uarch/plots and SV3/plots that are not
referenced by any markdown after regeneration.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # SV3/
TOOLS = Path(__file__).resolve().parent
PLOTS = TOOLS.parent / "plots"
ANALYSIS_PLOTS = ROOT / "plots"
SINGLE = ROOT / "single-op" / "build"
SCHED = ROOT / "schedules" / "runs"
PYV = Path("/home/onasismelchiorjeyakumar/venv/bin/python")
IPC = TOOLS / "instr_vs_ipc.py"
WARP = TOOLS / "warpchart.py"


def pick_single(tag: str) -> Path | None:
    hits = sorted((SINGLE / f"npusim_{tag}").rglob("instr.bin")) if (SINGLE / f"npusim_{tag}").exists() else []
    if not hits:
        hits = sorted(SINGLE.glob(f"npusim_{tag}/**/instr.bin"))
    # also allow direct find
    if not hits:
        hits = sorted(Path(SINGLE).rglob("instr.bin"))
        hits = [h for h in hits if f"npusim_{tag}" in str(h)]
    return hits[0] if hits else None


def pick_sched(name: str) -> Path | None:
    d = SCHED / name
    if not d.exists():
        return None
    hits = sorted(d.rglob("instr.bin"))
    return hits[0] if hits else None


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd[-6:]))
    subprocess.run(cmd, check=True)


def pair(bin_path: Path, chart: str, label: str, kind: str = "simt", vf: int = 0, warps: bool = True):
    out = PLOTS / chart
    run([str(PYV), str(IPC), str(bin_path), kind, str(out), "--label", label, "--vf", str(vf)])
    if warps and kind == "simt" and vf >= 0:
        run([str(PYV), str(WARP), str(bin_path), str(PLOTS / f"{chart}_warps"),
             "--label", f"{label}, warp activity", "--vf", str(vf)])


def md_refs() -> set[str]:
    refs = set()
    for p in ROOT.rglob("*.md"):
        t = p.read_text(errors="ignore")
        for m in re.finditer(r"\(plots/([^)]+\.png)\)", t):
            refs.add(m.group(1))
    return refs


def main() -> int:
    PLOTS.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PLOTS.mkdir(parents=True, exist_ok=True)
    missing = []

    # ---- SV3-theoretical schedule charts (2nd VF except base) ----
    schedule_charts = [
        ("ub-n2048-k8-t256", "case_ub_k8", "Figure 1. Base case, ub-n2048-k8-t256", 0),
        ("methodo-n2048-k8-t256", "methodo_ub_k8", "Figure 2. Op-matched, methodo-n2048-k8-t256", 1),
        ("serial-n2048-k8-t256", "serial_ub_k8", "Figure 3. Serial schedule, serial-n2048-k8-t256", 1),
        ("warp_pipe-n2048-k8-t256", "warp_pipe_ub_k8", "Figure 4. Warp-pipeline, warp_pipe-n2048-k8-t256", 1),
        ("dbl_buf-n2048-k8-t256", "dbl_buf_ub_k8", "Figure 5. Double-buffered strip pipeline, dbl_buf-n2048-k8-t256", 1),
        ("dbl_buf_quad-n2048-k8-t256", "dbl_buf_quad_ub_k8", "Figure 6. Quad-buffered strip pipeline, dbl_buf_quad-n2048-k8-t256", 1),
        ("quad_acc-n2048-k8-t256-nosync", "quad_acc_ub_k8", "Figure 7. Quad-acc (ILP chains), quad_acc-n2048-k8-t256-nosync", 1),
    ]
    for run_name, chart, label, vf in schedule_charts:
        b = pick_sched(run_name)
        if not b:
            missing.append(chart)
            print(f"MISS schedule {run_name}")
            continue
        pair(b, chart, label, vf=vf, warps=True)

    # ---- simt-uarch.md + lds_fp32_vs_bf16.md single-op charts ----
    single = [
        # (tag, chart, label, vf, warps)
        ("pure_ldsk0_t256", "lds_s1_scalar_t256", "LDS S1 scalar fp16 (pure), 256 threads", 0, True),
        ("pure_ldsk1_t256", "lds_s2_u32_t256", "LDS S2 uint32 pack (pure), 256 threads", 0, True),
        ("pure_ldsk3_t256", "lds_s4_bf16x2_t256", "LDS S4 bfloat16x2 typed (pure), 256t", 0, True),
        ("pure_f2fk0_t256", "f2f_s1_scalar_t256", "F2F S1 scalar convert (pure), 256t", 0, False),
        ("pure_f2fk2_t256", "f2f_s3_word_t256", "F2F S3 word-preload (pure), 256t", 0, False),
        ("pure_f2fk3_t256", "f2f_s4_barrier_t256", "F2F S4 barrier-aligned (pure), 256t", 0, True),
        ("pure_ffmak0_t256", "ffma_s1_scalar_t256", "FFMA S1 scalar (pure), 256 threads", 0, False),
        ("pure_ffmak2_t256", "ffma_s3_barrier_t256", "FFMA S3 barrier-aligned (pure), 256t", 0, True),
        ("hide32_sts_t256", "sts_s1_t256", "STS S1 coalesced (pure), 256 threads", 0, True),
        ("hide32_sts_t512", "hide32_sts_t512", "STS coalesced, 512 threads", 0, False),
        ("hide32_sts_t1024", "hide32_sts_t1024", "STS coalesced, 1024 threads", 0, False),
        ("pure_f2f_s4_barrier_t512", "pure_f2f_s4_barrier_t512", "F2F S4 barrier, 512 threads", 0, True),
        ("pure_f2f_s4_barrier_t1024", "pure_f2f_s4_barrier_t1024", "F2F S4 barrier, 1024 threads", 0, True),
        ("pure_ffma_s3_barrier_t512", "pure_ffma_s3_barrier_t512", "FFMA S3 barrier, 512 threads", 0, False),
        ("pure_ffma_s3_barrier_t1024", "pure_ffma_s3_barrier_t1024", "FFMA S3 barrier, 1024 threads", 0, False),
        ("lds_peak_fp32_t256_n24576", "lds_fp32_peak_t256", "LDS fp32 peak, 256 threads", 0, True),
        ("lds_peak_fp32_t512_n49152", "lds_fp32_peak_t512", "LDS fp32 peak, 512 threads", 0, True),
        ("lds_pure_fp32_t256", "lds_fp32_t256", "LDS fp32 scalar (pure), 256 threads", 0, False),
        ("lds_pure_bf16_t256", "lds_bf16_scalar_t256", "LDS bf16 scalar (pure), 256 threads", 0, False),
        ("lds_pure_bf16x2_t256", "lds_bf16x2_matched_t256", "LDS bf16x2 matched (pure), 256 threads", 0, False),
        # I-cache
        ("pure_ffmak0_t256", "icache_ffma_s1", "FFMA S1 (I-cache)", 0, False),
        ("pure_ffmak1_t256", "icache_ffma_s2", "FFMA S2 (I-cache)", 0, False),
        ("pure_ffmak2_t256", "icache_ffma_s3_single", "FFMA S3 single (I-cache)", 0, True),
        ("ffma_s3_dual_vf_t256", "icache_ffma_s3_dual", "FFMA S3 dual VF0 (I-cache)", 0, False),
    ]
    for tag, chart, label, vf, warps in single:
        b = pick_single(tag)
        if not b:
            missing.append(chart)
            print(f"MISS single {tag} -> {chart}")
            continue
        pair(b, chart, label, vf=vf, warps=warps)

    # dual full window (both VFs)
    b = pick_single("ffma_s3_dual_vf_t256")
    if b:
        pair(b, "icache_ffma_s3_dual_full", "FFMA S3 dual full window (I-cache)", vf=-1, warps=False)
    else:
        missing.append("icache_ffma_s3_dual_full")

    # ---- ANALYSIS.md charts via updated plot_charts / plot_compare ----
    try:
        run([str(PYV), str(ROOT / "schedules" / "plot_charts.py")])
    except subprocess.CalledProcessError as e:
        print("WARN plot_charts failed:", e)
        missing.append("ANALYSIS plot_charts")
    try:
        run([str(PYV), str(ROOT / "cce" / "plot_compare.py")])
    except subprocess.CalledProcessError as e:
        print("WARN plot_compare failed:", e)
        missing.append("ANALYSIS plot_compare")

    # ---- delete obsolete PNGs not referenced by any MD ----
    refs = md_refs()
    print(f"\nMD references {len(refs)} PNGs")
    deleted = []
    for folder in (PLOTS, ANALYSIS_PLOTS):
        if not folder.exists():
            continue
        for png in folder.glob("*.png"):
            if png.name not in refs:
                png.unlink()
                deleted.append(f"{folder.name}/{png.name}")
    print(f"Deleted {len(deleted)} obsolete PNGs")
    for d in deleted[:30]:
        print("  -", d)
    if len(deleted) > 30:
        print(f"  ... +{len(deleted)-30} more")

    # verify all refs exist
    still_missing = []
    for name in sorted(refs):
        if (PLOTS / name).exists() or (ANALYSIS_PLOTS / name).exists():
            continue
        still_missing.append(name)
    print("\nStill missing from disk:", still_missing or "none")
    print("Regen missing bins:", missing or "none")
    return 1 if still_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
