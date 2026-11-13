"""Compile a GEMV schedule kernel → schedules/runs/<name>-…/kernel.so

Usage:
  python compile.py <name> N K [threads] [--nosync]

Names (see kernels.py — one TileLang source for all schedules):
  published (§3 in SV3-theoretical.md):
    methodo | serial | warp_pipe | dbl_buf | dbl_buf_quad | quad_acc
  exploration (§5):
    dbl_buf_twice | vfsplit | quad_reord | quad_ilv
  base case (§3.1):
    ub   → kernels.py::build_thread_ub

--nosync sets tl.disable_thread_storage_sync=True (drops compiler-inserted
asc_syncthreads). Safe for these schedules: out_ub writes are thread-private;
a_ub/x_ub are MTE2-fed with pipe syncs; VF→MTE3 order uses asc_sync_notify/wait.

Working recipe on this box: tilelang.compile for codegen (cython adapter may
fail), recover device source from the error text, then bisheng-compile a .asc
file with the host wrapper launch.py expects (init / call / get_last_error).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang

from kernels import (
    build_thread_ub,
    build_ub_methodo,
    build_ub_serial,
    build_ub_warp_pipe,
    build_ub_dbl_buf,
    build_ub_dbl_buf_quad,
    build_ub_quad_reord,
    build_ub_quad_ilv,
    build_ub_quad_acc,
    build_ub_vfsplit,
    build_ub_dbl_buf_twice,
)

HERE = Path(__file__).resolve().parent
BUILDERS = {
    "ub": build_thread_ub,
    "methodo": build_ub_methodo,
    "serial": build_ub_serial,
    "warp_pipe": build_ub_warp_pipe,
    "dbl_buf": build_ub_dbl_buf,
    "dbl_buf_quad": build_ub_dbl_buf_quad,
    "quad_reord": build_ub_quad_reord,
    "quad_ilv": build_ub_quad_ilv,
    "quad_acc": build_ub_quad_acc,
    "vfsplit": build_ub_vfsplit,
    "dbl_buf_twice": build_ub_dbl_buf_twice,
}
SRC = "/home/onasismelchiorjeyakumar/workspace/tilelang/src"


def _host_wrapper(smem: int) -> str:
    return f"""

#ifdef _WIN32
#define TL_EXPORT __declspec(dllexport)
#else
#define TL_EXPORT
#endif
#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];
extern "C" TL_EXPORT const char* get_last_error() {{ return error_buf; }}
extern "C" TL_EXPORT int init() {{ error_buf[0] = '\\0'; return 0; }}
extern "C" TL_EXPORT int call(void* A, void* X, void* Out, void* stream) {{
  main_kernel<<<(1 * 1 * 1), {smem}, stream>>>((__gm__ half*)A, (__gm__ float*)Out, (__gm__ half*)X);
  return 0;
}}
"""


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    nosync = "--nosync" in args
    args = [a for a in args if a != "--nosync"]
    if len(args) not in (3, 4):
        print(
            "usage: compile.py <name> N K [threads] [--nosync]\n"
            "  name: ub|methodo|serial|warp_pipe|dbl_buf|dbl_buf_quad|quad_acc\n"
            "        |dbl_buf_twice|vfsplit|quad_reord|quad_ilv",
            flush=True,
        )
        return 2
    kind, N, K = args[0], int(args[1]), int(args[2])
    threads = int(args[3]) if len(args) == 4 else None
    if kind not in BUILDERS:
        print(f"unknown name {kind!r}; see --help names above", flush=True)
        return 2
    name = f"{kind}-n{N}-k{K}"
    if threads is not None:
        name = f"{name}-t{threads}"
    if nosync:
        name += "-nosync"
    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    kernel = BUILDERS[kind](N, K, threads) if threads is not None else BUILDERS[kind](N, K)
    smem = None
    pass_configs = {"tl.disable_data_race_check": True}
    if nosync:
        pass_configs["tl.disable_thread_storage_sync"] = True
    try:
        compiled = tilelang.compile(
            kernel,
            target="ascend",
            out_idx=-1,
            execution_backend="cython",
            pass_configs=pass_configs,
        )
        src = compiled.get_kernel_source()
    except RuntimeError as e:
        # The adapter build fails; the device source is embedded in the error.
        text = str(e)
        marker = "#include <tl_templates/ascend/common.h>"
        idx = text.find(marker)
        if idx < 0:
            raise
        src = text[idx:].strip()
        import re
        m = re.search(r"<<<\(1 \* 1 \* 1\),\s*(\d+),\s*stream>>>", text)
        if m:
            smem = int(m.group(1))
    if smem is None:
        # a_ub (slab,K) fp16 + x_ub (K) fp16 + out_ub (N) fp32; slab = N/2
        smem = (N // 2) * K * 2 + K * 2 + N * 4
    (out / "kernel.cpp").write_text(src)
    asc = out / "kernel.asc"
    asc.write_text(src + _host_wrapper(smem))

    cmd = [
        "bisheng", "-O2", "-fPIC", "-std=c++20",
        "-mllvm", "-cce-aicore-dcpreload-args=false",
        "--npu-arch=dav-3510",
        "-Wl,--no-mmap-output-file", "--shared",
        str(asc), "-I", SRC, "-o", str(out / "kernel.so"),
    ]
    subprocess.run(cmd, check=True)
    print(f"OK {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
