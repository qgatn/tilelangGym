"""Compile lds-boost V kernels → <outdir>/<tag>/kernel.so

Usage: compile_flip.py <v0_base|v1_burst|v2_pair|v3_row> N K T <outdir> <nosync 0|1>
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kernels_flip import BUILDERS, GM_DTYPE  # noqa: E402

SRC = "/home/onasismelchiorjeyakumar/workspace/tilelang/src"

_CT = {"float16": "half", "uint32": "uint32_t", "uint64": "uint64_t"}


def host_wrapper(smem: int, gm_dtype: str) -> str:
    ct = _CT[gm_dtype]
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
extern "C" TL_EXPORT int call(void* a, void* x, void* o, void* stream) {{
  main_kernel<<<(1 * 1 * 1), {smem}, stream>>>((__gm__ {ct}*)a, (__gm__ float*)o, (__gm__ half*)x);
  return 0;
}}
"""


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 6:
        print("usage: compile_flip.py <name> N K T <outdir> <nosync>")
        return 2
    kind, N, K, T_, outdir, ns = (
        args[0], int(args[1]), int(args[2]), int(args[3]), args[4], args[5] == "1",
    )
    name = f"{kind}-n{N}-k{K}-t{T_}" + ("-nosync" if ns else "")
    out = Path(outdir) / name
    out.mkdir(parents=True, exist_ok=True)
    kernel = BUILDERS[kind](N, K, T_)
    pass_configs = {"tl.disable_data_race_check": True}
    if ns:
        pass_configs["tl.disable_thread_storage_sync"] = True
    try:
        compiled = tilelang.compile(
            kernel, target="ascend", out_idx=-1,
            execution_backend="cython", pass_configs=pass_configs,
        )
        src = compiled.get_kernel_source()
        smem = None
    except RuntimeError as e:
        text = str(e)
        marker = "#include <tl_templates/ascend/common.h>"
        idx = text.find(marker)
        if idx < 0:
            raise
        src = text[idx:].strip()
        m = re.search(r"<<<\(1 \* 1 \* 1\),\s*(\d+),\s*stream>>>", text)
        smem = int(m.group(1)) if m else None
    if smem is None:
        smem = (N // 2) * K * 2 + K * 2 + N * 4
    (out / "kernel.cpp").write_text(src)
    asc = out / "kernel.asc"
    asc.write_text(src + host_wrapper(smem, GM_DTYPE[kind]))
    subprocess.run(
        ["bisheng", "-O2", "-fPIC", "-std=c++20",
         "-mllvm", "-cce-aicore-dcpreload-args=false",
         "--npu-arch=dav-3510",
         "-Wl,--no-mmap-output-file", "--shared",
         str(asc), "-I", SRC, "-o", str(out / "kernel.so")],
        check=True,
    )
    print(f"OK {name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
