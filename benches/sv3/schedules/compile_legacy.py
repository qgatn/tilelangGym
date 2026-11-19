"""Compile one ANALYSIS kernel. Usage: compile_legacy.py thread|ub|warp|simd N K [threads]"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

torch.utils.rename_privateuse1_backend("npu")

import tilelang

from kernels_legacy import build_simd, build_thread, build_thread_ub, build_warp

HERE = Path(__file__).resolve().parent
BUILDERS = {"thread": build_thread, "ub": build_thread_ub, "warp": build_warp, "simd": build_simd}
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
    if len(args) not in (3, 4):
        print("usage: compile.py thread|ub|warp|simd N K [threads]", flush=True)
        return 2
    kind, N, K = args[0], int(args[1]), int(args[2])
    threads = int(args[3]) if len(args) == 4 else None
    if threads is not None and kind not in ("thread", "ub"):
        print("threads is only valid for thread and ub", flush=True)
        return 2
    name = f"{kind}-n{N}-k{K}"
    if threads is not None:
        name = f"{name}-t{threads}"
    out = HERE / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    kernel = BUILDERS[kind](N, K, threads) if threads is not None else BUILDERS[kind](N, K)
    smem = None
    try:
        compiled = tilelang.compile(
            kernel,
            target="ascend",
            out_idx=-1,
            execution_backend="cython",
            pass_configs={"tl.disable_data_race_check": True},
        )
        src = compiled.get_kernel_source()
        (out / "kernel.cpp").write_text(src)
        shutil.copy(compiled.adapter.libpath, out / "kernel.so")
        print(f"OK {name}", flush=True)
        return 0
    except RuntimeError as e:
        text = str(e)
        marker = "#include <tl_templates/ascend/common.h>"
        idx = text.find(marker)
        if idx < 0:
            raise
        src = text[idx:].strip()
        m = re.search(r"<<<\(1 \* 1 \* 1\),\s*(\d+),\s*stream>>>", text)
        if m:
            smem = int(m.group(1))
    if smem is None:
        smem = N * K * 2 + K * 2 + N * 4
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
